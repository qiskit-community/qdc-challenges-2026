"""
Fragment chemical-identity module for the QDC 2026 Proteasome challenge.

IAO atomic fragmentation (vayesta ``add_all_atomic_fragments``) assigns exactly
one fragment per atom in the input geometry.  Fragment *i* therefore corresponds
to atom *i* in the XYZ file, and its center element is recorded directly in the
HDF5 dumpfile as the ``name`` attribute of the ``fragment_i`` group.

This module:
- reads those element assignments from the HDF5 ``name`` attrs (ground truth),
- reads atomic coordinates from the XYZ geometry file,
- builds a bond graph using covalent-radius thresholds,
- derives ``bonded_neighbors``, ``chemical_environment``, and
  ``contains_transition_metal`` for every fragment.

No orbital-projection analysis is performed; the metadata describes only the
chemically defined *fragment center* atom, not the embedded DMET cluster that
surrounds it.  The cluster orbitals include bath contributions from neighboring
atoms; those are NOT assigned a literal atom composition here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "build_fragment_chemistry",
    "load_fragment_chemistry_json",
    "save_fragment_chemistry_json",
]

# ---------------------------------------------------------------------------
# Covalent radii (Alvarez 2008 / Pyykkö & Atsumi 2009, Å)
# ---------------------------------------------------------------------------
_COVALENT_RADIUS: Dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.77, "N": 0.71,
    "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76,
    # 3d transition metals
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.61,
    "Fe": 1.52, "Co": 1.50, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
    "I": 1.39,
    # 4d/5d transition metals (common)
    "Mo": 1.54, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45,
    "Pt": 1.36, "Au": 1.36,
}

_TRANSITION_METALS = frozenset({
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
})

# Tolerance factor: bond detected when dist < TOL*(r_i + r_j)
_BOND_TOL = 1.3


def _read_xyz(xyz_path: Path) -> List[Dict[str, Any]]:
    """Parse a standard XYZ file, return list of {sym, xyz} dicts (0-indexed)."""
    lines = Path(xyz_path).read_text().splitlines()
    n_atoms = int(lines[0].strip())
    atoms: List[Dict[str, Any]] = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if not parts:
            continue
        sym = parts[0]
        coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        atoms.append({"sym": sym, "xyz": coords})
    if len(atoms) != n_atoms:
        raise ValueError(
            f"XYZ header says {n_atoms} atoms but found {len(atoms)} coordinate lines."
        )
    return atoms


def _build_bonds(atoms: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    """Return adjacency list (atom index → sorted list of bonded atom indices)."""
    n = len(atoms)
    bonds: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        ri = _COVALENT_RADIUS.get(atoms[i]["sym"], 0.77)
        for j in range(i + 1, n):
            rj = _COVALENT_RADIUS.get(atoms[j]["sym"], 0.77)
            dist = float(np.linalg.norm(atoms[i]["xyz"] - atoms[j]["xyz"]))
            if dist < _BOND_TOL * (ri + rj):
                bonds[i].append(j)
                bonds[j].append(i)
    for i in bonds:
        bonds[i].sort()
    return bonds


def _chemical_environment(center_sym: str, neighbor_syms: List[str]) -> str:
    """Build a concise chemical-environment descriptor.

    Format: ``ELEMENT(N₁E₁N₂E₂…)`` where Eₖ is an element symbol and Nₖ its
    count (omitted when 1).  Heavy atoms come first (alphabetical), then H.

    Examples::
        C with neighbors [C, C, H]     → "C(C2H)"
        N with neighbors [C, C, H]     → "N(C2H)"
        B with neighbors [O, O, C]     → "B(CO2)"   (O×2 alphabetically after C)
        Cl with neighbor [C]           → "Cl(C)"
    """
    counts: Dict[str, int] = {}
    for s in neighbor_syms:
        counts[s] = counts.get(s, 0) + 1

    heavy_syms = sorted(s for s in counts if s != "H")
    parts = []
    for s in heavy_syms:
        c = counts[s]
        parts.append(f"{s}{c}" if c > 1 else s)
    if "H" in counts:
        c = counts["H"]
        parts.append(f"H{c}" if c > 1 else "H")

    return f"{center_sym}({''.join(parts)})"


def build_fragment_chemistry(
    h5_path: Path,
    xyz_path: Path,
    ligand: str,
) -> List[Dict[str, Any]]:
    """Build per-fragment chemical metadata for all fragments in *h5_path*.

    Parameters
    ----------
    h5_path : Path
        Stage 2 EWF dumpfile (``*_ewf_dumpfile.h5``).
    xyz_path : Path
        Molecular geometry in XYZ format.
    ligand : str
        Ligand name (stored verbatim in the returned rows).

    Returns
    -------
    List of dicts with one entry per fragment, sorted by fragment_id:

    ``ligand``, ``fragment_id``, ``center_atom_idx``, ``center_element``,
    ``bonded_neighbor_indices``, ``bonded_neighbor_elements``,
    ``chemical_environment``, ``contains_transition_metal``,
    ``norb_dmet_cluster``, ``assignment_method``

    Notes
    -----
    * ``center_atom_idx`` and ``center_element`` come directly from the HDF5
      ``name`` attribute (ground truth from vayesta IAO fragmentation).
    * ``bonded_neighbor_*`` describe atoms bonded to the *fragment center* only,
      not the full DMET cluster bath.
    * ``norb_dmet_cluster`` is reported so the caller can distinguish the
      chemically defined fragment from its embedding bath.
    * ``assignment_method = "iao_atomic_hdf5_name"`` records that the element
      was read from the HDF5 dumpfile, not guessed.
    """
    import h5py

    atoms = _read_xyz(xyz_path)
    bonds = _build_bonds(atoms)
    n_atoms = len(atoms)

    rows: List[Dict[str, Any]] = []

    with h5py.File(h5_path, "r") as hf:
        frag_keys = sorted(
            [k for k in hf.keys() if k.startswith("fragment_")],
            key=lambda k: int(k.split("_")[1]),
        )
        for fk in frag_keys:
            g = hf[fk]
            fid = int(fk.split("_")[1])
            attrs = dict(g.attrs)

            # Element from HDF5 name attr (authoritative — set by vayesta IAO)
            center_element: str = str(attrs.get("name", "?")).strip()

            # The i-th IAO atomic fragment always corresponds to atom i.
            # Validate: atom symbol from XYZ must match HDF5 name.
            center_atom_idx: int = fid
            if center_atom_idx < n_atoms:
                xyz_sym = atoms[center_atom_idx]["sym"]
                if xyz_sym != center_element:
                    # Inconsistency — report but do not fabricate
                    center_element = f"?({center_element}/{xyz_sym})"
                    center_atom_idx_verified = False
                else:
                    center_atom_idx_verified = True
            else:
                center_atom_idx_verified = False  # XYZ has fewer atoms

            # Bonded neighbors of the center atom
            if center_atom_idx_verified:
                nbr_indices = bonds[center_atom_idx]
                nbr_elements = [atoms[j]["sym"] for j in nbr_indices]
            else:
                nbr_indices = []
                nbr_elements = []

            # Chemical environment descriptor
            env = (
                _chemical_environment(center_element, nbr_elements)
                if center_atom_idx_verified
                else center_element
            )

            # Transition metal flag — center atom only (bath may differ)
            has_tm = center_element in _TRANSITION_METALS

            norb_dmet = int(attrs.get("norb_dmet_cluster",
                                      attrs.get("norb_dmet", -1)))

            rows.append({
                "ligand":                   ligand,
                "fragment_id":              fid,
                "center_atom_idx":          center_atom_idx,
                "center_element":           center_element
                                            if center_atom_idx_verified else f"?",
                "bonded_neighbor_indices":  nbr_indices,
                "bonded_neighbor_elements": nbr_elements,
                "chemical_environment":     env,
                "contains_transition_metal": bool(has_tm),
                "norb_dmet_cluster":        norb_dmet,
                "assignment_method":        "iao_atomic_hdf5_name",
            })

    return rows


def load_fragment_chemistry_json(json_path: Path) -> List[Dict[str, Any]]:
    """Load pre-computed fragment chemistry from a JSON file."""
    with Path(json_path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data


def save_fragment_chemistry_json(rows: List[Dict[str, Any]], json_path: Path) -> None:
    """Persist fragment chemistry rows to *json_path*."""
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(json_path).open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
