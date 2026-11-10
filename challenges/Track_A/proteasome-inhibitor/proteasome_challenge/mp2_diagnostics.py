"""
mp2_diagnostics — MP2 doubles-amplitude diagnostics for embedded EWF clusters.

Each embedded cluster already has a Fock matrix (canonical cluster MOs) and
two-electron repulsion integrals (ERIs) stored in the Stage 2 HDF5 dump.
This module computes MP2 correlation-hardness metrics from those stored
quantities — no additional Vayesta/PySCF run is required.

ERI convention
--------------
PySCF chemist notation:  eri[p, q, r, s] = (pq|rs) = ∫ φ_p(1)φ_q(1) 1/r₁₂ φ_r(2)φ_s(2)

Amplitude formula
-----------------
    t_ijab = <ij||ab> / (ε_i + ε_j - ε_a - ε_b)
where
    <ij||ab> = (ia|jb) - (ib|ja)          antisymmetrized integral
    ε_p      = F_pp                         diagonal Fock elements (canonical MOs)

Orbital ordering within the stored Fock/ERI arrays: occupied first (0…nocc-1),
then virtual (nocc…nocc+nvir-1).  This follows the convention used by both
Vayesta and PySCF for the embedded-cluster Hamiltonians.

Exported functions
------------------
compute_fragment_mp2_diagnostics(fock, eris, nocc, nvir)
    Core numerical computation for a single fragment.

compute_all_diagnostics(h5_path, ligand)
    Process every fragment in one HDF5 dumpfile and return a list of dicts.

load_or_compute_diagnostics(ligand, h5_path, cache_path)
    Load from a CSV cache if fresh; otherwise recompute and save.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Columns in the output CSV / diagnostic dicts
_DIAGNOSTIC_COLS = [
    "ligand",
    "fragment_id",
    "n_orb",
    "n_elec",
    "n_occ",
    "n_vir",
    "mp2_max_abs_t2",
    "mp2_t2_norm",
    "mp2_t2_rms",
    "mp2_corr_energy_ha",
    "hf_gap_ha",
    "diagnostic_status",
    "diagnostic_error",
]

# Threshold below which an orbital-energy denominator is considered degenerate.
_MIN_DENOM_THRESHOLD = 1.0e-10


def compute_fragment_mp2_diagnostics(
    fock: np.ndarray,
    eris: np.ndarray,
    nocc: int,
    nvir: int,
) -> dict[str, float]:
    """Compute MP2 doubles-amplitude diagnostics for one embedded cluster.

    Parameters
    ----------
    fock : (norb, norb) ndarray
        Fock matrix in the canonical cluster MO basis (nearly diagonal).
    eris : (norb, norb, norb, norb) ndarray
        Two-electron repulsion integrals in PySCF chemist notation
        ``(pq|rs)``.  Orbital ordering: occupied (0…nocc-1) then virtual
        (nocc…nocc+nvir-1).
    nocc : int
        Number of occupied spatial orbitals.
    nvir : int
        Number of virtual spatial orbitals.

    Returns
    -------
    dict with keys:

    mp2_max_abs_t2 : float
        Primary screening metric C_F = max_{ijab} |t_ij^ab|.
    mp2_t2_norm : float
        Frobenius norm of the full t₂ tensor.
    mp2_t2_rms : float
        RMS amplitude = ||t₂||_F / sqrt(nocc² × nvir²).
    mp2_corr_energy_ha : float
        MP2 correlation energy = (1/4) Σ_{ijab} t_ij^ab <ij||ab>  (Ha).
    hf_gap_ha : float
        HOMO–LUMO orbital-energy gap ε_{LUMO} − ε_{HOMO}  (Ha).

    Raises
    ------
    ValueError
        If a near-zero orbital-energy denominator is detected.
    """
    fock = np.asarray(fock, dtype=np.float64)
    eris = np.asarray(eris, dtype=np.float64)

    eps = np.diag(fock)
    eps_o = eps[:nocc]
    eps_v = eps[nocc : nocc + nvir]

    # ── Orbital energy denominators ε_i + ε_j − ε_a − ε_b ───────────────────
    # Shape: (nocc, nocc, nvir, nvir)
    denom = (
        eps_o[:, None, None, None]
        + eps_o[None, :, None, None]
        - eps_v[None, None, :, None]
        - eps_v[None, None, None, :]
    )

    min_denom = float(np.min(np.abs(denom)))
    if min_denom < _MIN_DENOM_THRESHOLD:
        raise ValueError(
            f"Near-zero orbital energy denominator detected: {min_denom:.3e} Ha. "
            "This may indicate quasi-degenerate occupied and virtual orbitals in the "
            "embedded cluster, which would make the MP2 expansion ill-defined."
        )

    # ── Antisymmetrized integrals <ij||ab> = (ia|jb) − (ib|ja) ──────────────
    # Extract occ–vir–occ–vir block: eri_ovov[i, a, j, b] = (ia|jb)
    # Shape (nocc, nvir, nocc, nvir)
    eri_ovov = eris[:nocc, nocc : nocc + nvir, :nocc, nocc : nocc + nvir]

    # Reorder to (nocc, nocc, nvir, nvir)
    iajb = np.einsum("iajb->ijab", eri_ovov)   # (ia|jb)
    ibja = np.einsum("ibja->ijab", eri_ovov)   # (ib|ja)
    g_ijab = iajb - ibja                        # <ij||ab>

    # ── MP2 doubles amplitudes t_ijab = <ij||ab> / denom ─────────────────────
    t2 = g_ijab / denom

    # ── Diagnostics ───────────────────────────────────────────────────────────
    mp2_energy  = 0.25 * float(np.einsum("ijab,ijab->", t2, g_ijab))
    max_abs_t2  = float(np.max(np.abs(t2)))
    t2_norm     = float(np.linalg.norm(t2))
    n_elem      = t2.size
    t2_rms      = float(t2_norm / np.sqrt(n_elem))
    hf_gap      = float(eps_v[0] - eps_o[-1])

    return {
        "mp2_max_abs_t2":     max_abs_t2,
        "mp2_t2_norm":        t2_norm,
        "mp2_t2_rms":         t2_rms,
        "mp2_corr_energy_ha": mp2_energy,
        "hf_gap_ha":          hf_gap,
    }


def compute_all_diagnostics(h5_path: "str | Path", ligand: str) -> list[dict[str, Any]]:
    """Compute MP2 diagnostics for every fragment in one HDF5 dumpfile.

    Parameters
    ----------
    h5_path : path to the Stage 2 HDF5 dumpfile.
    ligand  : ``"ixazomib"`` or ``"bortezomib"`` — written into each row.

    Returns
    -------
    list of dicts, one per fragment, with keys matching ``_DIAGNOSTIC_COLS``.
    The list is sorted by fragment_id.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for diagnostic extraction.") from exc

    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 dumpfile not found: {h5_path}")

    rows: list[dict[str, Any]] = []

    with h5py.File(h5_path, "r") as hf:
        frag_keys = sorted(hf.keys(), key=lambda k: int(hf[k].attrs["id"]))
        for key in frag_keys:
            g = hf[key]
            fid   = int(g.attrs["id"])
            norb  = int(g.attrs["norb"])
            nocc  = int(g.attrs["nocc"])
            nvir  = int(g.attrs["nvir"])
            n_elec = 2 * nocc

            row: dict[str, Any] = {
                "ligand":      ligand,
                "fragment_id": fid,
                "n_orb":       norb,
                "n_elec":      n_elec,
                "n_occ":       nocc,
                "n_vir":       nvir,
            }

            try:
                fock = g["fock"][:]
                eri  = g["eris"][:]
                diag = compute_fragment_mp2_diagnostics(fock, eri, nocc, nvir)
                row.update(diag)
                row["diagnostic_status"] = "ok"
                row["diagnostic_error"]  = ""
            except Exception as exc:
                logger.warning(
                    "MP2 diagnostics failed for %s fragment %d: %s",
                    ligand, fid, exc,
                )
                row.update({
                    "mp2_max_abs_t2":     float("nan"),
                    "mp2_t2_norm":        float("nan"),
                    "mp2_t2_rms":         float("nan"),
                    "mp2_corr_energy_ha": float("nan"),
                    "hf_gap_ha":          float("nan"),
                    "diagnostic_status":  "failed",
                    "diagnostic_error":   str(exc)[:500],
                })

            rows.append(row)

    rows.sort(key=lambda r: r["fragment_id"])
    return rows


def save_diagnostics_csv(rows: list[dict[str, Any]], path: "str | Path") -> None:
    """Write diagnostic rows to a CSV file.

    Parameters
    ----------
    rows : output of ``compute_all_diagnostics`` (one or more ligands combined).
    path : destination CSV path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=_DIAGNOSTIC_COLS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            # Serialize floats with full double precision; NaN → empty string
            out = {}
            for col in _DIAGNOSTIC_COLS:
                val = row.get(col, "")
                if isinstance(val, float):
                    out[col] = "" if (val != val) else f"{val:.15g}"
                else:
                    out[col] = val
            writer.writerow(out)


def load_diagnostics_csv(path: "str | Path") -> list[dict[str, Any]]:
    """Load a diagnostics CSV written by ``save_diagnostics_csv``.

    Empty fields are decoded as ``float('nan')`` for numeric columns.

    Returns
    -------
    list of dicts with the same structure as ``compute_all_diagnostics``.
    """
    _FLOAT_COLS = {
        "mp2_max_abs_t2", "mp2_t2_norm", "mp2_t2_rms",
        "mp2_corr_energy_ha", "hf_gap_ha",
    }
    _INT_COLS = {"fragment_id", "n_orb", "n_elec", "n_occ", "n_vir"}

    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out: dict[str, Any] = {}
            for k, v in row.items():
                if k in _INT_COLS:
                    out[k] = int(v) if v.strip() else None
                elif k in _FLOAT_COLS:
                    out[k] = float(v) if v.strip() else float("nan")
                else:
                    out[k] = v
            rows.append(out)
    return rows


def load_or_compute_diagnostics(
    ligand: str,
    h5_path: "str | Path",
    cache_path: "str | Path",
    *,
    force_recompute: bool = False,
) -> list[dict[str, Any]]:
    """Return diagnostics from cache if available, else compute and cache.

    Parameters
    ----------
    ligand         : ``"ixazomib"`` or ``"bortezomib"``.
    h5_path        : Stage 2 HDF5 dumpfile.
    cache_path     : CSV cache path.  If the file exists and covers all
                     expected fragment IDs it is loaded directly.
    force_recompute: If True, always recompute even if the cache exists.

    Returns
    -------
    list of diagnostic dicts for *ligand*.
    """
    cache_path = Path(cache_path)

    if not force_recompute and cache_path.is_file():
        existing = [r for r in load_diagnostics_csv(cache_path) if r["ligand"] == ligand]
        if existing:
            logger.debug("Loaded %d %s diagnostics from cache %s", len(existing), ligand, cache_path)
            return existing

    rows = compute_all_diagnostics(h5_path, ligand)
    logger.info("Computed %d %s diagnostics; saving to %s", len(rows), ligand, cache_path)

    # Merge with any existing rows from other ligands already in the cache
    if cache_path.is_file() and not force_recompute:
        other = [r for r in load_diagnostics_csv(cache_path) if r["ligand"] != ligand]
        all_rows = other + rows
    else:
        all_rows = rows

    # Sort for deterministic output: ligand alphabetical, then fragment_id
    all_rows.sort(key=lambda r: (str(r["ligand"]), int(r["fragment_id"])))
    save_diagnostics_csv(all_rows, cache_path)

    return rows
