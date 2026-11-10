"""
Step 1 — Mean-field (restricted Hartree-Fock) calculation.

The calculation is defined entirely by a ``Stage1Spec`` (molecule + HF configuration, see ``stage1_spec``).  The
geometry is the XYZ text embedded in the specification, so nothing depends on a path existing where this runs.
A result is returned only if it passes independent numerical validation; a non-converged or inconsistent state raises.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Union

from .stage1_spec import (
    Stage1Spec, Stage1SpecError, build_result_record, check_basis_available, sha256_file, validate_scf_state,
)


def _build_mol(spec: Stage1Spec):
    from pyscf import gto

    m = spec.molecule
    problems = check_basis_available(spec.config.basis, [a[0] for a in m.atoms])
    if problems:
        raise Stage1SpecError("; ".join(problems))
    mol = gto.Mole()
    # Records exactly as parsed (symbol x y z); the original text is never altered.
    mol.atom = "\n".join(f"{s} {x!r} {y!r} {z!r}" for s, x, y, z in m.atoms)
    mol.unit = m.units
    mol.basis = spec.config.basis
    mol.charge = m.charge
    mol.spin = m.spin
    mol.verbose = spec.config.verbose
    mol.build()
    return mol


def run_meanfield(
    spec: Union[Stage1Spec, Mapping[str, Any]],
    data_dir: str,
    pyscf_version: str | None = None,
) -> dict:
    """
    Run restricted HF for ``spec`` and write ``mf_data.pkl`` (``{"mol", "mf"}``) into ``data_dir``.

    Returns the full provenance record (``stage1_spec.build_result_record``).  Raises ``Stage1SpecError`` for an
    unsupported request and ``Stage1ValidationError`` if the SCF result fails validation (the pickle is then not kept).
    """
    import pyscf
    from pyscf import scf

    if not isinstance(spec, Stage1Spec):
        spec = Stage1Spec.from_dict(spec)
    cfg = spec.config

    mol = _build_mol(spec)
    print(f"[step1] {spec.tag}: {mol.natm} atoms, {mol.nao} AOs, {mol.nelectron} electrons, "
          f"charge {mol.charge}, spin {mol.spin}, {cfg.method}/{cfg.basis}")

    # Only reached for a closed-shell spec (Stage1Spec rejects spin != 0 for RHF), so scf.RHF is truly RHF.
    mf = scf.RHF(mol)
    mf.conv_tol = cfg.conv_tol
    mf.conv_tol_grad = cfg.conv_tol_grad
    mf.max_cycle = cfg.max_cycle
    mf.init_guess = cfg.init_guess
    mf.level_shift = cfg.level_shift
    mf.damp = cfg.damp
    mf.kernel()

    diagnostics = validate_scf_state(mf, mol, spec)

    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "mf_data.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump({"mol": mol, "mf": mf}, fh)
    print(f"[step1] {spec.tag}: wrote {pkl_path}  HF energy = {mf.e_tot:.10f} Ha, {mf.cycles} cycles")

    execution = {
        "pyscf_version": pyscf_version or pyscf.__version__,
        "scf_cycles": int(mf.cycles),
        "converged": bool(mf.converged),
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return build_result_record(
        spec, mol, mf, diagnostics,
        mf_sha256=sha256_file(pkl_path), mf_size=pkl_path.stat().st_size, execution=execution)
