"""
Step 2 — EWF fragmentation (IAO + MP2 / DMET bath).

Uses the QFWorkflow-compatible EWF embedder from quantum_fragment_methods (backed by
vendored Vayesta) to produce per-fragment cluster Hamiltonians and persist them to HDF5.
Validates the exported Hamiltonians and integral tensors immediately upon completion.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

_bundle_dir = Path(__file__).resolve().parent.parent
if str(_bundle_dir) not in sys.path:
    sys.path.insert(0, str(_bundle_dir))
_vendor_vayesta = _bundle_dir / "vendor" / "vayesta"
if _vendor_vayesta.is_dir() and str(_vendor_vayesta) not in sys.path:
    sys.path.insert(0, str(_vendor_vayesta))

from .stage1_spec import spec_from_pyscf
from .stage2_spec import (
    BathConfig,
    EWFConfig,
    FragmentationConfig,
    Stage2Spec,
    Stage2SpecError,
    build_stage2_result_record,
    validate_ewf_dumpfile,
)


def run_fragments(
    data_dir: str,
    run_name: str,
    *,
    ewf_config: Optional[EWFConfig | Mapping[str, Any]] = None,
    bath_type: str = "mp2",
    truncation: float = 1e-4,
    fragmentation: str = "atomic",
    verbose: int = 3,
    spec: Optional[Stage2Spec] = None,
) -> dict[str, Any]:
    """
    Run Vayesta EWF fragmentation on an RHF reference loaded from *data_dir*.

    Parameters
    ----------
    data_dir      : directory containing mf_data.pkl (from step 1)
    run_name      : short label for logging
    ewf_config    : validated EWFConfig (preferred) or mapping
    bath_type     : Vayesta bath type — 'mp2' (default) or 'dmet'
    truncation    : MP2 bath truncation threshold (default: 1e-4)
    fragmentation : fragmentation scheme — 'atomic' (default) or 'iao'
    verbose       : Vayesta verbosity
    spec          : optional full Stage2Spec for exact provenance

    Returns
    -------
    dict with keys: stage, run_name, n_fragments, dumpfile, sidecar
    """
    from quantum_fragment_methods.application.embedding.ewf import EWF

    # Resolve scientific EWFConfig
    if ewf_config is not None:
        if isinstance(ewf_config, EWFConfig):
            cfg = ewf_config
        else:
            cfg = EWFConfig.from_dict(ewf_config)
    else:
        frag_cfg = FragmentationConfig(scheme=fragmentation, basis="iao", iao_minao="auto")
        bath_cfg = BathConfig(method=bath_type, truncation=truncation)
        cfg = EWFConfig(fragmentation=frag_cfg, bath=bath_cfg, verbose=verbose)

    data_path = Path(data_dir)
    pkl_path = data_path / "mf_data.pkl"
    print(f"[step2] {run_name}: loading mean-field from {pkl_path}")
    with open(pkl_path, "rb") as fh:
        mf_data = pickle.load(fh)
    mf = mf_data["mf"]
    mol = mf.mol

    # Recover Stage2Spec if not explicitly passed
    stage1_res = mf_data.get("stage1_result")
    if spec is None:
        try:
            s1_spec = spec_from_pyscf(mol, mf, name=run_name.split("_")[0])
            spec = Stage2Spec(stage1=s1_spec, config=cfg)
        except Exception:
            spec = None

    dumpfile = str(data_path / "ewf_dumpfile.h5")
    dump_path = Path(dumpfile)
    if dump_path.exists():
        dump_path.unlink()

    print(
        f"[step2] {run_name}: running EWF fragmentation  scheme={cfg.fragmentation.scheme}  "
        f"bath_method={cfg.bath.method}  truncation={cfg.bath.truncation}"
    )

    # Configure EWF embedder with explicit validated settings
    bath_kwargs: dict[str, Any] = {
        "bath_type": cfg.bath.method,
        "dmet_threshold": cfg.bath.dmet_threshold,
        "canonicalize": cfg.bath.canonicalize,
        "occupation_tolerance": cfg.bath.occupation_tolerance,
    }
    if cfg.bath.method == "mp2":
        bath_kwargs["truncation"] = cfg.bath.truncation
        bath_kwargs["project_dmet_order"] = cfg.bath.project_dmet_order
        bath_kwargs["project_dmet_mode"] = cfg.bath.project_dmet_mode
        bath_kwargs["addbuffer"] = cfg.bath.addbuffer

    ewf_embedder = EWF(
        dumpfile=dumpfile,
        verbose=cfg.verbose,
        iao_minao=cfg.fragmentation.iao_minao,
        **bath_kwargs,
    )

    ewf_embedder.create_fragments(
        mf,
        fragmentation=cfg.fragmentation.scheme,
        iao_minao=cfg.fragmentation.iao_minao,
        dumpfile=dumpfile,
    )

    # Independent scientific validation gate
    diag = validate_ewf_dumpfile(dumpfile, spec=spec, mf=mf, mol=mol)
    n_frag = diag["n_fragments"]
    print(
        f"[step2] {run_name}: validation PASSED  n_fragments={n_frag}  "
        f"norb_range=[{diag['norb_min']}, {diag['norb_max']}]  "
        f"max_herm_err={diag['max_hermiticity_residual']:.2e}  "
        f"max_sym_err={diag['max_symmetry_residual']:.2e}"
    )

    # Build and persist sidecar record
    sidecar = None
    if spec is not None:
        sidecar = build_stage2_result_record(
            spec,
            dumpfile,
            stage1_result=stage1_res,
            validation_diag=diag,
        )
        sidecar_path = data_path / "ewf_sidecar.json"
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2)

    return {
        "stage": "fragmentation",
        "run_name": run_name,
        "n_fragments": n_frag,
        "dumpfile": dumpfile,
        "sidecar": sidecar,
        "diagnostics": diag,
    }
