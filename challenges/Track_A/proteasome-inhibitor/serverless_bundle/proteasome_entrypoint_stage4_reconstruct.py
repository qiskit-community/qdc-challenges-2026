"""
proteasome_entrypoint_stage4_reconstruct.py — Stage 4 Serverless Entrypoint.

Accepts serialized artifact references (no SQLite dependency on remote worker).
CPU only (resource family 'l' or 'm').
Returns ReconstructionSummary as JSON / dict.
No GPU, no live QPU calls.

Two specification modes
-----------------------
* **Legacy path mode** (Task 5.8): ``data_dir`` / ``fragment_result_refs`` carry worker-local paths.
  Retained unchanged for compatibility; result schema ``stage4_result/1``.
* **Transport mode** (Task 5.9D): ``schema == "stage4_spec/1"``.  The specification carries *no
  filesystem path*; every scientific input (Stage 1 mean-field pickle, Stage 2 EWF dump and one frs/1
  result per required fragment) is a ``TransportDescriptor`` that the worker downloads through the
  transport adapter and authenticates (size + SHA-256) into a private scratch directory that is always
  removed.  The worker re-checks identity, Hamiltonian SHA-256 (against the downloaded Stage 2 dump),
  payload digests, RDM validity and provenance with the same rules as the organizer, delegates to the
  inherited ``run_reconstruct_from_refs`` and returns the *complete* reconstruction result in the
  normal job result (``stage4_result/2``).  It never touches SQLite and never returns a path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional

ENTRYPOINT_SCHEMA = "stage4_result/1"
TRANSPORT_SPEC_SCHEMA = "stage4_spec/1"
TRANSPORT_RESULT_SCHEMA = "stage4_result/2"
FUNCTION_TITLE = "qfm-proteasome-stage4-reconstruct"
_TRACEBACK_LIMIT = 4000
_HF_TOL = 1e-8
_FORBIDDEN_PATH_KEYS = ("data_dir", "results_dir", "ewf_dumpfile_path", "result_dir", "result_path",
                        "fragment_result_refs", "mf_data_path")

# Ensure serverless_bundle/ modules are importable
_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE, _HERE.parent, _HERE.parent.parent]:
    if (_candidate / "serverless_bundle").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
    if (_candidate / "steps").is_dir() and (_candidate.name == "serverless_bundle"):
        if str(_candidate.parent) not in sys.path:
            sys.path.insert(0, str(_candidate.parent))
        break

_vendor_sqd = _HERE / "vendor" / "qiskit-addon-sqd"
if _vendor_sqd.is_dir() and str(_vendor_sqd) not in sys.path:
    sys.path.insert(0, str(_vendor_sqd))

try:
    from steps.artifact_transport import (
        ServerlessTransportAdapter,
        TransportDescriptor,
        TransportError,
        build_worker_file_client,
    )
    from steps.fragment_result_store import FragmentResultReference, load_fragment_result_from_ref
    from steps.reconstruction_coordinator import (
        BarrierReport,
        ReconstructionSummary,
        inspect_fragment_result,
        reconstruct_from_registry,
        validate_completion_barrier,
    )
    from steps.step4_reconstruct import QuantumReconstructionError, run_reconstruct_from_refs
except ImportError:
    try:
        from serverless_bundle.steps.artifact_transport import (
            ServerlessTransportAdapter,
            TransportDescriptor,
            TransportError,
            build_worker_file_client,
        )
        from serverless_bundle.steps.fragment_result_store import (
            FragmentResultReference,
            load_fragment_result_from_ref,
        )
        from serverless_bundle.steps.reconstruction_coordinator import (
            BarrierReport,
            ReconstructionSummary,
            inspect_fragment_result,
            reconstruct_from_registry,
            validate_completion_barrier,
        )
        from serverless_bundle.steps.step4_reconstruct import (
            QuantumReconstructionError,
            run_reconstruct_from_refs,
        )
    except ImportError:
        pass


def _validate_spec(spec: dict) -> None:
    """Validate that the input specification contains all required fields."""
    if not isinstance(spec, dict):
        raise ValueError(f"Specification must be a dict, got {type(spec).__name__}")

    for key in ("run_id", "data_dir", "fragment_result_refs", "required_fragments"):
        if key not in spec:
            raise ValueError(f"Missing required key {key!r} in specification")

    if not isinstance(spec["run_id"], str) or not spec["run_id"].strip():
        raise ValueError("spec['run_id'] must be a non-empty string")

    if not isinstance(spec["fragment_result_refs"], (dict, list)):
        raise ValueError("spec['fragment_result_refs'] must be a dict or list")

    if not isinstance(spec["required_fragments"], (list, tuple)):
        raise ValueError("spec['required_fragments'] must be a list or tuple of ints")


class _NoFileClient:
    """Placeholder used when no File API is reachable."""

    def _fail(self, *_a, **_k):
        raise TransportError("no Serverless File API client is available in this worker")

    file_upload = file_download = file_delete = files = _fail


def _is_transport_spec(spec: Any) -> bool:
    return isinstance(spec, dict) and spec.get("schema") == TRANSPORT_SPEC_SCHEMA


def _validate_transport_spec(spec: dict) -> None:
    forbidden = [k for k in _FORBIDDEN_PATH_KEYS if k in spec]
    if forbidden:
        raise ValueError(f"Transport specification must not carry filesystem paths: {forbidden}")
    missing = [k for k in ("run_id", "required_fragments", "solver_assignments", "hamiltonian_hashes",
                           "mf_input", "ewf_input", "fragment_inputs") if spec.get(k) is None]
    if missing:
        raise ValueError(f"Transport specification missing required keys: {missing}")
    if not isinstance(spec["run_id"], str) or not spec["run_id"].strip():
        raise ValueError("spec['run_id'] must be a non-empty string")
    req = spec["required_fragments"]
    if not isinstance(req, (list, tuple)) or not req or any(isinstance(f, bool) or not isinstance(f, int) for f in req):
        raise ValueError("spec['required_fragments'] must be a non-empty list of integers")
    if len(set(req)) != len(req):
        raise ValueError("spec['required_fragments'] contains duplicates")
    for key in ("solver_assignments", "hamiltonian_hashes", "fragment_inputs"):
        got = set(str(k) for k in spec[key])
        if got != {str(f) for f in req}:
            raise ValueError(f"spec[{key!r}] must cover exactly the required fragments")
    for key in ("mf_input", "ewf_input"):
        if not isinstance(spec[key], dict):
            raise ValueError(f"spec[{key!r}] must be a transport descriptor dictionary")


def _dump_hamiltonian_hashes(ewf_path: Path, fids: list[int]) -> dict[int, str]:
    """sha256(heff.tobytes() + eris.tobytes()) per fragment — same definition as the classical worker."""
    import h5py

    out: dict[int, str] = {}
    with h5py.File(ewf_path, "r") as hf:
        for fid in fids:
            key = f"fragment_{fid}"
            if key not in hf:
                raise ValueError(f"Stage 2 dump has no group {key!r}")
            g = hf[key]
            out[fid] = hashlib.sha256(g["heff"][:].tobytes() + g["eris"][:].tobytes()).hexdigest()
    return out


def _transport_failure(spec_like: Any, started: float, exc: BaseException) -> dict:
    s = spec_like if isinstance(spec_like, dict) else {}
    return {
        "schema": TRANSPORT_RESULT_SCHEMA,
        "status": "FAILED",
        "run_id": str(s.get("run_id", "unknown_run")),
        "attempt": s.get("attempt", 1),
        "elapsed_s": time.time() - started,
        "error": f"{type(exc).__name__}: {exc}"[:2000],
        "traceback": traceback.format_exc()[-_TRACEBACK_LIMIT:],
    }


def run_stage4_transport(spec: dict, transport_client: Any = None, *, workdir: Optional[Path] = None) -> dict:
    """Transport-mode Stage 4 reconstruction (see module docstring).  Returns a path-free dictionary."""
    started = time.time()
    scratch: Optional[Path] = None
    try:
        if not isinstance(spec, dict):
            raise ValueError("Specification must be a dict")
        _validate_transport_spec(spec)
        run_id = spec["run_id"]
        req = sorted(int(f) for f in spec["required_fragments"])
        assign = {int(k): str(v) for k, v in spec["solver_assignments"].items()}
        hashes = {int(k): str(v) for k, v in spec["hamiltonian_hashes"].items()}

        client = transport_client if transport_client is not None else build_worker_file_client(FUNCTION_TITLE)
        adapter = ServerlessTransportAdapter(client if client is not None else _NoFileClient())
        scratch = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="qfm_stage4_"))
        data_dir = scratch / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # 1. Authenticated retrieval of the scientific inputs (size + SHA-256 are mandatory).
        descs = {}
        for role, dest in (("mf_input", "mf_data.pkl"), ("ewf_input", "ewf_dumpfile.h5")):
            d = TransportDescriptor.from_dict(spec[role])
            if d.run_id != run_id:
                raise ValueError(f"{role} belongs to a different run")
            adapter.fetch_descriptor(d, data_dir / dest)
            descs[role] = d
        ewf_path = data_dir / "ewf_dumpfile.h5"

        # 2. Stage 2 Hamiltonian identity of every required fragment (mandatory, never optional).
        actual_hashes = _dump_hamiltonian_hashes(ewf_path, req)
        bad = [f for f in req if actual_hashes[f] != hashes[f]]
        if bad:
            raise ValueError(f"Hamiltonian SHA-256 of the downloaded Stage 2 dump differs for fragments {bad}")

        # 3. Authenticated retrieval + full content inspection of every fragment result.
        refs: dict[int, FragmentResultReference] = {}
        problems: list[str] = []
        frag_dir = scratch / "fragments"
        frag_dir.mkdir(exist_ok=True)
        for fid in req:
            d = TransportDescriptor.from_dict(spec["fragment_inputs"][str(fid)])
            if (d.run_id, d.fragment_id, d.solver, d.hamiltonian_sha256) != (run_id, fid, assign[fid], hashes[fid]):
                raise ValueError(f"fragment {fid}: descriptor identity differs from the specification")
            if not d.payload_digest:
                raise ValueError(f"fragment {fid}: descriptor carries no payload digest")
            local = adapter.fetch_descriptor(d, frag_dir / f"frag_{fid}.h5")
            _res, probs = inspect_fragment_result(
                local, run_id=run_id, fragment_id=fid, solver=assign[fid], hamiltonian_sha256=hashes[fid],
                expected_payload_digest=d.payload_digest, expected_file_sha256=d.file_sha256)
            for items in probs.values():
                problems.extend(items)
            refs[fid] = FragmentResultReference(
                path=local, file_sha256=d.file_sha256, payload_digest=d.payload_digest, fragment_id=fid,
                run_id=run_id, solver=assign[fid], hamiltonian_sha256=hashes[fid])
        if problems:
            raise ValueError("worker-side completion barrier failed: " + "; ".join(problems[:20]))

        # 4. Inherited EWF cumulant reconstruction (unchanged mathematics).
        res = run_reconstruct_from_refs(
            data_dir=str(data_dir), run_name=run_id, fragment_result_refs=refs,
            results_dir=str(scratch / "results"), expected_fragment_ids=req)
        e_hf = float(res["e_hf"])
        expected_hf = spec.get("expected_e_hf")
        if expected_hf is not None and abs(e_hf - float(expected_hf)) > _HF_TOL:
            raise ValueError(f"E_HF of the downloaded Stage 1 pickle ({e_hf!r}) differs from the "
                             f"authenticated value ({expected_hf!r})")
        fragments = {}
        for k, v in res["fragments"].items():
            fragments[str(int(k))] = {
                "e_corr": float(v["e_corr"]), "solver": str(v.get("solver", "")),
                "n_orb": int(v.get("n_orb", 0)), "converged": bool(v.get("converged", True)),
                "rdm_integrated": bool(v.get("rdm_integrated", False)),
            }
        e_corr_total = float(res["e_corr"])
        return {
            "schema": TRANSPORT_RESULT_SCHEMA,
            "status": "COMPLETE",
            "run_id": run_id,
            "attempt": int(spec.get("attempt", 1)),
            "manifest_digest": spec.get("manifest_digest", ""),
            "e_hf": e_hf,
            "e_corr_total": e_corr_total,
            "e_total": float(res["e_total"]),
            "fragments": fragments,
            "solver_used": {str(f): assign[f] for f in req},
            "hamiltonian_sha256s": {str(f): hashes[f] for f in req},
            "payload_digests": {str(f): refs[f].payload_digest for f in req},
            "mf_file_sha256": descs["mf_input"].file_sha256,
            "ewf_file_sha256": descs["ewf_input"].file_sha256,
            "elapsed_s": time.time() - started,
            "software": _software_versions(),
        }
    except Exception as exc:  # noqa: BLE001 — every failure becomes a FAILED descriptor (no raw path/trace leak)
        return _transport_failure(spec, started, exc)
    finally:
        if scratch is not None and workdir is None:
            shutil.rmtree(scratch, ignore_errors=True)


def _software_versions() -> dict:
    from importlib import metadata

    out = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "pyscf", "h5py"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def run_stage4_reconstruct(spec: dict, transport_client: Any = None) -> dict:
    """Execute Stage 4 independent EWF energy reconstruction on the worker.

    A specification with ``schema == "stage4_spec/1"`` selects the path-free transport mode
    (:func:`run_stage4_transport`).  Otherwise the legacy path mode below is used.

    Parameters
    ----------
    spec : dict
        Legacy task specification containing:
        - run_id: str
        - data_dir: str (path containing mf_data.pkl and ewf_dumpfile.h5)
        - fragment_result_refs: dict (mapping of fid -> ref dict or FragmentResultReference)
        - required_fragments: list[int]
        - results_dir: str (optional, default: data_dir/../../results/<run_id>)
        - solver_assignments: dict[int, str] (optional)
        - hamiltonian_hashes: dict[int, str] (optional)

    Returns
    -------
    dict
        Structured result record with schema 'stage4_result/1' (legacy) or 'stage4_result/2'.
    """
    if _is_transport_spec(spec):
        return run_stage4_transport(spec, transport_client)
    t_start = time.time()
    run_id = spec.get("run_id", "unknown_run")

    try:
        _validate_spec(spec)

        data_dir = str(spec["data_dir"])
        req_frags = [int(f) for f in spec["required_fragments"]]
        raw_refs = spec["fragment_result_refs"]

        # Normalize raw_refs to dict[int, FragmentResultReference]
        refs_dict: dict[int, FragmentResultReference] = {}
        if isinstance(raw_refs, list):
            for item in raw_refs:
                if isinstance(item, dict):
                    fid = int(item["fragment_id"])
                    refs_dict[fid] = FragmentResultReference(
                        path=Path(item["path"]),
                        file_sha256=item["file_sha256"],
                        payload_digest=item.get("payload_digest", ""),
                        fragment_id=fid,
                        run_id=item.get("run_id", run_id),
                        solver=item.get("solver", ""),
                        hamiltonian_sha256=item.get("hamiltonian_sha256", ""),
                        extra=item.get("extra", {}),
                    )
                elif hasattr(item, "fragment_id"):
                    fid = int(item.fragment_id)
                    refs_dict[fid] = item
        elif isinstance(raw_refs, dict):
            for k, v in raw_refs.items():
                fid = int(k)
                if isinstance(v, dict):
                    refs_dict[fid] = FragmentResultReference(
                        path=Path(v["path"]),
                        file_sha256=v["file_sha256"],
                        payload_digest=v.get("payload_digest", ""),
                        fragment_id=fid,
                        run_id=v.get("run_id", run_id),
                        solver=v.get("solver", ""),
                        hamiltonian_sha256=v.get("hamiltonian_sha256", ""),
                        extra=v.get("extra", {}),
                    )
                elif isinstance(v, (str, Path)):
                    # Plain path string/Path
                    p = Path(v)
                    from steps.reconstruction_coordinator import _sha256_file
                    refs_dict[fid] = FragmentResultReference(
                        path=p,
                        file_sha256=_sha256_file(p),
                        payload_digest="",
                        fragment_id=fid,
                        run_id=run_id,
                        solver=spec.get("solver_assignments", {}).get(fid, ""),
                        hamiltonian_sha256=spec.get("hamiltonian_hashes", {}).get(fid, ""),
                    )
                else:
                    refs_dict[fid] = v

        results_dir = spec.get("results_dir")
        if results_dir is None:
            results_dir = str(Path(data_dir).parent.parent / "results" / run_id)
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # Run reconstruction via authoritative step4_reconstruct
        res_dict = run_reconstruct_from_refs(
            data_dir=data_dir,
            run_name=run_id,
            fragment_result_refs=refs_dict,
            results_dir=results_dir,
            expected_fragment_ids=req_frags,
        )

        summary_file = Path(results_dir) / "summary.json"
        elapsed = time.time() - t_start

        return {
            "status": "COMPLETE",
            "run_id": run_id,
            "e_hf": float(res_dict["e_hf"]),
            "e_corr_total": float(res_dict["e_corr"]),
            "e_total": float(res_dict["e_total"]),
            "summary_artifact_uri": str(summary_file),
            "fragments": res_dict.get("fragments", {}),
            "elapsed_s": elapsed,
            "schema": ENTRYPOINT_SCHEMA,
        }

    except Exception as exc:
        elapsed = time.time() - t_start
        tb = traceback.format_exc()[-_TRACEBACK_LIMIT:]
        return {
            "status": "FAILED",
            "run_id": run_id,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
            "elapsed_s": elapsed,
            "schema": ENTRYPOINT_SCHEMA,
        }


def main() -> None:
    """Entrypoint function for Qiskit Serverless execution."""
    try:
        from qiskit_serverless import get_arguments, save_result
        args = get_arguments()
    except ImportError:
        args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}

    result = run_stage4_reconstruct(args)

    try:
        from qiskit_serverless import save_result
        save_result(result)
    except ImportError:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
