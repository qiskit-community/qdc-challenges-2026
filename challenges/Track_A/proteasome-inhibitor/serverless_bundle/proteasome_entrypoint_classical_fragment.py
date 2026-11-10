"""
proteasome_entrypoint_classical_fragment.py — Serverless entrypoint for ONE classical fragment (Task 5.9C.1).

Executes one FCI or CCSD fragment on a Serverless worker.  It never touches SQLite, never samples a
QPU, and never receives or returns a filesystem path.

Specification (JSON-serialisable; contains no organizer path)
-------------------------------------------------------------
  run_id              : str   run namespace
  fragment_id         : int   0-based fragment index
  solver              : "FCI" | "CCSD"   frozen assignment (anything else is rejected)
  hamiltonian_sha256  : str   expected sha256(h1e.tobytes() + h2e.tobytes()) of the fragment
  ewf_input           : dict  TransportDescriptor of the immutable Stage 2 ``ewf_dumpfile.h5``
                              (remote_name, file_sha256, file_size_bytes, ...)
  solver_config       : dict  optional allow-listed overrides (conv_tol, max_cycle[, diis_space])
  attempt             : int   dispatch attempt (>= 1); scopes the output transport name
  result_mode         : "auto" | "file_api" | "inline"   how the result is published (default auto)
  output_transport_name : str optional; if given it MUST equal the deterministic name

Returned descriptor (``classical_fragment_result/1``)
-----------------------------------------------------
  status "COMPLETE": scientific summary + ``result`` = TransportDescriptor of the published frs/1 file
  (remote name, SHA-256, size, payload_digest, transport mode; inline bytes only for inline mode).
  status "FAILED"  : ``error`` and a truncated ``traceback``.
  A worker-local path is NEVER returned.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Optional

ENTRYPOINT_SCHEMA = "classical_fragment_result/1"
FUNCTION_TITLE = "qfm-proteasome-classical-fragment"
_TRACEBACK_LIMIT = 4000

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

try:
    from steps.artifact_transport import (
        ServerlessTransportAdapter,
        TransportDescriptor,
        TransportError,
        build_worker_file_client,
        result_transport_name,
    )
    from steps.classical_fragment_runner import (
        CLASSICAL_SOLVERS,
        ClassicalFragmentError,
        load_fragment_hamiltonian,
        run_classical_fragment,
        validate_solver_config,
    )
except ImportError:
    from serverless_bundle.steps.artifact_transport import (  # type: ignore[no-redef]
        ServerlessTransportAdapter,
        TransportDescriptor,
        TransportError,
        build_worker_file_client,
        result_transport_name,
    )
    from serverless_bundle.steps.classical_fragment_runner import (  # type: ignore[no-redef]
        CLASSICAL_SOLVERS,
        ClassicalFragmentError,
        load_fragment_hamiltonian,
        run_classical_fragment,
        validate_solver_config,
    )


class _NoFileClient:
    """Placeholder used when no File API is reachable; only inline publication can work."""

    def _fail(self, *_a, **_k):
        raise TransportError("no Serverless File API client is available in this worker")

    file_upload = file_download = file_delete = files = _fail


def _validate_spec(spec: dict[str, Any]) -> None:
    required = ["run_id", "fragment_id", "solver", "hamiltonian_sha256", "ewf_input"]
    missing = [k for k in required if spec.get(k) is None]
    if missing:
        raise ValueError(f"Specification missing required keys: {missing}")
    forbidden = [k for k in ("ewf_dumpfile_path", "result_dir", "result_path") if k in spec]
    if forbidden:
        raise ValueError(f"Specification must not carry filesystem paths: {forbidden}")
    if not isinstance(spec["run_id"], str) or not spec["run_id"].strip():
        raise ValueError("spec['run_id'] must be a non-empty string")
    if isinstance(spec["fragment_id"], bool) or not isinstance(spec["fragment_id"], int) or spec["fragment_id"] < 0:
        raise ValueError("spec['fragment_id'] must be a non-negative integer")
    if spec["solver"] not in CLASSICAL_SOLVERS:
        raise ValueError(
            f"solver must be exactly one of {sorted(CLASSICAL_SOLVERS)}, got {spec['solver']!r}; "
            "this worker never substitutes a solver")
    attempt = spec.get("attempt", 1)
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("spec['attempt'] must be an integer >= 1")
    if spec.get("result_mode", "auto") not in ("auto", "file_api", "inline"):
        raise ValueError("spec['result_mode'] must be auto, file_api or inline")
    if not isinstance(spec["ewf_input"], dict):
        raise ValueError("spec['ewf_input'] must be a transport descriptor dictionary")


def _failure(spec_like: Any, started: float, exc: BaseException) -> dict[str, Any]:
    s = spec_like if isinstance(spec_like, dict) else {}
    try:
        fid = int(s.get("fragment_id", -1))
    except (TypeError, ValueError):
        fid = -1
    return {
        "schema": ENTRYPOINT_SCHEMA,
        "status": "FAILED",
        "run_id": str(s.get("run_id", "unknown_run")),
        "fragment_id": fid,
        "solver": str(s.get("solver", "")),
        "attempt": s.get("attempt", 1),
        "elapsed_seconds": time.time() - started,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc()[-_TRACEBACK_LIMIT:],
    }


def run_classical_fragment_worker(
    spec: dict[str, Any],
    transport_client: Any = None,
    *,
    workdir: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute one classical fragment and publish the result.

    ``transport_client`` is the logical-name file client (see ``artifact_transport``); when ``None``
    one is built from the gateway environment if available.  ``workdir`` is a *worker-local* scratch
    directory (tests only); by default a private temporary directory is created and always removed.
    """
    started = time.time()
    if not isinstance(spec, dict):
        return _failure({}, started, TypeError("Specification must be a dictionary"))

    scratch: Optional[Path] = None
    try:
        _validate_spec(spec)
        run_id = spec["run_id"]
        fragment_id = int(spec["fragment_id"])
        solver = spec["solver"]
        attempt = int(spec.get("attempt", 1))
        expected_ham = str(spec["hamiltonian_sha256"])
        config = validate_solver_config(solver, spec.get("solver_config"))

        expected_out = result_transport_name(run_id, fragment_id, solver, attempt)
        if spec.get("output_transport_name") not in (None, expected_out):
            raise ValueError("output_transport_name does not match the deterministic run/fragment/attempt name")

        client = transport_client
        if client is None:
            client = build_worker_file_client(FUNCTION_TITLE)
        adapter = ServerlessTransportAdapter(client if client is not None else _NoFileClient())

        scratch = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="qfm_classical_"))
        scratch.mkdir(parents=True, exist_ok=True)

        # 1. Retrieve + authenticate the immutable Stage 2 input (size and SHA-256 are mandatory).
        ewf_desc = TransportDescriptor.from_dict(spec["ewf_input"])
        if ewf_desc.run_id != run_id:
            raise ValueError("ewf_input belongs to a different run")
        ewf_path = adapter.fetch_descriptor(ewf_desc, scratch / "input_ewf.h5")

        # 2. Authenticate the fragment Hamiltonian identity.
        ham = load_fragment_hamiltonian(ewf_path, fragment_id)
        if ham["hamiltonian_sha256"] != expected_ham:
            raise ClassicalFragmentError(
                f"Fragment {fragment_id} Hamiltonian SHA-256 mismatch: expected {expected_ham[:12]}…, "
                f"got {ham['hamiltonian_sha256'][:12]}…")

        # 3. Execute the inherited solver; persist complete frs/1 (energy + RDMs) in scratch.
        local_result = scratch / f"result_f{fragment_id}_{solver.lower()}_a{attempt}.h5"
        res = run_classical_fragment(
            ewf_dumpfile_path=ewf_path, fragment_id=fragment_id, solver=solver,
            result_h5_path=local_result, run_id=run_id, compute_rdms=True,
            solver_config=config, extra={"attempt": attempt})

        # 4. Publish through the transport adapter; the descriptor is path-free.
        desc = adapter.publish_result(
            local_result, run_id=run_id, fragment_id=fragment_id, solver=solver, attempt=attempt,
            payload_digest=res.fragment_ref.payload_digest, hamiltonian_sha256=res.hamiltonian_sha256,
            mode=spec.get("result_mode", "auto"))

        return {
            "schema": ENTRYPOINT_SCHEMA,
            "status": "COMPLETE",
            "run_id": run_id,
            "fragment_id": fragment_id,
            "solver": solver,
            "attempt": attempt,
            "energy_ha": float(res.energy),
            "e_corr_ha": float(res.e_corr),
            "converged": bool(res.converged),
            "n_orb": int(res.n_orb),
            "n_occ": int(res.n_occ),
            "hamiltonian_sha256": res.hamiltonian_sha256,
            "payload_digest": res.fragment_ref.payload_digest,
            "result": desc.as_dict(),
            "elapsed_seconds": time.time() - started,
            "software": res.software_versions,
        }
    except Exception as exc:  # noqa: BLE001 — any failure is reported as a FAILED descriptor
        return _failure(spec, started, exc)
    finally:
        if scratch is not None and workdir is None:
            shutil.rmtree(scratch, ignore_errors=True)


# Alias used by test harnesses and the dispatcher's function registry
proteasome_entrypoint_classical_fragment = run_classical_fragment_worker


def main() -> None:
    """Entry point for IBM Serverless invocation."""
    try:
        from qiskit_serverless import get_arguments, save_result
    except ImportError:
        args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
        print(json.dumps(run_classical_fragment_worker(args), indent=2, default=str))
        return
    save_result(run_classical_fragment_worker(get_arguments()))


if __name__ == "__main__":
    main()
