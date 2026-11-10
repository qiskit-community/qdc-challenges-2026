"""
proteasome_entrypoint_3qc_postprocess.py — Stage 3Q-C Serverless entrypoint.

One fragment per invocation.  No QPU submission.  No EWF reconstruction.
No SQLite registry dependency inside the compute worker.

Design
------
This entrypoint receives a self-contained task specification dict, performs
the SQD/TrimSQD diagonalization independently, persists the result, and
returns a small structured result dict.

Registry coordination (POSTPROCESS_QUEUED → POSTPROCESSING → COMPLETE)
is NOT done here.  The organizer/controller (notebook or local coordinator)
performs registry state transitions before and after this worker executes.

This separation follows the architecture specified in Task 5.7 section 9:
  - Compute worker: receives spec, runs science, returns result.
  - Controller: claims task, dispatches worker, records result, commits state.

Input specification keys (all required):
  run_id               : str
  fragment_id          : int
  solver               : "SQD" or "TrimSQD"
  qpu_job_id           : str
  hamiltonian_sha256   : str
  ewf_dumpfile_path    : str (accessible from the worker's filesystem)
  result_dir           : str (writable directory for outputs)
  sqd_config           : dict (SQD configuration)
  trim_config          : dict (TrimSQD trim config — only required for TrimSQD)
  counts_sha256        : str (expected file SHA; empty if no prior storage)
  n_qubits             : int (2 × norb; used for bit-width verification)
  backend              : str (for provenance)

Optional:
  qpu_service_token    : str (IBM Quantum token for live counts retrieval)
  qpu_service_instance : str
  qpu_service_channel  : str (default "ibm_cloud")
  worker_id            : str (default "3qc_serverless")

Returns a dict:
  status               : "COMPLETE" | "FAILED"
  fragment_id          : int
  solver               : str
  energy               : float (if COMPLETE)
  artifact_uri         : str (path to frs/1 HDF5; if COMPLETE)
  payload_digest       : str (frs/1 payload digest; if COMPLETE)
  counts_sha256        : str (counts file SHA-256; if COMPLETE)
  elapsed_s            : float
  error                : str (if FAILED)
  schema               : "3qc_result/1"
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

ENTRYPOINT_SCHEMA = "3qc_result/1"
FUNCTION_TITLE = "qfm-proteasome-3qc-postprocess"
_TRACEBACK_LIMIT = 4000

# Ensure serverless_bundle/ modules are importable when this file is
# the entrypoint (working directory varies across deployment environments).
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


class _NoFileClient:
    """Placeholder used when no File API is reachable; only inline publication can work."""

    def _fail(self, *_a, **_k):
        raise RuntimeError("no Serverless File API client is available in this worker")

    file_upload = file_download = file_delete = files = _fail


def _import_transport():
    try:
        from serverless_bundle.steps.artifact_transport import (
            ServerlessTransportAdapter, TransportDescriptor, build_worker_file_client, result_transport_name)
    except ImportError:  # pragma: no cover - flat deployment layout
        from steps.artifact_transport import (  # type: ignore[no-redef]
            ServerlessTransportAdapter, TransportDescriptor, build_worker_file_client, result_transport_name)
    return ServerlessTransportAdapter, TransportDescriptor, build_worker_file_client, result_transport_name


def _validate_transport_spec(spec: dict) -> None:
    """Path-free ``trn/1`` specification: inputs arrive as descriptors, never as organizer paths."""
    required = ["run_id", "fragment_id", "solver", "qpu_job_id", "hamiltonian_sha256",
                "ewf_input", "counts_input", "sqd_config"]
    missing = [k for k in required if spec.get(k) is None or spec.get(k) == ""]
    if missing:
        raise ValueError(f"Specification missing required keys: {missing}")
    forbidden = [k for k in ("ewf_dumpfile_path", "result_dir", "result_path", "counts_path") if k in spec]
    if forbidden:
        raise ValueError(f"Specification must not carry filesystem paths: {forbidden}")
    if spec["solver"] not in ("SQD", "TrimSQD"):
        raise ValueError(f"solver must be SQD or TrimSQD, got {spec['solver']!r}")
    fid = spec["fragment_id"]
    if isinstance(fid, bool) or not isinstance(fid, int) or fid < 0:
        raise ValueError("spec['fragment_id'] must be a non-negative integer")
    attempt = spec.get("attempt", 1)
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("spec['attempt'] must be an integer >= 1")
    if not isinstance(spec["ewf_input"], dict) or not isinstance(spec["counts_input"], dict):
        raise ValueError("ewf_input and counts_input must be transport descriptor dictionaries")
    # Fail closed: the frozen protocol configuration must be supplied; the worker never falls back to defaults.
    if not isinstance(spec["sqd_config"], dict) or not spec["sqd_config"]:
        raise ValueError("a non-empty sqd_config from the frozen protocol is required")
    if spec["solver"] == "TrimSQD" and (not isinstance(spec.get("trim_config"), dict) or not spec["trim_config"]):
        raise ValueError("a non-empty trim_config from the frozen protocol is required for TrimSQD")


def _transport_failure(spec_like: Any, started: float, exc: BaseException) -> dict:
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
        "elapsed_s": time.time() - started,
        "elapsed_seconds": time.time() - started,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc()[-_TRACEBACK_LIMIT:],
    }


def _run_transport_mode(spec: dict, transport_client: Optional[Any], workdir: Optional[Path]) -> dict:
    """Descriptor-based 3Q-C execution.  No organizer path is read; no worker path is returned."""
    (ServerlessTransportAdapter, TransportDescriptor,
     build_worker_file_client, result_transport_name) = _import_transport()
    started = time.time()
    scratch: Optional[Path] = None
    try:
        _validate_transport_spec(spec)
        run_id = spec["run_id"]
        fragment_id = int(spec["fragment_id"])
        solver = spec["solver"]
        attempt = int(spec.get("attempt", 1))
        qpu_job_id = str(spec["qpu_job_id"])

        expected_out = result_transport_name(run_id, fragment_id, solver, attempt)
        if spec.get("output_transport_name") not in (None, expected_out):
            raise ValueError("output_transport_name does not match the deterministic run/fragment/attempt name")

        client = transport_client
        if client is None:
            client = build_worker_file_client(FUNCTION_TITLE)
        adapter = ServerlessTransportAdapter(client if client is not None else _NoFileClient())

        scratch = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="qfm_3qc_"))
        scratch.mkdir(parents=True, exist_ok=True)

        # 1. Retrieve + authenticate the immutable inputs (size and SHA-256 mandatory).
        ewf_desc = TransportDescriptor.from_dict(spec["ewf_input"])
        counts_desc = TransportDescriptor.from_dict(spec["counts_input"])
        if ewf_desc.run_id != run_id or counts_desc.run_id != run_id:
            raise ValueError("an input descriptor belongs to a different run")
        want_counts = spec.get("counts_sha256")
        if want_counts and counts_desc.file_sha256 != want_counts:
            raise ValueError("counts descriptor SHA-256 differs from the dispatched counts_sha256")
        ewf_path = adapter.fetch_descriptor(ewf_desc, scratch / "input_ewf.h5")
        counts_path = adapter.fetch_descriptor(
            counts_desc, scratch / f"counts_frag{fragment_id}_{qpu_job_id[:12]}.npy")

        # 2. Post-process from the authenticated counts using the frozen protocol configuration.
        try:
            from serverless_bundle.steps.quantum_postprocess import postprocess_quantum_fragment
        except ModuleNotFoundError as exc:
            if exc.name not in ('serverless_bundle', 'serverless_bundle.steps'):
                raise
            from steps.quantum_postprocess import postprocess_quantum_fragment
        dispatch_record = {
            "task_id": spec.get("task_id", "remote_task"),
            "run_id": run_id,
            "fragment_id": fragment_id,
            "solver": solver,
            "qpu_job_id": qpu_job_id,
            "backend": spec.get("backend", ""),
            "hamiltonian_sha256": spec["hamiltonian_sha256"],
            "ewf_dumpfile_path": str(ewf_path),
            "submission_record": {
                "n_qubits": int(spec.get("n_qubits") or 0),
                "counts_sha256": counts_desc.file_sha256,
                "sqd_config": dict(spec["sqd_config"]),
                "trim_config": dict(spec.get("trim_config") or {}),
            },
            "measurement_authenticated": False,
        }
        registry = _OfflineRegistryProxy(spec)
        pp = postprocess_quantum_fragment(
            dispatch_record=dispatch_record, registry=registry, result_dir=scratch, qpu_service=None,
            counts=None, task_id=spec.get("task_id", "remote_task"),
            worker_id=spec.get("worker_id", "3qc_serverless"),
            sqd_config=dict(spec["sqd_config"]),
            trim_config=dict(spec.get("trim_config") or {}) if solver == "TrimSQD" else None,
        )

        # 3. Publish through the transport adapter; the descriptor is path-free.
        local_result = Path(pp.result_ref_dict.get("path", ""))
        desc = adapter.publish_result(
            local_result, run_id=run_id, fragment_id=fragment_id, solver=solver, attempt=attempt,
            payload_digest=pp.result_ref_dict.get("payload_digest", ""),
            hamiltonian_sha256=pp.hamiltonian_sha256, mode=spec.get("result_mode", "auto"))
        return {
            "schema": ENTRYPOINT_SCHEMA,
            "status": "COMPLETE",
            "run_id": run_id,
            "fragment_id": fragment_id,
            "solver": solver,
            "attempt": attempt,
            "qpu_job_id": qpu_job_id,
            "energy_ha": float(pp.energy),
            "hamiltonian_sha256": pp.hamiltonian_sha256,
            "payload_digest": pp.result_ref_dict.get("payload_digest", ""),
            "counts_sha256": counts_desc.file_sha256,
            "protocol_digest": spec.get("protocol_digest", ""),
            "result": desc.as_dict(),
            "compute_seconds": float(pp.elapsed_s),
            "elapsed_seconds": time.time() - started,
            "diagnostics": {k: v for k, v in (pp.diagnostics or {}).items() if k != "sqd_config"},
        }
    except Exception as exc:  # noqa: BLE001 — any failure is reported as a FAILED record
        return _transport_failure(spec, started, exc)
    finally:
        if scratch is not None and workdir is None:
            shutil.rmtree(scratch, ignore_errors=True)


def run_3qc_postprocess(spec: dict, transport_client: Optional[Any] = None,
                        *, workdir: Optional[Path] = None) -> dict:
    """Execute one independent quantum post-processing task.

    Two modes, selected by the specification:

    * ``trn/1`` (production): ``ewf_input`` / ``counts_input`` descriptors, no filesystem paths; the result
      is published through the transport adapter and returned as a path-free descriptor.
    * legacy local-path mode (``ewf_dumpfile_path`` + ``result_dir``): retained for compatibility.

    Parameters
    ----------
    spec             : dict — task specification (see module docstring for required keys)
    transport_client : optional file client for downloading inputs and uploading results

    Returns
    -------
    dict — result record (always returned, even on failure)
    """
    if isinstance(spec, dict) and ("ewf_input" in spec or "counts_input" in spec):
        return _run_transport_mode(spec, transport_client, workdir)
    import hashlib
    t_start = time.time()
    fragment_id = spec.get("fragment_id", -1)
    solver = spec.get("solver", "")
    run_id = spec.get("run_id", "")

    try:
        _validate_spec(spec)

        # ── Resolve result directory ─────────────────────────────────────────
        result_dir = Path(spec["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        qpu_job_id = spec.get("qpu_job_id", f"job_frag{fragment_id}")

        # ── Resolve Stage 2 HDF5 dumpfile ─────────────────────────────────────
        ewf_dumpfile_path = spec.get("ewf_dumpfile_path")
        if not ewf_dumpfile_path or not Path(ewf_dumpfile_path).is_file():
            if transport_client is not None and spec.get("ewf_transport_name"):
                downloaded = result_dir / f"stage2_input_{fragment_id}.h5"
                transport_client.file_download(spec["ewf_transport_name"], str(downloaded))
                ewf_dumpfile_path = str(downloaded)
            elif not ewf_dumpfile_path:
                raise ValueError("Neither 'ewf_dumpfile_path' nor 'ewf_transport_name' is accessible")

        # ── Build QPU service if token provided ───────────────────────────────
        qpu_service = None
        if spec.get("qpu_service_token"):
            qpu_service = _build_qpu_service(spec)

        # ── Build dispatch_record from spec ───────────────────────────────────
        dispatch_record = {
            "task_id":             spec.get("task_id", ""),
            "run_id":              run_id,
            "fragment_id":         fragment_id,
            "solver":              solver,
            "qpu_job_id":          qpu_job_id,
            "backend":             spec.get("backend", ""),
            "hamiltonian_sha256":  spec["hamiltonian_sha256"],
            "ewf_dumpfile_path":   str(ewf_dumpfile_path),
            "submission_record": {
                "n_qubits":    spec.get("n_qubits", 0),
                "counts_sha256": spec.get("counts_sha256", ""),
                "sqd_config":  spec.get("sqd_config", {}),
                "trim_config": spec.get("trim_config", {}),
            },
            "measurement_authenticated": False,
        }

        # ── Import post-processing function ───────────────────────────────────
        try:
            from serverless_bundle.steps.quantum_postprocess import (
                postprocess_quantum_fragment,
                retrieve_and_store_counts,
            )
        except ModuleNotFoundError as exc:
            if exc.name not in ('serverless_bundle', 'serverless_bundle.steps'):
                raise
            from steps.quantum_postprocess import (
                postprocess_quantum_fragment,
                retrieve_and_store_counts,
            )

        # ── Retrieve or load counts ───────────────────────────────────────────
        counts_dict: dict[str, int] = {}
        if spec.get("counts_dict") is not None:
            counts_dict = spec["counts_dict"]
        elif spec.get("counts") is not None:
            counts_dict = spec["counts"]
        else:
            try:
                counts_path = result_dir / f"counts_frag{fragment_id}_{qpu_job_id[:12]}.npy"
                if spec.get("counts_transport_name") and transport_client is not None and not counts_path.is_file():
                    transport_client.file_download(spec["counts_transport_name"], str(counts_path))

                counts_artifact = retrieve_and_store_counts(
                    qpu_job_id=qpu_job_id,
                    qpu_service=qpu_service,
                    counts_path=counts_path,
                    norb=spec.get("n_qubits", 0) // 2,
                    expected_sha256=spec.get("counts_sha256") or None,
                )

                import numpy as np
                raw = np.load(counts_path, allow_pickle=True).item()
                norb = spec.get("n_qubits", 0) // 2
                num_bits = 2 * norb
                for k, v in raw.items():
                    if isinstance(k, (int,)):
                        counts_dict[format(int(k), f"0{num_bits}b")] = int(v)
                    else:
                        counts_dict[str(k)] = int(v)
            except Exception:
                counts_dict = {}

        # ── Create a dummy registry-less task record ──────────────────────────
        registry = _OfflineRegistryProxy(spec)

        # ── Execute post-processing ───────────────────────────────────────────
        pp_result = postprocess_quantum_fragment(
            dispatch_record=dispatch_record,
            registry=registry,
            result_dir=result_dir,
            qpu_service=qpu_service,
            counts=counts_dict,
            task_id=spec.get("task_id", "remote_task"),
            worker_id=spec.get("worker_id", "3qc_serverless"),
            sqd_config=spec.get("sqd_config"),
            trim_config=spec.get("trim_config"),
        )

        res_path_str = pp_result.result_ref_dict.get("path", "")
        file_sha256 = ""
        file_size_bytes = 0
        remote_transport_name = None
        if res_path_str and Path(res_path_str).is_file():
            file_bytes = Path(res_path_str).read_bytes()
            file_sha256 = hashlib.sha256(file_bytes).hexdigest()
            file_size_bytes = len(file_bytes)

            # Upload if transport is requested
            out_name = spec.get("output_transport_name")
            if out_name and transport_client is not None:
                remote_transport_name = transport_client.file_upload(res_path_str, remote_name=out_name)

        elapsed_s = time.time() - t_start
        return {
            "schema":                ENTRYPOINT_SCHEMA,
            "status":                "COMPLETE",
            "run_id":                run_id,
            "fragment_id":           pp_result.fragment_id,
            "solver":                pp_result.solver,
            "energy":                pp_result.energy,
            "energy_ha":             pp_result.energy,
            "artifact_uri":          res_path_str,
            "payload_digest":        pp_result.result_ref_dict.get("payload_digest", ""),
            "file_sha256":           file_sha256,
            "file_size_bytes":       file_size_bytes,
            "remote_transport_name": remote_transport_name,
            "counts_sha256":         pp_result.counts_artifact.counts_sha256 if pp_result.counts_artifact else "",
            "hamiltonian_sha256":    pp_result.hamiltonian_sha256,
            "elapsed_s":             elapsed_s,
            "elapsed_seconds":       elapsed_s,
            "diagnostics":           pp_result.diagnostics,
        }

    except Exception as exc:
        elapsed_s = time.time() - t_start
        return {
            "schema":      ENTRYPOINT_SCHEMA,
            "status":      "FAILED",
            "fragment_id": fragment_id,
            "solver":      solver,
            "elapsed_s":   elapsed_s,
            "error":       f"{type(exc).__name__}: {exc}",
            "traceback":   traceback.format_exc()[-_TRACEBACK_LIMIT:],
        }


def _validate_spec(spec: dict) -> None:
    required = [
        "run_id", "fragment_id", "solver", "qpu_job_id",
        "hamiltonian_sha256", "ewf_dumpfile_path", "result_dir",
    ]
    missing = [k for k in required if not spec.get(k) and spec.get(k) != 0]
    if missing:
        raise ValueError(f"Missing required spec keys: {missing}")
    if spec["solver"] not in ("SQD", "TrimSQD"):
        raise ValueError(f"solver must be SQD or TrimSQD, got {spec['solver']!r}")


def _build_qpu_service(spec: dict) -> Any:
    """Build a QiskitRuntimeService from spec credentials."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        return QiskitRuntimeService(
            channel=spec.get("qpu_service_channel", "ibm_cloud"),
            token=spec["qpu_service_token"],
            instance=spec.get("qpu_service_instance", ""),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to build QPU service: {exc}") from exc


class _OfflineRegistryProxy:
    """Minimal registry proxy for Serverless worker execution.

    The Serverless worker cannot write to the organizer's local SQLite file.
    This proxy records the claimed/completed state in-memory so that
    postprocess_quantum_fragment can call the registry methods it expects,
    but the actual state transitions are reconciled by the controller.
    """

    def __init__(self, spec: dict):
        self._task_id = spec.get("task_id", "remote_task")
        self._state = "POSTPROCESS_QUEUED"
        self._result_uri = ""
        self._result_digest = ""

    def get_task(self, task_id: str):
        return _ProxyTaskRecord(
            task_id=self._task_id,
            state=self._state,
            run_id="",
            fragment_id=-1,
        )

    def transition(self, task_id: str, from_state: str, to_state: str, **kwargs):
        self._state = to_state
        return self.get_task(task_id)

    def fail_task(self, task_id: str, *, reason: str = "", **kwargs):
        self._state = "FAILED"
        return self.get_task(task_id)

    def complete_task(self, task_id: str, *, artifact_uri: str = "",
                      payload_digest: str = "", **kwargs):
        self._state = "COMPLETE"
        self._result_uri = artifact_uri
        self._result_digest = payload_digest
        return self.get_task(task_id)

    def record_result(self, *args, **kwargs):
        return self.get_task(self._task_id)


class _ProxyTaskRecord:
    """Minimal task record for the offline proxy."""

    def __init__(self, *, task_id: str, state: str, run_id: str, fragment_id: int):
        self.task_id    = task_id
        self.state      = state
        self.run_id     = run_id
        self.fragment_id = fragment_id
        self.lease_token = ""
        self.qpu_job_id  = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Serverless entry point registration
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for IBM Serverless invocation."""
    try:
        from qiskit_serverless import get_arguments, save_result
    except ImportError:
        if len(sys.argv) < 2:
            print("Usage: python proteasome_entrypoint_3qc_postprocess.py <spec.json>")
            sys.exit(1)
        result = run_3qc_postprocess(json.loads(Path(sys.argv[1]).read_text()))
        print(json.dumps(result, indent=2, default=str))
        return
    save_result(run_3qc_postprocess(get_arguments()))


if __name__ == "__main__":
    main()
