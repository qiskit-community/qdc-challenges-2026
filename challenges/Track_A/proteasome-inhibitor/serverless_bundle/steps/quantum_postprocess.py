"""
quantum_postprocess — Stage 3Q-C: Independent quantum fragment post-processing.

Accepts authenticated measurement counts from a completed QPU job and runs
the inherited SQD or TrimSQD diagonalization independently, producing a
complete SolverResult persisted through the frs/1 store.

Design constraints
------------------
* Never submits a QPU job.  Counts retrieval is the only network operation.
* Never synchronously waits for QPU completion — caller must ensure SAMPLES_READY.
* Heavy scientific imports (SQD/TrimSQD diagonalization) are deferred until the
  run function is actually called, not at module level.
* Measurement identity is verified before diagonalization (provenance check).
* Write-once measurement storage: retrieved counts are persisted before
  diagonalization; a failed diagonalization retry reuses the stored counts.
* Registry lifecycle: POSTPROCESS_QUEUED → POSTPROCESSING → COMPLETE.
  Only COMPLETE when result is fully verified and artifact persisted.
* A failed post-processing step leaves the task in FAILED but does NOT destroy
  authenticated measurement counts.

Public API
----------
``postprocess_quantum_fragment(
    dispatch_record, registry, result_dir, *,
    qpu_service=None, counts=None,
    task_id=None, worker_id=None,
) -> QuantumPostprocessResult``

``retrieve_and_store_counts(
    qpu_job_id, qpu_service, counts_path, *, norb, expected_sha256=None
) -> CountsArtifact``

``verify_measurement_identity(
    counts_artifact, dispatch_record
) -> None``
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "QuantumPostprocessResult",
    "CountsArtifact",
    "QuantumPostprocessError",
    "MeasurementAuthenticationError",
    "postprocess_quantum_fragment",
    "retrieve_and_store_counts",
    "verify_measurement_identity",
    "POSTPROCESS_SCHEMA",
]

POSTPROCESS_SCHEMA = "qpp/1"
_VALID_SOLVERS = frozenset({"SQD", "TrimSQD"})


# ─────────────────────────────────────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class QuantumPostprocessError(RuntimeError):
    """Post-processing failed — see message for reason.

    A failed diagonalization leaves authenticated counts intact so the retry
    can reuse them without a new QPU job.
    """


class MeasurementAuthenticationError(QuantumPostprocessError):
    """The measurement counts do not match the expected provenance.

    This is a fail-closed condition — diagonalization must not proceed.
    """


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CountsArtifact:
    """Write-once measurement record.

    Attributes
    ----------
    counts_path   : path to the .npy file holding the counts dict
    counts_sha256 : SHA-256 of the file bytes (write-once identity)
    n_shots       : total shot count
    n_bits        : register width (must equal 2 * norb)
    qpu_job_id    : IBM Quantum Job ID that produced these counts
    backend       : backend name
    retrieved_utc : ISO-8601 UTC timestamp of retrieval
    """
    counts_path:   Path
    counts_sha256: str
    n_shots:       int
    n_bits:        int
    qpu_job_id:    str
    backend:       str
    retrieved_utc: str = ""

    def as_dict(self) -> dict:
        return {
            "schema":        "counts/1",
            "counts_path":   str(self.counts_path),
            "counts_sha256": self.counts_sha256,
            "n_shots":       self.n_shots,
            "n_bits":        self.n_bits,
            "qpu_job_id":    self.qpu_job_id,
            "backend":       self.backend,
            "retrieved_utc": self.retrieved_utc,
        }


@dataclass
class QuantumPostprocessResult:
    """Summary returned by postprocess_quantum_fragment.

    Attributes
    ----------
    task_id          : FragmentTaskRegistry task ID
    run_id           : run scoping namespace
    fragment_id      : 0-based fragment index
    solver           : "SQD" or "TrimSQD"
    energy           : raw embedded solver energy (Ha)
    hamiltonian_sha256 : SHA-256 of (h1e, h2e) used for diagonalization
    counts_artifact  : measurement provenance
    result_ref_dict  : FragmentResultReference.as_dict()
    postprocess_utc  : ISO-8601 UTC completion timestamp
    elapsed_s        : wall-clock seconds for diagonalization
    diagnostics      : solver-specific metadata (iterations, energy tol, …)
    """
    task_id:             str
    run_id:              str
    fragment_id:         int
    solver:              str
    energy:              float
    hamiltonian_sha256:  str
    counts_artifact:     CountsArtifact
    result_ref_dict:     dict
    postprocess_utc:     str
    elapsed_s:           float
    diagnostics:         dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "schema":              POSTPROCESS_SCHEMA,
            "task_id":             self.task_id,
            "run_id":              self.run_id,
            "fragment_id":         self.fragment_id,
            "solver":              self.solver,
            "energy":              self.energy,
            "hamiltonian_sha256":  self.hamiltonian_sha256,
            "counts_artifact":     self.counts_artifact.as_dict(),
            "result_ref":          self.result_ref_dict,
            "postprocess_utc":     self.postprocess_utc,
            "elapsed_s":           self.elapsed_s,
            "diagnostics":         self.diagnostics,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_counts(counts: dict) -> str:
    """Deterministic SHA-256 of a counts dict (sorted keys, canonical JSON)."""
    canonical = json.dumps(
        {str(k): int(v) for k, v in sorted(counts.items())},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sha256_hamiltonian(h1e, h2e) -> str:
    """Same convention as quantum_submission.load_fragment_hamiltonian_for_qpu."""
    import numpy as np
    return hashlib.sha256(
        np.asarray(h1e).tobytes() + np.asarray(h2e).tobytes()
    ).hexdigest()


def _load_fragment_hamiltonian(ewf_dumpfile_path: str | Path, fragment_id: int) -> dict:
    """Load Hamiltonian integrals from Stage 2 HDF5.

    Returns dict with h1e, h2e, norb, nocc, nelec, hamiltonian_sha256.
    """
    import h5py
    import numpy as np

    path = Path(ewf_dumpfile_path)
    if not path.is_file():
        raise QuantumPostprocessError(f"EWF dumpfile not found: {path}")

    frag_key = f"fragment_{int(fragment_id)}"
    try:
        with h5py.File(path, "r") as hf:
            if frag_key not in hf:
                raise QuantumPostprocessError(
                    f"Fragment {fragment_id} not found in {path} "
                    f"(available: {sorted(hf.keys())})"
                )
            g    = hf[frag_key]
            h1e  = g["heff"][:]
            h2e  = g["eris"][:]
            norb = int(g.attrs["norb"])
            nocc = int(g.attrs["nocc"])
    except QuantumPostprocessError:
        raise
    except Exception as exc:
        raise QuantumPostprocessError(
            f"Failed to read fragment {fragment_id} from {path}: {exc}"
        ) from exc

    ham_sha = _sha256_hamiltonian(h1e, h2e)
    nelec = (nocc, nocc)
    return {
        "h1e": h1e, "h2e": h2e,
        "norb": norb, "nocc": nocc,
        "nelec": nelec, "hamiltonian_sha256": ham_sha,
    }


def _normalise_counts_keys(counts: dict, num_bits: int) -> dict:
    """Ensure all keys are zero-padded binary strings of width ``num_bits``."""
    import numpy as np
    result = {}
    for k, v in counts.items():
        if isinstance(k, (int, np.integer)):
            result[format(int(k), f"0{num_bits}b")] = int(v)
        else:
            result[str(k)] = int(v)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Counts retrieval and write-once storage
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_and_store_counts(
    qpu_job_id: str,
    qpu_service: Any,
    counts_path: str | Path,
    *,
    norb: int,
    expected_sha256: str | None = None,
) -> CountsArtifact:
    """Retrieve measurement counts from a completed QPU job and store write-once.

    If ``counts_path`` already exists, the stored file is verified against
    ``expected_sha256`` (if provided) and returned without re-querying the provider.
    This makes retries safe: the original counts are never overwritten.

    Parameters
    ----------
    qpu_job_id      : IBM Quantum Job ID (must already be DONE)
    qpu_service     : active Qiskit Runtime service with .job() method
    counts_path     : write-once .npy file path for the counts dict
    norb            : fragment orbital count (determines expected bit width)
    expected_sha256 : if provided, verify the stored/retrieved file against this hash

    Returns
    -------
    :class:`CountsArtifact`

    Raises
    ------
    QuantumPostprocessError
        - provider query fails
        - job is not in DONE state
        - stored file hash does not match expected_sha256
    """
    import numpy as np

    counts_path = Path(counts_path)
    expected_n_bits = 2 * norb

    # ── If already stored, verify and return ─────────────────────────────────
    if counts_path.is_file():
        file_sha = _sha256_file(counts_path)
        if expected_sha256 and file_sha != expected_sha256:
            raise QuantumPostprocessError(
                f"Stored counts file {counts_path} has SHA-256 {file_sha!r} "
                f"but expected {expected_sha256!r} — file may have been tampered with."
            )
        raw = np.load(counts_path, allow_pickle=True).item()
        normalised = _normalise_counts_keys(raw, expected_n_bits)
        return CountsArtifact(
            counts_path=counts_path,
            counts_sha256=file_sha,
            n_shots=sum(normalised.values()),
            n_bits=expected_n_bits,
            qpu_job_id=qpu_job_id,
            backend="",   # backend unknown from stored file alone
            retrieved_utc=_utc_now(),
        )

    # ── Retrieve from provider ────────────────────────────────────────────────
    if qpu_service is None:
        raise QuantumPostprocessError(
            "counts_path does not exist and qpu_service is None — "
            "cannot retrieve counts without a live provider service."
        )

    try:
        job = qpu_service.job(qpu_job_id)
    except Exception as exc:
        raise QuantumPostprocessError(
            f"Failed to retrieve job {qpu_job_id!r} from provider: {exc}"
        ) from exc

    try:
        raw_status = job.status()
        status_str = raw_status.name.upper() if hasattr(raw_status, "name") else str(raw_status).upper()
        if "." in status_str:
            status_str = status_str.split(".")[-1]
    except Exception as exc:
        raise QuantumPostprocessError(
            f"Failed to query job status for {qpu_job_id!r}: {exc}"
        ) from exc

    if status_str not in ("DONE", "COMPLETED", "RESULT_AVAILABLE", "COMPLETE"):
        raise QuantumPostprocessError(
            f"QPU job {qpu_job_id!r} is not DONE (status={status_str!r}). "
            "3Q-C must only be called after the monitoring step confirms completion."
        )

    try:
        result = job.result()
        raw_counts = result[0].data.meas.get_counts()
    except Exception as exc:
        raise QuantumPostprocessError(
            f"Failed to retrieve measurement counts for job {qpu_job_id!r}: {exc}"
        ) from exc

    # Normalise integer keys if present
    if raw_counts and not isinstance(next(iter(raw_counts)), str):
        try:
            num_bits_raw = result[0].data.meas.num_bits
        except Exception:
            num_bits_raw = expected_n_bits
        raw_counts = {
            format(int(k), f"0{num_bits_raw}b"): int(v)
            for k, v in raw_counts.items()
        }
    else:
        raw_counts = {str(k): int(v) for k, v in raw_counts.items()}

    # ── Write-once save ───────────────────────────────────────────────────────
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(counts_path, raw_counts)

    file_sha = _sha256_file(counts_path)
    if expected_sha256 and file_sha != expected_sha256:
        raise QuantumPostprocessError(
            f"Freshly retrieved counts file SHA-256 {file_sha!r} "
            f"does not match expected {expected_sha256!r}."
        )

    # Attempt to read backend name from job metadata
    try:
        backend_name = str(job.backend().name)
    except Exception:
        backend_name = ""

    return CountsArtifact(
        counts_path=counts_path,
        counts_sha256=file_sha,
        n_shots=sum(raw_counts.values()),
        n_bits=expected_n_bits,
        qpu_job_id=qpu_job_id,
        backend=backend_name,
        retrieved_utc=_utc_now(),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Measurement identity verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_measurement_identity(
    counts_artifact: CountsArtifact,
    dispatch_record: dict,
    *,
    expected_n_bits: int | None = None,
) -> None:
    """Verify that the counts artifact matches the dispatch record provenance.

    Checks:
    - Non-empty counts
    - Register width (n_bits == 2 * norb from dispatch_record)
    - QPU Job ID matches
    - If expected_counts_sha256 is in dispatch_record, verify it

    Raises
    ------
    MeasurementAuthenticationError
        If any check fails.  Diagonalization must not proceed.
    """
    def _fail(reason: str) -> None:
        raise MeasurementAuthenticationError(
            f"Measurement authentication failed for fragment "
            f"{dispatch_record.get('fragment_id', '?')} "
            f"(job {dispatch_record.get('qpu_job_id', '?')!r}): {reason}"
        )

    # 1. Non-empty counts
    if counts_artifact.n_shots <= 0:
        _fail("counts are empty (0 shots)")

    # 2. QPU Job ID match
    expected_job_id = dispatch_record.get("qpu_job_id", "")
    if expected_job_id and counts_artifact.qpu_job_id != expected_job_id:
        _fail(
            f"Job ID mismatch: counts from {counts_artifact.qpu_job_id!r} "
            f"but dispatch expects {expected_job_id!r}"
        )

    # 3. Register width
    n_bits_expected = expected_n_bits
    if n_bits_expected is None:
        # Derive from dispatch_record's hamiltonian context if available
        sub_rec = dispatch_record.get("submission_record") or {}
        n_bits_expected = sub_rec.get("n_qubits") or None
    if n_bits_expected is not None and counts_artifact.n_bits != n_bits_expected:
        _fail(
            f"Register width mismatch: counts have {counts_artifact.n_bits} bits "
            f"but expected {n_bits_expected} bits (2 × norb)"
        )

    # 4. Counts SHA-256 match if recorded
    sub_rec = dispatch_record.get("submission_record") or {}
    recorded_sha = sub_rec.get("counts_sha256", "")
    if recorded_sha and counts_artifact.counts_sha256 != recorded_sha:
        _fail(
            f"Counts file SHA-256 mismatch: stored {counts_artifact.counts_sha256!r} "
            f"but expected {recorded_sha!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _claim_postprocess_task(registry, task_id: str, worker_id: str) -> tuple:
    """Claim a POSTPROCESS_QUEUED task for execution.

    Returns (TaskRecord, lease_token).
    """
    # Task states are serialized string values. The Serverless worker
    # must not depend on the organizer-only proteasome_challenge package.
    _QUEUED = "POSTPROCESS_QUEUED"
    _PROCESSING = "POSTPROCESSING"

    task = registry.get_task(task_id)

    # Idempotency: already in POSTPROCESSING (worker restart)
    if task.state == _PROCESSING:
        return task, task.lease_token or ""

    if task.state != _QUEUED:
        raise QuantumPostprocessError(
            f"Task {task_id!r}: cannot claim for post-processing in state {task.state!r}. "
            f"Expected POSTPROCESS_QUEUED."
        )

    try:
        task = registry.transition(
            task_id, _QUEUED, _PROCESSING,
            worker_id=worker_id,
            reason="3Q-C: post-processing claimed",
        )
        return task, task.lease_token
    except Exception as exc:
        raise QuantumPostprocessError(
            f"Failed to claim task {task_id!r} for post-processing: {exc}"
        ) from exc


def _fail_postprocess_task(registry, task_id: str, reason: str,
                           lease_token: str = "") -> None:
    """Mark the post-processing task FAILED without destroying counts."""
    try:
        registry.fail_task(task_id, reason=reason, lease_token=lease_token or "")
    except Exception:
        pass  # Swallow — task may already be in a terminal state


# ─────────────────────────────────────────────────────────────────────────────
#  Core post-processing function
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_quantum_fragment(
    dispatch_record: dict,
    registry,
    result_dir: str | Path,
    *,
    qpu_service: Any = None,
    counts: dict | None = None,
    task_id: str | None = None,
    worker_id: str = "3qc_worker",
    sqd_config: dict | None = None,
    trim_config: dict | None = None,
) -> QuantumPostprocessResult:
    """Execute independent SQD or TrimSQD post-processing for one quantum fragment.

    This is the authoritative entry point for Stage 3Q-C.  It accepts a durable
    dispatch record (from Task 5.6 ``create_dispatch_record``), retrieves and
    authenticates measurement counts, runs the inherited SQD/TrimSQD
    diagonalization, and persists a complete SolverResult via the frs/1 store.

    **Never submits a QPU job.**

    Parameters
    ----------
    dispatch_record     : dict from Task 5.6 ``create_dispatch_record``
    registry            : :class:`FragmentTaskRegistry` instance
    result_dir          : directory for result HDF5 and counts .npy files
    qpu_service         : active IBM Quantum service (required if counts not provided)
    counts              : pre-loaded counts dict {bitstring: int} (bypasses retrieval)
    task_id             : override task_id from dispatch_record
    worker_id           : identifier logged in registry audit trail
    sqd_config          : override SQD configuration (uses dispatch_record defaults if None)
    trim_config         : override TrimSQD trimming configuration

    Returns
    -------
    :class:`QuantumPostprocessResult`

    Raises
    ------
    QuantumPostprocessError
        - dispatch_record missing required fields
        - counts cannot be retrieved or authenticated
        - SQD/TrimSQD diagonalization fails
        - result cannot be persisted
    MeasurementAuthenticationError
        - counts do not match expected provenance
    """
    import numpy as np

    # ── 1. Resolve task identity from dispatch_record ─────────────────────────
    resolved_task_id = task_id or dispatch_record.get("task_id", "")
    run_id      = dispatch_record.get("run_id", "")
    fragment_id = int(dispatch_record.get("fragment_id", -1))
    solver      = dispatch_record.get("solver", "")
    qpu_job_id  = dispatch_record.get("qpu_job_id", "")
    backend     = dispatch_record.get("backend", "")
    ham_sha_expected = dispatch_record.get("hamiltonian_sha256", "")
    ewf_dumpfile_path = dispatch_record.get("ewf_dumpfile_path", "")

    if not resolved_task_id:
        raise QuantumPostprocessError("dispatch_record missing 'task_id'")
    if fragment_id < 0:
        raise QuantumPostprocessError("dispatch_record missing or invalid 'fragment_id'")
    if solver not in _VALID_SOLVERS:
        raise QuantumPostprocessError(
            f"dispatch_record solver {solver!r} must be one of {sorted(_VALID_SOLVERS)}"
        )
    if not qpu_job_id:
        raise QuantumPostprocessError("dispatch_record missing 'qpu_job_id'")
    if not ewf_dumpfile_path:
        raise QuantumPostprocessError("dispatch_record missing 'ewf_dumpfile_path'")

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Claim registry task (POSTPROCESS_QUEUED → POSTPROCESSING) ──────────
    task, lease_token = _claim_postprocess_task(registry, resolved_task_id, worker_id)

    # ── 3. Load fragment Hamiltonian from Stage 2 HDF5 ────────────────────────
    try:
        ham = _load_fragment_hamiltonian(ewf_dumpfile_path, fragment_id)
    except QuantumPostprocessError as exc:
        _fail_postprocess_task(registry, resolved_task_id,
                               f"3Q-C HDF5 load failed: {exc}", lease_token)
        raise

    h1e   = ham["h1e"]
    h2e   = ham["h2e"]
    norb  = ham["norb"]
    nocc  = ham["nocc"]
    nelec = ham["nelec"]
    ham_sha_actual = ham["hamiltonian_sha256"]

    # ── 4. Verify Hamiltonian identity ────────────────────────────────────────
    if ham_sha_expected and ham_sha_actual != ham_sha_expected:
        reason = (
            f"3Q-C Hamiltonian mismatch: fragment {fragment_id} in "
            f"{ewf_dumpfile_path} has SHA-256 {ham_sha_actual[:12]}... "
            f"but dispatch expects {ham_sha_expected[:12]}..."
        )
        _fail_postprocess_task(registry, resolved_task_id, reason, lease_token)
        raise QuantumPostprocessError(reason)

    # ── 5. Load or retrieve measurement counts ────────────────────────────────
    counts_path = result_dir / f"counts_frag{fragment_id}_{qpu_job_id[:12]}.npy"

    if counts is not None:
        # Caller-supplied counts (test mode or counts already loaded)
        np.save(counts_path, counts)
        file_sha = _sha256_file(counts_path)
        counts_artifact = CountsArtifact(
            counts_path=counts_path,
            counts_sha256=file_sha,
            n_shots=sum(counts.values()),
            n_bits=2 * norb,
            qpu_job_id=qpu_job_id,
            backend=backend,
            retrieved_utc=_utc_now(),
        )
    else:
        try:
            counts_artifact = retrieve_and_store_counts(
                qpu_job_id, qpu_service, counts_path, norb=norb,
            )
        except QuantumPostprocessError as exc:
            _fail_postprocess_task(registry, resolved_task_id,
                                   f"3Q-C counts retrieval failed: {exc}", lease_token)
            raise
        # Load normalised counts from artifact
        raw = np.load(counts_path, allow_pickle=True).item()
        counts = _normalise_counts_keys(raw, 2 * norb)

    # ── 6. Verify measurement identity ────────────────────────────────────────
    try:
        verify_measurement_identity(
            counts_artifact, dispatch_record,
            expected_n_bits=2 * norb,
        )
    except MeasurementAuthenticationError as exc:
        _fail_postprocess_task(registry, resolved_task_id,
                               f"3Q-C measurement authentication failed: {exc}", lease_token)
        raise

    if counts is None:
        # Load normalised counts from the write-once file
        raw = np.load(counts_path, allow_pickle=True).item()
        counts = _normalise_counts_keys(raw, 2 * norb)

    if not counts:
        reason = f"3Q-C fragment {fragment_id}: counts dict is empty after retrieval"
        _fail_postprocess_task(registry, resolved_task_id, reason, lease_token)
        raise QuantumPostprocessError(reason)

    # ── 7. Validate bit width ─────────────────────────────────────────────────
    _check_counts_bit_width(counts, norb, fragment_id)

    # ── 8. Build solver configuration ────────────────────────────────────────
    sub_rec = dispatch_record.get("submission_record") or {}
    _effective_sqd_config = _build_sqd_config(dispatch_record, sub_rec, sqd_config)
    _effective_trim_config = _build_trim_config(dispatch_record, sub_rec, trim_config)

    # ── 9. Run SQD or TrimSQD diagonalization ────────────────────────────────
    t_start = time.time()
    try:
        solver_result = _run_diagonalization(
            solver=solver,
            h1e=h1e, h2e=h2e,
            counts=counts,
            norb=norb, nelec=nelec,
            sqd_config=_effective_sqd_config,
            trim_config=_effective_trim_config,
            result_dir=result_dir,
        )
    except Exception as exc:
        reason = f"3Q-C diagonalization failed for fragment {fragment_id}: {exc}"
        _fail_postprocess_task(registry, resolved_task_id, reason, lease_token)
        # Counts artifact is preserved — retry can reuse it
        raise QuantumPostprocessError(reason) from exc
    elapsed_s = time.time() - t_start

    # ── 10. Verify result completeness ────────────────────────────────────────
    _verify_solver_result(solver_result, norb, nelec)

    # ── Stamp result metadata for Stage 4 reconstruction and frs/1 persistence ─
    if hasattr(solver_result, "metadata") and isinstance(solver_result.metadata, dict):
        solver_result.metadata["solver_label"] = solver
        solver_result.metadata["hamiltonian_sha256"] = ham_sha_actual
        if "nocc" not in solver_result.metadata:
            solver_result.metadata["nocc"] = int(nelec[0]) if isinstance(nelec, (tuple, list)) else int(nelec)

    # ── 11. Persist SolverResult via frs/1 ───────────────────────────────────
    result_path = result_dir / f"result_frag{fragment_id}_{solver.lower()}_{qpu_job_id[:12]}.h5"
    try:
        _sys_path_ensure_bundle()
        try:
            from serverless_bundle.steps.fragment_result_store import save_fragment_result
        except ModuleNotFoundError as exc:
            if exc.name not in ("serverless_bundle", "serverless_bundle.steps"):
                raise
            from steps.fragment_result_store import save_fragment_result
        ref = save_fragment_result(
            solver_result, result_path,
            fragment_id=fragment_id,
            run_id=run_id,
            solver=solver,
            hamiltonian_sha256=ham_sha_actual,
        )
    except Exception as exc:
        reason = f"3Q-C result persistence failed for fragment {fragment_id}: {exc}"
        _fail_postprocess_task(registry, resolved_task_id, reason, lease_token)
        raise QuantumPostprocessError(reason) from exc

    # ── 12. Record result in registry and transition to COMPLETE ──────────────
    try:
        registry.complete_task(
            resolved_task_id,
            artifact_uri=str(ref.path),
            payload_digest=ref.payload_digest,
            lease_token=lease_token or "",
            reason=(
                f"3Q-C complete: fragment {fragment_id} {solver} "
                f"E={solver_result.energy:.8f} Ha"
            ),
        )
    except Exception as exc:
        raise QuantumPostprocessError(
            f"Registry COMPLETE transition failed for task {resolved_task_id!r}: {exc}"
        ) from exc

    return QuantumPostprocessResult(
        task_id=resolved_task_id,
        run_id=run_id,
        fragment_id=fragment_id,
        solver=solver,
        energy=solver_result.energy,
        hamiltonian_sha256=ham_sha_actual,
        counts_artifact=counts_artifact,
        result_ref_dict=ref.as_dict(),
        postprocess_utc=_utc_now(),
        elapsed_s=elapsed_s,
        diagnostics={
            "norb": norb,
            "nelec": nelec,
            "n_shots": counts_artifact.n_shots,
            "sqd_config": _effective_sqd_config,
            "counts_sha256": counts_artifact.counts_sha256,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_counts_bit_width(counts: dict, norb: int, fragment_id: int) -> None:
    """Verify that all bitstring keys have width 2 * norb."""
    expected = 2 * norb
    widths = {len(k) for k in counts if isinstance(k, str)}
    if widths and widths != {expected}:
        raise QuantumPostprocessError(
            f"Fragment {fragment_id}: counts bit width {sorted(widths)} "
            f"does not match expected {expected} (2 × {norb} orbitals). "
            "These counts may belong to a different fragment."
        )


def _build_sqd_config(dispatch_record: dict, sub_rec: dict,
                      override: dict | None) -> dict:
    """Resolve SQD configuration from dispatch record or override."""
    if override is not None:
        return override
    # Pull from submission_record if available
    sr_sqd = sub_rec.get("sqd_config") or {}
    dr_sqd = (dispatch_record.get("submission_record") or {}).get("sqd_config") or {}
    base = {**sr_sqd, **dr_sqd}
    # Defaults that match the established SQD canonical config
    defaults = {
        "classical_backend": "python",
        "symmetrize_spin": True,
        "n_batches": 5,
        "iterations": 5,
        "samples_per_batch": 300,
        "energy_tol": 1e-8,
        "occupancies_tol": 1e-5,
        "carryover_threshold": 1e-4,
        "seed": 20260626,   # CANONICAL_SEED
    }
    return {**defaults, **base}


def _build_trim_config(dispatch_record: dict, sub_rec: dict,
                       override: dict | None) -> dict:
    """Resolve TrimSQD configuration from dispatch record or override."""
    if override is not None:
        return override
    sr_trim = sub_rec.get("trim_config") or {}
    dr_trim = (dispatch_record.get("submission_record") or {}).get("trim_config") or {}
    base = {**sr_trim, **dr_trim}
    defaults = {
        "trim_ratio": 0.5,
        "sector_trim": False,
        "partition_carryover": True,
    }
    return {**defaults, **base}


def _sys_path_ensure_bundle() -> None:
    """Ensure serverless_bundle is importable."""
    candidates = [
        Path(__file__).resolve().parents[2],          # steps/../../ == serverless_bundle/..
        Path(__file__).resolve().parents[1].parent,
    ]
    for c in candidates:
        bundle = c / "serverless_bundle"
        if bundle.is_dir():
            s = str(c)
            if s not in sys.path:
                sys.path.insert(0, s)
            return


def _run_diagonalization(
    *,
    solver: str,
    h1e, h2e,
    counts: dict,
    norb: int,
    nelec: tuple,
    sqd_config: dict,
    trim_config: dict,
    result_dir: Path,
) -> Any:
    """Instantiate the correct solver and call _sbd_postprocessing directly.

    This is the single point of contact with SQD/TrimSQD science.
    It reuses the inherited _sbd_postprocessing method without modification.
    """
    _sys_path_ensure_bundle()
    bundle_root = Path(__file__).resolve().parents[2]

    # Add quantum_fragment_methods to path if needed
    qfm_parent = bundle_root / "serverless_bundle"
    if str(qfm_parent) not in sys.path:
        sys.path.insert(0, str(qfm_parent))

    try:
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import SQDSolver
        from quantum_fragment_methods.application.solvers.quantum_zoo.trim_sqd import TrimSQDSolver
        from quantum_fragment_methods.qpu.qiskit_ibm_runtime import IBMQuantumBackend
    except ImportError as exc:
        raise QuantumPostprocessError(
            f"Cannot import SQD/TrimSQD solvers: {exc}"
        ) from exc

    # Build a minimal backend object (no live QPU connection needed for post-processing)
    dummy_cfg = {
        "provider":     "ibm_quantum",
        "backend_name": "offline",
        "credentials":  {"channel": "ibm_cloud", "token": "", "instance": ""},
        "sampler_options": {},
    }
    dummy_backend = IBMQuantumBackend(dummy_cfg)
    dummy_backend.service         = None
    dummy_backend.backend         = None
    dummy_backend._qiskit_backend = None

    # Full solver config
    full_config = {
        **sqd_config,
        "lucj": {},
        "classical_backend": sqd_config.get("classical_backend", "python"),
        "sampler_options": {},
    }
    if solver == "TrimSQD":
        full_config["trim"] = trim_config

    if solver == "TrimSQD":
        s = TrimSQDSolver(qpu_backend=dummy_backend, config=full_config)
    else:
        s = SQDSolver(qpu_backend=dummy_backend, config=full_config)

    # Call _sbd_postprocessing directly — no QPU submission involved
    return s._sbd_postprocessing(h1e, h2e, counts, norb, nelec, result_dir)


def _verify_solver_result(result: Any, norb: int, nelec: tuple) -> None:
    """Verify the SolverResult contains a complete RDM set.

    A result missing RDMs cannot participate in EWF reconstruction and must
    not be committed to the registry as COMPLETE.
    """
    if result is None:
        raise QuantumPostprocessError("Diagonalization returned None")

    energy = getattr(result, "energy", None)
    if energy is None or not isinstance(energy, (int, float)):
        raise QuantumPostprocessError(
            f"SolverResult.energy is not a finite scalar: {energy!r}"
        )
    if not (energy == energy):  # NaN check
        raise QuantumPostprocessError(
            f"SolverResult.energy is NaN — diagonalization did not converge"
        )

    rdm1 = getattr(result, "rdm1", None)
    rdm2 = getattr(result, "rdm2", None)
    if rdm1 is None:
        raise QuantumPostprocessError(
            "SolverResult.rdm1 is None — 1-RDM required for EWF reconstruction"
        )
    if rdm2 is None:
        raise QuantumPostprocessError(
            "SolverResult.rdm2 is None — 2-RDM required for EWF reconstruction"
        )

    # Shape checks
    import numpy as np
    rdm1 = np.asarray(rdm1)
    rdm2 = np.asarray(rdm2)
    if rdm1.shape != (norb, norb):
        raise QuantumPostprocessError(
            f"SolverResult.rdm1 shape {rdm1.shape} does not match "
            f"expected ({norb}, {norb})"
        )
    if rdm2.shape != (norb, norb, norb, norb):
        raise QuantumPostprocessError(
            f"SolverResult.rdm2 shape {rdm2.shape} does not match "
            f"expected ({norb}, {norb}, {norb}, {norb})"
        )

    # Check for NaN/Inf in RDMs
    if not np.all(np.isfinite(rdm1)):
        raise QuantumPostprocessError("SolverResult.rdm1 contains NaN or Inf")
    if not np.all(np.isfinite(rdm2)):
        raise QuantumPostprocessError("SolverResult.rdm2 contains NaN or Inf")
