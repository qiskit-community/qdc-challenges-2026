"""
quantum_submission — Stage 3Q-A: Independent QPU submission.

Prepare → validate → build LUCJ circuit → submit → persist QPU Job ID → return immediately.

This module is the authoritative entry point for **3Q-A** (asynchronous QPU submission).
It separates QPU job submission from QPU completion waiting and SQD/TrimSQD
post-processing, which remain in the existing ``step3_solve.run_solve`` path until
Tasks 5.6 (3Q-B monitoring) and 5.7 (3Q-C post-processing) replace them.

Design constraints
------------------
* Organizer-side submission (Option A): the notebook/controller calls this function,
  registers the attempt in the SQLite registry (Task 5.3), submits, and immediately
  persists the QPU Job ID.  No distributed database required.
* Never retrieves counts, diagonalizes a Hamiltonian, constructs RDMs, or runs EWF
  reconstruction.
* Never imports SQD/TrimSQD diagonalization at module level.
* Preserves the existing ``SQDSolver._build_lucj_circuit`` and
  ``SQDSolver._submit_to_qpu`` — the circuit science is unchanged.
* Uses the existing measurement plan and measurement routing infrastructure
  (``proteasome_challenge/measurement_plan.py``) for provenance.
* Uses the Task 5.3 ``FragmentTaskRegistry`` for durable task state.
* Reads fragment Hamiltonians via the same HDF5 convention as ``step3_solve``.

Submission boundary decision (Option A vs B)
--------------------------------------------
**Option A — organizer-side direct QPU submission** is chosen.

The existing ``MeasurementRoutingMixin.submit_measurement_plan`` already demonstrates
this pattern: the notebook prepares the job, calls the QPU API, and writes a JSON job
record to ``checkpoints/measurements/jobs/``.  That write is the Job ID persistence.

Option B (lightweight Serverless worker) adds a distributed handoff between the worker
and the notebook's SQLite registry, creating the exact ambiguous-acceptance failure mode
described in the mission.  With Option A the organizer holds both the Qiskit service
object AND the SQLite file, so no cross-process transport is needed for Job ID
persistence.

The 3Q-A function can be called from a notebook cell or a controller script.  It requires
no Serverless deployment for the submission step itself; only the post-processing
(Task 5.7 / 3Q-C) requires a Serverless compute worker.

Submission ambiguity
--------------------
If the provider accepts the job (returns a Job ID) but the controller crashes before
persisting the Job ID in the registry, the task is left in SUBMITTING state.  An
AMBIGUOUS_SUBMISSION sub-state is set via the registry ``reason`` field, and the Job ID
(if recovered) is stored in the failure metadata.  Automatic re-submission is NEVER
performed.  Recovery requires manual reconciliation:

1. Query the provider using the ``reconcile_submission`` function in this module.
2. Verify the Job ID by listing recent IBM Quantum jobs filtered by the submission tags.
3. Call ``resolve_ambiguous_submission`` to record the recovered Job ID and advance to
   SUBMITTED.

Public API
----------
``submit_quantum_fragment(
    ewf_dumpfile_path, fragment_id, solver, run_id, backend_name, *,
    n_shots, n_reps, registry, task_id, lease_token, hamiltonian_sha256,
    worker_id=None, job_tags=None, qpu_service=None,
    qpu_record_path=None | campaign_auth=None,   # one is MANDATORY
) -> SubmissionRecord``

``reconcile_submission(registry, task_id, *, qpu_service, tags) -> str | None``

``resolve_ambiguous_submission(registry, task_id, recovered_job_id, *, lease_token) -> SubmissionRecord``

``build_submission_tags(run_id, fragment_id, solver, basis) -> list[str]``
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "SubmissionRecord",
    "QuantumSubmissionError",
    "AmbiguousSubmissionError",
    "DuplicateSubmissionError",
    "QpuBudgetError",
    "check_qpu_budget",
    "record_qpu_sampling",
    "reserve_qpu_sampling",
    "check_unreconciled_quantum_tasks",
    "get_qpu_sampling_count",
    "submit_quantum_fragment",
    "build_submission_tags",
    "reconcile_submission",
    "resolve_ambiguous_submission",
    "load_fragment_hamiltonian_for_qpu",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_QPU_RECORD_PATH = _REPO_ROOT / "checkpoints" / ".qpu_sampling_record.json"

VALID_QUANTUM_SOLVERS: frozenset[str] = frozenset({"SQD", "TrimSQD"})


# ─────────────────────────────────────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class QuantumSubmissionError(RuntimeError):
    """A quantum fragment submission failed or was given an invalid request."""


class AmbiguousSubmissionError(QuantumSubmissionError):
    """QPU provider accepted the job but the Job ID was not durably persisted.

    The submission is in an AMBIGUOUS state.  Automatic resubmission is prohibited.
    Use ``reconcile_submission`` and ``resolve_ambiguous_submission`` to recover.
    """
    def __init__(self, message: str, *, recovered_job_id: str | None = None):
        super().__init__(message)
        self.recovered_job_id = recovered_job_id


class DuplicateSubmissionError(QuantumSubmissionError):
    """A submission was attempted for a fragment task that already has a Job ID.

    Automatic resubmission that would silently consume a QPU allocation is prohibited.
    """


class QpuBudgetError(QuantumSubmissionError):
    """A live QPU submission was attempted without budget authorization."""


# ─────────────────────────────────────────────────────────────────────────────
#  Persistent QPU budget ledger management
# ─────────────────────────────────────────────────────────────────────────────

def get_qpu_sampling_count(qpu_record_path: str | Path | None = None) -> int:
    """Read the current QPU sampling count from the persistent ledger file."""
    path = Path(qpu_record_path) if qpu_record_path is not None else _DEFAULT_QPU_RECORD_PATH
    if not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("count", 0))
    except Exception as exc:
        raise QpuBudgetError(f"Failed to read QPU sampling record at {path}: {exc}") from exc


def _resolve_ledger_path(qpu_record_path: str | Path | None) -> Path:
    """Resolve the legacy ledger path and refuse the real ledger under pytest."""
    path = Path(qpu_record_path) if qpu_record_path is not None else _DEFAULT_QPU_RECORD_PATH
    if os.environ.get("PYTEST_CURRENT_TEST") and path.resolve() == _DEFAULT_QPU_RECORD_PATH.resolve():
        raise QpuBudgetError(
            "Offline tests must not touch the persistent QPU ledger "
            f"({_DEFAULT_QPU_RECORD_PATH.name}); pass a temporary qpu_record_path."
        )
    return path


@contextlib.contextmanager
def _ledger_file_lock(path: Path):
    """Exclusive inter-process lock for the legacy ``{"count": N}`` ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _write_count_atomic(path: Path, count: int) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"count": count}, indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _journal(path: Path, event: dict) -> None:
    """Append-only audit journal beside the legacy ledger (never read for budget decisions)."""
    with open(path.with_name(path.name + ".journal.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def check_qpu_budget(
    qpu_record_path: str | Path | None = None,
    *,
    allow_second_qpu: bool = False,
    campaign_auth: Optional[Any] = None,
) -> int:
    """Verify that QPU budget allows a new live submission; raises QpuBudgetError if blocked.

    Read-only advisory check.  The authoritative, race-free check happens again
    under the ledger lock inside :func:`reserve_qpu_sampling` / ``reserve_submission``.
    """
    if campaign_auth is not None:
        try:
            campaign_auth.check_budget()
            return campaign_auth.get_submission_count()
        except Exception as exc:
            raise QpuBudgetError(f"Campaign QPU budget gate: {exc}") from exc

    path = _resolve_ledger_path(qpu_record_path)
    count = get_qpu_sampling_count(path)
    if count >= 1 and not allow_second_qpu:
        raise QpuBudgetError(
            f"QPU budget gate: {count} live QPU sampling job(s) already recorded in {path.name}.\n"
            "Reuse saved counts or set allow_second_qpu=True to explicitly authorize another QPU job."
        )
    return count


def reserve_qpu_sampling(
    qpu_record_path: str | Path,
    *,
    allow_second_qpu: bool = False,
    attempt_id: str = "",
    fragment_id: Optional[int] = None,
    task_id: Optional[str] = None,
) -> int:
    """Atomically check the budget AND consume one unit BEFORE the provider call.

    Conservative accounting: the unit stays consumed whether or not the provider
    accepts the job.  Never decrements.  Check and increment happen under one
    exclusive lock so two concurrent submitters cannot both pass the gate.
    """
    path = _resolve_ledger_path(qpu_record_path)
    with _ledger_file_lock(path):
        count = get_qpu_sampling_count(path)
        if count >= 1 and not allow_second_qpu:
            raise QpuBudgetError(
                f"QPU budget gate: {count} live QPU sampling job(s) already recorded in {path.name}.\n"
                "Reuse saved counts or set allow_second_qpu=True to explicitly authorize another QPU job."
            )
        _write_count_atomic(path, count + 1)
        _journal(path, {"event": "RESERVED", "count": count + 1, "attempt_id": attempt_id,
                        "fragment_id": fragment_id, "task_id": task_id, "utc": _utc_now()})
    return count + 1


def record_qpu_sampling(
    qpu_record_path: str | Path | None = None,
    *,
    campaign_auth: Optional[Any] = None,
    fragment_id: Optional[int] = None,
    task_id: Optional[str] = None,
    job_id: Optional[str] = None,
    backend: Optional[str] = None,
    attempt_id: Optional[str] = None,
    n_shots: Optional[int] = None,
) -> int:
    """Increment the persistent QPU sampling count by 1 (or record campaign submission).

    Exceptions are never swallowed.  Legacy ledger: locked read-modify-write with
    an atomic replace.  Campaign ledger: confirms an existing reservation for the
    same ``attempt_id`` instead of consuming a second unit.
    """
    if campaign_auth is not None:
        if fragment_id is not None and job_id is not None:
            kwargs = {}
            if attempt_id is not None:
                kwargs["attempt_id"] = attempt_id
            if n_shots is not None:
                kwargs["n_shots"] = n_shots
            return campaign_auth.record_submission(
                fragment_id=fragment_id,
                task_id=task_id or "",
                job_id=job_id,
                backend=backend or "",
                **kwargs,
            )
        return campaign_auth.get_submission_count()

    path = _resolve_ledger_path(qpu_record_path)
    with _ledger_file_lock(path):
        count = get_qpu_sampling_count(path) + 1
        _write_count_atomic(path, count)
        _journal(path, {"event": "RECORDED", "count": count, "attempt_id": attempt_id,
                        "fragment_id": fragment_id, "task_id": task_id, "job_id": job_id,
                        "utc": _utc_now()})
    return count


def check_unreconciled_quantum_tasks(registry, run_id: str, *, exclude_task_id: str = "") -> list[str]:
    """Return task IDs whose earlier QPU submission is unresolved.

    A quantum task is unreconciled when it is AMBIGUOUS, or SUBMITTING with no
    provider Job ID recorded.  No further QPU allocation may proceed while any
    such task exists, because the earlier attempt may have consumed provider quota.
    """
    blocked: list[str] = []
    for t in registry.list_tasks(run_id, task_type="quantum"):
        if t.task_id == exclude_task_id:
            continue
        if t.state == "AMBIGUOUS" or (t.state == "SUBMITTING" and not t.qpu_job_id):
            blocked.append(t.task_id)
    return blocked


# ─────────────────────────────────────────────────────────────────────────────
#  SubmissionRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubmissionRecord:
    """Complete provenance of one QPU submission attempt.

    All fields required for 3Q-B monitoring and 3Q-C post-processing retrieval.

    Attributes
    ----------
    run_id              : run scoping namespace
    fragment_id         : 0-based fragment index
    task_id             : logical task ID in the FragmentTaskRegistry
    submission_attempt  : monotonic attempt ID (UUID)
    solver              : "SQD" or "TrimSQD"
    hamiltonian_sha256  : sha256(h1e.tobytes() + h2e.tobytes())
    circuit_digest      : sha256 of the serialized ISA circuit bytes
    lucj_config_digest  : sha256 of the LUCJ config dict (n_reps, etc.)
    backend_name        : IBM Quantum backend name (e.g. "ibm_fez")
    n_shots             : requested shot count
    n_reps              : LUCJ repetitions
    job_id              : IBM Quantum Job ID (None if submission failed pre-provider)
    submission_tags     : IBM Quantum job tags applied at submission time
    submitted_utc       : ISO-8601 UTC timestamp of submission
    submission_state    : "SUBMITTED" | "AMBIGUOUS" | "FAILED"
    failure_reason      : failure or ambiguity description (None if SUBMITTED)
    n_qubits            : 2 * norb (circuit width)
    norb                : number of spatial orbitals
    nocc                : number of occupied orbitals
    code_version        : dict of relevant software versions
    """
    run_id:             str
    fragment_id:        int
    task_id:            str
    submission_attempt: str
    solver:             str
    hamiltonian_sha256: str
    circuit_digest:     str
    lucj_config_digest: str
    backend_name:       str
    n_shots:            int
    n_reps:             int
    job_id:             Optional[str]
    submission_tags:    list
    submitted_utc:      str
    submission_state:   str
    failure_reason:     Optional[str]
    n_qubits:           int
    norb:               int
    nocc:               int
    code_version:       dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "run_id":             self.run_id,
            "fragment_id":        self.fragment_id,
            "task_id":            self.task_id,
            "submission_attempt": self.submission_attempt,
            "solver":             self.solver,
            "hamiltonian_sha256": self.hamiltonian_sha256,
            "circuit_digest":     self.circuit_digest,
            "lucj_config_digest": self.lucj_config_digest,
            "backend_name":       self.backend_name,
            "n_shots":            self.n_shots,
            "n_reps":             self.n_reps,
            "job_id":             self.job_id,
            "submission_tags":    self.submission_tags,
            "submitted_utc":      self.submitted_utc,
            "submission_state":   self.submission_state,
            "failure_reason":     self.failure_reason,
            "n_qubits":           self.n_qubits,
            "norb":               self.norb,
            "nocc":               self.nocc,
            "code_version":       self.code_version,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _config_digest(cfg: dict) -> str:
    """Deterministic SHA-256 of a config dict (same convention as step3_solve)."""
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()


def _circuit_digest(isa_circuit: Any) -> str:
    """SHA-256 of the circuit's QPY serialization bytes.

    Falls back to a digest of the circuit's ``__repr__`` when QPY is unavailable,
    so offline tests can compute a stable deterministic digest from a mock circuit.
    """
    try:
        import io
        from qiskit.qpy import dump as qpy_dump
        buf = io.BytesIO()
        qpy_dump(isa_circuit, buf)
        return hashlib.sha256(buf.getvalue()).hexdigest()
    except Exception:
        return hashlib.sha256(repr(isa_circuit).encode()).hexdigest()


def _software_versions() -> dict:
    """Versions of relevant packages."""
    out: dict = {}
    for name in ("qiskit", "qiskit_ibm_runtime", "ffsim", "numpy"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "unavailable"
    return out


def build_submission_tags(
    run_id: str,
    fragment_id: int,
    solver: str,
    basis: str = "sto-3g",
) -> list[str]:
    """Canonical IBM Quantum job tags for a quantum fragment submission.

    Format (≤5 tags, ≤20 chars each, matching existing step3_solve convention):
        [run_id[:20], "fragment_<id>", solver.lower(), basis[:20]]

    These tags are used for provider-side job reconciliation when the Job ID
    is lost after submission.  They must be deterministic and unique per
    (run_id, fragment_id, solver).
    """
    return [
        run_id[:20],
        f"fragment_{int(fragment_id)}",
        solver.lower()[:20],
        basis[:20],
    ]


def load_fragment_hamiltonian_for_qpu(
    ewf_dumpfile_path: str | Path,
    fragment_id: int,
) -> dict:
    """Extract Hamiltonian integrals from Stage 2 HDF5 for QPU circuit construction.

    Returns dict with: h1e, h2e, norb, nocc, hamiltonian_sha256, nelec.

    Raises QuantumSubmissionError if the file or fragment is missing.
    """
    import h5py
    import numpy as np

    path = Path(ewf_dumpfile_path)
    if not path.is_file():
        raise QuantumSubmissionError(f"EWF dumpfile not found: {path}")

    frag_key = f"fragment_{int(fragment_id)}"
    try:
        with h5py.File(path, "r") as hf:
            if frag_key not in hf:
                raise QuantumSubmissionError(
                    f"Fragment {fragment_id} not found in {path} "
                    f"(available: {sorted(hf.keys())})"
                )
            g    = hf[frag_key]
            h1e  = g["heff"][:]
            h2e  = g["eris"][:]
            norb = int(g.attrs["norb"])
            nocc = int(g.attrs["nocc"])
    except QuantumSubmissionError:
        raise
    except Exception as exc:
        raise QuantumSubmissionError(
            f"Failed to read fragment {fragment_id} from {path}: {exc}"
        ) from exc

    ham_sha = hashlib.sha256(h1e.tobytes() + h2e.tobytes()).hexdigest()
    nelec = (nocc, nocc)   # RHF: equal alpha/beta occupation
    return {
        "h1e": h1e,
        "h2e": h2e,
        "norb": norb,
        "nocc": nocc,
        "nelec": nelec,
        "hamiltonian_sha256": ham_sha,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Registry transition helpers
# ─────────────────────────────────────────────────────────────────────────────

def _registry_transition(registry, task_id, from_state, to_state, *,
                          worker_id, lease_token, reason):
    """Wrap a registry transition; raises QuantumSubmissionError on failure."""
    try:
        registry.transition(
            task_id, from_state, to_state,
            worker_id=worker_id or "",
            lease_token=lease_token or "",
            reason=reason,
        )
    except Exception as exc:
        raise QuantumSubmissionError(
            f"Registry transition {from_state}→{to_state} for task {task_id!r} failed: {exc}"
        ) from exc


def _registry_fail(registry, task_id, reason, *, lease_token):
    """Mark a task FAILED for confirmed pre-submission errors.

    Use only when the QPU was definitely NOT contacted (e.g. HDF5 load failure,
    Hamiltonian mismatch, circuit build failure).  Never use after _submit_circuit
    has been called — use _registry_mark_ambiguous instead.
    """
    try:
        registry.fail_task(task_id, reason=reason, lease_token=lease_token or "")
    except Exception:
        pass  # swallow secondary exceptions to not mask the primary error


def _registry_mark_ambiguous(
    registry,
    task_id: str,
    reason: str,
    *,
    lease_token: str,
    candidate_job_id: str = "",
) -> None:
    """Transition task to AMBIGUOUS after uncertain QPU submission.

    Called when the QPU provider may have accepted a job but the Job ID
    could not be durably persisted.  Unlike _registry_fail this does NOT
    mark the task as terminally failed — it records the uncertainty so
    the operator can recover via registry.recover_ambiguous_submission().

    The candidate_job_id (if available) is stored in failure_reason for
    operator reference, even though the registry transition itself may
    not be able to persist a formal qpu_job_id.
    """
    full_reason = reason
    if candidate_job_id:
        full_reason = f"{reason}  candidate_job_id={candidate_job_id!r}"
    try:
        registry.transition(
            task_id, "SUBMITTING", "AMBIGUOUS",
            lease_token=lease_token or "",
            reason=full_reason,
        )
    except Exception:
        # If we cannot even transition to AMBIGUOUS, fall back to FAILED
        # so the task is not permanently stuck in SUBMITTING.
        try:
            registry.fail_task(
                task_id,
                reason=(
                    f"AMBIGUOUS_SUBMISSION (registry transition to AMBIGUOUS failed): {full_reason}"
                ),
                lease_token=lease_token or "",
            )
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Core submission function
# ─────────────────────────────────────────────────────────────────────────────

def submit_quantum_fragment(
    ewf_dumpfile_path: str | Path,
    fragment_id: int,
    solver: str,
    run_id: str,
    backend_name: str,
    *,
    n_shots: int,
    n_reps: int,
    registry,
    task_id: str,
    lease_token: str | None = None,
    hamiltonian_sha256: str | None = None,
    worker_id: str | None = None,
    job_tags: list[str] | None = None,
    qpu_service: Any = None,
    basis: str = "sto-3g",
    lucj_config: dict | None = None,
    qpu_record_path: str | Path | None = None,
    allow_second_qpu: bool = False,
    allow_sampling: bool = False,
    campaign_auth: Optional[Any] = None,
) -> SubmissionRecord:
    """Submit one QPU fragment job and persist the Job ID.  Returns immediately.

    This is the authoritative entry point for Stage 3Q-A.  It is deliberately
    generic — it accepts any fragment ID and any valid quantum solver.  Fragment
    selection validation (which fragments are authorised for QPU sampling) is the
    responsibility of the caller (notebook, orchestrator, or Challenge client), not
    this function.

    Steps performed:
    1. Validates the requested solver (SQD or TrimSQD only).
    2. Checks for an existing Job ID in the registry (duplicate prevention).
    3. Claims the submission attempt via the registry (READY → SUBMITTING).
    4. Loads the fragment Hamiltonian from the Stage 2 HDF5.
    5. Optionally verifies the Hamiltonian SHA-256 against the caller-supplied value.
    6. Builds the inherited LUCJ circuit (reuses ``SQDSolver._build_lucj_circuit``).
    7. Computes a deterministic circuit digest.
    8. Transitions the registry to RUNNING.
    9. Submits to the QPU via the existing ``SQDSolver._submit_to_qpu``.
    10. Persists the returned Job ID in the registry (→ COMPLETE).
    11. Returns a ``SubmissionRecord`` immediately — **no polling, no counts retrieval**.

    If the provider accepts the job but the registry transition to COMPLETE fails,
    the task is marked FAILED with an AMBIGUOUS_SUBMISSION reason and an
    ``AmbiguousSubmissionError`` is raised carrying the recovered Job ID.  The caller
    must use ``reconcile_submission`` + ``resolve_ambiguous_submission`` to recover.

    Parameters
    ----------
    ewf_dumpfile_path  : path to the Stage 2 ``ewf_dumpfile.h5``
    fragment_id        : which fragment to submit
    solver             : ``"SQD"`` or ``"TrimSQD"``
    run_id             : run scoping namespace (matches the registry task)
    backend_name       : IBM Quantum backend name (e.g. ``"ibm_fez"``)
    n_shots            : requested shot count
    n_reps             : LUCJ circuit repetitions
    registry           : ``FragmentTaskRegistry`` instance (Task 5.3)
    task_id            : task ID in the registry
    lease_token        : fencing token from ``claim_task``
    hamiltonian_sha256 : expected Hamiltonian SHA-256 (verified if supplied)
    worker_id          : identifier of the calling worker (for audit)
    job_tags           : IBM Quantum job tags (auto-built if None)
    qpu_service        : an already-initialised Qiskit service or
                         ``IBMQuantumBackend`` instance; when None the
                         ``SQDSolver`` is built in offline mode (tests only)
    basis              : basis set label for job tags
    lucj_config        : LUCJ configuration dict; ``{"n_reps": n_reps}`` when None
    qpu_record_path /
    campaign_auth      : **one is mandatory.**  The explicit budget ledger.  There is no
                         default ledger and no offline bypass; omitting both raises
                         ``QpuBudgetError`` before any provider contact.
    allow_second_qpu   : legacy ledger only; authorises count >= 1
    allow_sampling     : retained for call-site compatibility; it has no effect on
                         budget enforcement (the gate is unconditional)

    Returns
    -------
    :class:`SubmissionRecord`

    Raises
    ------
    QuantumSubmissionError
        - solver is not SQD or TrimSQD
        - EWF dumpfile missing or fragment not found
        - Hamiltonian SHA-256 mismatch
        - Registry transition fails before submission
        - Provider rejects the job
    DuplicateSubmissionError
        - registry task already has a Job ID (prevents duplicate QPU allocation)
    AmbiguousSubmissionError
        - provider accepted the job but the Job ID could not be persisted
    """
    if solver not in VALID_QUANTUM_SOLVERS:
        raise QuantumSubmissionError(
            f"submit_quantum_fragment only supports {sorted(VALID_QUANTUM_SOLVERS)}, "
            f"got {solver!r}.  Use run_classical_fragment for FCI/CCSD."
        )
    if not run_id:
        raise QuantumSubmissionError("run_id must be a non-empty string")
    if not backend_name:
        raise QuantumSubmissionError("backend_name must be a non-empty string")
    if not task_id:
        raise QuantumSubmissionError("task_id is required")
    if not registry:
        raise QuantumSubmissionError("registry is required")

    submission_attempt = str(uuid.uuid4())
    effective_lucj_config = lucj_config or {"n_reps": n_reps}
    lucj_digest = _config_digest(effective_lucj_config)
    tags = job_tags or build_submission_tags(run_id, fragment_id, solver, basis)

    # ── Duplicate prevention: check registry for existing Job ID ─────────────
    try:
        task = registry.get_task(task_id)
    except Exception as exc:
        raise QuantumSubmissionError(
            f"Failed to read task {task_id!r} from registry: {exc}"
        ) from exc

    # TaskRecord.qpu_job_id is the dedicated field for the provider Job ID.
    # A non-empty value means a prior submission already recorded a Job ID.
    existing_job_id = getattr(task, "qpu_job_id", None) or ""
    if existing_job_id:
        raise DuplicateSubmissionError(
            f"Fragment {fragment_id} task {task_id!r} already has a QPU Job ID "
            f"({existing_job_id!r}).  Automatic resubmission would silently consume "
            "a QPU allocation.  Cancel the existing task explicitly if a retry is intended."
        )

    # ── MANDATORY QPU budget gate (no bypass) ─────────────────────────────────
    # Every call passes the budget mechanism.  An explicit ledger is required: either
    # a CampaignAuthorization or a legacy ``qpu_record_path``.  There is no implicit
    # default ledger and no "offline" mode that skips the check.
    if campaign_auth is None and qpu_record_path is None:
        raise QpuBudgetError(
            "submit_quantum_fragment requires an explicit QPU budget ledger "
            "(campaign_auth or qpu_record_path); refusing to contact the provider. "
            "The ledger is never bypassed, reset or implicitly selected."
        )

    # Earlier uncertain submissions must be reconciled before any new allocation.
    unreconciled = check_unreconciled_quantum_tasks(registry, run_id, exclude_task_id=task_id)
    if unreconciled:
        raise QpuBudgetError(
            f"Unreconciled QPU submission(s) {sorted(unreconciled)} (AMBIGUOUS or SUBMITTING without a "
            "Job ID) must be resolved by an operator before any further QPU allocation."
        )

    if campaign_auth is not None:
        try:
            campaign_auth.validate_submission(
                fragment_id=fragment_id,
                backend=backend_name,
                n_shots=n_shots,
            )
            unresolved = [e for e in campaign_auth.unresolved_reservations()
                          if e.task_id != task_id]
        except Exception as exc:
            _registry_fail(registry, task_id, f"Campaign authorization blocked submission: {exc}",
                           lease_token=lease_token)
            raise
        if unresolved:
            raise QpuBudgetError(
                f"Campaign ledger has unresolved reservation(s) for fragment(s) "
                f"{sorted(e.fragment_id for e in unresolved)}; reconcile them before further allocation."
            )

    active_ledger: Path | None = None
    if campaign_auth is None:
        active_ledger = _resolve_ledger_path(qpu_record_path)
    try:
        check_qpu_budget(active_ledger, allow_second_qpu=allow_second_qpu, campaign_auth=campaign_auth)
    except QpuBudgetError as exc:
        _registry_fail(registry, task_id, f"QPU budget gate blocked submission: {exc}",
                       lease_token=lease_token)
        raise

    # ── Registry: READY → SUBMITTING ─────────────────────────────────────────
    _registry_transition(
        registry, task_id, "READY", "SUBMITTING",
        worker_id=worker_id, lease_token=lease_token,
        reason=f"3Q-A submission starting (attempt={submission_attempt})",
    )

    # ── Load Hamiltonian from Stage 2 HDF5 ───────────────────────────────────
    try:
        ham = load_fragment_hamiltonian_for_qpu(ewf_dumpfile_path, fragment_id)
    except QuantumSubmissionError:
        _registry_fail(registry, task_id,
                       f"HDF5 load failed for fragment {fragment_id}",
                       lease_token=lease_token)
        raise
    except Exception as exc:
        msg = f"Unexpected error loading fragment {fragment_id}: {exc}"
        _registry_fail(registry, task_id, msg, lease_token=lease_token)
        raise QuantumSubmissionError(msg) from exc

    h1e     = ham["h1e"]
    h2e     = ham["h2e"]
    norb    = ham["norb"]
    nocc    = ham["nocc"]
    nelec   = ham["nelec"]
    ham_sha = ham["hamiltonian_sha256"]

    # ── Optionally verify the Hamiltonian identity ────────────────────────────
    if hamiltonian_sha256 is not None and hamiltonian_sha256 != ham_sha:
        msg = (
            f"Fragment {fragment_id}: Hamiltonian SHA-256 mismatch — "
            f"caller supplied {hamiltonian_sha256[:12]}... but the dumpfile "
            f"contains {ham_sha[:12]}...  The submission is for a different "
            "embedding; nothing was submitted."
        )
        _registry_fail(registry, task_id, msg, lease_token=lease_token)
        raise QuantumSubmissionError(msg)

    # ── Build LUCJ circuit (reuses SQDSolver._build_lucj_circuit) ────────────
    try:
        isa_circuit, circuit_dig = _build_circuit(
            h1e, h2e, norb, nelec,
            backend_name=backend_name,
            n_shots=n_shots,
            lucj_config=effective_lucj_config,
            qpu_service=qpu_service,
            solver=solver,
        )
    except QuantumSubmissionError:
        _registry_fail(registry, task_id,
                       f"Circuit build failed for fragment {fragment_id}",
                       lease_token=lease_token)
        raise
    except Exception as exc:
        msg = f"Fragment {fragment_id}: circuit build failed: {exc}"
        _registry_fail(registry, task_id, msg, lease_token=lease_token)
        raise QuantumSubmissionError(msg) from exc

    submitted_utc = _utc_now()

    # ── Reserve quota BEFORE the provider call (conservative accounting) ──────
    # The provider has not been contacted yet, so a reservation failure is a clean
    # pre-submission failure.  Once reserved, the unit stays consumed even if the
    # provider outcome is unknown; only an operator can reconcile it.
    try:
        if campaign_auth is not None:
            campaign_auth.reserve_submission(
                fragment_id=fragment_id, task_id=task_id, backend=backend_name,
                attempt_id=submission_attempt, n_shots=n_shots,
            )
        else:
            reserve_qpu_sampling(
                active_ledger, allow_second_qpu=allow_second_qpu,
                attempt_id=submission_attempt, fragment_id=fragment_id, task_id=task_id,
            )
    except Exception as exc:
        msg = f"QPU quota reservation failed for fragment {fragment_id}: {exc}"
        _registry_fail(registry, task_id, msg, lease_token=lease_token)
        if isinstance(exc, QpuBudgetError):
            raise
        raise QpuBudgetError(msg) from exc

    # ── Submit to QPU ─────────────────────────────────────────────────────────
    # Ambiguity boundary: after _submit_circuit() returns a Job ID the QPU
    # allocation may be consumed.  A provider exception does NOT prove rejection —
    # a network timeout can occur after the provider accepted the job.  We therefore
    # treat ALL exceptions from _submit_circuit as AMBIGUOUS unless the provider's
    # error payload explicitly signals a pre-acceptance validation failure (which
    # the provider SDK is not guaranteed to distinguish).
    # Only confirmed pre-acceptance failures (e.g. auth errors) may be treated as clean.
    job_id: str | None = None
    try:
        job_id = _submit_circuit(
            isa_circuit, backend_name, tags,
            n_shots=n_shots,
            qpu_service=qpu_service,
        )
    except Exception as exc:
        # Treat all provider call exceptions as AMBIGUOUS: we cannot prove the job
        # was never accepted.  Do not mark FAILED — that would block recovery.
        ambiguous_reason = (
            f"AMBIGUOUS_SUBMISSION: QPU submission call for fragment {fragment_id} "
            f"raised an exception — provider acceptance is uncertain: {exc}.  "
            "Use reconcile_submission() to query the provider, then "
            "recover_ambiguous_submission() or abandon_ambiguous_and_retry()."
        )
        _registry_mark_ambiguous(
            registry, task_id, ambiguous_reason,
            lease_token=lease_token or "",
            candidate_job_id="",
        )
        raise AmbiguousSubmissionError(ambiguous_reason, recovered_job_id=None) from exc

    # ── Atomically persist the Job ID (still in SUBMITTING) ──────────────────
    # record_qpu_job_id writes the qpu_job_id field while the task is still in
    # SUBMITTING state so it is never lost even if the next transition fails.
    try:
        registry.record_qpu_job_id(
            task_id,
            qpu_job_id=job_id,
            backend=backend_name,
            lease_token=lease_token or "",
        )
    except Exception as exc:
        ambiguous_reason = (
            f"AMBIGUOUS_SUBMISSION: QPU provider accepted fragment {fragment_id} "
            f"with Job ID {job_id!r} but the registry could not persist it: {exc}.  "
            "DO NOT resubmit automatically — use reconcile_submission() to recover, "
            "then recover_ambiguous_submission() with the confirmed Job ID."
        )
        _registry_mark_ambiguous(
            registry, task_id, ambiguous_reason,
            lease_token=lease_token or "",
            candidate_job_id=job_id or "",
        )
        raise AmbiguousSubmissionError(
            ambiguous_reason, recovered_job_id=job_id
        ) from exc

    # ── Confirm the reservation with the provider Job ID ──────────────────────
    # Legacy ledger: the unit was consumed at reservation time; nothing more to do.
    # Campaign ledger: attach the Job ID to the reservation.  Failures are NOT
    # swallowed: the task becomes AMBIGUOUS (with the Job ID retained) so the
    # operator reconciles the ledger.
    if campaign_auth is not None:
        try:
            record_qpu_sampling(
                campaign_auth=campaign_auth,
                fragment_id=fragment_id,
                task_id=task_id,
                job_id=job_id,
                backend=backend_name,
                attempt_id=submission_attempt,
                n_shots=n_shots,
            )
        except Exception as exc:
            ambiguous_reason = (
                f"AMBIGUOUS_SUBMISSION: Job ID {job_id!r} accepted but the campaign ledger could not "
                f"confirm the reservation: {exc}.  Reconcile the ledger before further allocation."
            )
            _registry_mark_ambiguous(
                registry, task_id, ambiguous_reason,
                lease_token=lease_token or "", candidate_job_id=job_id or "",
            )
            raise AmbiguousSubmissionError(ambiguous_reason, recovered_job_id=job_id) from exc
    else:
        try:
            _journal(active_ledger, {"event": "ACCEPTED", "attempt_id": submission_attempt,
                                     "fragment_id": fragment_id, "task_id": task_id,
                                     "job_id": job_id, "utc": _utc_now()})
        except OSError:
            pass  # audit journal only; the count was already consumed at reservation

    # ── SUBMITTING → SUBMITTED ────────────────────────────────────────────────
    # 3Q-A ends here. The task stays in SUBMITTED until 3Q-B detects QPU completion.
    # Job ID is already durably stored; if this transition fails the task ends up
    # AMBIGUOUS with the Job ID still accessible, allowing operator recovery.
    try:
        _registry_transition(
            registry, task_id, "SUBMITTING", "SUBMITTED",
            worker_id=worker_id, lease_token=lease_token,
            reason=f"QPU Job ID {job_id!r} persisted; 3Q-A complete",
        )
    except Exception as exc:
        # Job ID already written; transition failure leaves task in SUBMITTING
        # (not SUBMITTED). Attempt AMBIGUOUS transition so the operator can recover.
        ambiguous_reason = (
            f"AMBIGUOUS_SUBMISSION: Job ID {job_id!r} persisted but SUBMITTING→SUBMITTED "
            f"transition failed: {exc}.  Recovery: use recover_ambiguous_submission() "
            f"with job_id={job_id!r} for task {task_id!r}."
        )
        _registry_mark_ambiguous(
            registry, task_id, ambiguous_reason,
            lease_token=lease_token or "",
            candidate_job_id=job_id or "",
        )
        raise AmbiguousSubmissionError(
            ambiguous_reason, recovered_job_id=job_id
        ) from exc

    return SubmissionRecord(
        run_id=run_id,
        fragment_id=int(fragment_id),
        task_id=task_id,
        submission_attempt=submission_attempt,
        solver=solver,
        hamiltonian_sha256=ham_sha,
        circuit_digest=circuit_dig,
        lucj_config_digest=lucj_digest,
        backend_name=backend_name,
        n_shots=n_shots,
        n_reps=n_reps,
        job_id=job_id,
        submission_tags=list(tags),
        submitted_utc=submitted_utc,
        submission_state="SUBMITTED",
        failure_reason=None,
        n_qubits=2 * norb,
        norb=norb,
        nocc=nocc,
        code_version=_software_versions(),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Circuit construction — reuses SQDSolver internals
# ─────────────────────────────────────────────────────────────────────────────

def _build_circuit(
    h1e, h2e, norb: int, nelec: tuple,
    *,
    backend_name: str,
    n_shots: int,
    lucj_config: dict,
    qpu_service: Any,
    solver: str,
) -> tuple:
    """Build the ISA LUCJ circuit for one fragment.  Returns (circuit, circuit_digest).

    Delegates entirely to ``SQDSolver._build_lucj_circuit``, which is the single
    maintenance point for LUCJ construction science.  We instantiate a minimal
    ``SQDSolver`` and use its existing ``_build_lucj_circuit`` without importing
    SQD diagonalization.

    The ``qpu_service`` argument is either:
    - an ``IBMQuantumBackend`` instance with ``._qiskit_backend`` already set, OR
    - a raw ``QiskitRuntimeService`` object (we call ``.backend(backend_name)``), OR
    - None (offline/test mode: the circuit is built from amplitudes only and the
      Qiskit backend is not required — raises ImportError from ffsim gracefully
      if ffsim is unavailable, which the test suite stubs out).
    """
    try:
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import SQDSolver
        from quantum_fragment_methods.qpu.qiskit_ibm_runtime import IBMQuantumBackend
    except ImportError as exc:
        raise QuantumSubmissionError(
            f"Cannot import SQDSolver (is serverless_bundle on sys.path?): {exc}"
        ) from exc

    # nelec is (n_alpha, n_beta); for RHF n_alpha == n_beta == nocc
    nocc = nelec[0]

    # Compute CCSD amplitudes from the fragment integrals (same as SQDSolver.solve)
    try:
        from pyscf import cc as pyscf_cc, scf as pyscf_scf
        import numpy as np

        # Build a minimal RHF object to run CCSD
        from pyscf import gto
        mol = gto.M(verbose=0)
        mol.nelectron = sum(nelec)
        mol.spin = 0
        mf = pyscf_scf.RHF(mol)
        mf.get_hcore = lambda *a, **k: h1e
        mf.get_ovlp  = lambda *a, **k: np.eye(norb)
        from pyscf import ao2mo as _ao2mo
        mf._eri = _ao2mo.restore(8, h2e, norb) if (hasattr(h2e, "ndim") and h2e.ndim == 4) else h2e
        mf.mo_energy = np.zeros(norb)
        mf.mo_coeff  = np.eye(norb)
        mf.mo_occ    = np.array([2.0] * nocc + [0.0] * (norb - nocc))
        # Use the two-body integrals directly via the CCSD integral transform
        mycc = pyscf_cc.CCSD(mf)
        mycc.verbose = 0
        mycc.run()
        t1, t2 = mycc.t1, mycc.t2
    except Exception as exc:
        raise QuantumSubmissionError(
            f"Cannot compute CCSD amplitudes for circuit construction: {exc}"
        ) from exc

    # Build or reuse backend object
    backend_obj = _make_backend_obj(qpu_service, backend_name, n_shots)

    # Build solver (no diagonalization imports occur until solve() is called)
    config = {
        "lucj": lucj_config,
        "classical_backend": "python",
        "sampler_options": {"default_shots": n_shots},
    }
    sqd_solver = SQDSolver(qpu_backend=backend_obj, config=config)

    # Build the ISA circuit
    isa_circuit = sqd_solver._build_lucj_circuit(t1, t2, norb, nelec)
    circuit_dig = _circuit_digest(isa_circuit)
    return isa_circuit, circuit_dig


def _make_backend_obj(qpu_service: Any, backend_name: str, n_shots: int) -> Any:
    """Wrap qpu_service into an IBMQuantumBackend usable by SQDSolver."""
    try:
        from quantum_fragment_methods.qpu.qiskit_ibm_runtime import IBMQuantumBackend
    except ImportError as exc:
        raise QuantumSubmissionError(
            f"Cannot import IBMQuantumBackend: {exc}"
        ) from exc

    cfg = {
        "provider":     "ibm_quantum",
        "backend_name": backend_name,
        "credentials":  {"channel": "ibm_cloud", "token": "", "instance": ""},
        "sampler_options": {"default_shots": n_shots},
    }
    backend_obj = IBMQuantumBackend(cfg)

    if qpu_service is None:
        # Offline / test mode: _qiskit_backend stays None; _build_lucj_circuit
        # will raise RuntimeError when it reaches the ffsim pass manager step.
        # Tests replace _build_lucj_circuit with a mock.
        backend_obj.service = None
        backend_obj.backend = None
        backend_obj._qiskit_backend = None
    elif hasattr(qpu_service, "_qiskit_backend"):
        # Already an IBMQuantumBackend
        backend_obj.service          = qpu_service.service
        backend_obj.backend          = qpu_service.backend
        backend_obj._qiskit_backend  = qpu_service._qiskit_backend
    else:
        # Raw QiskitRuntimeService
        qiskit_backend = qpu_service.backend(backend_name)
        backend_obj.service          = qpu_service
        backend_obj.backend          = qiskit_backend
        backend_obj._qiskit_backend  = qiskit_backend

    return backend_obj


def _submit_circuit(
    isa_circuit: Any,
    backend_name: str,
    tags: list[str],
    *,
    n_shots: int,
    qpu_service: Any,
) -> str:
    """Submit the ISA circuit and return the IBM Quantum Job ID.

    Reuses ``SQDSolver._submit_to_qpu`` which handles token refresh and sampler
    options correctly for the Serverless/Fleets environment.
    """
    try:
        from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import SQDSolver
    except ImportError as exc:
        raise QuantumSubmissionError(
            f"Cannot import SQDSolver for submission: {exc}"
        ) from exc

    backend_obj = _make_backend_obj(qpu_service, backend_name, n_shots)
    config = {
        "lucj": {},
        "classical_backend": "python",
        "sampler_options": {"default_shots": n_shots},
    }
    sqd_solver = SQDSolver(qpu_backend=backend_obj, config=config)
    return sqd_solver._submit_to_qpu(isa_circuit, job_tags=tags)


# ─────────────────────────────────────────────────────────────────────────────
#  Ambiguity recovery
# ─────────────────────────────────────────────────────────────────────────────

def reconcile_submission(
    registry,
    task_id: str,
    *,
    qpu_service: Any,
    tags: list[str],
    max_age_hours: int = 24,
) -> str | None:
    """Query the QPU provider for a recent job matching the given tags.

    This is a MANUAL RECOVERY aid — do not call automatically.

    Searches ``qpu_service`` for recent jobs whose tags match ALL of ``tags``.
    Returns the first matching Job ID string, or None if no match is found.

    The caller must verify the returned Job ID is for the intended fragment and
    call ``resolve_ambiguous_submission`` to durably record it.

    No registry state is changed here.

    Parameters
    ----------
    registry    : FragmentTaskRegistry (read-only in this call)
    task_id     : task ID to reconcile
    qpu_service : an active Qiskit Runtime service with a ``jobs()`` method
    tags        : submission tags used during the original submission attempt
    max_age_hours : only search jobs submitted within this many hours

    Returns
    -------
    str | None — IBM Quantum Job ID, or None if not found
    """
    if qpu_service is None:
        raise QuantumSubmissionError(
            "reconcile_submission requires an active qpu_service (cannot reconcile offline)"
        )
    try:
        # IBM Quantum job listing: filter by tags
        # The service.jobs() API accepts `job_tags` and `limit` kwargs.
        jobs = qpu_service.jobs(job_tags=list(tags), limit=10)
    except Exception as exc:
        raise QuantumSubmissionError(
            f"Failed to list jobs from provider for reconciliation: {exc}"
        ) from exc

    if not jobs:
        return None

    # Return the most recently submitted matching job ID
    # IBM Quantum returns jobs newest-first by default
    try:
        return str(jobs[0].job_id())
    except Exception:
        return None


def resolve_ambiguous_submission(
    registry,
    task_id: str,
    recovered_job_id: str,
    *,
    lease_token: str | None = None,
    circuit_digest: str = "",
    backend_name: str = "",
    submission_tags: list | None = None,
    campaign_auth: Optional[Any] = None,
) -> SubmissionRecord:
    """Audited operator recovery: record a verified Job ID and advance AMBIGUOUS → SUBMITTED.

    This must only be called after the operator has:
    1. Used reconcile_submission() to locate the Job ID on the provider side.
    2. Verified the Job ID belongs to the correct fragment and Hamiltonian.
    3. Confirmed it was not a duplicate submission (check the QPU dashboard).

    Delegates to ``registry.recover_ambiguous_submission()`` which is the single
    authorised transactional path from AMBIGUOUS → SUBMITTED with a full audit trail.
    No direct SQL access to the registry is used here.

    Parameters
    ----------
    registry           : FragmentTaskRegistry
    task_id            : task ID in AMBIGUOUS state
    recovered_job_id   : the verified IBM Quantum Job ID
    lease_token        : fencing token (verified by the registry if the task carries one)
    circuit_digest     : circuit digest for provenance (stored in registry extra_json)
    backend_name       : backend name for provenance
    submission_tags    : original submission tags (stored in registry extra_json)

    Returns
    -------
    SubmissionRecord with ``submission_state="SUBMITTED"``

    Raises
    ------
    QuantumSubmissionError
        - recovered_job_id is empty
        - task is not in AMBIGUOUS state
        - task already carries a conflicting Job ID
    """
    if not recovered_job_id:
        raise QuantumSubmissionError("recovered_job_id must be a non-empty string")

    try:
        task = registry.get_task(task_id)
    except Exception as exc:
        raise QuantumSubmissionError(
            f"Cannot read task {task_id!r} from registry: {exc}"
        ) from exc

    recovery_reason = (
        f"Operator recovery via resolve_ambiguous_submission(): "
        f"Job ID {recovered_job_id!r} on backend {backend_name!r}."
    )
    try:
        task = registry.recover_ambiguous_submission(
            task_id,
            qpu_job_id=recovered_job_id,
            backend=backend_name or task.backend or "unknown",
            recovery_reason=recovery_reason,
            lease_token=lease_token or "",
            circuit_digest=circuit_digest,
            submission_tags=submission_tags or [],
        )
    except Exception as exc:
        raise QuantumSubmissionError(
            f"Registry recovery for task {task_id!r} failed: {exc}"
        ) from exc

    if campaign_auth is not None:
        for entry in campaign_auth.unresolved_reservations():
            if entry.task_id == task_id:
                campaign_auth.confirm_submission(entry.attempt_id, recovered_job_id)

    return SubmissionRecord(
        run_id=str(getattr(task, "run_id", "")),
        fragment_id=int(getattr(task, "fragment_id", -1)),
        task_id=task_id,
        submission_attempt="recovered",
        solver=str(getattr(task, "solver", "")),
        hamiltonian_sha256="",
        circuit_digest=circuit_digest,
        lucj_config_digest="",
        backend_name=backend_name or getattr(task, "backend", ""),
        n_shots=0,
        n_reps=0,
        job_id=recovered_job_id,
        submission_tags=submission_tags or [],
        submitted_utc=_utc_now(),
        submission_state="SUBMITTED",
        failure_reason=None,
        n_qubits=0,
        norb=0,
        nocc=0,
        code_version=_software_versions(),
    )
