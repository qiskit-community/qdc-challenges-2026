"""
classical_dispatcher — Organizer-side coordinator for independent classical fragment tasks (Task 5.9C.1).

Runs in the organizer notebook environment only.  It owns the SQLite registry; Serverless workers never
see it.  Submission and reconciliation are separate operations.

Safety contracts
----------------
1. **Bounded concurrency.**  SUBMITTING + RUNNING classical tasks (including uncertain ones) count
   against ``max_concurrent``.
2. **Atomic claims.**  ``registry.claim_task`` (compare-and-set); only READY tasks are ever claimed.
3. **Durable intent before side effect.**  A ``SUBMIT_INTENT`` event is written to the task's history
   *before* ``service.run`` is called, and the provider Job ID is recorded *immediately after*.
4. **No automatic resubmission of uncertain jobs.**  If ``run`` raises, or the process dies between
   intent and Job ID, the task stays SUBMITTING and is reported ``SUBMISSION_UNCERTAIN``.  Nothing is ever
   resubmitted for it until an operator calls :func:`resolve_uncertain_submission` (attach the real Job ID,
   or explicitly abandon the attempt).  A lease expiring or the notebook restarting changes nothing.
   Residual window: a job accepted by the provider whose ID never reached SQLite is invisible to the
   organizer; the operator must find it in the provider console.
5. **Workers return descriptors, not paths.**  A path in a worker result is ignored.  Results are fetched
   from the transport identifier named by the descriptor, verified for size and SHA-256, validated as
   frs/1 with run/fragment/solver/Hamiltonian/payload identity, imported into the ``LocalArtifactStore``,
   and only then is the task COMPLETE.
6. **Append-only history.**  Every attempt event (intent, acceptance, uncertainty, failure, verification
   error, abandonment, requeue, cache reuse) is appended to ``extra_json["dispatch_events"]``.
7. **Failures are explicit.**  Certain failures move a task to RETRY_PENDING; nothing requeues until
   :func:`requeue_classical_tasks` is called.
"""

from __future__ import annotations

import datetime
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from proteasome_challenge.run_control_plane import ControlPlaneError, RunContext, load_stage_ref
from proteasome_challenge.task_registry import (
    ClaimConflictError,
    InvalidTransitionError,
    RegistryError,
    TaskRecord,
    TaskState,
    TaskType,
)
from serverless_bundle.steps.artifact_store import KIND_FRAGMENT_H5
from serverless_bundle.steps.artifact_transport import (
    ServerlessFileApiClient,
    ServerlessTransportAdapter,
    TransportChecksumError,
    TransportDescriptor,
    TransportError,
    TransportNotFoundError,
    result_transport_name,
    validate_fragment_result_file,
    verify_and_import_fragment_result,
)
from serverless_bundle.steps.classical_fragment_runner import (
    CLASSICAL_SOLVERS,
    ClassicalFragmentError,
    validate_solver_config,
)

__all__ = [
    "CLASSICAL_FUNCTION_TITLE",
    "VALID_RESOURCE_FAMILIES",
    "DispatcherError",
    "DispatchReport",
    "ReconciliationReport",
    "CampaignSummary",
    "CatalogServerlessService",
    "make_live_transport",
    "dispatch_classical_tasks",
    "reconcile_classical_tasks",
    "resolve_uncertain_submission",
    "requeue_classical_tasks",
    "classical_status",
    "run_notebook_step",
    "run_classical_campaign",
]

CLASSICAL_FUNCTION_TITLE = "qfm-proteasome-classical-fragment"
VALID_RESOURCE_FAMILIES = frozenset({"s", "m", "l"})
EVENTS_KEY = "dispatch_events"

_PROVIDER_SUCCESS = frozenset({"DONE", "COMPLETED", "SUCCEEDED"})
_PROVIDER_FAILED = frozenset({"ERROR", "FAILED"})
_PROVIDER_CANCELED = frozenset({"CANCELED", "CANCELLED", "STOPPED"})
_ACTIVE = (TaskState.SUBMITTING, TaskState.RUNNING)
_MAX_RETRIEVAL_FAILURES = 3


class DispatcherError(ControlPlaneError):
    """Base error for classical dispatch operations."""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  Reports
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DispatchReport:
    dispatched_tasks: list[dict[str, Any]]
    reused_from_cache: list[dict[str, Any]]
    uncertain_tasks: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    active_count: int
    available_slots: int
    max_concurrent: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconciliationReport:
    completed_tasks: list[dict[str, Any]]
    retry_pending_tasks: list[dict[str, Any]]
    running_tasks: list[dict[str, Any]]
    uncertain_tasks: list[dict[str, Any]]
    pending_retrieval: list[dict[str, Any]]
    active_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignSummary:
    run_id: str
    total_classical_tasks: int
    completed_count: int
    failed_count: int
    retry_pending_count: int
    running_count: int
    ready_count: int
    uncertain_count: int
    all_complete: bool
    elapsed_seconds: float
    passes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
#  Live (real Qiskit Serverless) adapters.  Constructed only when the operator supplies a client;
#  nothing here contacts the network at import time and no test calls them with a real client.
# ─────────────────────────────────────────────────────────────────────────────

class CatalogServerlessService:
    """``run`` / ``job`` facade over a ``qiskit_ibm_catalog.QiskitServerless`` (or compatible) client."""

    def __init__(self, client: Any):
        self._client = client

    def run(self, function_title: str, arguments: Mapping[str, Any], function_size: Optional[str] = None, **_: Any):
        fn = self._client.load(function_title)
        return fn.run(function_size=function_size, **dict(arguments))

    def job(self, job_id: str):
        return self._client.job(job_id)


def make_live_transport(client: Any, function_title: str = CLASSICAL_FUNCTION_TITLE) -> ServerlessTransportAdapter:
    """Build a File API transport adapter for an authenticated Serverless client (organizer side)."""
    from qiskit_serverless.core.function import QiskitFunction
    return ServerlessTransportAdapter(
        ServerlessFileApiClient(client, QiskitFunction(title=function_title)))


# ─────────────────────────────────────────────────────────────────────────────
#  History helpers
# ─────────────────────────────────────────────────────────────────────────────

def _event(ctx: RunContext, task: TaskRecord, kind: str, **fields: Any) -> None:
    ctx.registry.update_extra(
        task.task_id, EVENTS_KEY,
        {"event": kind, "at": _now(), "attempt": fields.pop("attempt", task.attempt), **fields},
        append=True)


def _events(task: TaskRecord) -> list[dict[str, Any]]:
    value = task.extra.get(EVENTS_KEY, [])
    return value if isinstance(value, list) else []


def _classical_tasks(ctx: RunContext) -> list[TaskRecord]:
    return ctx.registry.list_tasks(ctx.run_id, task_type=TaskType.CLASSICAL)


def _check_manifest_assignment(ctx: RunContext, task: TaskRecord) -> Optional[str]:
    """Return a conflict description, or ``None`` when the task agrees with the frozen manifest."""
    if ctx.manifest is None:
        return "no frozen manifest"
    try:
        frozen = ctx.manifest.solver_for(task.fragment_id)
    except Exception as exc:  # noqa: BLE001
        return f"fragment {task.fragment_id} is not in the frozen manifest ({exc})"
    if frozen != task.solver:
        return f"task solver {task.solver!r} differs from frozen assignment {frozen!r}"
    if task.solver not in CLASSICAL_SOLVERS:
        return f"solver {task.solver!r} is not a classical solver"
    return None


def _cached_artifact(ctx: RunContext, task: TaskRecord):
    """Return ``(path, payload_digest)`` of a valid stored result for this task, else ``None``."""
    try:
        path = ctx.store._artifact_path(ctx.run_id, KIND_FRAGMENT_H5, f"frag_{task.fragment_id}_{task.solver.lower()}.h5")
    except Exception:  # noqa: BLE001
        return None
    if not path.is_file():
        return None
    try:
        digest = validate_fragment_result_file(
            path, run_id=ctx.run_id, fragment_id=task.fragment_id, expected_solver=task.solver,
            expected_hamiltonian_sha256=task.hamiltonian_sha256)
    except TransportError:
        return None
    return path, digest


# ─────────────────────────────────────────────────────────────────────────────
#  Submission
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_classical_tasks(
    ctx: RunContext,
    serverless_service: Any,
    transport_adapter: ServerlessTransportAdapter,
    *,
    max_concurrent: int = 4,
    resource_family: Optional[str] = None,
    function_title: str = CLASSICAL_FUNCTION_TITLE,
    solver_configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    result_mode: str = "auto",
    worker_id: str = "organizer_dispatcher",
) -> DispatchReport:
    """Dispatch READY classical tasks, one Serverless job per fragment.  Never touches non-READY tasks."""
    if ctx.manifest is None:
        raise DispatcherError(f"Run {ctx.run_id!r} has no frozen manifest; freeze it before dispatch.")
    if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent < 1:
        raise DispatcherError("max_concurrent must be an integer >= 1")
    if resource_family is not None and resource_family not in VALID_RESOURCE_FAMILIES:
        raise DispatcherError(f"resource_family must be one of {sorted(VALID_RESOURCE_FAMILIES)}")
    configs = {str(k): validate_solver_config(str(k), dict(v)) for k, v in (solver_configs or {}).items()}

    tasks = _classical_tasks(ctx)
    active = [t for t in tasks if t.state in _ACTIVE]
    ready = [t for t in tasks if t.state == TaskState.READY]

    reused: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    dispatched: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []

    # 1. Conflicts and completed-fragment cache reuse never consume a worker slot.
    pending: list[TaskRecord] = []
    for t in ready:
        problem = _check_manifest_assignment(ctx, t)
        if problem:
            conflicts.append({"task_id": t.task_id, "fragment_id": t.fragment_id, "problem": problem})
            continue
        hit = _cached_artifact(ctx, t)
        if hit is not None:
            reused.append(_complete_from_cache(ctx, t, hit, worker_id))
            continue
        pending.append(t)

    available = max(0, max_concurrent - len(active))
    batch = pending[:available]

    if batch:
        # 2. Upload the immutable Stage 2 input once (content-addressed, so repeated uploads are no-ops).
        ewf_ref = load_stage_ref(ctx, "embedding_ewf")
        if ewf_ref is None:
            raise DispatcherError("Stage 2 'embedding_ewf' reference is not published for this run")
        if ewf_ref.file_sha256 != ctx.manifest.embedding_artifact_ref.get("file_sha256"):
            raise DispatcherError("published Stage 2 artifact differs from the frozen manifest")
        verification = ctx.store.verify(ewf_ref)
        if not verification.ok:
            raise DispatcherError(f"Stage 2 artifact failed local verification: {verification.reason}")
        ewf_desc = transport_adapter.upload_input(
            Path(ewf_ref.uri), run_id=ctx.run_id, file_role="ewf_dumpfile",
            expected_sha256=ewf_ref.file_sha256)

        for t in batch:
            try:
                claimed, lease = ctx.registry.claim_task(t.task_id, worker_id=worker_id)
            except ClaimConflictError:
                continue
            rec = ctx.registry.transition(
                t.task_id, TaskState.READY, TaskState.SUBMITTING, worker_id=worker_id,
                lease_token=lease, reason="claimed for Serverless dispatch")
            attempt = rec.attempt
            size = resource_family or t.resource_family or "l"
            if size not in VALID_RESOURCE_FAMILIES:
                size = "l"
            out_name = result_transport_name(ctx.run_id, t.fragment_id, t.solver, attempt)
            spec = {
                "run_id": ctx.run_id,
                "fragment_id": t.fragment_id,
                "solver": t.solver,
                "hamiltonian_sha256": t.hamiltonian_sha256,
                "ewf_input": ewf_desc.as_dict(),
                "solver_config": configs.get(t.solver, {}),
                "attempt": attempt,
                "result_mode": result_mode,
                "output_transport_name": out_name,
            }
            try:
                json.dumps(spec)
                _event(ctx, rec, "SUBMIT_INTENT", function=function_title, function_size=size,
                       output_transport_name=out_name, solver_config=spec["solver_config"])
            except Exception as exc:  # noqa: BLE001 — nothing has been sent yet: safe to give the slot back
                ctx.registry.transition(t.task_id, TaskState.SUBMITTING, TaskState.RETRY_PENDING,
                                        lease_token=lease, reason=f"dispatch aborted before submission: {exc}")
                raise DispatcherError(f"could not record submission intent for fragment {t.fragment_id}: {exc}") from exc

            try:
                job = serverless_service.run(function_title, arguments=spec, function_size=size)
                job_id = str(getattr(job, "job_id", "") or "")
                if not job_id:
                    raise DispatcherError("provider returned a job handle without a job_id")
            except Exception as exc:  # noqa: BLE001
                # The request may or may not have reached the provider: UNCERTAIN.  Never auto-resubmit.
                _event(ctx, rec, "SUBMIT_UNCERTAIN", error=f"{type(exc).__name__}: {exc}")
                uncertain.append({"task_id": t.task_id, "fragment_id": t.fragment_id, "attempt": attempt,
                                  "error": f"{type(exc).__name__}: {exc}"})
                continue

            # Durable Job ID first, then the state change.  A crash in between leaves SUBMITTING + job id,
            # which reconcile promotes to RUNNING.
            ctx.registry.record_serverless_job_id(t.task_id, serverless_job_id=job_id, lease_token=lease)
            _event(ctx, rec, "SUBMIT_ACCEPTED", serverless_job_id=job_id)
            ctx.registry.transition(t.task_id, TaskState.SUBMITTING, TaskState.RUNNING, worker_id=worker_id,
                                    lease_token=lease, reason=f"dispatched Serverless job {job_id}")
            dispatched.append({"task_id": t.task_id, "fragment_id": t.fragment_id, "solver": t.solver,
                               "attempt": attempt, "serverless_job_id": job_id, "function_size": size})

    # Uncertain submissions left over from earlier passes are surfaced, never retried.
    for t in _classical_tasks(ctx):
        if t.state == TaskState.SUBMITTING and not t.serverless_job_id and \
                all(u["task_id"] != t.task_id for u in uncertain):
            uncertain.append({"task_id": t.task_id, "fragment_id": t.fragment_id, "attempt": t.attempt,
                              "error": "SUBMISSION_UNCERTAIN (no provider Job ID recorded)"})

    active_now = sum(1 for t in _classical_tasks(ctx) if t.state in _ACTIVE)
    return DispatchReport(
        dispatched_tasks=dispatched, reused_from_cache=reused, uncertain_tasks=uncertain,
        conflicts=conflicts, active_count=active_now,
        available_slots=max(0, max_concurrent - active_now), max_concurrent=max_concurrent)


def _complete_from_cache(ctx: RunContext, t: TaskRecord, hit, worker_id: str) -> dict[str, Any]:
    path, digest = hit
    claimed, lease = ctx.registry.claim_task(t.task_id, worker_id=worker_id)
    rec = ctx.registry.transition(t.task_id, TaskState.READY, TaskState.SUBMITTING, worker_id=worker_id,
                                  lease_token=lease, reason="completed-fragment cache reuse")
    ctx.registry.transition(t.task_id, TaskState.SUBMITTING, TaskState.RUNNING, worker_id=worker_id,
                            lease_token=lease, reason="cache reuse: verified stored artifact")
    _event(ctx, rec, "CACHE_REUSE", payload_digest=digest)
    ctx.registry.complete_task(t.task_id, artifact_uri=str(path), payload_digest=digest, lease_token=lease,
                               reason="verified stored artifact reused; no worker dispatched")
    return {"task_id": t.task_id, "fragment_id": t.fragment_id, "solver": t.solver, "payload_digest": digest}


# ─────────────────────────────────────────────────────────────────────────────
#  Reconciliation
# ─────────────────────────────────────────────────────────────────────────────

def reconcile_classical_tasks(
    ctx: RunContext,
    serverless_service: Any,
    transport_adapter: ServerlessTransportAdapter,
    *,
    max_retrieval_failures: int = _MAX_RETRIEVAL_FAILURES,
) -> ReconciliationReport:
    """Observe provider job status and import verified results.  Never submits anything."""
    completed: list[dict[str, Any]] = []
    retry_pending: list[dict[str, Any]] = []
    running: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    pending_retrieval: list[dict[str, Any]] = []

    for t in [x for x in _classical_tasks(ctx) if x.state in _ACTIVE]:
        if not t.serverless_job_id:
            uncertain.append({"task_id": t.task_id, "fragment_id": t.fragment_id, "attempt": t.attempt,
                              "status": "SUBMISSION_UNCERTAIN"})
            continue
        if t.state == TaskState.SUBMITTING:   # crash between Job-ID record and state change
            t = ctx.registry.transition(t.task_id, TaskState.SUBMITTING, TaskState.RUNNING,
                                        lease_token=t.lease_token, reason="recovered: Job ID is durable")
        job_id = t.serverless_job_id

        try:
            job = serverless_service.job(job_id)
            if job is None:
                raise DispatcherError("provider reports no such job")
            status = str(job.status()).upper()
        except Exception as exc:  # noqa: BLE001 — a failed query is never evidence of failure
            running.append({"task_id": t.task_id, "fragment_id": t.fragment_id, "serverless_job_id": job_id,
                            "status": "PROVIDER_QUERY_ERROR", "error": f"{type(exc).__name__}: {exc}"})
            continue
        ctx.registry.update_provider_status(t.task_id, provider_status=status)

        if status in _PROVIDER_SUCCESS:
            outcome = _collect_result(ctx, t, job, transport_adapter, max_retrieval_failures)
            {"completed": completed, "retry": retry_pending, "pending": pending_retrieval}[outcome[0]].append(outcome[1])
        elif status in _PROVIDER_FAILED or status in _PROVIDER_CANCELED:
            reason = f"Serverless job {job_id} ended with provider status {status}"
            _event(ctx, t, "PROVIDER_TERMINAL_FAILURE", serverless_job_id=job_id, provider_status=status)
            ctx.registry.request_retry(t.task_id, reason=reason, max_attempts=10**6)
            retry_pending.append({"task_id": t.task_id, "fragment_id": t.fragment_id, "reason": reason})
        else:
            running.append({"task_id": t.task_id, "fragment_id": t.fragment_id, "serverless_job_id": job_id,
                            "status": status})

    return ReconciliationReport(
        completed_tasks=completed, retry_pending_tasks=retry_pending, running_tasks=running,
        uncertain_tasks=uncertain, pending_retrieval=pending_retrieval,
        active_count=sum(1 for x in _classical_tasks(ctx) if x.state in _ACTIVE))


def _reject(ctx: RunContext, t: TaskRecord, kind: str, reason: str) -> tuple[str, dict[str, Any]]:
    _event(ctx, t, kind, error=reason)
    ctx.registry.request_retry(t.task_id, reason=f"{kind}: {reason}", max_attempts=10**6)
    return "retry", {"task_id": t.task_id, "fragment_id": t.fragment_id, "reason": f"{kind}: {reason}"}


def _collect_result(ctx, t: TaskRecord, job, adapter: ServerlessTransportAdapter, max_retrieval_failures: int,
                    *, attempt: Optional[int] = None, rejector=None):
    """Authenticate and import one worker result.

    ``attempt`` overrides the task's dispatch attempt (3Q-C scopes its own post-processing attempt);
    ``rejector`` replaces the default RETRY_PENDING rejection (3Q-C tasks cannot enter RETRY_PENDING).
    """
    reject = rejector or _reject
    try:
        payload = job.result()
        if isinstance(payload, (str, bytes)):
            payload = json.loads(payload)
    except Exception as exc:  # noqa: BLE001
        return reject(ctx, t, "RESULT_UNAVAILABLE", f"{type(exc).__name__}: {exc}")
    if not isinstance(payload, dict):
        return reject(ctx, t, "RESULT_MALFORMED", f"payload is {type(payload).__name__}, not a dictionary")
    if payload.get("status") != "COMPLETE":
        return reject(ctx, t, "WORKER_REPORTED_FAILURE", str(payload.get("error", "unknown worker error"))[:500])

    attempt = t.attempt if attempt is None else attempt
    try:
        # Cross-worker scientific identity: everything the worker claims must equal what was dispatched.
        for key, want in (("run_id", ctx.run_id), ("fragment_id", t.fragment_id), ("solver", t.solver),
                          ("attempt", attempt), ("hamiltonian_sha256", t.hamiltonian_sha256)):
            if payload.get(key) != want:
                raise TransportError(f"worker result {key}={payload.get(key)!r} does not match dispatched {want!r}")
        desc = TransportDescriptor.from_dict(payload.get("result") or {})
        if (desc.run_id, desc.fragment_id, desc.solver, desc.attempt) != (ctx.run_id, t.fragment_id, t.solver, attempt):
            raise TransportError("result descriptor identity does not match the dispatched task")
        if desc.remote_name != result_transport_name(ctx.run_id, t.fragment_id, t.solver, attempt):
            raise TransportError("result transport name is not the deterministic name for this attempt")
        if desc.payload_digest != payload.get("payload_digest") or desc.hamiltonian_sha256 != t.hamiltonian_sha256:
            raise TransportError("result descriptor digests disagree with the worker summary")
    except TransportError as exc:
        return reject(ctx, t, "RESULT_IDENTITY_MISMATCH", str(exc))

    staging = ctx.paths.run_dir / "transport_staging" / f"f{t.fragment_id}_a{attempt}"
    try:
        local = adapter.fetch_descriptor(desc, staging / "result.h5")
        ref = verify_and_import_fragment_result(
            ctx.store, local, run_id=ctx.run_id, fragment_id=t.fragment_id, expected_solver=t.solver,
            expected_hamiltonian_sha256=t.hamiltonian_sha256, expected_file_sha256=desc.file_sha256,
            expected_payload_digest=desc.payload_digest,
            extra={"serverless_job_id": t.serverless_job_id, "attempt": attempt,
                   "transport_mode": desc.transport_mode, "remote_name": desc.remote_name})
    except (TransportNotFoundError, TransportChecksumError) as exc:
        failures = int(t.extra.get("retrieval_failures", 0)) + 1
        ctx.registry.update_extra(t.task_id, "retrieval_failures", failures)
        _event(ctx, t, "RETRIEVAL_FAILED", error=f"{type(exc).__name__}: {exc}", consecutive=failures)
        if failures >= max_retrieval_failures:
            return reject(ctx, t, "RETRIEVAL_EXHAUSTED", f"{failures} failed retrievals; last: {exc}")
        return "pending", {"task_id": t.task_id, "fragment_id": t.fragment_id, "failures": failures,
                           "error": f"{type(exc).__name__}: {exc}"}
    except (TransportError, ClassicalFragmentError, OSError) as exc:
        return reject(ctx, t, "VERIFICATION_FAILED", f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    elapsed = payload.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
        # worker-reported compute time (never invented): lets the organizer separate waiting from compute
        ctx.registry.update_extra(t.task_id, "worker_elapsed_seconds", float(elapsed))
    ctx.registry.complete_task(
        t.task_id, artifact_uri=str(ref.path), payload_digest=ref.payload_digest, lease_token=t.lease_token,
        reason=f"{t.solver} fragment {t.fragment_id} verified and imported")
    _event(ctx, t, "RESULT_IMPORTED", payload_digest=ref.payload_digest, file_sha256=ref.file_sha256,
           transport_mode=desc.transport_mode)
    return "completed", {"task_id": t.task_id, "fragment_id": t.fragment_id, "solver": t.solver,
                         "energy_ha": payload.get("energy_ha"), "artifact_uri": str(ref.path),
                         "payload_digest": ref.payload_digest}


# ─────────────────────────────────────────────────────────────────────────────
#  Explicit operator actions (the only paths out of uncertainty / failure)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_uncertain_submission(
    ctx: RunContext,
    task_id: str,
    *,
    reason: str,
    serverless_job_id: Optional[str] = None,
    abandon: bool = False,
    serverless_service: Any = None,
) -> TaskRecord:
    """Operator resolution of a SUBMITTING task that has no recorded provider Job ID.

    Exactly one of ``serverless_job_id`` (the job really exists: attach it and continue as RUNNING) or
    ``abandon=True`` (the operator asserts nothing was accepted: the attempt is closed as RETRY_PENDING and
    can later be requeued).  The history is preserved either way.
    """
    if not reason or not reason.strip():
        raise DispatcherError("an operator reason is required")
    if bool(serverless_job_id) == bool(abandon):
        raise DispatcherError("provide exactly one of serverless_job_id or abandon=True")
    t = ctx.registry.get_task(task_id)
    if t.run_id != ctx.run_id or t.task_type != TaskType.CLASSICAL:
        raise DispatcherError("task does not belong to this run's classical set")
    if t.state != TaskState.SUBMITTING or t.serverless_job_id:
        raise DispatcherError(
            f"task is {t.state!r} with job id {t.serverless_job_id!r}; only a SUBMITTING task without a "
            "recorded Job ID is an uncertain submission")
    if abandon:
        _event(ctx, t, "SUBMIT_ABANDONED_BY_OPERATOR", reason=reason)
        return ctx.registry.transition(task_id, TaskState.SUBMITTING, TaskState.RETRY_PENDING,
                                       lease_token=t.lease_token, reason=f"operator abandoned: {reason}")
    if serverless_service is not None:
        try:
            if serverless_service.job(serverless_job_id) is None:
                raise DispatcherError("provider does not know this job id")
        except DispatcherError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DispatcherError(f"could not confirm job {serverless_job_id!r} with the provider: {exc}") from exc
    ctx.registry.record_serverless_job_id(task_id, serverless_job_id=serverless_job_id, lease_token=t.lease_token)
    _event(ctx, t, "SUBMIT_RECOVERED_BY_OPERATOR", serverless_job_id=serverless_job_id, reason=reason)
    return ctx.registry.transition(task_id, TaskState.SUBMITTING, TaskState.RUNNING, lease_token=t.lease_token,
                                   reason=f"operator attached job {serverless_job_id}: {reason}")


def requeue_classical_tasks(ctx: RunContext, *, max_attempts: int = 3, reason: str = "operator requeue") -> dict[str, list[int]]:
    """Move RETRY_PENDING classical tasks back to READY, or to FAILED once ``max_attempts`` is exhausted."""
    out: dict[str, list[int]] = {"requeued": [], "failed": []}
    for t in ctx.registry.list_tasks(ctx.run_id, state=TaskState.RETRY_PENDING, task_type=TaskType.CLASSICAL):
        if t.attempt >= max_attempts:
            ctx.registry.fail_task(t.task_id, reason=f"max_attempts ({max_attempts}) reached; last: {t.failure_reason}")
            _event(ctx, t, "MAX_ATTEMPTS_EXHAUSTED")
            out["failed"].append(t.fragment_id)
            continue
        token = t.lease_token
        ctx.registry.reset_for_retry(t.task_id)
        if token:                       # a stale lease would otherwise block the next claim
            ctx.registry.release_lease(t.task_id, lease_token=token)
        _event(ctx, t, "REQUEUED", reason=reason)
        out["requeued"].append(t.fragment_id)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Status and bounded driver
# ─────────────────────────────────────────────────────────────────────────────

def classical_status(ctx: RunContext) -> dict[str, Any]:
    """Read-only classical campaign status (safe to call at any time; no provider contact)."""
    rows = []
    counts: dict[str, int] = {}
    for t in _classical_tasks(ctx):
        counts[t.state] = counts.get(t.state, 0) + 1
        ev = _events(t)
        rows.append({
            "fragment_id": t.fragment_id, "solver": t.solver, "state": t.state, "attempt": t.attempt,
            "serverless_job_id": t.serverless_job_id, "provider_status": t.last_provider_status,
            "uncertain": t.state == TaskState.SUBMITTING and not t.serverless_job_id,
            "last_event": ev[-1]["event"] if ev else "", "failure_reason": t.failure_reason,
        })
    return {"run_id": ctx.run_id, "by_state": counts, "tasks": rows, "total": len(rows)}


def run_classical_campaign(
    ctx: RunContext,
    serverless_service: Any,
    transport_adapter: ServerlessTransportAdapter,
    *,
    max_concurrent: int = 4,
    max_passes: int = 100,
    poll_interval: float = 0.0,
    **dispatch_kwargs: Any,
) -> CampaignSummary:
    """Bounded reconcile→dispatch loop.  Stops when complete, or when only operator action can progress."""
    started = time.time()
    passes = 0
    for passes in range(1, max_passes + 1):
        reconcile_classical_tasks(ctx, serverless_service, transport_adapter)
        dispatch_classical_tasks(ctx, serverless_service, transport_adapter,
                                 max_concurrent=max_concurrent, **dispatch_kwargs)
        tasks = _classical_tasks(ctx)
        states = [t.state for t in tasks]
        if states and all(s == TaskState.COMPLETE for s in states):
            break
        waiting = [t for t in tasks if t.state in (TaskState.READY, TaskState.RUNNING) or
                   (t.state == TaskState.SUBMITTING and t.serverless_job_id)]
        if not waiting:
            break      # only uncertain / retry-pending / failed tasks remain: operator action needed
        if poll_interval:
            time.sleep(poll_interval)

    tasks = _classical_tasks(ctx)
    n = lambda *s: sum(1 for t in tasks if t.state in s)  # noqa: E731
    return CampaignSummary(
        run_id=ctx.run_id, total_classical_tasks=len(tasks), completed_count=n(TaskState.COMPLETE),
        failed_count=n(TaskState.FAILED), retry_pending_count=n(TaskState.RETRY_PENDING),
        running_count=n(*_ACTIVE), ready_count=n(TaskState.READY),
        uncertain_count=sum(1 for t in tasks if t.state == TaskState.SUBMITTING and not t.serverless_job_id),
        all_complete=bool(tasks) and n(TaskState.COMPLETE) == len(tasks),
        elapsed_seconds=time.time() - started, passes=passes)


# ─────────────────────────────────────────────────────────────────────────────
#  Notebook entry point (thin; the cell contains no orchestration)
# ─────────────────────────────────────────────────────────────────────────────

def run_notebook_step(
    ctx: RunContext,
    client: Any,
    *,
    reconcile: bool = False,
    dispatch: bool = False,
    max_concurrent: int = 4,
    resource_family: str = "l",
    function_title: str = CLASSICAL_FUNCTION_TITLE,
    solver_configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """One explicit operator step: optionally reconcile, then optionally dispatch (in that order).

    ``client`` is an authenticated ``QiskitServerless`` (organizer side).  Nothing is contacted unless
    ``reconcile`` or ``dispatch`` is true.  Reconciliation always precedes dispatch so that freed
    concurrency slots and finished results are accounted for before new work is submitted.  Each call is
    a single pass; the notebook re-runs the cell to make progress (no hidden polling loop).
    """
    out: dict[str, Any] = {}
    if not (reconcile or dispatch):
        return out
    service = CatalogServerlessService(client)
    adapter = make_live_transport(client, function_title)
    if reconcile:
        out["reconciliation"] = reconcile_classical_tasks(ctx, service, adapter).as_dict()
    if dispatch:
        out["dispatch"] = dispatch_classical_tasks(
            ctx, service, adapter, max_concurrent=max_concurrent, resource_family=resource_family,
            function_title=function_title, solver_configs=solver_configs).as_dict()
    return out
