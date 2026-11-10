"""
quantum_coordinator.py — Organizer-side coordinator for generalized quantum campaign orchestration (Task 5.9C.2).

Orchestrates the four-stage quantum lifecycle across the organizer–QPU–Serverless boundaries:
  3Q-A Submit → 3Q-B Monitor → 3Q-C Post-process → Authenticate & Import SolverResult

Invariants & Safety Guarantees
-------------------------------
1. Authoritative ownership: The SQLite registry (`trv/1`) is owned exclusively by the organizer.
   Compute workers operate via explicit inputs without SQLite or shared memory dependencies.
2. Distinct campaign authorization: Quantum execution requires explicit `CampaignAuthorization`.
   The historical participant ledger (`checkpoints/.qpu_sampling_record.json` = 5) is preserved.
3. Manifest-driven configuration: Targets, solvers (SQD/TrimSQD), shot counts, and Hamiltonians
   are derived dynamically from the frozen `RunManifest`, supporting arbitrary fragment subsets,
   non-contiguous IDs, diverse molecular systems, or zero-quantum experiments without hardcoded constants.
4. Strict lifecycle separation:
   - Submission (3Q-A) submits authorized circuits and records provider Job IDs durably.
   - Monitoring (3Q-B) refreshes provider status without dispatching computations.
   - Post-processing (3Q-C) executes diagonalization independently on Serverless compute workers.
   - Collection & Import authenticates results (Hamiltonian SHA-256, solver metadata, RDMs) before
     transitioning tasks to COMPLETE. Provider DONE != scientific COMPLETE.
5. Duplicate prevention: Already-submitted fragments or tasks with recorded Job IDs cannot be
   silently resubmitted. Ambiguous submissions are surfaced for explicit recovery.
6. Replay provenance: Replay measurements are authenticated against Hamiltonian SHA-256 and
   circuit digests, and visibly stamped with `is_replay=True`.
7. Post-processing retry: Failed diagonalization reuses existing authenticated counts; it never
   triggers implicit QPU resubmission.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from proteasome_challenge.campaign_authorization import (
    DEFAULT_MAX_SUBMISSIONS,
    DEFAULT_SHOTS_PER_FRAGMENT,
    FROZEN_QUANTUM_FRAGMENT_IDS,
    CampaignAuthorization,
    CampaignAuthorizationError,
    QuantumDuplicateSubmissionError,
    create_organizer_campaign_authorization,
)
from proteasome_challenge.classical_dispatcher import (
    _PROVIDER_CANCELED,
    _PROVIDER_FAILED,
    _PROVIDER_SUCCESS,
    CatalogServerlessService,
    _collect_result,
    _event,
    make_live_transport,
)
from proteasome_challenge.protocol_transfer import protocol_digest as _protocol_digest
from proteasome_challenge.run_control_plane import (
    ControlPlaneError,
    RunContext,
    load_stage_ref,
)
from proteasome_challenge.solver_assignments import (
    QUANTUM_SOLVERS,
    VALID_SOLVERS,
)
from proteasome_challenge.task_registry import (
    AmbiguousSubmissionError as RegistryAmbiguousError,
    ClaimConflictError,
    DuplicateTaskError,
    FragmentTaskRegistry,
    InvalidTransitionError,
    TaskNotFoundError,
    TaskRecord,
    TaskState,
    TaskType,
)
from serverless_bundle.steps.artifact_store import KIND_EMBEDDING, KIND_FRAGMENT_H5, LocalArtifactStore
from serverless_bundle.steps.artifact_transport import (
    MockServerlessFileClient,
    ServerlessTransportAdapter,
    TransportChecksumError,
    TransportError,
    result_transport_name,
    verify_and_import_fragment_result,
)
from serverless_bundle.steps.fragment_result_store import (
    FragmentResultReference,
    load_fragment_result,
    save_fragment_result,
)
from serverless_bundle.steps.quantum_monitor import (
    MonitoringSnapshot,
    ProviderStatus,
    QuantumStatusTable,
    build_quantum_status_table,
    create_dispatch_record,
    list_dispatch_eligible,
    refresh_quantum_jobs,
)
from serverless_bundle.steps.quantum_postprocess import (
    CountsArtifact,
    MeasurementAuthenticationError,
    QuantumPostprocessError,
    retrieve_and_store_counts,
    verify_measurement_identity,
)
from serverless_bundle.steps.quantum_submission import (
    AmbiguousSubmissionError,
    DuplicateSubmissionError,
    QpuBudgetError,
    QuantumSubmissionError,
    SubmissionRecord,
    check_qpu_budget,
    load_fragment_hamiltonian_for_qpu,
    record_qpu_sampling,
    submit_quantum_fragment,
)

QUANTUM_POSTPROCESS_FUNCTION_TITLE = "qfm-proteasome-3qc-postprocess"
PP_ATTEMPT_KEY = "pp_attempt"            # per-task 3Q-C post-processing attempt counter (organizer-owned)
PP_COUNTS_SHA_KEY = "pp_counts_sha256"   # SHA-256 of the authenticated counts dispatched to 3Q-C
PP_COUNTS_PATH_KEY = "pp_counts_path"    # organizer-local counts file (never sent to a worker)
_REPLAY_BACKEND = "replay"
_REPLAY_JOB_PREFIX = "replay_"

__all__ = [
    "QUANTUM_POSTPROCESS_FUNCTION_TITLE",
    "QuantumCoordinatorError",
    "QuantumCampaignBudgetError",
    "QuantumReplayError",
    "QuantumPostprocessVerificationError",
    "QuantumCampaignStatus",
    "QuantumSubmissionReport",
    "QuantumPostprocessDispatchReport",
    "QuantumPostprocessReconciliationReport",
    "build_qpu_service_from_env",
    "is_replay_task",
    "collect_quantum_measurements",
    "requeue_failed_postprocessing",
    "resolve_uncertain_postprocess",
    "quantum_status",
    "inspect_quantum_campaign",
    "submit_quantum_campaign",
    "refresh_quantum_status",
    "apply_replay_measurements",
    "dispatch_postprocessing_campaign",
    "reconcile_postprocessing_campaign",
    "retry_failed_postprocessing_task",
    "reconcile_ambiguous_quantum_task",
    "run_quantum_postprocess_campaign",
    "run_notebook_step",
]


class QuantumCoordinatorError(ControlPlaneError):
    """Base error for organizer quantum coordination operations."""


class QuantumCampaignBudgetError(QuantumCoordinatorError):
    """Raised when campaign budget limits are violated."""


class QuantumReplayError(QuantumCoordinatorError):
    """Raised when replay measurement data fails validation."""


class QuantumPostprocessVerificationError(QuantumCoordinatorError):
    """Raised when a quantum worker result fails scientific verification."""


@dataclass(frozen=True)
class QuantumCampaignStatus:
    """Read-only status overview of the quantum campaign."""
    run_id: str
    total_quantum_tasks: int
    ready_count: int
    submitting_count: int
    submitted_count: int
    running_count: int
    samples_ready_count: int
    postprocess_queued_count: int
    postprocessing_count: int
    complete_count: int
    failed_count: int
    ambiguous_count: int
    remaining_allocation: int
    authorized_backend: str
    is_complete: bool
    as_of_utc: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_table(self) -> str:
        lines = [
            f"Quantum Campaign Status — run={self.run_id!r}  ({self.as_of_utc})",
            f"  Backend: {self.authorized_backend or 'N/A'}",
            f"  Remaining QPU Allocation: {self.remaining_allocation}",
            f"  Progress: {self.complete_count}/{self.total_quantum_tasks} COMPLETE  (is_complete={self.is_complete})",
            "  State Breakdown:",
            f"    READY              : {self.ready_count}",
            f"    SUBMITTING         : {self.submitting_count}",
            f"    SUBMITTED          : {self.submitted_count}",
            f"    RUNNING            : {self.running_count}",
            f"    SAMPLES_READY      : {self.samples_ready_count}",
            f"    POSTPROCESS_QUEUED : {self.postprocess_queued_count}",
            f"    POSTPROCESSING     : {self.postprocessing_count}",
            f"    COMPLETE           : {self.complete_count}",
            f"    FAILED             : {self.failed_count}",
            f"    AMBIGUOUS          : {self.ambiguous_count}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class QuantumSubmissionReport:
    """Summary of QPU submissions attempted in a coordinator pass."""
    submitted_tasks: List[Dict[str, Any]]
    skipped_tasks: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    remaining_allocation: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuantumPostprocessDispatchReport:
    """Summary of 3Q-C post-processing workers dispatched."""
    dispatched_tasks: List[Dict[str, Any]]
    active_count: int
    available_slots: int
    max_concurrent: int
    blocked_tasks: List[Dict[str, Any]] = field(default_factory=list)
    uncertain_tasks: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuantumPostprocessReconciliationReport:
    """Summary of 3Q-C post-processing workers reconciled."""
    completed_tasks: List[Dict[str, Any]]
    failed_tasks: List[Dict[str, Any]]
    running_tasks: List[Dict[str, Any]]
    active_count: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_quantum_task(task: TaskRecord, manifest: Optional[Any] = None) -> bool:
    """Identify if a task is a quantum task based on task_type, solver, or manifest."""
    if task.task_type == TaskType.QUANTUM or str(task.task_type).lower() == "quantum":
        return True
    if task.solver in QUANTUM_SOLVERS:
        return True
    if manifest is not None and hasattr(manifest, "solver_assignments"):
        solver = manifest.solver_assignments.get(str(task.fragment_id)) or manifest.solver_assignments.get(task.fragment_id)
        if solver in QUANTUM_SOLVERS:
            return True
    # Legacy fallback for bare test tasks. Never applied once a manifest is available: a manifest-driven
    # run derives its quantum population from the frozen solver assignments only.
    if manifest is None and task.fragment_id in FROZEN_QUANTUM_FRAGMENT_IDS and task.task_type != TaskType.CLASSICAL:
        return True
    return False


def is_replay_task(task: TaskRecord) -> bool:
    """True when the task's measurements are a replay, not a genuine provider job."""
    return str(task.backend or "").lower() == _REPLAY_BACKEND or str(task.qpu_job_id or "").startswith(_REPLAY_JOB_PREFIX)


def build_qpu_service_from_env() -> Any:
    """Organizer-side ``QiskitRuntimeService`` from the environment (never from Serverless arguments).

    Raises ``QuantumCoordinatorError`` for missing or placeholder credentials. Nothing is logged.
    """
    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE", "")
    if not token or not instance or token.startswith("<") or instance.startswith("<"):
        raise QuantumCoordinatorError(
            "IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE are missing or placeholders; no QPU service was created.")
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService(channel="ibm_cloud", token=token, instance=instance)


PROTOCOL_FILENAME = "shared_protocol.json"


def persist_protocol(ctx: RunContext, protocol: Mapping[str, Any]) -> Path:
    """Write-once copy of the frozen shared protocol inside the run directory (digest-checked)."""
    _protocol_configs(ctx, protocol)  # validates against the frozen manifest
    path = ctx.paths.run_dir / PROTOCOL_FILENAME
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")).get("protocol_digest") != protocol.get("protocol_digest"):
            raise QuantumCoordinatorError("a different shared protocol is already persisted for this run")
        return path
    path.write_text(json.dumps(dict(protocol), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _protocol_configs(ctx: RunContext, protocol: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Return ``(sqd_config, trim_config, protocol_digest)`` verified against the frozen manifest."""
    if protocol is None:
        p = ctx.paths.run_dir / PROTOCOL_FILENAME
        if not p.is_file():
            raise QuantumCoordinatorError(
                "no shared protocol is available for 3Q-C; the frozen protocol configuration is mandatory "
                "(solver defaults are never substituted)")
        protocol = json.loads(p.read_text(encoding="utf-8"))
    tr = protocol.get("transferable") or {}
    sqd, trim = dict(tr.get("sqd_config") or {}), dict(tr.get("trim_config") or {})
    if not sqd or not trim:
        raise QuantumCoordinatorError("the shared protocol carries no transferable sqd_config / trim_config")
    digest = _protocol_digest(sqd, trim, list(protocol.get("adaptations") or []))
    if protocol.get("protocol_digest") not in (None, digest):
        raise QuantumCoordinatorError("shared protocol digest does not match its own configuration")
    if ctx.manifest is None or digest != ctx.manifest.protocol_digest:
        raise QuantumCoordinatorError("shared protocol digest differs from the frozen RunManifest protocol_digest")
    return sqd, trim, digest


def _resolve_stage2_path(ctx: RunContext) -> Path:
    """Resolve the Stage 2 HDF5 dumpfile path from manifest or persisted refs."""
    if ctx.manifest is not None and ctx.manifest.embedding_artifact_ref:
        ref_d = ctx.manifest.embedding_artifact_ref
        if isinstance(ref_d, dict):
            p = ref_d.get("path") or ref_d.get("uri")
            if p and Path(p).is_file():
                return Path(p)
    ref = load_stage_ref(ctx, "embedding_ewf")
    if ref is not None:
        p = getattr(ref, "path", None) or getattr(ref, "uri", None)
        if p and Path(p).is_file():
            return Path(p)
    emb_dir = ctx.paths.run_dir / KIND_EMBEDDING
    if emb_dir.is_dir():
        candidates = list(emb_dir.glob("*.h5"))
        if candidates:
            return candidates[0]
    if hasattr(ctx.paths, "embedding_dir") and ctx.paths.embedding_dir.is_dir():
        candidates = list(ctx.paths.embedding_dir.glob("*.h5"))
        if candidates:
            return candidates[0]
    if ctx.manifest is not None:
        lig = getattr(ctx.manifest, "ligand", "ixazomib")
        basis = getattr(ctx.manifest, "basis", "sto-3g")
        p = emb_dir / f"{lig}_ligand_{basis}__ewf_dumpfile.h5"
        if p.is_file():
            return p
    return emb_dir / "ixazomib_ligand_sto-3g__ewf_dumpfile.h5"


# ─────────────────────────────────────────────────────────────────────────────
#  1. Inspection & Read-only Status
# ─────────────────────────────────────────────────────────────────────────────

def inspect_quantum_campaign(
    ctx: RunContext,
    campaign_auth: Optional[CampaignAuthorization] = None,
) -> QuantumCampaignStatus:
    """Inspect the SQLite task registry for all quantum tasks in a run.

    Derives quantum tasks dynamically from manifest and registry state.
    Purely read-only; safe to call at any time without network or QPU side effects.
    """
    all_tasks = ctx.registry.list_tasks(ctx.run_id)
    quantum_tasks = [t for t in all_tasks if _is_quantum_task(t, ctx.manifest)]

    counts: Dict[str, int] = {
        TaskState.READY: 0,
        TaskState.SUBMITTING: 0,
        TaskState.SUBMITTED: 0,
        TaskState.RUNNING: 0,
        TaskState.SAMPLES_READY: 0,
        TaskState.POSTPROCESS_QUEUED: 0,
        TaskState.POSTPROCESSING: 0,
        TaskState.COMPLETE: 0,
        TaskState.FAILED: 0,
        TaskState.AMBIGUOUS: 0,
    }
    for t in quantum_tasks:
        counts[t.state] = counts.get(t.state, 0) + 1

    remaining_alloc = campaign_auth.remaining_allocation() if campaign_auth is not None else 0
    backend = campaign_auth.authorized_backend if campaign_auth is not None else ""
    is_complete = (len(quantum_tasks) == 0) or (counts[TaskState.COMPLETE] == len(quantum_tasks))

    return QuantumCampaignStatus(
        run_id=ctx.run_id,
        total_quantum_tasks=len(quantum_tasks),
        ready_count=counts[TaskState.READY],
        submitting_count=counts[TaskState.SUBMITTING],
        submitted_count=counts[TaskState.SUBMITTED],
        running_count=counts[TaskState.RUNNING],
        samples_ready_count=counts[TaskState.SAMPLES_READY],
        postprocess_queued_count=counts[TaskState.POSTPROCESS_QUEUED],
        postprocessing_count=counts[TaskState.POSTPROCESSING],
        complete_count=counts[TaskState.COMPLETE],
        failed_count=counts[TaskState.FAILED],
        ambiguous_count=counts[TaskState.AMBIGUOUS],
        remaining_allocation=remaining_alloc,
        authorized_backend=backend,
        is_complete=is_complete,
    )


def quantum_status(
    ctx: RunContext,
    campaign_auth: Optional[CampaignAuthorization] = None,
) -> Dict[str, Any]:
    """Summary dictionary of quantum tasks for notebook rendering and programmatic checks."""
    all_tasks = ctx.registry.list_tasks(ctx.run_id)
    quantum_tasks = [t for t in all_tasks if _is_quantum_task(t, ctx.manifest)]
    by_state: Dict[str, int] = {}
    for t in quantum_tasks:
        by_state[t.state] = by_state.get(t.state, 0) + 1

    rows = []
    for t in sorted(quantum_tasks, key=lambda x: x.fragment_id):
        solver = t.solver or (ctx.manifest.solver_assignments.get(str(t.fragment_id)) if ctx.manifest else "TrimSQD") or "TrimSQD"
        rows.append({
            "task_id": t.task_id,
            "fragment_id": t.fragment_id,
            "solver": solver,
            "state": t.state,
            "qpu_job_id": t.qpu_job_id,
            "serverless_job_id": t.serverless_job_id,
            "uncertain": t.state == TaskState.AMBIGUOUS or (t.state == TaskState.SUBMITTING and not t.qpu_job_id),
            "failure_reason": t.failure_reason,
            "payload_digest": getattr(t, "result_payload_digest", getattr(t, "payload_digest", "")),
            "artifact_uri": getattr(t, "result_artifact_uri", getattr(t, "artifact_uri", "")),
        })

    return {
        "run_id": ctx.run_id,
        "total": len(quantum_tasks),
        "by_state": by_state,
        "tasks": rows,
        "is_complete": bool(quantum_tasks) and by_state.get(TaskState.COMPLETE, 0) == len(quantum_tasks),
        "remaining_allocation": campaign_auth.remaining_allocation() if campaign_auth is not None else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  2. 3Q-A Quantum Submission
# ─────────────────────────────────────────────────────────────────────────────

def submit_quantum_campaign(
    ctx: RunContext,
    campaign_auth: Optional[CampaignAuthorization] = None,
    *,
    qpu_service: Any = None,
    allow_sampling: bool = False,
    allow_second_qpu: bool = False,
    worker_id: str = "organizer_qpu_dispatcher",
    max_submissions: Optional[int] = None,
    time_box_seconds: Optional[float] = None,
) -> QuantumSubmissionReport:
    """Submit approved READY quantum tasks to the QPU subject to campaign authorization.

    Parameters
    ----------
    ctx             : RunContext with registry and manifest
    campaign_auth   : Explicit CampaignAuthorization for this run (optional if 0 quantum tasks)
    qpu_service     : QPU service client (mock or live)
    allow_sampling  : Confirmation flag for live sampling
    allow_second_qpu: Second QPU authorization flag
    worker_id       : Audit identifier for the submitting worker
    max_submissions : optional bound on submission attempts in THIS call (remaining READY tasks stay READY)
    time_box_seconds: optional wall-clock bound; no new submission starts after it elapses

    Returns
    -------
    QuantumSubmissionReport
    """
    if ctx.manifest is None:
        raise QuantumCoordinatorError(f"Run {ctx.run_id!r} has no frozen manifest; freeze manifest before submission.")

    all_tasks = ctx.registry.list_tasks(ctx.run_id)
    quantum_tasks = [t for t in all_tasks if _is_quantum_task(t, ctx.manifest)]

    if len(quantum_tasks) == 0:
        return QuantumSubmissionReport(
            submitted_tasks=[],
            skipped_tasks=[],
            errors=[],
            remaining_allocation=campaign_auth.remaining_allocation() if campaign_auth is not None else 0,
        )

    if campaign_auth is None:
        raise QuantumCampaignBudgetError(
            f"Campaign authorization required for quantum task submission in run {ctx.run_id!r}."
        )

    # Check manifest digest
    if campaign_auth.manifest_digest and ctx.manifest.manifest_digest and campaign_auth.manifest_digest != ctx.manifest.manifest_digest:
        raise QuantumCoordinatorError(
            f"Manifest digest mismatch: campaign authorized for {campaign_auth.manifest_digest[:16]}..., "
            f"but current manifest is {ctx.manifest.manifest_digest[:16]}..."
        )

    # Resolve Stage 2 HDF5 dumpfile path
    ewf_dumpfile_path = _resolve_stage2_path(ctx)
    if not ewf_dumpfile_path.is_file():
        raise QuantumCoordinatorError(f"Stage 2 dumpfile not found at {ewf_dumpfile_path}")

    submitted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    attempts = 0
    t_begin = time.monotonic()
    for task in sorted(quantum_tasks, key=lambda t: t.fragment_id):
        fid = task.fragment_id
        solver_name = task.solver or (ctx.manifest.solver_assignments.get(str(fid)) if ctx.manifest else "TrimSQD") or "TrimSQD"

        # Skip tasks that are not in READY state
        if task.state != TaskState.READY:
            skipped.append({
                "task_id": task.task_id,
                "fragment_id": fid,
                "state": task.state,
                "reason": f"Task in state {task.state!r} (not READY)",
            })
            continue

        # Skip tasks not authorized in campaign
        if not campaign_auth.is_authorized_fragment(fid):
            skipped.append({
                "task_id": task.task_id,
                "fragment_id": fid,
                "state": task.state,
                "reason": "Fragment not in authorized campaign set",
            })
            continue

        # Check existing job_id
        if task.qpu_job_id:
            skipped.append({
                "task_id": task.task_id,
                "fragment_id": fid,
                "state": task.state,
                "qpu_job_id": task.qpu_job_id,
                "reason": "Task already has a recorded QPU Job ID",
            })
            continue

        # Bounded call: unattempted READY tasks are left untouched for the next call.
        if (max_submissions is not None and attempts >= max_submissions) or (
                time_box_seconds is not None and time.monotonic() - t_begin >= time_box_seconds):
            skipped.append({"task_id": task.task_id, "fragment_id": fid, "state": task.state,
                            "reason": "deferred: per-call submission bound reached"})
            continue
        attempts += 1

        # Submit single quantum fragment
        n_reps = 1
        try:
            sub_record = submit_quantum_fragment(
                ewf_dumpfile_path=ewf_dumpfile_path,
                fragment_id=fid,
                run_id=ctx.run_id,
                solver=solver_name,
                task_id=task.task_id,
                registry=ctx.registry,
                qpu_service=qpu_service,
                backend_name=campaign_auth.authorized_backend,
                n_shots=campaign_auth.shots_per_fragment,
                n_reps=n_reps,
                allow_sampling=allow_sampling,
                allow_second_qpu=allow_second_qpu,
                campaign_auth=campaign_auth,
                hamiltonian_sha256=task.hamiltonian_sha256,
                worker_id=worker_id,
            )
            submitted.append({
                "task_id": task.task_id,
                "fragment_id": fid,
                "solver": solver_name,
                "qpu_job_id": sub_record.job_id,
                "backend": sub_record.backend_name,
                "attempt_id": sub_record.submission_attempt,
                "circuit_digest": sub_record.circuit_digest,
            })
        except (AmbiguousSubmissionError, DuplicateSubmissionError, QpuBudgetError, QuantumSubmissionError) as exc:
            errors.append({
                "task_id": task.task_id,
                "fragment_id": fid,
                "solver": solver_name,
                "error": f"{type(exc).__name__}: {exc}",
            })
        except Exception as exc:
            errors.append({
                "task_id": task.task_id,
                "fragment_id": fid,
                "solver": solver_name,
                "error": f"Unexpected error: {exc}",
            })

    return QuantumSubmissionReport(
        submitted_tasks=submitted,
        skipped_tasks=skipped,
        errors=errors,
        remaining_allocation=campaign_auth.remaining_allocation(),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  3. 3Q-B Monitoring & Status Refresh
# ─────────────────────────────────────────────────────────────────────────────

def refresh_quantum_status(
    ctx: RunContext,
    *,
    qpu_service: Any = None,
    max_queries: int = 50,
) -> MonitoringSnapshot:
    """Bounded status refresh for active quantum tasks in SUBMITTED or RUNNING state.

    If qpu_service is None, returns a snapshot of stored state without network queries.
    """
    if qpu_service is None:
        table = build_quantum_status_table(ctx.registry, ctx.run_id)
        all_tasks = ctx.registry.list_tasks(ctx.run_id)
        dispatch_ready = [t.task_id for t in all_tasks if t.state == TaskState.SAMPLES_READY and _is_quantum_task(t, ctx.manifest)]
        return MonitoringSnapshot(
            run_id=ctx.run_id,
            refreshed_utc=table.as_of_utc,
            queried_count=0,
            updated_count=0,
            refresh_errors=[],
            state_counts=table.counts,
            dispatch_ready=dispatch_ready,
            provider_raw={},
            live_refresh=False,
        )

    return refresh_quantum_jobs(
        ctx.registry,
        ctx.run_id,
        qpu_service=qpu_service,
        max_queries=max_queries,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  4. Replay Mode Integration
# ─────────────────────────────────────────────────────────────────────────────

def apply_replay_measurements(
    ctx: RunContext,
    measurement_plan: Dict[str, Any],
    *,
    counts_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Authenticate and apply replay measurement counts to READY quantum tasks.

    Advances eligible tasks from READY → SAMPLES_READY with replay provenance.
    """
    if ctx.manifest is None:
        raise QuantumCoordinatorError(f"Run {ctx.run_id!r} has no frozen manifest.")

    ewf_dumpfile_path = _resolve_stage2_path(ctx)

    specs = measurement_plan.get("specs", [])
    applied: List[Dict[str, Any]] = []

    for spec in specs:
        fid = int(spec["fragment_id"])
        mode = spec.get("mode", "replay")
        if mode != "replay":
            continue

        task_id = f"frag_{fid:02d}_quantum"
        try:
            task = ctx.registry.get_task(task_id)
        except TaskNotFoundError:
            matching = [t for t in ctx.registry.list_tasks(ctx.run_id) if t.fragment_id == fid and _is_quantum_task(t, ctx.manifest)]
            if not matching:
                continue
            task = matching[0]

        if task.state != TaskState.READY and task.state != TaskState.SAMPLES_READY:
            continue

        # Resolve counts artifact
        artifact_path_str = spec.get("artifact_path", "")
        counts_path = Path(artifact_path_str) if artifact_path_str else None
        if counts_path is None or not counts_path.is_file():
            if counts_dir is not None:
                candidates = list(counts_dir.glob(f"*frag{fid}*.npy")) + list(counts_dir.glob(f"*fragment_{fid}*.npy"))
                if candidates:
                    counts_path = candidates[0]

        if counts_path is None or not counts_path.is_file():
            raise QuantumReplayError(
                f"Replay counts artifact not found for fragment {fid} (spec artifact: {artifact_path_str!r})"
            )

        # Validate counts array and bit width
        raw_counts = np.load(counts_path, allow_pickle=True).item()
        ham = load_fragment_hamiltonian_for_qpu(ewf_dumpfile_path, fid)
        norb = ham["norb"]
        num_bits = 2 * norb

        # Verify bit keys format
        for k in list(raw_counts.keys())[:10]:
            bit_str = format(int(k), f"0{num_bits}b") if isinstance(k, (int,)) else str(k)
            if len(bit_str) != num_bits:
                raise QuantumReplayError(
                    f"Fragment {fid} replay bit-width mismatch: expected {num_bits} bits, got {len(bit_str)}"
                )

        # Verify Hamiltonian SHA-256 matches Stage 2
        ham_sha = ham["hamiltonian_sha256"]
        if task.hamiltonian_sha256 and task.hamiltonian_sha256 != ham_sha:
            raise QuantumReplayError(
                f"Fragment {fid} Hamiltonian SHA mismatch: task={task.hamiltonian_sha256[:12]} vs stage2={ham_sha[:12]}"
            )

        counts_bytes = counts_path.read_bytes()
        counts_sha = hashlib.sha256(counts_bytes).hexdigest()

        # Check expected counts sha in spec if present
        if spec.get("counts_sha256") and spec["counts_sha256"] != counts_sha:
            raise QuantumReplayError(
                f"Fragment {fid} counts SHA mismatch: expected {spec['counts_sha256'][:12]} vs actual {counts_sha[:12]}"
            )

        if not dry_run and task.state == TaskState.READY:
            lease = "replay_lease"
            replay_job_id = f"replay_frag{fid}_{counts_sha[:8]}"
            ctx.registry.transition(task.task_id, TaskState.READY, TaskState.SUBMITTING, worker_id="replay", lease_token=lease)
            ctx.registry.record_qpu_job_id(task.task_id, qpu_job_id=replay_job_id, backend="replay", lease_token=lease)
            ctx.registry.transition(task.task_id, TaskState.SUBMITTING, TaskState.SUBMITTED, worker_id="replay", lease_token=lease)
            ctx.registry.transition(
                task.task_id,
                TaskState.SUBMITTED,
                TaskState.SAMPLES_READY,
                worker_id="replay",
                lease_token=lease,
                reason=f"Replay counts authenticated from {counts_path.name} (sha={counts_sha[:8]})",
            )

        applied.append({
            "fragment_id": fid,
            "task_id": task.task_id,
            "counts_path": str(counts_path),
            "counts_sha256": counts_sha,
            "norb": norb,
            "is_replay": True,
        })

    return applied


# ─────────────────────────────────────────────────────────────────────────────
#  5. 3Q-C Post-processing Dispatch & Reconciliation
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_postprocessing_campaign(
    ctx: RunContext,
    serverless_service: Any,
    *,
    transport_adapter: Optional[ServerlessTransportAdapter] = None,
    max_concurrent: int = 4,
    resource_family: str = "m",
    function_title: str = QUANTUM_POSTPROCESS_FUNCTION_TITLE,
    worker_id: str = "organizer_3qc_dispatcher",
) -> QuantumPostprocessDispatchReport:
    """Identify SAMPLES_READY quantum tasks and dispatch Serverless 3Q-C workers.

    With a ``transport_adapter`` the production ``trn/1`` path is used (path-free specification, authenticated
    counts uploaded as an input, frozen protocol configuration, attempt-scoped result name). Without one the
    legacy local-path specification is kept for compatibility.
    """
    if transport_adapter is not None:
        return _dispatch_postprocessing_transport(
            ctx, serverless_service, transport_adapter, max_concurrent=max_concurrent,
            resource_family=resource_family, function_title=function_title, worker_id=worker_id)
    return _dispatch_postprocessing_legacy(
        ctx, serverless_service, max_concurrent=max_concurrent, resource_family=resource_family,
        function_title=function_title, worker_id=worker_id)


def _dispatch_postprocessing_legacy(
    ctx: RunContext,
    serverless_service: Any,
    *,
    max_concurrent: int = 4,
    resource_family: str = "m",
    function_title: str = QUANTUM_POSTPROCESS_FUNCTION_TITLE,
    worker_id: str = "organizer_3qc_dispatcher",
) -> QuantumPostprocessDispatchReport:
    if ctx.manifest is None:
        raise QuantumCoordinatorError(f"Run {ctx.run_id!r} has no frozen manifest.")

    all_tasks = ctx.registry.list_tasks(ctx.run_id)
    quantum_tasks = [t for t in all_tasks if _is_quantum_task(t, ctx.manifest)]

    active_states = {TaskState.POSTPROCESS_QUEUED, TaskState.POSTPROCESSING}
    active_tasks = [t for t in quantum_tasks if t.state in active_states]
    active_count = len(active_tasks)

    available_slots = max(0, max_concurrent - active_count)
    if available_slots == 0:
        return QuantumPostprocessDispatchReport(
            dispatched_tasks=[],
            active_count=active_count,
            available_slots=0,
            max_concurrent=max_concurrent,
        )

    # Eligible tasks are those in SAMPLES_READY
    eligible = [t for t in quantum_tasks if t.state == TaskState.SAMPLES_READY]
    ewf_dumpfile_path = _resolve_stage2_path(ctx)

    dispatched: List[Dict[str, Any]] = []

    for task in eligible[:available_slots]:
        fid = task.fragment_id
        solver_name = task.solver or (ctx.manifest.solver_assignments.get(str(fid)) if ctx.manifest else "TrimSQD") or "TrimSQD"

        # 1. Claim task: SAMPLES_READY -> POSTPROCESS_QUEUED -> POSTPROCESSING
        lease = task.lease_token or f"lease_3qc_{task.task_id}_{time.time()}"
        try:
            ctx.registry.transition(
                task.task_id,
                TaskState.SAMPLES_READY,
                TaskState.POSTPROCESS_QUEUED,
                worker_id=worker_id,
                lease_token=lease,
                reason="Dispatching 3Q-C post-processing worker",
            )
            ctx.registry.transition(
                task.task_id,
                TaskState.POSTPROCESS_QUEUED,
                TaskState.POSTPROCESSING,
                worker_id=worker_id,
                lease_token=lease,
                reason="3Q-C Serverless worker claimed task",
            )
        except (ClaimConflictError, InvalidTransitionError):
            continue

        # 2. Build task specification
        work_dir = tempfile.mkdtemp(prefix=f"worker_{ctx.run_id}_frag{fid}_3qc_")
        out_transport_name = result_transport_name(ctx.run_id, fid, solver_name)
        spec = {
            "run_id": ctx.run_id,
            "fragment_id": fid,
            "solver": solver_name,
            "qpu_job_id": task.qpu_job_id or f"qjob_frag{fid}",
            "hamiltonian_sha256": task.hamiltonian_sha256,
            "ewf_dumpfile_path": str(ewf_dumpfile_path),
            "result_dir": work_dir,
            "output_transport_name": out_transport_name,
            "task_id": task.task_id,
            "backend": task.backend or "ibm_fez",
            "sqd_config": {},
            "trim_config": {},
            "n_qubits": 0,  # worker resolves norb from HDF5
        }

        # 3. Submit job to Serverless service
        fn_name = function_title or "run_3qc_postprocess"
        job = serverless_service.run(
            fn_name,
            arguments=spec,
            function_size=resource_family,
        )

        # 4. Atomically persist Serverless Job ID
        job_id_str = str(job.job_id) if hasattr(job, "job_id") else str(job)
        ctx.registry.record_serverless_job_id(
            task.task_id,
            serverless_job_id=job_id_str,
            lease_token=lease,
        )

        dispatched.append({
            "task_id": task.task_id,
            "fragment_id": fid,
            "solver": solver_name,
            "serverless_job_id": job_id_str,
            "state": TaskState.POSTPROCESSING,
        })

    new_active_count = active_count + len(dispatched)
    return QuantumPostprocessDispatchReport(
        dispatched_tasks=dispatched,
        active_count=new_active_count,
        available_slots=max(0, max_concurrent - new_active_count),
        max_concurrent=max_concurrent,
    )


def reconcile_postprocessing_campaign(
    ctx: RunContext,
    serverless_service: Any,
    *,
    transport_adapter: Optional[ServerlessTransportAdapter] = None,
    worker_id: str = "organizer_3qc_reconciler",
) -> QuantumPostprocessReconciliationReport:
    """Reconcile Serverless 3Q-C jobs, verify results, and commit COMPLETE state.

    Tasks dispatched through the ``trn/1`` path (they carry a ``pp_attempt``) are collected through the
    transport adapter; legacy-dispatched tasks keep the legacy handling.
    """
    if ctx.manifest is None:
        raise QuantumCoordinatorError(f"Run {ctx.run_id!r} has no frozen manifest.")

    all_tasks = ctx.registry.list_tasks(ctx.run_id)
    running_tasks = [t for t in all_tasks if t.state == TaskState.POSTPROCESSING and _is_quantum_task(t, ctx.manifest)]

    completed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    still_running: List[Dict[str, Any]] = []

    for task in running_tasks:
        s_job_id = task.serverless_job_id
        fid = task.fragment_id
        solver_name = task.solver or (ctx.manifest.solver_assignments.get(str(fid)) if ctx.manifest else "TrimSQD") or "TrimSQD"

        if task.extra.get(PP_ATTEMPT_KEY):
            if transport_adapter is None:
                still_running.append({"task_id": task.task_id, "fragment_id": fid, "serverless_job_id": s_job_id,
                                      "status": "AWAITING_TRANSPORT_ADAPTER"})
            else:
                _reconcile_transport_task(ctx, task, serverless_service, transport_adapter,
                                          completed, failed, still_running)
            continue

        if not s_job_id:
            ctx.registry.fail_task(
                task.task_id,
                reason="Task in POSTPROCESSING state without Serverless Job ID",
                lease_token=task.lease_token,
            )
            failed.append({"task_id": task.task_id, "fragment_id": fid, "error": "Missing serverless_job_id"})
            continue

        try:
            job = serverless_service.job(s_job_id)
            status_str = str(job.status()).upper()
        except Exception as exc:
            still_running.append({"task_id": task.task_id, "fragment_id": fid, "serverless_job_id": s_job_id, "query_error": str(exc)})
            continue

        if status_str not in _PROVIDER_SUCCESS and status_str not in _PROVIDER_FAILED \
                and status_str not in _PROVIDER_CANCELED:
            # RUNNING / QUEUED / INITIALIZING / any non-terminal provider state: keep waiting.
            still_running.append({"task_id": task.task_id, "fragment_id": fid, "serverless_job_id": s_job_id, "status": status_str})
            continue

        if status_str in _PROVIDER_SUCCESS:
            try:
                res = job.result()
                if not isinstance(res, dict) or res.get("status") != "COMPLETE":
                    raise QuantumPostprocessVerificationError(f"Worker reported non-complete status: {res.get('error', 'Unknown error')}")

                # Verify result file / payload
                artifact_uri = res.get("artifact_uri", "")
                result_path = Path(artifact_uri) if artifact_uri else None
                file_sha = res.get("file_sha256", "")

                remote_name = res.get("remote_transport_name")
                if transport_adapter is not None and remote_name:
                    size = res.get("file_size_bytes")
                    if not file_sha or not isinstance(size, int):
                        raise QuantumPostprocessVerificationError(
                            "Worker result lacks file_sha256/file_size_bytes; cannot authenticate transport")
                    staging = Path(tempfile.mkdtemp(prefix=f"dl_{ctx.run_id}_frag{fid}_3qc_"))
                    try:
                        downloaded = transport_adapter.download_file(
                            remote_name, staging / "result.h5",
                            expected_sha256=file_sha, expected_size=size)
                        imported_ref = verify_and_import_fragment_result(
                            ctx.store,
                            downloaded,
                            run_id=ctx.run_id,
                            fragment_id=fid,
                            expected_solver=solver_name,
                            expected_hamiltonian_sha256=task.hamiltonian_sha256,
                            expected_file_sha256=file_sha,
                        )
                    finally:
                        shutil.rmtree(staging, ignore_errors=True)
                else:
                    if result_path is None or not result_path.is_file():
                        raise QuantumPostprocessVerificationError(f"Result file not found at {result_path}")
                    imported_ref = verify_and_import_fragment_result(
                        ctx.store,
                        result_path,
                        run_id=ctx.run_id,
                        fragment_id=fid,
                        expected_solver=solver_name,
                        expected_hamiltonian_sha256=task.hamiltonian_sha256,
                    )

                # Commit task to COMPLETE
                ctx.registry.complete_task(
                    task.task_id,
                    artifact_uri=str(imported_ref.path),
                    payload_digest=imported_ref.payload_digest,
                    lease_token=task.lease_token,
                    reason=f"3Q-C complete: E={res.get('energy', 0.0):.8f} Ha",
                )

                completed.append({
                    "task_id": task.task_id,
                    "fragment_id": fid,
                    "solver": solver_name,
                    "energy": res.get("energy"),
                    "artifact_uri": str(imported_ref.path),
                })
            except Exception as exc:
                ctx.registry.fail_task(
                    task.task_id,
                    reason=f"Result verification failed: {exc}",
                    lease_token=task.lease_token,
                )
                failed.append({
                    "task_id": task.task_id,
                    "fragment_id": fid,
                    "error": f"Verification error: {exc}",
                })
        else:
            # Definite failure from Serverless
            ctx.registry.fail_task(
                task.task_id,
                reason=f"Serverless job failed with status: {status_str}",
                lease_token=task.lease_token,
            )
            failed.append({
                "task_id": task.task_id,
                "fragment_id": fid,
                "error": f"Serverless job failed: {status_str}",
            })

    return QuantumPostprocessReconciliationReport(
        completed_tasks=completed,
        failed_tasks=failed,
        running_tasks=still_running,
        active_count=len(still_running),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  5b. Production 3Q-C path: authenticated counts + path-free trn/1 specification
# ─────────────────────────────────────────────────────────────────────────────

def _quantum_tasks(ctx: RunContext) -> List[TaskRecord]:
    return [t for t in ctx.registry.list_tasks(ctx.run_id) if _is_quantum_task(t, ctx.manifest)]


def _solver_of(ctx: RunContext, task: TaskRecord) -> str:
    return task.solver or (ctx.manifest.solver_assignments.get(str(task.fragment_id)) if ctx.manifest else None) or "TrimSQD"


def collect_quantum_measurements(
    ctx: RunContext,
    qpu_service: Any,
    *,
    max_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    """Retrieve and authenticate the counts of every SAMPLES_READY (provider-DONE) genuine QPU job.

    Organizer-side only: the provider credentials never leave this process. Counts are stored write-once
    under ``<run_dir>/qpu_counts`` and their SHA-256 is recorded on the task. Replay measurements are never
    collected here. A failure is recorded and reported; it never changes task state and never touches the QPU.
    """
    collected: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    already = 0
    if ctx.manifest is None:
        raise QuantumCoordinatorError(f"Run {ctx.run_id!r} has no frozen manifest.")
    ewf_path = _resolve_stage2_path(ctx)
    for task in sorted(_quantum_tasks(ctx), key=lambda t: t.fragment_id):
        if task.state != TaskState.SAMPLES_READY:
            continue
        if is_replay_task(task):
            rejected.append({"task_id": task.task_id, "fragment_id": task.fragment_id,
                             "reason": "replay measurements are not accepted in a live trial"})
            continue
        if task.extra.get(PP_COUNTS_SHA_KEY):
            already += 1
            continue
        if max_tasks is not None and len(collected) >= max_tasks:
            break
        fid = task.fragment_id
        try:
            ham = load_fragment_hamiltonian_for_qpu(ewf_path, fid)
            if task.hamiltonian_sha256 and ham["hamiltonian_sha256"] != task.hamiltonian_sha256:
                raise QuantumPostprocessError("Stage 2 Hamiltonian differs from the task's Hamiltonian SHA-256")
            norb = int(ham["norb"])
            counts_path = ctx.paths.run_dir / "qpu_counts" / f"counts_frag{fid}_{task.qpu_job_id[:12]}.npy"
            art = retrieve_and_store_counts(task.qpu_job_id, qpu_service, counts_path, norb=norb)
            verify_measurement_identity(
                art, {"fragment_id": fid, "qpu_job_id": task.qpu_job_id,
                      "submission_record": {"n_qubits": 2 * norb}}, expected_n_bits=2 * norb)
            ctx.registry.update_extra(task.task_id, PP_COUNTS_SHA_KEY, art.counts_sha256)
            ctx.registry.update_extra(task.task_id, PP_COUNTS_PATH_KEY, str(counts_path))
            ctx.registry.update_extra(task.task_id, "pp_n_qubits", 2 * norb)
            ctx.registry.update_extra(task.task_id, "pp_counts_shots", int(art.n_shots))
            _event(ctx, task, "MEASUREMENT_COLLECTED", counts_sha256=art.counts_sha256, n_shots=int(art.n_shots))
            collected.append({"task_id": task.task_id, "fragment_id": fid, "counts_sha256": art.counts_sha256,
                              "n_shots": int(art.n_shots)})
        except Exception as exc:  # noqa: BLE001 — recorded; state unchanged; retried on the next pass
            errors.append({"task_id": task.task_id, "fragment_id": fid, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    return {"collected": collected, "errors": errors, "rejected": rejected, "already_collected": already}


def _upload_ewf_input(ctx: RunContext, adapter: ServerlessTransportAdapter):
    ref = load_stage_ref(ctx, "embedding_ewf")
    if ref is None:
        raise QuantumCoordinatorError("Stage 2 'embedding_ewf' reference is not published for this run")
    if ref.file_sha256 != ctx.manifest.embedding_artifact_ref.get("file_sha256"):
        raise QuantumCoordinatorError("published Stage 2 artifact differs from the frozen manifest")
    verification = ctx.store.verify(ref)
    if not verification.ok:
        raise QuantumCoordinatorError(f"Stage 2 artifact failed local verification: {verification.reason}")
    return adapter.upload_input(Path(ref.uri), run_id=ctx.run_id, file_role="ewf_dumpfile",
                                expected_sha256=ref.file_sha256)


def _dispatch_postprocessing_transport(
    ctx: RunContext,
    serverless_service: Any,
    adapter: ServerlessTransportAdapter,
    *,
    max_concurrent: int,
    resource_family: str,
    function_title: str,
    worker_id: str,
) -> QuantumPostprocessDispatchReport:
    if ctx.manifest is None:
        raise QuantumCoordinatorError(f"Run {ctx.run_id!r} has no frozen manifest.")
    if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent < 0:
        raise QuantumCoordinatorError("max_concurrent must be an integer >= 0")
    tasks = _quantum_tasks(ctx)
    active = [t for t in tasks if t.state in (TaskState.POSTPROCESS_QUEUED, TaskState.POSTPROCESSING)]
    uncertain = [{"task_id": t.task_id, "fragment_id": t.fragment_id, "attempt": t.extra.get(PP_ATTEMPT_KEY),
                  "error": "3Q-C SUBMISSION_UNCERTAIN (no Serverless Job ID recorded)"}
                 for t in active if t.extra.get(PP_ATTEMPT_KEY) and not t.serverless_job_id]
    available = max(0, max_concurrent - len(active))
    eligible = sorted((t for t in tasks if t.state == TaskState.SAMPLES_READY), key=lambda t: t.fragment_id)
    blocked: List[Dict[str, Any]] = []
    ready: List[TaskRecord] = []
    for t in eligible:
        if is_replay_task(t):
            blocked.append({"task_id": t.task_id, "fragment_id": t.fragment_id,
                            "reason": "replay measurements are not accepted"})
        elif not t.extra.get(PP_COUNTS_SHA_KEY):
            blocked.append({"task_id": t.task_id, "fragment_id": t.fragment_id,
                            "reason": "measurement counts not yet collected and authenticated"})
        else:
            ready.append(t)
    dispatched: List[Dict[str, Any]] = []
    batch = ready[:available]
    if batch:
        sqd, trim, pdigest = _protocol_configs(ctx)
        ewf_desc = _upload_ewf_input(ctx, adapter)
        size = resource_family if resource_family in ("s", "m", "l") else "m"
        for task in batch:
            fid = task.fragment_id
            solver = _solver_of(ctx, task)
            counts_path = Path(str(task.extra.get(PP_COUNTS_PATH_KEY, "")))
            counts_sha = str(task.extra[PP_COUNTS_SHA_KEY])
            if not counts_path.is_file():
                blocked.append({"task_id": task.task_id, "fragment_id": fid, "reason": "authenticated counts file is missing"})
                continue
            counts_desc = adapter.upload_input(counts_path, run_id=ctx.run_id, file_role=f"qpucounts{fid}",
                                               expected_sha256=counts_sha)
            attempt = int(task.extra.get(PP_ATTEMPT_KEY, 0)) + 1
            lease = task.lease_token or f"lease_3qc_{task.task_id}_{time.time()}"
            # Recorded BEFORE the transitions: a crash after them leaves a POSTPROCESSING task that still
            # carries its attempt, so reconciliation reports an uncertain submission rather than a legacy failure.
            ctx.registry.update_extra(task.task_id, PP_ATTEMPT_KEY, attempt)
            try:
                ctx.registry.transition(task.task_id, TaskState.SAMPLES_READY, TaskState.POSTPROCESS_QUEUED,
                                        worker_id=worker_id, lease_token=lease, reason="Dispatching 3Q-C post-processing worker")
                rec = ctx.registry.transition(task.task_id, TaskState.POSTPROCESS_QUEUED, TaskState.POSTPROCESSING,
                                              worker_id=worker_id, lease_token=lease, reason="3Q-C Serverless worker claimed task")
            except (ClaimConflictError, InvalidTransitionError):
                continue
            out_name = result_transport_name(ctx.run_id, fid, solver, attempt)
            spec = {
                "run_id": ctx.run_id, "fragment_id": fid, "solver": solver, "task_id": task.task_id,
                "qpu_job_id": task.qpu_job_id, "backend": task.backend or "",
                "hamiltonian_sha256": task.hamiltonian_sha256,
                "ewf_input": ewf_desc.as_dict(), "counts_input": counts_desc.as_dict(),
                "counts_sha256": counts_sha, "n_qubits": int(task.extra.get("pp_n_qubits", 0)),
                "sqd_config": sqd, "trim_config": trim if solver == "TrimSQD" else {},
                "protocol_digest": pdigest, "attempt": attempt, "output_transport_name": out_name,
            }
            try:
                json.dumps(spec)
                _event(ctx, rec, "PP_SUBMIT_INTENT", pp_attempt=attempt, function=function_title,
                       function_size=size, output_transport_name=out_name)
                run = getattr(serverless_service, "run", None)
                if callable(run):
                    # Existing mock/legacy service interface.
                    job = run(
                        function_title, arguments=spec, function_size=size
                    )
                else:
                    # Live QiskitServerless: load the function, then execute.
                    job = serverless_service.load(function_title).run(
                        **spec, function_size=size
                    )
                job_id = str(getattr(job, "job_id", "") or "")
                if not job_id:
                    raise QuantumCoordinatorError("provider returned a job handle without a job_id")
            except Exception as exc:  # noqa: BLE001 — may or may not have reached the provider: UNCERTAIN
                _event(ctx, rec, "PP_SUBMIT_UNCERTAIN", pp_attempt=attempt, error=f"{type(exc).__name__}: {str(exc)[:300]}")
                uncertain.append({"task_id": task.task_id, "fragment_id": fid, "attempt": attempt,
                                  "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                continue
            ctx.registry.record_serverless_job_id(task.task_id, serverless_job_id=job_id, lease_token=lease)
            _event(ctx, rec, "PP_SUBMIT_ACCEPTED", pp_attempt=attempt, serverless_job_id=job_id)
            dispatched.append({"task_id": task.task_id, "fragment_id": fid, "solver": solver,
                               "serverless_job_id": job_id, "attempt": attempt, "state": TaskState.POSTPROCESSING})
    n_active = sum(1 for t in _quantum_tasks(ctx) if t.state in (TaskState.POSTPROCESS_QUEUED, TaskState.POSTPROCESSING))
    return QuantumPostprocessDispatchReport(
        dispatched_tasks=dispatched, active_count=n_active, available_slots=max(0, max_concurrent - n_active),
        max_concurrent=max_concurrent, blocked_tasks=blocked, uncertain_tasks=uncertain)


def _pp_reject(ctx: RunContext, t: TaskRecord, kind: str, reason: str):
    """3Q-C rejection: POSTPROCESSING cannot enter RETRY_PENDING, so the task FAILS; counts and QPU Job ID stay."""
    _event(ctx, t, kind, error=reason)
    ctx.registry.fail_task(t.task_id, reason=f"{kind}: {reason}"[:500], lease_token=t.lease_token)
    return "failed", {"task_id": t.task_id, "fragment_id": t.fragment_id, "error": f"{kind}: {reason}"[:500]}


def _reconcile_transport_task(ctx, task, service, adapter, completed, failed, running) -> None:
    fid = task.fragment_id
    attempt = int(task.extra[PP_ATTEMPT_KEY])
    if not task.serverless_job_id:
        running.append({"task_id": task.task_id, "fragment_id": fid, "status": "SUBMISSION_UNCERTAIN"})
        return
    try:
        job = service.job(task.serverless_job_id)
        if job is None:
            raise QuantumCoordinatorError("provider reports no such job")
        status = str(job.status()).upper()
    except Exception as exc:  # noqa: BLE001 — a failed query is never evidence of failure
        running.append({"task_id": task.task_id, "fragment_id": fid, "serverless_job_id": task.serverless_job_id,
                        "status": "PROVIDER_QUERY_ERROR", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        return
    ctx.registry.update_provider_status(task.task_id, provider_status=status)
    if status in _PROVIDER_SUCCESS:
        kind, info = _collect_result(ctx, task, job, adapter, 3, attempt=attempt, rejector=_pp_reject)
        if kind == "completed":
            completed.append(info)
        elif kind == "pending":
            running.append({**info, "status": "PENDING_RETRIEVAL"})
        else:
            failed.append(info)
    elif status in _PROVIDER_FAILED or status in _PROVIDER_CANCELED:
        reason = f"Serverless 3Q-C job {task.serverless_job_id} ended with provider status {status}"
        _event(ctx, task, "PP_PROVIDER_TERMINAL_FAILURE", provider_status=status)
        ctx.registry.fail_task(task.task_id, reason=reason, lease_token=task.lease_token)
        failed.append({"task_id": task.task_id, "fragment_id": fid, "error": reason})
    else:
        running.append({"task_id": task.task_id, "fragment_id": fid, "serverless_job_id": task.serverless_job_id,
                        "status": status})


def _reset_to_samples_ready(ctx: RunContext, task_id: str, from_state: str, reason: str) -> TaskRecord:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with ctx.registry._conn:
        cur = ctx.registry._conn.execute(
            """UPDATE fragment_tasks SET state = ?, failure_reason = '', worker_id = '', lease_token = '',
               serverless_job_id = '', updated_iso8601 = ? WHERE task_id = ? AND state = ?""",
            (TaskState.SAMPLES_READY, now, task_id, from_state))
        if cur.rowcount != 1:
            raise QuantumCoordinatorError(f"Task {task_id!r} is no longer in state {from_state!r}")
        ctx.registry._append_transition(task_id, from_state, TaskState.SAMPLES_READY, "", reason, now)
    return ctx.registry.get_task(task_id)


def requeue_failed_postprocessing(ctx: RunContext, *, max_attempts: int = 3,
                                  reason: str = "operator requeue of failed 3Q-C from preserved counts") -> Dict[str, List[int]]:
    """FAILED quantum tasks with a QPU Job ID and authenticated counts -> SAMPLES_READY. Never touches the QPU."""
    requeued: List[int] = []
    exhausted: List[int] = []
    for t in _quantum_tasks(ctx):
        if t.state != TaskState.FAILED or not t.qpu_job_id or not t.extra.get(PP_COUNTS_SHA_KEY):
            continue
        if int(t.extra.get(PP_ATTEMPT_KEY, 0)) >= max_attempts:
            exhausted.append(t.fragment_id)
            continue
        _reset_to_samples_ready(ctx, t.task_id, TaskState.FAILED, reason)
        requeued.append(t.fragment_id)
    return {"requeued": requeued, "exhausted": exhausted}


def resolve_uncertain_postprocess(ctx: RunContext, task_id: str, *, reason: str,
                                  serverless_job_id: Optional[str] = None) -> TaskRecord:
    """Explicit operator resolution of an uncertain 3Q-C submission.

    With ``serverless_job_id`` the operator confirms the job exists; otherwise the task is returned to
    SAMPLES_READY for a fresh attempt (the QPU is never contacted).
    """
    if not reason or not reason.strip():
        raise QuantumCoordinatorError("a reason is required to resolve an uncertain 3Q-C submission")
    task = ctx.registry.get_task(task_id)
    if task.state != TaskState.POSTPROCESSING or task.serverless_job_id:
        raise QuantumCoordinatorError(f"Task {task_id!r} has no uncertain 3Q-C submission")
    if serverless_job_id:
        _event(ctx, task, "PP_SUBMIT_RESOLVED", serverless_job_id=serverless_job_id, reason=reason)
        return ctx.registry.record_serverless_job_id(task_id, serverless_job_id=serverless_job_id, lease_token=task.lease_token)
    _event(ctx, task, "PP_SUBMIT_RESOLVED", resubmit=True, reason=reason)
    return _reset_to_samples_ready(ctx, task_id, TaskState.POSTPROCESSING, reason)


# ─────────────────────────────────────────────────────────────────────────────
#  6. Recovery & Error Handling
# ─────────────────────────────────────────────────────────────────────────────

def retry_failed_postprocessing_task(
    ctx: RunContext,
    task_id: str,
    *,
    reason: str = "Operator requested 3Q-C postprocessing retry from preserved counts",
) -> TaskRecord:
    """Reset a FAILED quantum task back to SAMPLES_READY without resubmitting to QPU.

    Preserves the recorded QPU Job ID and authenticated counts.
    """
    task = ctx.registry.get_task(task_id)
    if task.state != TaskState.FAILED:
        raise QuantumCoordinatorError(f"Task {task_id!r} is in state {task.state!r} (must be FAILED to retry)")

    if not task.qpu_job_id:
        raise QuantumCoordinatorError(
            f"Task {task_id!r} has no recorded QPU Job ID; cannot retry post-processing without counts."
        )

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with ctx.registry._conn:
        ctx.registry._conn.execute(
            """
            UPDATE fragment_tasks
            SET state           = ?,
                failure_reason  = '',
                worker_id       = '',
                lease_token     = '',
                updated_iso8601 = ?
            WHERE task_id = ? AND state = ?
            """,
            (TaskState.SAMPLES_READY, now, task_id, TaskState.FAILED),
        )
        ctx.registry._append_transition(task_id, TaskState.FAILED, TaskState.SAMPLES_READY, "", reason, now)
    return ctx.registry.get_task(task_id)


def reconcile_ambiguous_quantum_task(
    ctx: RunContext,
    task_id: str,
    *,
    confirmed_job_id: Optional[str] = None,
    abandon_and_fail: bool = False,
    reason: str = "",
    campaign_auth: Optional[CampaignAuthorization] = None,
) -> TaskRecord:
    """Reconcile an AMBIGUOUS quantum task safely.

    If confirmed_job_id is provided, transitions AMBIGUOUS → SUBMITTED with the job ID.
    If abandon_and_fail is True, transitions AMBIGUOUS → FAILED.

    When ``campaign_auth`` is supplied the matching campaign-ledger reservation is
    settled too: confirmed with the Job ID, or marked ABANDONED (the quota unit stays
    consumed — conservative accounting).
    """
    task = ctx.registry.get_task(task_id)
    if task.state != TaskState.AMBIGUOUS:
        raise QuantumCoordinatorError(f"Task {task_id!r} is in state {task.state!r} (not AMBIGUOUS)")

    if abandon_and_fail:
        failed = ctx.registry.fail_task(
            task_id,
            reason=reason or "Operator abandoned ambiguous submission",
        )
        if campaign_auth is not None and any(
            e.task_id == task_id for e in campaign_auth.unresolved_reservations()
        ):
            campaign_auth.abandon_reservation(task.fragment_id, reason=reason)
        return failed

    effective_job_id = confirmed_job_id or task.qpu_job_id
    if not effective_job_id:
        raise QuantumCoordinatorError(
            f"Cannot recover ambiguous task {task_id!r} without a confirmed Job ID. "
            "Pass confirmed_job_id or set abandon_and_fail=True."
        )

    backend = task.backend or "ibm_fez"
    recovered = ctx.registry.recover_ambiguous_submission(
        task_id,
        qpu_job_id=effective_job_id,
        backend=backend,
        recovery_reason=reason or f"Recovered with confirmed QPU Job ID {effective_job_id!r}",
        lease_token=task.lease_token,
    )
    if campaign_auth is not None:
        for entry in campaign_auth.unresolved_reservations():
            if entry.task_id == task_id:
                campaign_auth.confirm_submission(entry.attempt_id, effective_job_id)
    return recovered


# ─────────────────────────────────────────────────────────────────────────────
#  7. Notebook Step Helper & End-to-end Postprocess Campaign Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_notebook_step(
    ctx: RunContext,
    serverless_client: Any = None,
    *,
    refresh: bool = False,
    dispatch: bool = False,
    reconcile: bool = False,
    qpu_service: Any = None,
    max_concurrent: int = 4,
    resource_family: str = "m",
    function_title: str = QUANTUM_POSTPROCESS_FUNCTION_TITLE,
    trn_transport: bool = False,
    collect: bool = False,
) -> Dict[str, Any]:
    """Single-pass notebook step for quantum 3Q-B monitoring and 3Q-C post-processing.

    ``trn_transport=True`` selects the production path (authenticated counts + path-free ``trn/1`` spec);
    it is required for a real Serverless client. ``collect=True`` retrieves and authenticates DONE counts
    from ``qpu_service`` (organizer side) before dispatch.
    """
    out: Dict[str, Any] = {}
    if refresh:
        snap = refresh_quantum_status(ctx, qpu_service=qpu_service)
        out["refresh"] = snap.as_dict() if hasattr(snap, "as_dict") else {
            "queried_count": snap.queried_count,
            "updated_count": snap.updated_count,
        }
    if collect and qpu_service is not None:
        out["collect"] = collect_quantum_measurements(ctx, qpu_service)
    if (reconcile or dispatch) and serverless_client is not None:
        if hasattr(serverless_client, "run") and hasattr(serverless_client, "job") and not hasattr(serverless_client, "load"):
            service = serverless_client
            adapter = None
        else:
            service = CatalogServerlessService(serverless_client)
            adapter = make_live_transport(serverless_client, function_title)
        if not trn_transport:
            adapter = None

        if reconcile:
            out["reconciliation"] = reconcile_postprocessing_campaign(ctx, service, transport_adapter=adapter).as_dict()
        if dispatch:
            out["dispatch"] = dispatch_postprocessing_campaign(
                ctx, service, transport_adapter=adapter,
                max_concurrent=max_concurrent, resource_family=resource_family,
                function_title=function_title,
            ).as_dict()
    return out


def run_quantum_postprocess_campaign(
    ctx: RunContext,
    serverless_service: Any,
    *,
    transport_adapter: Optional[ServerlessTransportAdapter] = None,
    max_concurrent: int = 4,
    poll_interval: float = 0.1,
    max_poll_seconds: float = 60.0,
) -> QuantumCampaignStatus:
    """Run post-processing dispatch and reconciliation until all SAMPLES_READY tasks complete."""
    t_start = time.time()
    while time.time() - t_start < max_poll_seconds:
        dispatch_postprocessing_campaign(
            ctx,
            serverless_service,
            transport_adapter=transport_adapter,
            max_concurrent=max_concurrent,
        )
        reconcile_postprocessing_campaign(
            ctx,
            serverless_service,
            transport_adapter=transport_adapter,
        )
        status = inspect_quantum_campaign(ctx)
        active = status.postprocessing_count + status.postprocess_queued_count + status.samples_ready_count
        if active == 0:
            break
        time.sleep(poll_interval)
    return inspect_quantum_campaign(ctx)
