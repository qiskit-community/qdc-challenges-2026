"""
full_trial.py — concurrent FULL_LIVE_TRIAL coordination (Task 5.9E).

A thin layer over the existing dispatchers.  It adds no scheduler, registry, storage service or scientific
code: classical dispatch/reconciliation, QPU submission/monitoring/collection and 3Q-C dispatch/reconciliation
are the unchanged ``classical_dispatcher`` / ``quantum_coordinator`` functions.  What this module adds is

* ``FullTrialProfile``       ceilings (not guarantees) for the three overlapping branches
* ``capacity_plan``          one shared Serverless budget with a reservation so 3Q-C is never starved
* ``verify_launch_conditions`` fail-closed preconditions (manifest, protocol, authorization, contamination)
* ``launch_full_trial``      dispatch classical work, then submit QPU work WITHOUT waiting for results
* ``advance_full_trial``     one bounded, non-blocking pass: reconcile -> refresh -> collect -> 3Q-C -> refill
* ``full_trial_status``      progress plus measured wall-clock overlap (waiting time kept apart from compute)

Nothing is hardcoded about the molecule: fragment populations come from the frozen RunManifest.
``advance_full_trial`` never submits to the QPU.  Stage 4 reconstruction is never started here.
Every step of a pass is isolated: one failing branch (or provider outage) never blocks another.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from proteasome_challenge import clean_slate
from proteasome_challenge.campaign_authorization import CampaignAuthorization, create_organizer_campaign_authorization
from proteasome_challenge.promotion_manifest import DECLARED_GRID_LABEL
from proteasome_challenge.classical_dispatcher import (
    CLASSICAL_FUNCTION_TITLE,
    CatalogServerlessService,
    classical_status,
    dispatch_classical_tasks,
    make_live_transport,
    reconcile_classical_tasks,
)
from proteasome_challenge.quantum_coordinator import (
    PP_COUNTS_SHA_KEY,
    QUANTUM_POSTPROCESS_FUNCTION_TITLE,
    QuantumCoordinatorError,
    _protocol_configs,
    _resolve_stage2_path,
    build_qpu_service_from_env,
    collect_quantum_measurements,
    dispatch_postprocessing_campaign,
    is_replay_task,
    reconcile_postprocessing_campaign,
    refresh_quantum_status,
    submit_quantum_campaign,
)
from proteasome_challenge.run_control_plane import ControlPlaneError, RunContext
from proteasome_challenge.solver_assignments import QUANTUM_SOLVERS
from proteasome_challenge.task_registry import TaskRecord, TaskState, TaskType

__all__ = [
    "PROFILE_NAME", "LEDGER_FILENAME", "FullTrialError", "FullTrialProfile", "capacity_plan",
    "build_live_interfaces", "verify_launch_conditions", "launch_full_trial", "advance_full_trial",
    "full_trial_status", "format_status", "declare_reference_protocol",
    "REFERENCE_SQD_CONFIG", "REFERENCE_TRIM_CONFIG",
    "stable_run_id", "ensure_clean_start", "authorize_stage_execution", "freeze_declared_protocol",
    "protocol_provenance", "trial_campaign", "describe_profile", "clean_slate_report", "format_clean_slate_report",
    "ceiling_note", "ceiling_history", "uncertain_submissions", "classify_submission_error",
    "notebook_launch", "notebook_advance", "notebook_poll_until_complete", "format_operation", "format_clean_start", "assert_legacy_flags_off",
]

PROFILE_NAME = "FULL_LIVE_TRIAL"
LEDGER_FILENAME = "full_trial_ledger.json"
_PP_ACTIVE = (TaskState.POSTPROCESS_QUEUED, TaskState.POSTPROCESSING)
_CLASSICAL_ACTIVE = (TaskState.SUBMITTING, TaskState.RUNNING)
_VALID_FAMILIES = ("s", "m", "l")


class FullTrialError(ControlPlaneError):
    """A full-trial precondition is violated; nothing was submitted."""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _short_err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


# ─────────────────────────────────────────────────────────────────────────────
#  Profile and shared-capacity policy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FullTrialProfile:
    """Ceilings for the concurrent branches.  ``None`` means "as many as the manifest has".

    These are organizer-set CEILINGS bounded by real service limits; they are not guarantees.
    ``serverless_max_total_jobs`` is the shared Serverless quota (classical + 3Q-C).  It is an organizer
    setting: no provider limit is documented in the repository, so ``None`` means "unbounded here".
    """
    classical_max_concurrent: Optional[int] = None
    classical_resource_family: str = "l"
    postprocess_max_concurrent: Optional[int] = None
    postprocess_resource_family: str = "l"
    serverless_max_total_jobs: Optional[int] = None
    postprocess_reserved_slots: int = 1
    qpu_submit_max_per_call: Optional[int] = None
    qpu_submit_time_box_seconds: Optional[float] = None
    max_status_queries: int = 100

    def __post_init__(self) -> None:
        for name in ("classical_max_concurrent", "postprocess_max_concurrent", "serverless_max_total_jobs",
                     "qpu_submit_max_per_call"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
                raise FullTrialError(f"{name} must be None or an integer >= 1, got {v!r}")
        for name in ("classical_resource_family", "postprocess_resource_family"):
            if getattr(self, name) not in _VALID_FAMILIES:
                raise FullTrialError(f"{name} must be one of {_VALID_FAMILIES}")
        if isinstance(self.postprocess_reserved_slots, bool) or not isinstance(self.postprocess_reserved_slots, int) \
                or self.postprocess_reserved_slots < 0:
            raise FullTrialError("postprocess_reserved_slots must be an integer >= 0")
        if self.qpu_submit_time_box_seconds is not None and not self.qpu_submit_time_box_seconds > 0:
            raise FullTrialError("qpu_submit_time_box_seconds must be None or > 0")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def capacity_plan(profile: FullTrialProfile, *, classical_active: int, classical_pending: int,
                  pp_active: int, pp_waiting: int, quantum_unfinished: int) -> dict[str, int]:
    """Per-dispatcher ``max_concurrent`` ceilings under ONE shared Serverless budget.

    ``pp_waiting`` is the number of authenticated measurements awaiting a 3Q-C worker and
    ``quantum_unfinished`` the number of quantum fragments not yet COMPLETE.  While any quantum fragment is
    unfinished, ``postprocess_reserved_slots`` stay reserved, so the classical branch can never occupy the
    last slots and starve post-processing.  Active jobs are never cancelled to make room.
    """
    c_ceiling = profile.classical_max_concurrent if profile.classical_max_concurrent is not None \
        else classical_active + classical_pending
    p_ceiling = profile.postprocess_max_concurrent if profile.postprocess_max_concurrent is not None \
        else pp_active + pp_waiting + max(0, quantum_unfinished)
    total = profile.serverless_max_total_jobs
    if total is None:
        return {"classical_cap": max(0, c_ceiling), "postprocess_cap": max(0, p_ceiling),
                "reserved_for_postprocess": 0, "total": 0}
    floor = min(p_ceiling, profile.postprocess_reserved_slots) if quantum_unfinished > 0 else 0
    reserved = min(p_ceiling, max(pp_active + pp_waiting, floor))
    return {
        "classical_cap": max(0, min(c_ceiling, total - reserved)),
        "postprocess_cap": max(0, min(p_ceiling, total - classical_active)),
        "reserved_for_postprocess": reserved,
        "total": total,
    }


def build_live_interfaces(client: Any) -> dict[str, Any]:
    """``run``/``job`` facade plus one File-API adapter per function for an authenticated Serverless client."""
    return {
        "service": CatalogServerlessService(client),
        "classical_adapter": make_live_transport(client, CLASSICAL_FUNCTION_TITLE),
        "postprocess_adapter": make_live_transport(client, QUANTUM_POSTPROCESS_FUNCTION_TITLE),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Registry views (all derived from the frozen manifest and the registry)
# ─────────────────────────────────────────────────────────────────────────────

def _manifest_ids(ctx: RunContext) -> tuple[list[int], list[int]]:
    quantum, classical = [], []
    for fid, solver in ctx.manifest.solver_assignments.items():
        (quantum if solver in QUANTUM_SOLVERS else classical).append(int(fid))
    return sorted(quantum), sorted(classical)


def _quantum_tasks(ctx: RunContext) -> list[TaskRecord]:
    return ctx.registry.list_tasks(ctx.run_id, task_type=TaskType.QUANTUM)


def _counts(ctx: RunContext) -> dict[str, int]:
    classical = ctx.registry.list_tasks(ctx.run_id, task_type=TaskType.CLASSICAL)
    quantum = _quantum_tasks(ctx)
    return {
        "classical_active": sum(1 for t in classical if t.state in _CLASSICAL_ACTIVE),
        "classical_pending": sum(1 for t in classical if t.state in (TaskState.READY, TaskState.RETRY_PENDING)),
        "pp_active": sum(1 for t in quantum if t.state in _PP_ACTIVE),
        "pp_waiting": sum(1 for t in quantum if t.state == TaskState.SAMPLES_READY
                          and t.extra.get(PP_COUNTS_SHA_KEY) and not is_replay_task(t)),
        "quantum_unfinished": sum(1 for t in quantum if t.state != TaskState.COMPLETE),
    }


def _plan(ctx: RunContext, profile: FullTrialProfile) -> dict[str, int]:
    return capacity_plan(profile, **_counts(ctx))


# ─────────────────────────────────────────────────────────────────────────────
#  Run-level ledger (small, append-only audit; not a scheduler and not a registry)
# ─────────────────────────────────────────────────────────────────────────────

def _ledger_path(ctx: RunContext) -> Path:
    return ctx.paths.run_dir / LEDGER_FILENAME


def _load_ledger(ctx: RunContext) -> dict[str, Any]:
    p = _ledger_path(ctx)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return {"schema": "full_trial_ledger/1", "run_id": ctx.run_id, "events": [], "corrupt": True}
    return {"schema": "full_trial_ledger/1", "run_id": ctx.run_id, "events": []}


def _append_ledger(ctx: RunContext, campaign_auth: Any, op: str, summary: Mapping[str, Any]) -> None:
    led = _load_ledger(ctx)
    led["profile"] = PROFILE_NAME
    led["campaign_id"] = getattr(campaign_auth, "campaign_id", "")
    led["manifest_digest"] = ctx.manifest.manifest_digest
    led["events"].append({"at": _now(), "op": op, **dict(summary)})
    fd, tmp = tempfile.mkstemp(dir=str(ctx.paths.run_dir), prefix=".ftl_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(led, fh, indent=1, sort_keys=True, default=str)
        os.replace(tmp, _ledger_path(ctx))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ─────────────────────────────────────────────────────────────────────────────
#  Preconditions
# ─────────────────────────────────────────────────────────────────────────────

def verify_launch_conditions(
    ctx: RunContext, campaign_auth: Any, profile: FullTrialProfile, *, armed: bool,
    qpu_service: Any = None, serverless_service: Any = None,
    deny: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Read-only, fail-closed verification.  ``ok`` is False when any blocker exists."""
    blockers: list[str] = []
    warnings: list[str] = []
    if not armed:
        blockers.append("ARM_FULL_LIVE_TRIAL is False")
    if ctx.manifest is None:
        return {"ok": False, "blockers": blockers + ["no frozen RunManifest"], "warnings": warnings}
    q_ids, c_ids = _manifest_ids(ctx)
    tasks = ctx.registry.list_tasks(ctx.run_id)
    if not tasks:
        blockers.append("no fragment tasks exist for this run (create them from the frozen manifest)")
    missing = sorted(set(q_ids + c_ids) - {t.fragment_id for t in tasks})
    if tasks and missing:
        blockers.append(f"manifest fragments without a task: {missing}")

    qtasks = _quantum_tasks(ctx)
    unsubmitted = [t for t in qtasks if t.state == TaskState.READY and not t.qpu_job_id]
    if q_ids:
        if campaign_auth is None:
            blockers.append("no CampaignAuthorization")
        else:
            if campaign_auth.run_id != ctx.run_id:
                blockers.append("CampaignAuthorization belongs to another run")
            if campaign_auth.trial_profile != PROFILE_NAME:
                blockers.append(f"CampaignAuthorization is not a {PROFILE_NAME} campaign")
            if campaign_auth.manifest_digest and campaign_auth.manifest_digest != ctx.manifest.manifest_digest:
                blockers.append("CampaignAuthorization is bound to a different manifest")
            absent = [f for f in q_ids if not campaign_auth.is_authorized_fragment(f)]
            if absent:
                blockers.append(f"quantum fragments not authorized: {absent}")
            if campaign_auth.remaining_allocation() < len(unsubmitted):
                blockers.append(f"campaign allocation {campaign_auth.remaining_allocation()} < "
                                f"{len(unsubmitted)} unsubmitted quantum fragments")
        try:
            _protocol_configs(ctx)
        except (QuantumCoordinatorError, OSError, ValueError) as exc:
            blockers.append(f"frozen protocol configuration unavailable: {_short_err(exc)}")
        if unsubmitted and qpu_service is None:
            blockers.append("no QPU service supplied for the unsubmitted quantum fragments")
        if any(is_replay_task(t) for t in qtasks):
            blockers.append("replayed measurements are not accepted in a live trial")
    if serverless_service is None:
        blockers.append("no Serverless service supplied")
    try:
        if not _resolve_stage2_path(ctx).is_file():
            blockers.append("Stage 2 artifact is not available locally")
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"Stage 2 artifact unresolved: {_short_err(exc)}")

    finds = clean_slate.guard_task_registry(ctx, deny if deny is not None else {"job_ids": [], "run_ids": []})
    if finds["historical_job_ids"]:
        blockers.append(f"historical Job IDs attached to this run: {finds['historical_job_ids'][:5]}")
    if finds["foreign_run_tasks"]:
        blockers.append("tasks from another run are present")
    if finds["historical_run_id"]:
        blockers.append("this RUN_ID was used by a previous run")
    cached = [t.fragment_id for t in tasks if t.state == TaskState.COMPLETE
              and not t.serverless_job_id and not t.qpu_job_id]
    if cached:
        blockers.append(f"tasks COMPLETE without any remote job (cached or imported results): {cached}")
    ambiguous = [t.fragment_id for t in tasks if t.state == TaskState.AMBIGUOUS
                 or (t.state == TaskState.SUBMITTING and not t.serverless_job_id and not t.qpu_job_id)]
    if ambiguous:
        warnings.append(f"ambiguous submissions need explicit resolution: {ambiguous}")
    return {"ok": not blockers, "blockers": blockers, "warnings": warnings,
            "quantum_fragment_ids": q_ids, "classical_fragment_ids": c_ids,
            "unsubmitted_quantum": [t.fragment_id for t in unsubmitted]}


# ─────────────────────────────────────────────────────────────────────────────
#  Launch
# ─────────────────────────────────────────────────────────────────────────────

def _run_step(name: str, fn: Callable[[], Any], errors: list[dict[str, str]]) -> Any:
    """Isolation: a failing step is recorded and returns ``None``; it never prevents the others."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — recorded in the report and the ledger
        errors.append({"step": name, "error": _short_err(exc)})
        return None


def _compact(report: Any) -> dict[str, Any]:
    d = report.as_dict() if hasattr(report, "as_dict") else dict(report or {})
    return {k: (len(v) if isinstance(v, list) else v) for k, v in d.items() if not isinstance(v, dict)}


def launch_full_trial(
    ctx: RunContext,
    campaign_auth: Any,
    *,
    profile: FullTrialProfile,
    armed: bool,
    service: Any,
    classical_adapter: Any,
    postprocess_adapter: Any = None,
    qpu_service: Any = None,
    deny: Optional[Mapping[str, Any]] = None,
    solver_configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    quantum_options: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Dispatch classical fragments and submit authorized quantum fragments.

    Classical dispatch returns after Serverless jobs are accepted,
    allowing QPU submissions to overlap ongoing classical execution.

    QPU submissions remain sequential within the campaign, while
    accepted provider jobs execute asynchronously.

    Repeated launches are restart-safe and idempotent through the
    existing task registry and campaign authorization machinery.

    Explicit quantum_options are rejected until the coordinator
    implements and validates their full acquisition contract.
    """

    # TODO(QPU acquisition protocol):
    # Extend quantum_coordinator.submit_quantum_campaign() and the
    # underlying quantum submission path to accept, validate, and apply
    # explicit QPU execution settings.
    #
    # Required controls include:
    #   - LUCJ circuit construction and transpilation
    #   - Initial layout, routing, and optimization settings
    #   - Dynamical decoupling and pulse sequence
    #   - Gate and measurement twirling
    #   - Sampler options, backend, shots, and repetitions
    #
    # Promote these settings into the frozen experiment protocol BEFORE
    # launch, with digest-bound provenance shared by all quantum fragments.
    # Record the effective execution settings and circuit provenance
    # for each submitted QPU job.
    #
    # Implement end-to-end propagation:
    #   protocol -> full_trial -> quantum_coordinator
    #            -> quantum_submission -> Qiskit Runtime
    #
    # Add regression tests verifying that requested settings are actually
    # applied, mismatches are rejected, and no defaults silently override
    # the frozen protocol.
    #
    # Until implemented, reject explicit overrides rather than silently
    # ignoring them. Preserve the existing validated submission path.


    # Fail closed on unsupported QPU acquisition overrides.
    # Never silently discard transpilation, DD, twirling, or other
    # execution controls supplied by the caller.
    if quantum_options is not None:
        raise FullTrialError(
            "Explicit quantum_options are not supported by "
            "submit_quantum_campaign. Refusing to silently discard "
            "QPU acquisition settings."
        )

    # Validate authorization, manifest identity, protocol, and services.
    check = verify_launch_conditions(
        ctx,
        campaign_auth,
        profile,
        armed=armed,
        qpu_service=qpu_service,
        serverless_service=service,
        deny=deny,
    )

    if not check["ok"]:
        return {
            "status": "BLOCKED",
            "blockers": check["blockers"],
            "warnings": check["warnings"],
        }

    started = _now()
    errors: list[dict[str, str]] = []

    out: dict[str, Any] = {
        "status": "LAUNCHED",
        "started": started,
        "warnings": check["warnings"],
    }

    # Derive concurrency from the frozen manifest and active profile.
    plan = _plan(ctx, profile)
    out["capacity"] = plan

    # ── Classical FCI / CCSD dispatch ────────────────────────────────────
    # Existing active jobs are handled by the registry-backed dispatcher.
    if plan["classical_cap"] >= 1 and check["classical_fragment_ids"]:
        rep = _run_step(
            "classical_dispatch",
            lambda: dispatch_classical_tasks(
                ctx,
                service,
                classical_adapter,
                max_concurrent=plan["classical_cap"],
                resource_family=profile.classical_resource_family,
                solver_configs=solver_configs,
            ),
            errors,
        )

        out["classical"] = (
            _compact(rep)
            if rep is not None
            else {"error": "see errors"}
        )

    # ── Fresh QPU acquisition ────────────────────────────────────────────
    # Submit only authorized, eligible quantum fragments.
    # Preparation/submission is bounded by the configured time box.
    # Accepted QPU jobs continue asynchronously after this call returns.
    if check["quantum_fragment_ids"]:
        rep = _run_step(
            "quantum_submission",
            lambda: submit_quantum_campaign(
                ctx,
                campaign_auth,
                qpu_service=qpu_service,
                allow_sampling=True,
                max_submissions=profile.qpu_submit_max_per_call,
                time_box_seconds=profile.qpu_submit_time_box_seconds,
            ),
            errors,
        )

        if rep is not None:
            out["quantum"] = {
                "submitted": [
                    {
                        "fragment_id": s["fragment_id"],
                        "qpu_job_id": s["qpu_job_id"],
                    }
                    for s in rep.submitted_tasks
                ],
                "deferred": sum(
                    1
                    for s in rep.skipped_tasks
                    if str(s.get("reason", "")).startswith("deferred")
                ),
                "errors": rep.errors,
                "remaining_allocation": rep.remaining_allocation,
            }

    # ── Record operation ────────────────────────────────────────────────
    out["errors"] = errors
    out["finished"] = _now()
    out["summary"] = _summary(ctx)

    _append_ledger(
        ctx,
        campaign_auth,
        "launch",
        {
            "started": started,
            "capacity": plan,
            "errors": errors,
            "summary": out["summary"],
            "serverless_max_total_jobs": (
                profile.serverless_max_total_jobs
            ),
        },
    )

    return out

# ─────────────────────────────────────────────────────────────────────────────
#  Advance
# ─────────────────────────────────────────────────────────────────────────────

def advance_full_trial(
    ctx: RunContext, campaign_auth: Any, *, profile: FullTrialProfile, armed: bool,
    service: Any, classical_adapter: Any, postprocess_adapter: Any, qpu_service: Any = None,
    solver_configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """One bounded, non-blocking pass.  Never submits to the QPU, never starts Stage 4.

    reconcile classical -> refresh QPU status -> collect + authenticate measurements -> reconcile 3Q-C ->
    dispatch 3Q-C -> refill classical.  Every step is isolated.
    """
    if not armed:
        raise FullTrialError("ARM_FULL_LIVE_TRIAL is False; advancing dispatches paid Serverless work")
    if ctx.manifest is None:
        raise FullTrialError("no frozen RunManifest")
    started = _now()
    errors: list[dict[str, str]] = []
    out: dict[str, Any] = {"started": started}
    q_ids, c_ids = _manifest_ids(ctx)

    if c_ids:
        out["classical_reconcile"] = _compact(_run_step(
            "classical_reconcile", lambda: reconcile_classical_tasks(ctx, service, classical_adapter), errors) or {})
    if q_ids:
        snap = _run_step("qpu_refresh", lambda: refresh_quantum_status(
            ctx, qpu_service=qpu_service, max_queries=profile.max_status_queries), errors) if qpu_service is not None else None
        out["qpu_refresh"] = ({"queried": snap.queried_count, "updated": snap.updated_count,
                               "refresh_errors": len(snap.refresh_errors)} if snap is not None else {"skipped": True})
        if qpu_service is not None:
            col = _run_step("collect_measurements", lambda: collect_quantum_measurements(ctx, qpu_service), errors)
            out["collect"] = ({k: (len(v) if isinstance(v, list) else v) for k, v in col.items()} if col else {})
        out["postprocess_reconcile"] = _compact(_run_step(
            "postprocess_reconcile",
            lambda: reconcile_postprocessing_campaign(ctx, service, transport_adapter=postprocess_adapter),
            errors) or {})
        plan = _plan(ctx, profile)
        if plan["postprocess_cap"] >= 1:
            out["postprocess_dispatch"] = _compact(_run_step(
                "postprocess_dispatch",
                lambda: dispatch_postprocessing_campaign(
                    ctx, service, transport_adapter=postprocess_adapter, max_concurrent=plan["postprocess_cap"],
                    resource_family=profile.postprocess_resource_family,
                    function_title=QUANTUM_POSTPROCESS_FUNCTION_TITLE), errors) or {})
    if c_ids:
        plan = _plan(ctx, profile)
        out["capacity"] = plan
        if plan["classical_cap"] >= 1:
            out["classical_dispatch"] = _compact(_run_step(
                "classical_dispatch",
                lambda: dispatch_classical_tasks(ctx, service, classical_adapter, max_concurrent=plan["classical_cap"],
                                                 resource_family=profile.classical_resource_family,
                                                 solver_configs=solver_configs), errors) or {})
    out["errors"] = errors
    out["finished"] = _now()
    out["summary"] = _summary(ctx)
    _append_ledger(ctx, campaign_auth, "advance", {"started": started, "errors": errors, "summary": out["summary"],
                                                   "serverless_max_total_jobs": profile.serverless_max_total_jobs})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Status, timing and overlap
# ─────────────────────────────────────────────────────────────────────────────

# Heuristic classification of a submission error text.  It is evidence for the operator, never an automatic
# trigger: nothing here lowers, raises or retries anything.  Order matters (authentication is tested first).
_ERROR_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("authentication", ("401", "403", "unauthor", "forbidden", "authenticat", "credential", "invalid token", "api key")),
    ("capacity", ("429", "too many", "quota", "capacity", "concurrent", "rate limit", "resource exhausted",
                  "limit exceeded", "maximum number of", "no available worker", "queue is full")),
    ("memory", ("out of memory", "oom", "memoryerror", "killed", "insufficient memory")),
    ("transport", ("timeout", "timed out", "connection", "ssl", "reset by peer", "502", "503", "504",
                   "gateway", "unavailable", "name resolution", "file api", "upload")),
    ("scientific", ("hamiltonian", "sha256", "digest", "manifest", "protocol", "assignment", "not json serializable")),
)


def classify_submission_error(text: Any) -> str:
    """``capacity`` | ``authentication`` | ``memory`` | ``transport`` | ``scientific`` | ``unclassified``.

    Only ``capacity`` is evidence of a concurrency limit, and even then it is *suspected*: a submission that
    ended ambiguously (timeout, dropped connection) is NOT capacity evidence and is classed ``transport``.
    """
    low = str(text or "").lower()
    for name, needles in _ERROR_CATEGORIES:
        # short tokens (status codes, "oom", "ssl") must stand alone so they never match inside ids or words
        if any((re.search(rf"(?<![0-9a-z]){re.escape(n)}(?![0-9a-z])", low) if len(n) <= 4 else n in low) for n in needles):
            return name
    return "unclassified"


_UNCERTAIN_EVENTS = ("SUBMIT_UNCERTAIN", "PP_SUBMIT_UNCERTAIN")


def uncertain_submissions(ctx: RunContext) -> list[dict[str, Any]]:
    """Serverless submissions whose outcome is unknown (no provider Job ID recorded).  Read-only; never retried here."""
    rows: list[dict[str, Any]] = []
    for t in ctx.registry.list_tasks(ctx.run_id):
        if t.serverless_job_id:
            continue
        quantum = t.task_type == TaskType.QUANTUM
        if quantum:
            if not (t.state == TaskState.POSTPROCESSING and t.extra.get("pp_attempt")):
                continue
        elif t.state != TaskState.SUBMITTING:
            continue
        events = [e for e in (t.extra.get("dispatch_events") or []) if isinstance(e, dict)
                  and e.get("event") in _UNCERTAIN_EVENTS]
        err = str(events[-1].get("error", "")) if events else ""
        rows.append({
            "fragment_id": t.fragment_id, "task_id": t.task_id, "branch": "3Q-C" if quantum else "classical",
            "state": t.state, "attempt": (t.extra.get("pp_attempt") if quantum else t.attempt),
            "category": classify_submission_error(err) if err else "unknown (no error recorded)",
            "error": err[:300],
            "reconciliation": ("OPERATOR ACTION REQUIRED: confirm with the provider whether the job exists; attach its Job ID "
                               "or abandon the attempt explicitly. It is never retried automatically."),
        })
    return sorted(rows, key=lambda r: (r["branch"], r["fragment_id"]))


def ceiling_history(ctx: RunContext, *, limit: int = 8) -> dict[str, Any]:
    """Shared-pool ceilings used by recent passes, from the run ledger (read-only).

    ``last_error_free`` is the most recent ceiling whose pass reported no step errors and no ambiguous task: a
    pointer for returning to a previously tolerated value, not proof of provider capacity.
    """
    passes = [{"at": e.get("at"), "op": e.get("op"), "ceiling": e.get("serverless_max_total_jobs"),
               "errors": len(e.get("errors") or []), "ambiguous": len((e.get("summary") or {}).get("ambiguous") or [])}
              for e in _load_ledger(ctx).get("events", []) if e.get("op") in ("launch", "advance")]
    clean = [p["ceiling"] for p in passes if p["ceiling"] is not None and not p["errors"] and not p["ambiguous"]]
    return {"recent": passes[-limit:], "last_error_free": clean[-1] if clean else None}


def ceiling_note(profile: Any, ramp: tuple[int, ...] = ()) -> str:
    """One line for the control panel: the active shared ceiling and its position in the operator's ramp.

    The ramp is guidance only.  Any positive integer is accepted: the ceiling is a requested value, not a
    verified provider limit, and the engine never advances it.
    """
    n = _as_profile(profile).serverless_max_total_jobs
    if n is None:
        return "Shared Serverless ceiling: NOT SET (launch/advance refuse until it is set)."
    where = f"ramp step {ramp.index(n) + 1} of {len(ramp)}" if n in ramp else "NOT one of the planned ramp values"
    return (f"Shared Serverless ceiling (classical + 3Q-C active jobs): {n} - {where}. Requested, not a verified provider "
            "limit; QPU submissions do not use these slots.")


def _summary(ctx: RunContext) -> dict[str, Any]:
    tasks = ctx.registry.list_tasks(ctx.run_id)
    by_state: dict[str, dict[str, int]] = {"classical": {}, "quantum": {}}
    for t in tasks:
        branch = "quantum" if t.task_type == TaskType.QUANTUM else "classical"
        by_state[branch][t.state] = by_state[branch].get(t.state, 0) + 1
    ambiguous = sorted(t.fragment_id for t in tasks if t.state == TaskState.AMBIGUOUS
                       or (t.state == TaskState.SUBMITTING and not t.serverless_job_id and not t.qpu_job_id)
                       or (t.state == TaskState.POSTPROCESSING and t.extra.get("pp_attempt") and not t.serverless_job_id))
    failed = sorted(t.fragment_id for t in tasks if t.state in (TaskState.FAILED, TaskState.RETRY_PENDING))
    done = sum(1 for t in tasks if t.state == TaskState.COMPLETE)
    return {"tasks": len(tasks), "complete": done, "failed_or_retry": failed, "ambiguous": ambiguous,
            "by_state": by_state, "all_complete": bool(tasks) and done == len(tasks)}


def _parse(ts: str) -> Optional[float]:
    try:
        return datetime.datetime.fromisoformat(ts).timestamp()
    except (TypeError, ValueError):
        return None


def _first(history: list[dict], states: tuple[str, ...]) -> Optional[float]:
    for row in history:
        if row["to_state"] in states:
            return _parse(row["occurred_iso8601"])
    return None


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for a, b in sorted(i for i in intervals if i[1] >= i[0]):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    total = 0.0
    for a0, a1 in a:
        for b0, b1 in b:
            total += max(0.0, min(a1, b1) - max(a0, b0))
    return total


def full_trial_status(ctx: RunContext, campaign_auth: Any = None, *, now: Optional[float] = None) -> dict[str, Any]:
    """Read-only progress and measured overlap.  Waiting time is reported apart from worker compute time.

    Branch intervals come from the registry's own transition log: classical = SUBMITTING..COMPLETE,
    QPU = SUBMITTED..SAMPLES_READY (queue + execution, not separable from here), 3Q-C =
    POSTPROCESS_QUEUED..COMPLETE.  ``compute_seconds`` is only what workers reported.
    """
    now = time.time() if now is None else now
    branches: dict[str, list[tuple[float, float]]] = {"classical": [], "qpu": [], "postprocess": []}
    wall: dict[str, float] = {k: 0.0 for k in branches}
    compute = {"classical": 0.0, "postprocess": 0.0}
    for t in ctx.registry.list_tasks(ctx.run_id):
        hist = ctx.registry.get_transition_history(t.task_id)
        if t.task_type == TaskType.QUANTUM:
            for name, start_states, end_states in (("qpu", (TaskState.SUBMITTED,), (TaskState.SAMPLES_READY,)),
                                                   ("postprocess", (TaskState.POSTPROCESS_QUEUED,),
                                                    (TaskState.COMPLETE, TaskState.FAILED))):
                s = _first(hist, start_states)
                if s is None:
                    continue
                e = _first(hist, end_states)
                branches[name].append((s, e if e is not None else now))
            if isinstance(t.extra.get("worker_elapsed_seconds"), (int, float)):
                compute["postprocess"] += float(t.extra["worker_elapsed_seconds"])
        else:
            s = _first(hist, (TaskState.SUBMITTING,))
            if s is not None:
                e = _first(hist, (TaskState.COMPLETE, TaskState.FAILED))
                branches["classical"].append((s, e if e is not None else now))
            if isinstance(t.extra.get("worker_elapsed_seconds"), (int, float)):
                compute["classical"] += float(t.extra["worker_elapsed_seconds"])
    merged = {k: _union(v) for k, v in branches.items()}
    for k, v in branches.items():
        wall[k] = sum(b - a for a, b in v)            # summed per-task time (task-seconds)
    span = {k: sum(b - a for a, b in v) for k, v in merged.items()}   # wall-clock covered by the branch
    allv = [i for v in merged.values() for i in v]
    overall = (max(b for _, b in allv) - min(a for a, _ in allv)) if allv else 0.0
    ov = {"classical_x_qpu": _overlap(merged["classical"], merged["qpu"]),
          "classical_x_postprocess": _overlap(merged["classical"], merged["postprocess"]),
          "qpu_x_postprocess": _overlap(merged["qpu"], merged["postprocess"])}
    tasks = ctx.registry.list_tasks(ctx.run_id)
    return {
        "run_id": ctx.run_id,
        "campaign_id": getattr(campaign_auth, "campaign_id", ""),
        "campaign_submissions": len(campaign_auth.submissions) if campaign_auth is not None else None,
        "remaining_allocation": campaign_auth.remaining_allocation() if campaign_auth is not None else None,
        "qpu_job_ids": {t.fragment_id: t.qpu_job_id for t in tasks if t.qpu_job_id},
        "serverless_job_count": sum(1 for t in tasks if t.serverless_job_id),
        "failures": {t.fragment_id: t.failure_reason for t in tasks
                     if t.state in (TaskState.FAILED, TaskState.RETRY_PENDING) and t.failure_reason},
        "summary": _summary(ctx),
        "capacity_counts": _counts(ctx),
        "timing": {
            "wall_clock_seconds": overall,
            "branch_wall_seconds": span,
            "branch_task_seconds": wall,
            "reported_compute_seconds": compute,
            "waiting_seconds_estimate": {
                "classical": max(0.0, wall["classical"] - compute["classical"]),
                "postprocess": max(0.0, wall["postprocess"] - compute["postprocess"]),
                "qpu": wall["qpu"]},
            "overlap_seconds": ov,
            "branches_overlapped": sum(1 for v in ov.values() if v > 0) > 0,
            "note": "QPU time is queue + execution as observed by the organizer; compute is worker-reported only.",
        },
        "classical": classical_status(ctx)["by_state"],
        "uncertain_submissions": uncertain_submissions(ctx),
        "ceiling_history": ceiling_history(ctx),
    }


def format_status(status: Mapping[str, Any]) -> str:
    s, tm = status["summary"], status["timing"]
    lines = [f"Full trial {status['run_id']}  campaign={status['campaign_id'] or '-'}  "
             f"QPU submissions={status['campaign_submissions']}",
             f"  tasks {s['complete']}/{s['tasks']} complete  classical={s['by_state']['classical']}  "
             f"quantum={s['by_state']['quantum']}"]
    lines.append(f"  remaining authorized QPU allocation: {status.get('remaining_allocation')}  "
                 f"Serverless jobs recorded: {status.get('serverless_job_count')}")
    lines.append(f"  concurrency now: {status.get('capacity_counts')}")
    if status.get("qpu_job_ids"):
        lines.append(f"  QPU Job IDs: {status['qpu_job_ids']}")
    if s["failed_or_retry"]:
        lines.append(f"  FAILED/RETRY fragments: {s['failed_or_retry']}  reasons: {status.get('failures')}")
    if s["ambiguous"]:
        lines.append(f"  AMBIGUOUS (explicit resolution required): {s['ambiguous']}")
    for u in status.get("uncertain_submissions", []):
        lines.append(f"  UNCERTAIN {u['branch']} submission: fragment {u['fragment_id']} ({u['state']}, attempt {u['attempt']}) "
                     f"- suspected cause: {u['category']}. {u['reconciliation']}")
    if any(u["category"] == "capacity" for u in status.get("uncertain_submissions", [])):
        lines.append("  Capacity rejection suspected: do NOT raise the ceiling. Lower serverless_max_total_jobs to the "
                     "last tolerated value, rerun the control panel, resolve the task, then ADVANCE.")
    hist = status.get("ceiling_history") or {}
    if hist.get("recent"):
        lines.append(f"  shared ceiling per recent pass: {[p['ceiling'] for p in hist['recent']]}  "
                     f"last pass without errors: {hist.get('last_error_free')}")
    lines.append(f"  wall-clock {tm['wall_clock_seconds']:.0f}s  overlap(s): {tm['overlap_seconds']}")
    lines.append(f"  Stage 4: {'ready to be authorized explicitly' if s['all_complete'] else 'not ready (barrier not satisfied)'}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Declared-reference protocol (fresh run, no QPU counts yet)
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_SQD_CONFIG: dict[str, Any] = {
    "classical_backend": "python", "symmetrize_spin": True, "n_batches": 5, "iterations": 6,
    "samples_per_batch": 3000, "energy_tol": 1e-8, "occupancies_tol": 1e-5,
    "carryover_threshold": 1e-4, "max_dim": 800, "seed": 20260626}
REFERENCE_TRIM_CONFIG: dict[str, Any] = {
    "trim_ratio": 0.3, "max_strings_per_trim": 300, "max_carryover": 400, "sector_trim": False}
DECLARED_RATIONALE = ("DECLARED reference configuration (the published trim_800_r30 settings). It was NOT "
                      "tuned or validated on this run's Hamiltonians; no tuning grid was executed and no "
                      "grid energy exists (grid_energy_ha is null by design).")


def declare_reference_protocol(
    lab: Any, *, source_fragment_id: Optional[int] = None, ligand: Optional[str] = None,
    sqd_config: Optional[Mapping[str, Any]] = None, trim_config: Optional[Mapping[str, Any]] = None,
    qpu_backend: Optional[str] = None, n_shots: int = 100_000, n_reps: int = 1, overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze a DECLARED protocol for a fresh run through the existing ``transfer_promoted_protocol``.

    Used when no Stage E tuning exists for the fresh Hamiltonians.  The record is bound to the FRESH
    baseline Hamiltonian hashes of the source fragment (never an archived hash), carries no counts, and is
    labelled as declared, not validated.  ``source_fragment_id=None`` picks the first selected fragment
    for which the protocol transfer validates.  Archived promotions are never read or copied.
    """
    from proteasome_challenge import SELECTED_LIGAND, ChallengeError, resolve_qpu_backend
    ligand = ligand or SELECTED_LIGAND
    sel = lab.load_final_selection()
    base = lab.reference_baseline(ligand)       # Stage 2 structural view in a full trial: no baseline job
    ham = {int(f["fragment_id"]): f["hamiltonian_hash"] for f in base["fragments"]}
    sqd = dict(sqd_config or REFERENCE_SQD_CONFIG)
    trim = dict(trim_config or REFERENCE_TRIM_CONFIG)
    selected = [int(i) for i in sel["selected_fragment_ids"]]
    candidates = [int(source_fragment_id)] if source_fragment_id is not None else selected
    last: Optional[Exception] = None
    for fid in candidates:
        if fid not in selected or fid not in ham:
            raise FullTrialError(f"source fragment {fid} is not in the frozen selection / baseline")
        record = {
            "selected_grid_label": DECLARED_GRID_LABEL, "grid_label": DECLARED_GRID_LABEL, "ligand": ligand,
            "fragment_id": fid, "solver_method": "trimsqd", "solver_mode": "trimsqd",
            "sqd_config": dict(sqd), "trim_config": dict(trim), "n_shots": int(n_shots), "n_reps": int(n_reps),
            "qpu_backend": resolve_qpu_backend(qpu_backend), "grid_energy_ha": None,
            "measurement_artifact": None, "hamiltonian_sha256": ham[fid], "counts_source": None,
            "counts_hash": None, "rationale": DECLARED_RATIONALE,
            "final_selection_digest": sel["manifest_digest"], "baseline_hamiltonian_hash": ham[fid],
        }
        try:
            return lab.transfer_promoted_protocol(record, ligand=ligand, overwrite=overwrite)
        except ChallengeError as exc:
            last = exc
            if source_fragment_id is not None:
                raise
    raise FullTrialError(f"no selected fragment accepts the declared protocol: {last}")


# ─────────────────────────────────────────────────────────────────────────────
#  Notebook glue (Task 5.9E): thin wrappers so the organizer cells stay short.  No new scheduler or store;
#  every operation delegates to the functions above, ``clean_slate`` and the existing dispatchers.
# ─────────────────────────────────────────────────────────────────────────────

_RUN_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{0,63}")
# Written by a deliberate bundle deploy BEFORE the run exists; its validity is enforced separately by the
# bundle-fingerprint check, so it does not make the checkpoint namespace "dirty".
_PRE_RUN_ALLOWED = frozenset({".deployed_bundle_sha256"})


def stable_run_id(path: str | Path, *, prefix: str = "ixazomib_live") -> str:
    """The trial RUN_ID, generated once and then read back from ``path`` on every later call.

    Surviving a kernel restart means re-running the control-panel cell never changes the identity.  Creation is
    atomic and exclusive; an unreadable or malformed file is an error (it is never silently regenerated).
    """
    p = Path(path)

    def _read() -> str:
        try:
            rid = str(json.loads(p.read_text(encoding="utf-8")).get("run_id", ""))
        except (OSError, ValueError, AttributeError) as exc:
            raise FullTrialError(f"{p.name} is unreadable ({_short_err(exc)}); fix or remove it deliberately") from exc
        if not _RUN_ID_RE.fullmatch(rid):
            raise FullTrialError(f"{p.name} holds an invalid run id {rid!r}")
        return rid

    if p.is_file():
        return _read()
    rid = clean_slate.new_trial_run_id(prefix)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".rid_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"schema": "full_live_run_id/1", "run_id": rid, "created_utc": _now()}, fh)
        try:
            os.link(tmp, p)                      # atomic; fails if another kernel created it first
        except FileExistsError:
            return _read()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return rid


def ensure_clean_start(lab: Any, run_id: str, *, notebook_path: Optional[str | Path] = None,
                       enforce: bool = False) -> dict[str, Any]:
    """Read-only clean-slate inventory; with ``enforce`` it refuses to create a NEW trial run on top of history.

    A run that already exists (restart) is resumed.  For a new run, any run-specific entry in ``checkpoints/``
    (prior Stage 1/2/3, phase5 runs, final selection, ...) or a previously used run id blocks creation:
    retiring that state is a separately approved action (``clean_slate.retire_active_state``) that this
    function never performs.
    """
    ckpt = Path(lab.ckpt_dir)
    root = ckpt.parent
    resumed = (ckpt / "phase5" / "runs" / run_id).is_dir()
    inv = clean_slate.inventory_active_state(root, notebook_path=notebook_path)
    deny = clean_slate.historical_deny_list(root, include_active=True)
    blocking = [] if resumed else [n for n in inv["to_retire"] if n not in _PRE_RUN_ALLOWED]
    reused = (not resumed) and run_id in deny["run_ids"]
    plan = clean_slate.retire_active_state(root, apply=False, known_remote_job_ids=())
    report = {"run_id": run_id, "resumed": resumed, "blocking_entries": blocking, "run_id_reused": reused,
              "proposed_retirement": plan["to_retire"], "kept": plan["to_keep"],
              "possibly_running_remote_jobs": plan["unacknowledged_remote_jobs"],
              "ledger": inv["historical_ledger"]}
    if enforce and (blocking or reused):
        raise FullTrialError(
            ("RUN_ID was used by a previous run: " + run_id + ". " if reused else "")
            + (f"prior execution state is present in checkpoints/: {blocking}. " if blocking else "")
            + "A clean-slate trial may not start on top of it. Retire it with the separately approved "
              "clean_slate.retire_active_state(..., apply=True) (acknowledging any possibly-running remote jobs "
              f"{plan['unacknowledged_remote_jobs'][:3]}...), then restart.")
    return report


def format_clean_start(report: Mapping[str, Any]) -> str:
    """Compact text for ``ensure_clean_start``: what blocks a new trial and what retirement would move."""
    lines = [f"Clean start for {report['run_id']}: "
             + ("RESUMING an existing trial run" if report["resumed"] else
                ("BLOCKED" if report["blocking_entries"] or report["run_id_reused"] else "clean (no prior execution state)"))]
    if report["run_id_reused"]:
        lines.append("  RUN_ID was used by a previous run")
    if report["blocking_entries"]:
        lines.append(f"  prior state that must be retired first (separately approved): {report['blocking_entries']}")
        jobs = report["possibly_running_remote_jobs"]
        lines.append(f"  possibly-running remote jobs to acknowledge before retirement: {len(jobs)}")
    lg = report["ledger"]
    lines.append(f"  legacy participant ledger (never modified): count={lg['lifetime_count']} sha256={str(lg['ledger_sha256'])[:12]}")
    return "\n".join(lines)


_REFRESHED_DATA = {"stage2": "fragment_catalog.csv"}   # Stage 1 (ensure_hf) writes only digest-keyed checkpoints
STAGE_OUTPUT_DIR = "stage_outputs"
STRUCTURAL_NOTE = ("No classical-baseline job is needed or run in the FULL_LIVE_TRIAL: the protocol, manifest and plan "
                   "bind to the Stage 2 structural reference (identity, Hamiltonian hashes, n_orb/n_occ; no energies). "
                   "The classical fragment energies come from the dispatched FCI/CCSD workers.")


def scope_trial_data(lab: Any, artifact_root: str | Path, run_id: str) -> dict[str, Any]:
    """Keep Stage 1/2 outputs of this trial under its own run identity; select the Stage 2 structural reference.

    Redirects ``lab.hf_path`` / ``lab.cat_path`` to ``<artifact_root>/runs/<run_id>/stage_outputs/`` so a fresh
    ``run_stage2`` can never rewrite the tracked ``challenge_data/hf_results.json`` / ``fragment_catalog.csv``
    (the artifact contracts are unchanged; only the location differs).  Nothing is created or copied here: the
    scoped files appear only when the fresh stages produce them, so the trial cannot silently read the tracked
    reference data as if it were its own.  Idempotent (safe on every cell re-run and after a kernel restart).
    """
    scope = Path(artifact_root) / "runs" / str(run_id) / STAGE_OUTPUT_DIR
    lab.hf_path = scope / "hf_results.json"
    lab.cat_path = scope / "fragment_catalog.csv"
    lab.trial_data_scope = str(scope)
    lab.use_structural_reference = True
    return {"scope": str(scope), "hf_results": str(lab.hf_path), "fragment_catalog": str(lab.cat_path)}


def authorize_stage_execution(stage: str, run_ctx: Any, *, allow_data_refresh: bool, lab: Any = None) -> None:
    """Fail-closed gate before a paid Stage 1/2/baseline/protocol step inside the full trial.

    The trial run must already exist (so the clean-start check has passed).  ``run_stage2`` rewrites its data
    files: when ``lab`` is scoped to this run (:func:`scope_trial_data`) those are the run's own files and nothing
    tracked can change; otherwise the tracked ``challenge_data`` files would be rewritten and the operator's
    explicit data-refresh authorization is required.  The paid classical baseline is never run in the trial.
    """
    if run_ctx is None:
        raise FullTrialError(f"{stage}: create the trial run first (RUN_INIT_RUN = True) so the clean-start check runs")
    if stage == "baseline":
        raise FullTrialError("RUN_CLASSICAL_BASELINE is not used by the FULL_LIVE_TRIAL (it would duplicate the "
                             "dispatched classical fragment computation). " + STRUCTURAL_NOTE)
    scoped = lab is not None and getattr(lab, "trial_data_scope", None) == str(
        Path(run_ctx.paths.run_dir) / STAGE_OUTPUT_DIR)
    if stage in _REFRESHED_DATA and not scoped and not allow_data_refresh:
        raise FullTrialError(
            f"{stage}: running it rewrites the tracked challenge_data/{_REFRESHED_DATA[stage]}. "
            "Set ALLOW_CHALLENGE_DATA_REFRESH = True only with explicit authorization.")


def freeze_declared_protocol(lab: Any, *, qpu_backend: Optional[str], n_shots: int, n_reps: int) -> dict[str, Any]:
    """Declare the reference protocol once; an existing manifest is accepted only if it is itself a declared one."""
    if not qpu_backend:
        raise FullTrialError("IBM_QUANTUM_BACKEND is not set; the protocol backend must be explicit (no silent default)")
    if Path(lab.promotion_manifest_path).is_file():
        rows = lab.shared_protocol_report()
        if not rows or not all(r.get("declared_reference") for r in rows):
            raise FullTrialError(
                "promotion_manifest.json exists but is not a declared-reference protocol for this trial "
                "(historical or tuned); it is not reused silently. Retire it through the approved clean-slate step.")
        return lab.load_promotion_manifest(require_complete=False)
    return declare_reference_protocol(lab, qpu_backend=qpu_backend, n_shots=n_shots, n_reps=n_reps)


def protocol_provenance(rows: Any) -> str:
    """One truthful sentence about where the protocol comes from, from ``shared_protocol_report()`` rows."""
    rows = list(rows or [])
    if not rows:
        return "Protocol provenance: no protocol rows."
    declared = [r for r in rows if r.get("declared_reference")]
    if len(declared) == len(rows):
        return ("Protocol provenance: DECLARED REFERENCE (published trim_800_r30 settings). NO tuning grid was "
                "executed for this run and no grid energy exists; the settings are not validated on this run's "
                "Hamiltonians.")
    if not declared:
        return ("Protocol provenance: SELECTED FROM MEASURED TUNING-GRID RESULTS (validated source "
                f"fragments: {sum(1 for r in rows if r.get('validated_result'))}).")
    return f"Protocol provenance: MIXED ({len(declared)} declared-reference, {len(rows) - len(declared)} other rows) - inspect."


def trial_campaign(run_ctx: RunContext, manifest: Any, *, backend: Optional[str], fragment_ids: list[int],
                   shots: int, project_root: str | Path) -> CampaignAuthorization:
    """Create (or reload and verify) the FULL_LIVE_TRIAL campaign.  Idempotent: repeating never adds allocation."""
    if not backend:
        raise FullTrialError("IBM_QUANTUM_BACKEND is not set; the campaign backend must be explicit")
    path = run_ctx.paths.run_dir / "campaign_authorization.json"
    ids = sorted(int(i) for i in fragment_ids)
    if path.is_file():
        auth = CampaignAuthorization.load(path)
        bad = [m for m, ok in (
            ("run_id", auth.run_id == run_ctx.run_id), ("trial_profile", auth.trial_profile == PROFILE_NAME),
            ("manifest_digest", auth.manifest_digest == manifest.manifest_digest),
            ("backend", auth.authorized_backend == backend), ("shots", auth.shots_per_fragment == int(shots)),
            ("fragments", sorted(auth.authorized_fragments) == ids)) if not ok]
        if bad:
            raise FullTrialError(f"existing campaign authorization disagrees with this trial on {bad}; not reused")
        return auth
    deny = clean_slate.historical_deny_list(project_root, include_active=True)
    return create_organizer_campaign_authorization(
        run_id=run_ctx.run_id, authorized_backend=backend, manifest_digest=manifest.manifest_digest,
        authorized_fragments=ids, shots_per_fragment=int(shots), max_submissions=len(ids), ledger_path=path,
        campaign_id=f"{run_ctx.run_id}_campaign", trial_profile=PROFILE_NAME,
        historical_accounting=clean_slate.historical_accounting_snapshot(project_root),
        forbidden_job_ids=deny["job_ids"])


def _as_profile(profile: Any) -> FullTrialProfile:
    return profile if isinstance(profile, FullTrialProfile) else FullTrialProfile(**dict(profile))


def assert_legacy_flags_off(flags: Mapping[str, Any]) -> None:
    """Consistency of the FULL_LIVE_TRIAL mode flags (fail closed).

    * ARM / LAUNCH / ADVANCE need ``USE_FULL_LIVE_TRIAL``; LAUNCH and ADVANCE are separate operations (never both),
      and either needs ARM.
    * In full-trial mode the ordinary dispatch/QPU flags must stay off: classical and 3Q-C work is dispatched only
      through the shared-pool launch/advance operations and the QPU only through the authorized campaign, so the
      ordinary paths would bypass those bounds.
    """
    use = bool(flags.get("USE_FULL_LIVE_TRIAL"))
    arm, launch, advance = (bool(flags.get(n)) for n in ("ARM_FULL_LIVE_TRIAL", "LAUNCH_FULL_TRIAL", "ADVANCE_FULL_TRIAL"))
    if (arm or launch or advance) and not use:
        raise FullTrialError("ARM/LAUNCH/ADVANCE_FULL_* need USE_FULL_LIVE_TRIAL = True")
    if launch and advance:
        raise FullTrialError("LAUNCH_FULL_TRIAL and ADVANCE_FULL_TRIAL are separate operations; set one at a time")
    if (launch or advance) and not arm:
        raise FullTrialError("LAUNCH/ADVANCE_FULL_TRIAL need ARM_FULL_LIVE_TRIAL = True")
    if use:
        names = ("RUN_CLASSICAL_RECONCILE", "RUN_CLASSICAL_DISPATCH", "LIVE_QPU_SAMPLING", "ALLOW_SECOND_QPU_SAMPLING",
                 "RUN_QUANTUM_SUBMISSION", "RUN_QUANTUM_REPLAY", "RUN_QUANTUM_STATUS_REFRESH", "RUN_QUANTUM_POSTPROCESS",
                 "RUN_QUANTUM_RECONCILE", "RUN_EWF_SQD")
        on = [n for n in names if flags.get(n)]
        if on:
            raise FullTrialError(f"USE_FULL_LIVE_TRIAL is on; these ordinary-mode flags must be False: {on}")


def describe_profile(profile: Any, ctx: Optional[RunContext] = None,
                     reference: Optional[Mapping[str, int]] = None) -> str:
    """Requested ceilings, manifest-derived populations and the effective caps; unresolved limits are named."""
    prof = _as_profile(profile)
    lines = [f"{PROFILE_NAME}: requested ceilings classical={prof.classical_max_concurrent} (family "
             f"{prof.classical_resource_family}), 3Q-C={prof.postprocess_max_concurrent} (family "
             f"{prof.postprocess_resource_family}), shared Serverless pool={prof.serverless_max_total_jobs}"]
    if ctx is not None and ctx.manifest is not None:
        q, c = _manifest_ids(ctx)
        lines.append(f"  populations from the frozen manifest: {len(c)} classical, {len(q)} quantum "
                     f"(not hardcoded)")
        if reference:
            exp = (reference.get("classical_fragments"), reference.get("quantum_fragments"))
            if exp != (len(c), len(q)):
                lines.append(f"  NOTE: the reference experiment expected {exp[0]} classical / {exp[1]} quantum fragments; "
                             "the frozen manifest governs.")
        lines.append(f"  effective caps now: {_plan(ctx, prof)}")
    if prof.serverless_max_total_jobs is None:
        lines.append("  UNRESOLVED: no shared Serverless pool limit is configured; the provider's own quota/queueing "
                     "decides, and rejected submissions surface as isolated errors.")
        lines.append("  The installed Qiskit Serverless / IBM Catalog clients expose no maximum-concurrent-jobs or "
                     "worker-pool API (ServerlessClient.usage() reports runtime-seconds quota only), so the limit "
                     "cannot be discovered programmatically: the organizer must set serverless_max_total_jobs.")
    return "\n".join(lines)


def _launched(ctx: RunContext, campaign_auth: Any = None) -> bool:
    """True once paid work has started.  A first ADVANCE (e.g. a single-fragment classical probe) counts: the
    pre-launch acceptance is evaluated *before* it, and would otherwise fail afterwards on its own probe result."""
    if any(e.get("op") in ("launch", "advance") for e in _load_ledger(ctx).get("events", [])):
        return True
    return bool(campaign_auth is not None and campaign_auth.submissions)


def _prelaunch_blockers(lab: Any, run_ctx: RunContext, campaign_auth: Any) -> list[str]:
    """Clean-slate acceptance H1-H10 before the first paid operation; empty once work has started or when it passes."""
    if _launched(run_ctx, campaign_auth):
        return []
    root = Path(lab.ckpt_dir).parent
    acc = clean_slate.assert_clean_slate(
        run_ctx, campaign_auth, project_root=root, deny=clean_slate.historical_deny_list(root, include_active=False),
        expected_ledger_sha256=((campaign_auth.historical_accounting or {}).get("ledger_sha256")
                                if campaign_auth is not None else None))
    return [] if acc["passed"] else [f"clean slate {c['code']}: {c['detail']}" for c in acc["checks"] if not c["passed"]]


def _require_pool_limit(prof: FullTrialProfile) -> None:
    """The shared Serverless pool is not documented anywhere in the repository, so it is never assumed."""
    if prof.serverless_max_total_jobs is None:
        raise FullTrialError(
            "FULL_LIVE_TRIAL['serverless_max_total_jobs'] is unresolved (None). The shared Serverless worker pool "
            "limit is an explicit organizer decision; set it (e.g. classical + 3Q-C workers) before launching.")


def clean_slate_report(lab: Any, run_ctx: RunContext, campaign_auth: Any, *,
                       notebook_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Pre-launch acceptance (H1-H10).  After the first launch the checks no longer apply and are not evaluated."""
    if _launched(run_ctx, campaign_auth):
        return {"phase": "LAUNCHED", "acceptance": None}
    root = Path(lab.ckpt_dir).parent
    snap = (campaign_auth.historical_accounting or {}) if campaign_auth is not None else {}
    return {"phase": "PRE_LAUNCH", "acceptance": clean_slate.assert_clean_slate(
        run_ctx, campaign_auth, project_root=root, deny=clean_slate.historical_deny_list(root, include_active=False),
        notebook_path=notebook_path, expected_ledger_sha256=snap.get("ledger_sha256"))}


def format_clean_slate_report(report: Mapping[str, Any]) -> str:
    if report["phase"] == "LAUNCHED":
        return "Clean-slate acceptance: already launched (the checks apply only before the first launch)."
    acc = report["acceptance"]
    lines = [f"Clean-slate acceptance for {acc['run_id']}: {'PASS' if acc['passed'] else 'FAIL'}"]
    lines += [f"  [{'ok' if c['passed'] else 'FAIL'}] {c['code']}: {c['detail']}" for c in acc["checks"]]
    return "\n".join(lines)


def notebook_launch(lab: Any, run_ctx: RunContext, campaign_auth: Any, *, profile: Any, armed: bool,
                    interfaces: Optional[Mapping[str, Any]] = None, qpu_service: Any = None,
                    solver_configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
                    quantum_options: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Launch classical dispatch and QPU submission in one bounded operation (nothing waits for results)."""
    if not armed:
        raise FullTrialError("ARM_FULL_LIVE_TRIAL is False; launching dispatches paid Serverless and QPU work")
    _require_pool_limit(_as_profile(profile))
    root = Path(lab.ckpt_dir).parent
    blockers = _prelaunch_blockers(lab, run_ctx, campaign_auth)
    if blockers:
        return {"status": "BLOCKED", "warnings": [], "blockers": blockers}
    if interfaces is None:
        lab._assert_bundle_deployed()
        interfaces = build_live_interfaces(lab._connect())
    if qpu_service is None:
        try:
            qpu_service = build_qpu_service_from_env()
        except QuantumCoordinatorError as exc:
            return {"status": "BLOCKED", "warnings": [], "blockers": [f"QPU service unavailable: {_short_err(exc)}"]}
    return launch_full_trial(
        run_ctx, campaign_auth, profile=_as_profile(profile), armed=armed, service=interfaces["service"],
        classical_adapter=interfaces["classical_adapter"], postprocess_adapter=interfaces.get("postprocess_adapter"),
        qpu_service=qpu_service, deny=clean_slate.historical_deny_list(root, include_active=False),
        solver_configs=solver_configs, quantum_options=quantum_options)


def notebook_advance(lab: Any, run_ctx: RunContext, campaign_auth: Any, *, profile: Any, armed: bool,
                     interfaces: Optional[Mapping[str, Any]] = None, qpu_service: Any = None,
                     solver_configs: Optional[Mapping[str, Mapping[str, Any]]] = None) -> dict[str, Any]:
    """One bounded reconcile/progress pass.  It cannot submit to the QPU and never starts Stage 4."""
    if not armed:
        raise FullTrialError("ARM_FULL_LIVE_TRIAL is False; advancing dispatches paid Serverless work")
    _require_pool_limit(_as_profile(profile))
    blockers = _prelaunch_blockers(lab, run_ctx, campaign_auth)       # the first ADVANCE (probe) is also a paid start
    if blockers:
        return {"status": "BLOCKED", "warnings": [], "blockers": blockers}
    warnings: list[str] = []
    if interfaces is None:
        lab._assert_bundle_deployed()
        interfaces = build_live_interfaces(lab._connect())
    if qpu_service is None:
        try:
            qpu_service = build_qpu_service_from_env()
        except QuantumCoordinatorError as exc:
            warnings.append(f"QPU service unavailable ({_short_err(exc)}); provider status/collection skipped this pass")
    out = advance_full_trial(
        run_ctx, campaign_auth, profile=_as_profile(profile), armed=armed, service=interfaces["service"],
        classical_adapter=interfaces["classical_adapter"], postprocess_adapter=interfaces["postprocess_adapter"],
        qpu_service=qpu_service, solver_configs=solver_configs)
    out["warnings"] = warnings
    return out


def notebook_poll_until_complete(
    lab: Any,
    run_ctx: RunContext,
    campaign_auth: Any,
    *,
    profile: Any,
    armed: bool,
    poll_seconds: float = 30.0,
    max_passes: int = 500,
    interfaces: Optional[Mapping[str, Any]] = None,
    qpu_service: Any = None,
) -> dict[str, Any]:
    """Repeatedly advance the full trial at a bounded polling interval until all tasks complete.

    Stops immediately if an ambiguous/uncertain submission is detected or if advancement is BLOCKED.
    """
    if not armed:
        raise FullTrialError("ARM_FULL_LIVE_TRIAL is False; advancing dispatches paid Serverless work")

    st = full_trial_status(run_ctx, campaign_auth)
    passes = 0
    while not st["summary"]["all_complete"] and passes < max_passes:
        passes += 1
        adv = notebook_advance(
            lab, run_ctx, campaign_auth, profile=profile, armed=armed,
            interfaces=interfaces, qpu_service=qpu_service,
        )
        if adv.get("status") == "BLOCKED":
            print(f"Advance BLOCKED: {adv.get('blockers')}")
            break
        st = full_trial_status(run_ctx, campaign_auth)
        s = st["summary"]
        print(f"Pass {passes}: {s['complete']}/{s['tasks']} complete "
              f"(classical: {s['by_state']['classical']}, quantum: {s['by_state']['quantum']})")
        if st["uncertain_submissions"]:
            print(f"OPERATOR ACTION REQUIRED: Uncertain submissions detected: {st['uncertain_submissions']}")
            break
        if s["failed_or_retry"]:
            print(f"Warning: Tasks in FAILED/RETRY: {s['failed_or_retry']} ({st.get('failures')})")
        if not st["summary"]["all_complete"]:
            time.sleep(poll_seconds)
    return st


def format_operation(result: Mapping[str, Any]) -> str:
    """Compact, honest text for a launch/advance result."""
    if result.get("status") == "BLOCKED":
        return "BLOCKED - nothing was submitted:\n" + "\n".join(f"  - {b}" for b in result["blockers"])
    lines = [f"{result.get('status', 'ADVANCED')}  {result.get('started', '')} -> {result.get('finished', '')}"]
    q = result.get("quantum")
    if q:
        lines.append(f"  QPU submitted: {[(s['fragment_id'], s['qpu_job_id']) for s in q['submitted']]}  "
                     f"deferred={q['deferred']}  remaining allocation={q['remaining_allocation']}")
        if q["deferred"]:
            lines.append(f"  {q['deferred']} authorized fragment(s) were NOT submitted this call (per-call bound or time "
                         "box). They stay READY: run LAUNCH_FULL_TRIAL again (ADVANCE never submits to the QPU).")
    for key in ("classical", "classical_dispatch", "classical_reconcile", "qpu_refresh", "collect",
                "postprocess_dispatch", "postprocess_reconcile"):
        if result.get(key):
            lines.append(f"  {key}: {result[key]}")
    if result.get("capacity"):
        lines.append(f"  capacity: {result['capacity']}")
    for w in result.get("warnings", []):
        lines.append(f"  WARNING: {w}")
    for e in result.get("errors", []):
        lines.append(f"  ERROR [{e['step']}]: {e['error']}")
    sm = result.get("summary")
    if sm:
        lines.append(f"  tasks {sm['complete']}/{sm['tasks']} complete  ambiguous={sm['ambiguous']}  "
                     f"failed/retry={sm['failed_or_retry']}")
    return "\n".join(lines)
