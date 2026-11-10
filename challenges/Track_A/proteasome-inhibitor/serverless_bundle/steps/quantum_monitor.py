"""
quantum_monitor — Stage 3Q-B: Lightweight QPU job monitoring and dispatch preparation.

Implements the notebook-native status-refresh-and-dispatch-preparation interface for
quantum fragment tasks that are in SUBMITTED or RUNNING state.  This module is the
authoritative implementation of Stage 3Q-B.

Design constraints
------------------
* No synchronous waiting for QPU completion.
* No heavy scientific imports at module level (no Vayesta, PySCF, SQD, CUDA).
* No automatic QPU resubmission — ambiguous tasks are surfaced, not retried.
* No Serverless worker allocation for monitoring.
* Reentrant and notebook-restart-safe: all state is persisted in the SQLite registry.
* A single bounded refresh operation: query once, persist observations, return.
* Partial failures are tolerated: successful observations are persisted even when
  some provider queries fail.

Key concepts
------------
Provider completion vs scientific readiness:
  * Provider completion (DONE): IBM Quantum reports the job finished.
  * Measurement verification (SAMPLES_READY): result is available for retrieval.
    This is NOT the same as authenticated counts — authentication is Task 5.7 (3Q-C).
  * Post-processing completion (COMPLETE): Task 5.7 has produced and persisted a
    validated FragmentResult.  This module never sets COMPLETE.

Registry state transitions owned by Task 5.6
---------------------------------------------
  SUBMITTED   → RUNNING        when provider reports RUNNING
  SUBMITTED   → SAMPLES_READY  when provider reports DONE/COMPLETED
  RUNNING     → SAMPLES_READY  when provider reports DONE/COMPLETED
  SUBMITTED   → FAILED         when provider reports ERROR/FAILED/CANCELED (†)
  RUNNING     → FAILED         same
  SAMPLES_READY → POSTPROCESS_QUEUED  when create_dispatch_record is called
  (†) FAILED only for definitively failed/canceled jobs — not for network errors.
      Network errors leave the task state unchanged and record the error in
      refresh_errors on the returned MonitoringSnapshot.

The monitoring module never transitions to COMPLETE.

Public API
----------
``refresh_quantum_jobs(registry, run_id, *, qpu_service, ...) -> MonitoringSnapshot``
``build_quantum_status_table(registry, run_id) -> QuantumStatusTable``
``create_dispatch_record(registry, task_id, *, fragment_id, submission_record) -> dict``
``list_dispatch_eligible(registry, run_id) -> list[dict]``
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "QuantumStatusTable",
    "MonitoringSnapshot",
    "RefreshError",
    "ProviderStatus",
    "refresh_quantum_jobs",
    "build_quantum_status_table",
    "create_dispatch_record",
    "list_dispatch_eligible",
    "normalize_provider_status",
    "MONITORING_SCHEMA",
]

MONITORING_SCHEMA = "qmon/1"

# ─────────────────────────────────────────────────────────────────────────────
#  Provider status normalization
# ─────────────────────────────────────────────────────────────────────────────

class ProviderStatus:
    """Normalized internal provider status categories."""
    QUEUED    = "QUEUED"
    RUNNING   = "RUNNING"
    DONE      = "DONE"        # IBM: DONE / JobStatus.DONE — counts available
    FAILED    = "FAILED"      # IBM: ERROR / JobStatus.ERROR
    CANCELED  = "CANCELED"    # IBM: CANCELED / JobStatus.CANCELLED
    UNKNOWN   = "UNKNOWN"     # temporarily unavailable or unrecognized
    ERROR     = "ERROR"       # local query error (network, auth) — NOT provider failure

    # Provider-reported terminal states that map to DONE (counts available)
    _DONE_ALIASES: frozenset[str] = frozenset({
        "DONE", "COMPLETED", "RESULT_AVAILABLE", "COMPLETE",
    })
    # Provider-reported terminal failure states
    _FAILED_ALIASES: frozenset[str] = frozenset({
        "ERROR", "FAILED", "FAILURE",
    })
    # Provider-reported cancellation states
    _CANCELED_ALIASES: frozenset[str] = frozenset({
        "CANCELED", "CANCELLED", "CANCELLING",
    })
    # Provider-reported active states
    _RUNNING_ALIASES: frozenset[str] = frozenset({
        "RUNNING", "IN_PROGRESS",
    })
    _QUEUED_ALIASES: frozenset[str] = frozenset({
        "QUEUED", "QUEUING", "PENDING", "INITIALIZING", "WAITING",
    })


def normalize_provider_status(raw: str) -> str:
    """Map a raw IBM Quantum job status string to a stable internal category.

    Parameters
    ----------
    raw : str
        Raw status from ``job.status().name.upper()`` or similar.
        May include enum prefix (e.g. ``"JOBSTATUS.DONE"``).

    Returns
    -------
    One of the ``ProviderStatus`` constants.
    """
    if not raw:
        return ProviderStatus.UNKNOWN

    # Strip enum prefix if present (e.g. "JobStatus.DONE" → "DONE")
    s = raw.upper()
    if "." in s:
        s = s.split(".")[-1]

    if s in ProviderStatus._DONE_ALIASES:
        return ProviderStatus.DONE
    if s in ProviderStatus._FAILED_ALIASES:
        return ProviderStatus.FAILED
    if s in ProviderStatus._CANCELED_ALIASES:
        return ProviderStatus.CANCELED
    if s in ProviderStatus._RUNNING_ALIASES:
        return ProviderStatus.RUNNING
    if s in ProviderStatus._QUEUED_ALIASES:
        return ProviderStatus.QUEUED
    return ProviderStatus.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RefreshError:
    """A non-fatal error encountered while querying a single QPU job."""
    task_id:    str
    fragment_id: int
    qpu_job_id: str
    error_type: str   # "network", "auth", "timeout", "unknown"
    message:    str
    occurred_utc: str = ""

    def as_dict(self) -> dict:
        return {
            "task_id":       self.task_id,
            "fragment_id":   self.fragment_id,
            "qpu_job_id":    self.qpu_job_id,
            "error_type":    self.error_type,
            "message":       self.message,
            "occurred_utc":  self.occurred_utc,
        }


@dataclass
class MonitoringSnapshot:
    """Result of one bounded monitoring refresh pass.

    Attributes
    ----------
    run_id          : the run scoped by this snapshot
    refreshed_utc   : ISO-8601 timestamp of the refresh
    queried_count   : number of provider queries attempted
    updated_count   : number of tasks whose state changed
    refresh_errors  : per-task query errors (non-fatal)
    state_counts    : mapping of task state → count
    dispatch_ready  : task_ids now eligible for post-processing dispatch
    provider_raw    : mapping of task_id → raw provider status string
    live_refresh    : True if provider was actually queried (vs stored-state only)
    """
    run_id:          str
    refreshed_utc:   str
    queried_count:   int
    updated_count:   int
    refresh_errors:  list[RefreshError]
    state_counts:    dict[str, int]
    dispatch_ready:  list[str]   # task_ids in SAMPLES_READY (eligible for 3Q-C)
    provider_raw:    dict[str, str]    # task_id → raw provider status
    live_refresh:    bool

    def summary_text(self) -> str:
        lines = [
            f"Quantum job refresh — run={self.run_id!r}  ({self.refreshed_utc})",
            f"  Queries attempted : {self.queried_count}",
            f"  Tasks updated     : {self.updated_count}",
            f"  Errors            : {len(self.refresh_errors)}",
            f"  Live provider poll: {'yes' if self.live_refresh else 'no (stored state)'}",
            "  State counts:",
        ]
        for state, cnt in sorted(self.state_counts.items()):
            lines.append(f"    {state:<24} {cnt}")
        if self.dispatch_ready:
            lines.append(f"  Post-processing eligible: {len(self.dispatch_ready)} fragment(s)")
            lines.append(f"    task_ids: {self.dispatch_ready}")
        if self.refresh_errors:
            lines.append("  Refresh errors:")
            for e in self.refresh_errors:
                lines.append(f"    fragment={e.fragment_id} job={e.qpu_job_id} : {e.message}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "run_id":         self.run_id,
            "refreshed_utc":  self.refreshed_utc,
            "queried_count":  self.queried_count,
            "updated_count":  self.updated_count,
            "refresh_errors": [e.as_dict() if hasattr(e, "as_dict") else str(e) for e in self.refresh_errors],
            "state_counts":   dict(self.state_counts),
            "dispatch_ready": list(self.dispatch_ready),
            "provider_raw":   dict(self.provider_raw),
            "live_refresh":   self.live_refresh,
        }


@dataclass
class _StatusRow:
    """One row of the status table."""
    fragment_id:          int
    solver:               str
    qpu_job_id:           str
    backend:              str
    provider_status:      str   # last stored provider status
    task_state:           str
    submitted_utc:        str
    last_refresh_utc:     str
    elapsed_s:            Optional[float]
    samples_ready:        bool
    postprocess_state:    str
    note:                 str   # reconciliation hint or error note

    def as_dict(self) -> dict:
        return {
            "fragment_id":       self.fragment_id,
            "solver":            self.solver,
            "qpu_job_id":        self.qpu_job_id or "—",
            "backend":           self.backend or "—",
            "provider_status":   self.provider_status or "(not polled)",
            "task_state":        self.task_state,
            "submitted_utc":     self.submitted_utc or "—",
            "last_refresh_utc":  self.last_refresh_utc or "—",
            "elapsed_s":         f"{self.elapsed_s:.0f}" if self.elapsed_s is not None else "—",
            "samples_ready":     "✓" if self.samples_ready else "—",
            "postprocess_state": self.postprocess_state or "—",
            "note":              self.note or "—",
        }


@dataclass
class QuantumStatusTable:
    """Notebook-friendly collection of quantum task status rows.

    Attributes
    ----------
    rows        : one :class:`_StatusRow` per quantum task
    as_of_utc   : snapshot timestamp (stored state, not live)
    run_id      : scoped run
    counts      : aggregate counts by category (queued, running, done, failed, ambiguous)
    """
    rows:       list[_StatusRow]
    as_of_utc:  str
    run_id:     str
    counts:     dict[str, int]

    def table(self) -> str:
        """Notebook-friendly plain-text table (call display(status.table()) or print it)."""
        if not self.rows:
            return (
                f"No quantum tasks for run {self.run_id!r} (as of {self.as_of_utc})\n"
                "Note: provider_status reflects STORED observations only — call "
                "refresh_quantum_jobs() or challenge.quantum_job_status(refresh=True) "
                "for a live update.\n"
            )

        cols = [
            "frag", "solver", "job_id", "backend",
            "provider", "state", "elapsed_s",
            "pp_state", "note",
        ]
        rows_data = []
        for r in sorted(self.rows, key=lambda x: x.fragment_id):
            note = r.note or ""
            if r.task_state == "AMBIGUOUS":
                note = "⚠ AMBIGUOUS — manual recovery required" + (f"; {note}" if note else "")
            elif r.task_state == "SUBMITTING":
                note = "⚠ stale SUBMITTING — investigate provider" + (f"; {note}" if note else "")
            rows_data.append([
                str(r.fragment_id),
                r.solver or "—",
                (r.qpu_job_id[-16:] if len(r.qpu_job_id or "") > 16 else (r.qpu_job_id or "—")),
                r.backend or "—",
                r.provider_status or "(not polled)",
                r.task_state,
                f"{r.elapsed_s:.0f}s" if r.elapsed_s is not None else "—",
                r.postprocess_state or "—",
                note[:40] if note else "—",
            ])

        widths = [max(len(cols[i]), max((len(r[i]) for r in rows_data), default=0))
                  for i in range(len(cols))]
        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        header = "|" + "|".join(f" {cols[i]:<{widths[i]}} " for i in range(len(cols))) + "|"

        lines = [
            f"Quantum job status — run={self.run_id!r}  (stored state as of {self.as_of_utc})",
            "Note: 'provider' column shows STORED status — not authenticated measurements.",
            "",
            sep, header, sep,
        ]
        for r in rows_data:
            lines.append("|" + "|".join(f" {r[i]:<{widths[i]}} " for i in range(len(cols))) + "|")
        lines.append(sep)
        lines.append("")

        # Aggregate summary
        c = self.counts
        lines.append(
            f"Summary: queued={c.get('queued', 0)}  running={c.get('running', 0)}  "
            f"samples_ready={c.get('samples_ready', 0)}  complete={c.get('complete', 0)}  "
            f"failed={c.get('failed', 0)}  ambiguous={c.get('ambiguous', 0)}"
        )
        return "\n".join(lines)

    def to_list(self) -> list[dict]:
        """Return a list of dicts suitable for pd.DataFrame() or JSON serialization."""
        return [r.as_dict() for r in sorted(self.rows, key=lambda x: x.fragment_id)]


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed(created_iso: str, updated_iso: str) -> Optional[float]:
    try:
        c = datetime.fromisoformat(created_iso)
        u = datetime.fromisoformat(updated_iso)
        return (u - c).total_seconds()
    except (ValueError, TypeError):
        return None


def _classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "auth" in msg or "token" in msg or "401" in msg or "403" in msg:
        return "auth"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "network" in msg or "connect" in msg or "connection" in msg:
        return "network"
    return "unknown"


def _query_provider_status(qpu_service: Any, job_id: str) -> tuple[str, str]:
    """Query the provider for a single job's status.

    Returns (normalized_status, raw_status_string).
    Raises on authentication/network errors so the caller can record them.
    """
    try:
        job = qpu_service.job(job_id)
    except Exception as exc:
        raise RuntimeError(f"Failed to retrieve job {job_id!r}: {exc}") from exc

    try:
        raw = job.status()
    except Exception as exc:
        raise RuntimeError(f"Failed to get status for job {job_id!r}: {exc}") from exc

    # IBM Runtime returns a JobStatus enum; .name gives the member name
    raw_str = raw.name.upper() if hasattr(raw, "name") else str(raw).upper()
    normalized = normalize_provider_status(raw_str)
    return normalized, raw_str


def _registry_transition_safe(registry, task_id: str, from_state: str, to_state: str,
                               reason: str) -> tuple[bool, Exception | None]:
    """Attempt a registry transition; return (True, None) on success or (False, exc).

    Does not raise — failures are returned to the caller so the caller can
    decide whether to record a RefreshError in the monitoring snapshot.
    """
    try:
        registry.transition(task_id, from_state, to_state, reason=reason)
        return True, None
    except Exception as exc:
        return False, exc


# ─────────────────────────────────────────────────────────────────────────────
#  Core monitoring function
# ─────────────────────────────────────────────────────────────────────────────

def refresh_quantum_jobs(
    registry,
    run_id: str,
    *,
    qpu_service: Any,
    max_queries: int = 50,
    query_timeout_seconds: float = 30.0,
) -> MonitoringSnapshot:
    """One bounded provider status refresh for all active quantum tasks.

    Queries only tasks in SUBMITTED or RUNNING state (tasks with recorded QPU Job IDs
    that are not yet terminal).  Updates registry state and returns a snapshot.

    Parameters
    ----------
    registry              : FragmentTaskRegistry
    run_id                : scoped run identifier
    qpu_service           : an active provider service with a ``.job(job_id)`` method
    max_queries           : maximum provider queries per call (default 50)
    query_timeout_seconds : per-query timeout hint (passed to provider if supported)

    Returns
    -------
    :class:`MonitoringSnapshot`

    Safety guarantees
    -----------------
    * Never submits a QPU job.
    * Never retrieves measurement counts.
    * Never runs SQD/TrimSQD diagonalization.
    * Partial failures do not prevent successful observations from being persisted.
    * Idempotent: repeating the call does not duplicate work or state transitions.
    * Ambiguous tasks are NOT automatically retried — they remain AMBIGUOUS.
    * Tasks without a recorded qpu_job_id are skipped and surfaced in the table.
    """
    refreshed_utc = _utc_now()
    errors: list[RefreshError] = []
    dispatch_ready: list[str] = []
    provider_raw: dict[str, str] = {}
    updated_count = 0
    queried_count = 0

    # States eligible for provider query (active but not terminal)
    _ACTIVE_STATES = {"SUBMITTED", "RUNNING"}

    tasks = registry.list_tasks(run_id, task_type="quantum")
    active_tasks = [t for t in tasks if t.state in _ACTIVE_STATES and t.qpu_job_id]
    stale_submitting = [t for t in tasks
                        if t.state == "SUBMITTING" and not t.qpu_job_id]

    for t in active_tasks[:max_queries]:
        queried_count += 1
        try:
            normalized, raw_str = _query_provider_status(qpu_service, t.qpu_job_id)
        except Exception as exc:
            errors.append(RefreshError(
                task_id=t.task_id,
                fragment_id=t.fragment_id,
                qpu_job_id=t.qpu_job_id,
                error_type=_classify_error(exc),
                message=str(exc)[:200],
                occurred_utc=_utc_now(),
            ))
            # Preserve existing state — do not transition on query error
            continue

        provider_raw[t.task_id] = raw_str

        # Always persist the observed provider status (idempotent field update)
        try:
            registry.update_provider_status(t.task_id, provider_status=raw_str)
        except Exception:
            pass  # Non-fatal: observation is best-effort

        # Determine registry state transition
        if normalized == ProviderStatus.RUNNING and t.state == "SUBMITTED":
            ok, exc = _registry_transition_safe(
                registry, t.task_id, "SUBMITTED", "RUNNING",
                reason=f"3Q-B: provider reports RUNNING (raw={raw_str!r})",
            )
            if ok:
                updated_count += 1
            else:
                errors.append(RefreshError(
                    task_id=t.task_id,
                    fragment_id=t.fragment_id,
                    qpu_job_id=t.qpu_job_id,
                    error_type="registry",
                    message=f"SUBMITTED→RUNNING transition failed: {exc}"[:200],
                    occurred_utc=_utc_now(),
                ))

        elif normalized == ProviderStatus.DONE:
            # Mark SAMPLES_READY regardless of whether in SUBMITTED or RUNNING
            if t.state in ("SUBMITTED", "RUNNING"):
                ok, exc = _registry_transition_safe(
                    registry, t.task_id, t.state, "SAMPLES_READY",
                    reason=f"3Q-B: provider reports DONE (raw={raw_str!r}); measurements available",
                )
                if ok:
                    updated_count += 1
                    # Re-read to get updated record
                    try:
                        updated_task = registry.get_task(t.task_id)
                        if updated_task.state == "SAMPLES_READY":
                            dispatch_ready.append(t.task_id)
                    except Exception:
                        dispatch_ready.append(t.task_id)
                else:
                    errors.append(RefreshError(
                        task_id=t.task_id,
                        fragment_id=t.fragment_id,
                        qpu_job_id=t.qpu_job_id,
                        error_type="registry",
                        message=f"{t.state}→SAMPLES_READY transition failed: {exc}"[:200],
                        occurred_utc=_utc_now(),
                    ))

        elif normalized in (ProviderStatus.FAILED, ProviderStatus.CANCELED):
            # Definitively failed/canceled at provider — mark FAILED in registry
            try:
                registry.fail_task(
                    t.task_id,
                    reason=f"3Q-B: provider reports {normalized} (raw={raw_str!r})",
                )
                updated_count += 1
            except Exception:
                pass  # Already in terminal state

    # Build state counts from fresh registry read
    all_tasks = registry.list_tasks(run_id, task_type="quantum")
    state_counts: dict[str, int] = {}
    for t in all_tasks:
        state_counts[t.state] = state_counts.get(t.state, 0) + 1

    # Also include any existing SAMPLES_READY tasks in dispatch_ready
    for t in all_tasks:
        if t.state == "SAMPLES_READY" and t.task_id not in dispatch_ready:
            dispatch_ready.append(t.task_id)

    # Map state counts to named categories
    counts = {
        "queued":        state_counts.get("SUBMITTED", 0),
        "running":       state_counts.get("RUNNING", 0),
        "samples_ready": state_counts.get("SAMPLES_READY", 0),
        "dispatched":    state_counts.get("POSTPROCESS_QUEUED", 0)
                         + state_counts.get("POSTPROCESSING", 0),
        "complete":      state_counts.get("COMPLETE", 0),
        "failed":        state_counts.get("FAILED", 0),
        "ambiguous":     state_counts.get("AMBIGUOUS", 0),
        "stale_submitting": len(stale_submitting),
    }

    return MonitoringSnapshot(
        run_id=run_id,
        refreshed_utc=refreshed_utc,
        queried_count=queried_count,
        updated_count=updated_count,
        refresh_errors=errors,
        state_counts=state_counts,
        dispatch_ready=dispatch_ready,
        provider_raw=provider_raw,
        live_refresh=(qpu_service is not None),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Stored-state status table (no provider calls)
# ─────────────────────────────────────────────────────────────────────────────

def build_quantum_status_table(
    registry,
    run_id: str,
) -> QuantumStatusTable:
    """Build a status table from stored registry state only (no provider calls).

    This is the read-only path: no provider queries, no state transitions.
    Safe to call any time, including before connecting to IBM Quantum.

    Parameters
    ----------
    registry : FragmentTaskRegistry
    run_id   : run to inspect

    Returns
    -------
    :class:`QuantumStatusTable`
    """
    as_of_utc = _utc_now()
    tasks = registry.list_tasks(run_id, task_type="quantum")
    rows: list[_StatusRow] = []

    _PP_STATES = {
        "POSTPROCESS_QUEUED": "queued",
        "POSTPROCESSING":     "running",
        "COMPLETE":           "complete",
    }
    _SAMPLES_STATES = {"SAMPLES_READY", "POSTPROCESS_QUEUED", "POSTPROCESSING", "COMPLETE"}

    for t in tasks:
        note = ""
        if t.state == "AMBIGUOUS":
            note = f"AMBIGUOUS — use recover_ambiguous_submission() or abandon_ambiguous_and_retry()"
            if t.failure_reason:
                note += f"; reason: {t.failure_reason[:60]}"
        elif t.state == "SUBMITTING" and not t.qpu_job_id:
            note = "stale SUBMITTING — process may have crashed; investigate provider"
        elif t.recovery_notes:
            note = f"recovery: {t.recovery_notes[:60]}"

        rows.append(_StatusRow(
            fragment_id=t.fragment_id,
            solver=t.solver,
            qpu_job_id=t.qpu_job_id,
            backend=t.backend,
            provider_status=t.last_provider_status,
            task_state=t.state,
            submitted_utc=t.created_iso8601,
            last_refresh_utc=t.updated_iso8601,
            elapsed_s=_elapsed(t.created_iso8601, t.updated_iso8601),
            samples_ready=(t.state in _SAMPLES_STATES),
            postprocess_state=_PP_STATES.get(t.state, ""),
            note=note,
        ))

    # Aggregate counts
    state_counts: dict[str, int] = {}
    for t in tasks:
        state_counts[t.state] = state_counts.get(t.state, 0) + 1

    counts = {
        "queued":        state_counts.get("SUBMITTED", 0),
        "running":       state_counts.get("RUNNING", 0),
        "samples_ready": state_counts.get("SAMPLES_READY", 0),
        "complete":      state_counts.get("COMPLETE", 0),
        "failed":        state_counts.get("FAILED", 0),
        "ambiguous":     state_counts.get("AMBIGUOUS", 0),
    }

    return QuantumStatusTable(rows=rows, as_of_utc=as_of_utc, run_id=run_id, counts=counts)


# ─────────────────────────────────────────────────────────────────────────────
#  Dispatch preparation
# ─────────────────────────────────────────────────────────────────────────────

def create_dispatch_record(
    registry,
    task_id: str,
    *,
    ewf_dumpfile_path: str,
    submission_record_dict: dict,
) -> dict:
    """Prepare a durable post-processing dispatch record for a SAMPLES_READY task.

    Transitions SAMPLES_READY → POSTPROCESS_QUEUED and writes a JSON dispatch
    outbox record containing all information 3Q-C needs to locate and process
    this fragment's measurements.

    Idempotent: if the task is already in POSTPROCESS_QUEUED, the existing
    dispatch record is returned without re-transitioning.

    Parameters
    ----------
    registry              : FragmentTaskRegistry
    task_id               : task in SAMPLES_READY state
    ewf_dumpfile_path     : path to the Stage 2 HDF5 used during submission
    submission_record_dict : dict from SubmissionRecord.as_dict()

    Returns
    -------
    dict — the dispatch record written to the registry

    Raises
    ------
    RuntimeError
        - task is not in SAMPLES_READY or POSTPROCESS_QUEUED state
        - transition fails for unexpected reasons
    """
    from proteasome_challenge.task_registry import TaskState

    task = registry.get_task(task_id)

    if task.state == TaskState.POSTPROCESS_QUEUED:
        # Already dispatched — return existing record (idempotent)
        try:
            existing = json.loads(task.dispatch_record)
            return existing
        except (json.JSONDecodeError, TypeError):
            pass

    if task.state != TaskState.SAMPLES_READY:
        raise RuntimeError(
            f"Task {task_id!r}: create_dispatch_record requires SAMPLES_READY state, "
            f"found {task.state!r}."
        )

    dispatch = {
        "schema":              "dispatch/1",
        "task_id":             task_id,
        "run_id":              task.run_id,
        "fragment_id":         task.fragment_id,
        "solver":              task.solver,
        "qpu_job_id":          task.qpu_job_id,
        "backend":             task.backend,
        "hamiltonian_sha256":  task.hamiltonian_sha256 or (
            submission_record_dict.get("hamiltonian_sha256", "")
            if isinstance(submission_record_dict, dict)
            else ""
        ),
        "ewf_dumpfile_path":   str(ewf_dumpfile_path),
        "submission_record":   submission_record_dict,
        "queued_utc":          _utc_now(),
        # 3Q-C note: provider completion ≠ authenticated counts.
        # The post-processing worker (Task 5.7) must retrieve and authenticate
        # measurements before performing diagonalization.
        "measurement_authenticated": False,
    }

    registry.mark_postprocess_queued(
        task_id,
        dispatch_record=dispatch,
        expected_state=TaskState.SAMPLES_READY,
    )
    return dispatch


def list_dispatch_eligible(registry, run_id: str) -> list[dict]:
    """Return all pending dispatch records for a run.

    These are tasks in POSTPROCESS_QUEUED state with a valid dispatch_record
    JSON blob.  The 3Q-C worker (Task 5.7) reads these to find work.

    Parameters
    ----------
    registry : FragmentTaskRegistry
    run_id   : run to inspect

    Returns
    -------
    list[dict] — dispatch record dicts sorted by fragment_id
    """
    records = registry.list_pending_dispatch(run_id)
    result = []
    for task_id, dispatch_dict in records:
        try:
            task = registry.get_task(task_id)
            dispatch_dict["_task_state"] = task.state
            dispatch_dict["_fragment_id"] = task.fragment_id
        except Exception:
            dispatch_dict["_task_state"] = "unknown"
            dispatch_dict["_fragment_id"] = dispatch_dict.get("fragment_id", -1)
        result.append(dispatch_dict)

    result.sort(key=lambda d: d.get("_fragment_id", 0))
    return result
