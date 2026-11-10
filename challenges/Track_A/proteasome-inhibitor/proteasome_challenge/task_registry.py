"""
task_registry — Durable fragment task registry for the EWF–TrimSQD pipeline.

Provides a narrow, transactional state store for per-fragment computational
tasks.  The local backend uses SQLite (via the standard library ``sqlite3``
module) which gives ACID transactions, conditional UPDATE semantics, and
zero external dependencies.

Backend boundary
----------------
All external interactions go through :class:`FragmentTaskRegistry`.  Callers
never touch SQLite directly.  A future remote backend (e.g. a shared DB
or managed scheduler) can implement the same :class:`RegistryBackend`
interface without changing client code.

Limits
------
SQLite is appropriate for single-machine orchestration and notebook-driven
workflows.  It is NOT accessible across independently-scheduled Serverless
workers unless they share a network filesystem.  See
:attr:`FragmentTaskRegistry.REMOTE_READINESS_NOTES` for the verified blocker
list.

Scientific data
---------------
The registry stores task identity, state, timestamps and artifact
*references*.  It does NOT store Hamiltonians, QPU measurement arrays,
density matrices, or SolverResult arrays.  Those are managed by
``serverless_bundle/steps/fragment_result_store.py`` and
``serverless_bundle/steps/artifact_store.py`` (Task 5.2).

QPU ledger
----------
The persistent QPU sampling ledger at
``checkpoints/.qpu_sampling_record.json`` is **never touched** by this
module.  QPU quota accounting remains the exclusive responsibility of the
existing :meth:`Challenge._record_qpu_sampling` / :meth:`Challenge._check_qpu_budget`
methods.

Schema version
--------------
``trv/1``  (task-registry version 1).

Usage
-----
    >>> from proteasome_challenge.task_registry import FragmentTaskRegistry
    >>> reg = FragmentTaskRegistry.open(":memory:")          # or a .db file path
    >>> task_id = reg.create_task(
    ...     run_id="ixazomib_sto-3g",
    ...     fragment_id=3,
    ...     task_type="quantum",
    ...     solver="TrimSQD",
    ...     hamiltonian_sha256="abc...",
    ...     resource_family="l",
    ... )
    >>> reg.transition(task_id, "READY", "SUBMITTING", worker_id="w1")
    >>> reg.record_qpu_job_id(task_id, qpu_job_id="crn:...", backend="ibm_kingston")
    >>> reg.transition(task_id, "SUBMITTING", "SUBMITTED")
    >>> snapshot = reg.quantum_job_status()
    >>> print(snapshot.to_text())
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

__all__ = [
    # Core
    "FragmentTaskRegistry",
    "RegistryError",
    "InvalidTransitionError",
    "TaskNotFoundError",
    "DuplicateTaskError",
    "ClaimConflictError",
    "AmbiguousSubmissionError",
    # Data
    "TaskRecord",
    "QuantumJobSnapshot",
    "StatusTable",
    # Constants
    "REGISTRY_SCHEMA",
    "TaskState",
    "TaskType",
]

REGISTRY_SCHEMA = "trv/1"

# ─────────────────────────────────────────────────────────────────────────────
#  State machine
# ─────────────────────────────────────────────────────────────────────────────

class TaskState:
    """Legal task states and their transition graph."""

    READY              = "READY"
    SUBMITTING         = "SUBMITTING"
    SUBMITTED          = "SUBMITTED"
    RUNNING            = "RUNNING"
    SAMPLES_READY      = "SAMPLES_READY"
    POSTPROCESS_QUEUED = "POSTPROCESS_QUEUED"
    POSTPROCESSING     = "POSTPROCESSING"
    COMPLETE           = "COMPLETE"
    # Non-forward states
    AMBIGUOUS          = "AMBIGUOUS"
    RETRY_PENDING      = "RETRY_PENDING"
    FAILED             = "FAILED"
    CANCELED           = "CANCELED"

    # All valid state names
    ALL: frozenset[str] = frozenset({
        "READY", "SUBMITTING", "SUBMITTED", "RUNNING",
        "SAMPLES_READY", "POSTPROCESS_QUEUED", "POSTPROCESSING", "COMPLETE",
        "AMBIGUOUS", "RETRY_PENDING", "FAILED", "CANCELED",
    })

    # Terminal states — no further transitions are permitted
    TERMINAL: frozenset[str] = frozenset({"COMPLETE", "FAILED", "CANCELED"})

    # Legal transitions: {from_state: frozenset(to_states)}
    TRANSITIONS: dict[str, frozenset[str]] = {
        "READY":              frozenset({"SUBMITTING", "FAILED", "CANCELED"}),
        "SUBMITTING":         frozenset({"SUBMITTED", "AMBIGUOUS", "RETRY_PENDING", "FAILED", "CANCELED"}),
        # SUBMITTED → SAMPLES_READY covers the case where the monitor first observes
        # a job as DONE without having seen a RUNNING intermediate state.
        "SUBMITTED":          frozenset({"RUNNING", "SAMPLES_READY", "FAILED", "CANCELED"}),
        "RUNNING":            frozenset({"SAMPLES_READY", "FAILED", "CANCELED"}),
        "SAMPLES_READY":      frozenset({"POSTPROCESS_QUEUED", "FAILED", "CANCELED"}),
        "POSTPROCESS_QUEUED": frozenset({"POSTPROCESSING", "FAILED", "CANCELED"}),
        "POSTPROCESSING":     frozenset({"COMPLETE", "FAILED", "CANCELED"}),
        "COMPLETE":           frozenset(),
        # AMBIGUOUS → SUBMITTED is the audited recovery path via recover_ambiguous_submission().
        # AMBIGUOUS → RETRY_PENDING requires explicit operator approval (abandon_ambiguous_and_retry).
        "AMBIGUOUS":          frozenset({"SUBMITTED", "RETRY_PENDING", "FAILED", "CANCELED"}),
        "RETRY_PENDING":      frozenset({"READY", "FAILED", "CANCELED"}),
        "FAILED":             frozenset(),
        "CANCELED":           frozenset(),
    }

    # Classical fragments only use this subset (bypasses QPU-specific states)
    CLASSICAL_TRANSITIONS: dict[str, frozenset[str]] = {
        "READY":          frozenset({"SUBMITTING", "FAILED", "CANCELED"}),
        "SUBMITTING":     frozenset({"RUNNING", "RETRY_PENDING", "FAILED", "CANCELED"}),
        "RUNNING":        frozenset({"COMPLETE", "RETRY_PENDING", "FAILED", "CANCELED"}),
        "COMPLETE":       frozenset(),
        "RETRY_PENDING":  frozenset({"READY", "FAILED", "CANCELED"}),
        "FAILED":         frozenset(),
        "CANCELED":       frozenset(),
    }

    @classmethod
    def is_valid_transition(
        cls,
        from_state: str,
        to_state: str,
        *,
        task_type: str = "quantum",
    ) -> bool:
        table = (
            cls.CLASSICAL_TRANSITIONS
            if task_type == "classical"
            else cls.TRANSITIONS
        )
        return to_state in table.get(from_state, frozenset())

    @classmethod
    def check_transition(
        cls,
        from_state: str,
        to_state: str,
        *,
        task_type: str = "quantum",
        task_id: str = "",
    ) -> None:
        if not cls.is_valid_transition(from_state, to_state, task_type=task_type):
            raise InvalidTransitionError(
                f"Task {task_id!r}: {from_state!r} → {to_state!r} is not a legal "
                f"transition for a {task_type} task."
            )


class TaskType:
    QUANTUM   = "quantum"
    CLASSICAL = "classical"
    ALL: frozenset[str] = frozenset({"quantum", "classical"})


# ─────────────────────────────────────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class RegistryError(RuntimeError):
    """Base class for all registry errors."""


class InvalidTransitionError(RegistryError):
    """Requested state transition is not in the legal transition graph."""


class TaskNotFoundError(RegistryError):
    """No task matching the requested identity exists."""


class DuplicateTaskError(RegistryError):
    """A logical task with the same (run_id, fragment_id, task_type) already exists."""


class ClaimConflictError(RegistryError):
    """The requested task is already claimed by another active worker."""


class AmbiguousSubmissionError(RegistryError):
    """Task is in AMBIGUOUS state — automatic resubmission is prohibited."""


# ─────────────────────────────────────────────────────────────────────────────
#  Data objects
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskRecord:
    """Immutable snapshot of one task row returned by registry queries."""
    task_id:              str
    run_id:               str
    fragment_id:          int
    task_type:            str
    solver:               str
    hamiltonian_sha256:   str
    resource_family:      str
    state:                str
    attempt:              int
    created_iso8601:      str
    updated_iso8601:      str
    worker_id:            str = ""
    lease_token:          str = ""
    lease_expires_iso8601: str = ""
    serverless_job_id:    str = ""
    qpu_job_id:           str = ""
    backend:              str = ""
    last_provider_status: str = ""
    result_artifact_uri:  str = ""
    result_payload_digest: str = ""
    failure_reason:       str = ""
    recovery_notes:       str = ""
    dispatch_record:      str = ""   # JSON blob for transactional outbox
    extra_json:           str = "{}"  # extensible JSON blob

    @property
    def extra(self) -> dict:
        try:
            return json.loads(self.extra_json or "{}")
        except json.JSONDecodeError:
            return {}

    def is_terminal(self) -> bool:
        return self.state in TaskState.TERMINAL

    def needs_qpu(self) -> bool:
        return self.task_type == TaskType.QUANTUM

    def as_dict(self) -> dict:
        return {
            "task_id":               self.task_id,
            "run_id":                self.run_id,
            "fragment_id":           self.fragment_id,
            "task_type":             self.task_type,
            "solver":                self.solver,
            "hamiltonian_sha256":    self.hamiltonian_sha256,
            "resource_family":       self.resource_family,
            "state":                 self.state,
            "attempt":               self.attempt,
            "created_iso8601":       self.created_iso8601,
            "updated_iso8601":       self.updated_iso8601,
            "worker_id":             self.worker_id,
            "lease_token":           self.lease_token,
            "lease_expires_iso8601": self.lease_expires_iso8601,
            "serverless_job_id":     self.serverless_job_id,
            "qpu_job_id":            self.qpu_job_id,
            "backend":               self.backend,
            "last_provider_status":  self.last_provider_status,
            "result_artifact_uri":   self.result_artifact_uri,
            "result_payload_digest": self.result_payload_digest,
            "failure_reason":        self.failure_reason,
            "recovery_notes":        self.recovery_notes,
            "dispatch_record":       self.dispatch_record,
            "extra_json":            self.extra_json,
        }


@dataclass
class QuantumJobSnapshot:
    """Per-fragment quantum status snapshot for notebook display."""
    fragment_id:          int
    qpu_job_id:           str
    backend:              str
    last_provider_status: str    # empty string = never queried
    task_state:           str
    submission_iso8601:   str    # time of SUBMITTING→SUBMITTED transition
    last_checked_iso8601: str    # time of last state update
    elapsed_seconds:      float | None
    postprocess_ready:    bool
    postprocess_state:    str
    failure_reason:       str
    is_ambiguous:         bool

    def as_dict(self) -> dict:
        return {
            "fragment_id":          self.fragment_id,
            "qpu_job_id":           self.qpu_job_id or "—",
            "backend":              self.backend or "—",
            "provider_status":      self.last_provider_status or "(not polled)",
            "task_state":           self.task_state,
            "submitted_at":         self.submission_iso8601 or "—",
            "last_update":          self.last_checked_iso8601,
            "elapsed_s":            f"{self.elapsed_seconds:.0f}" if self.elapsed_seconds is not None else "—",
            "pp_ready":             "✓" if self.postprocess_ready else "—",
            "pp_state":             self.postprocess_state or "—",
            "failure":              self.failure_reason or "—",
            "ambiguous":            "⚠" if self.is_ambiguous else "—",
        }


@dataclass
class StatusTable:
    """Collection of quantum snapshots with notebook-friendly rendering."""
    snapshots: list[QuantumJobSnapshot]
    as_of_iso8601: str
    run_id: str

    def to_text(self) -> str:
        """Plain-text table suitable for notebook cell output."""
        if not self.snapshots:
            return f"No quantum tasks found for run {self.run_id!r} (as of {self.as_of_iso8601})\n"

        cols = ["frag", "job_id", "backend", "provider", "state", "elapsed_s", "pp_state", "flag"]
        rows = []
        for s in sorted(self.snapshots, key=lambda x: x.fragment_id):
            flag = "⚠ AMBIGUOUS" if s.is_ambiguous else ("✗ FAILED" if "FAIL" in s.task_state else "")
            rows.append([
                str(s.fragment_id),
                (s.qpu_job_id[-16:] if len(s.qpu_job_id) > 16 else s.qpu_job_id) or "—",
                s.backend or "—",
                s.last_provider_status or "(not polled)",
                s.task_state,
                f"{s.elapsed_seconds:.0f}s" if s.elapsed_seconds is not None else "—",
                s.postprocess_state or "—",
                flag,
            ])

        widths = [max(len(cols[i]), max(len(r[i]) for r in rows)) for i in range(len(cols))]
        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        header = "|" + "|".join(f" {cols[i]:<{widths[i]}} " for i in range(len(cols))) + "|"
        lines = [
            f"Quantum job status — run={self.run_id!r}  (stored state as of {self.as_of_iso8601})",
            "",
            sep,
            header,
            sep,
        ]
        for r in rows:
            lines.append("|" + "|".join(f" {r[i]:<{widths[i]}} " for i in range(len(cols))) + "|")
        lines.append(sep)
        lines.append("")
        lines.append("Note: 'provider' column shows STORED status only (no live QPU poll performed).")
        return "\n".join(lines)

    def to_list(self) -> list[dict]:
        """Return a list of dicts suitable for pd.DataFrame() or JSON serialization."""
        return [s.as_dict() for s in self.snapshots]


# ─────────────────────────────────────────────────────────────────────────────
#  SQLite DDL
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fragment_tasks (
    task_id               TEXT PRIMARY KEY,
    run_id                TEXT NOT NULL,
    fragment_id           INTEGER NOT NULL,
    task_type             TEXT NOT NULL,
    solver                TEXT NOT NULL,
    hamiltonian_sha256    TEXT NOT NULL,
    resource_family       TEXT NOT NULL DEFAULT 'l',
    state                 TEXT NOT NULL DEFAULT 'READY',
    attempt               INTEGER NOT NULL DEFAULT 0,
    created_iso8601       TEXT NOT NULL,
    updated_iso8601       TEXT NOT NULL,
    worker_id             TEXT NOT NULL DEFAULT '',
    lease_token           TEXT NOT NULL DEFAULT '',
    lease_expires_iso8601 TEXT NOT NULL DEFAULT '',
    serverless_job_id     TEXT NOT NULL DEFAULT '',
    qpu_job_id            TEXT NOT NULL DEFAULT '',
    backend               TEXT NOT NULL DEFAULT '',
    last_provider_status  TEXT NOT NULL DEFAULT '',
    result_artifact_uri   TEXT NOT NULL DEFAULT '',
    result_payload_digest TEXT NOT NULL DEFAULT '',
    failure_reason        TEXT NOT NULL DEFAULT '',
    recovery_notes        TEXT NOT NULL DEFAULT '',
    dispatch_record       TEXT NOT NULL DEFAULT '',
    extra_json            TEXT NOT NULL DEFAULT '{}',
    UNIQUE (run_id, fragment_id, task_type)
);

CREATE INDEX IF NOT EXISTS idx_tasks_state
    ON fragment_tasks (state);
CREATE INDEX IF NOT EXISTS idx_tasks_run_id
    ON fragment_tasks (run_id);
CREATE INDEX IF NOT EXISTS idx_tasks_qpu_job_id
    ON fragment_tasks (qpu_job_id)
    WHERE qpu_job_id != '';

CREATE TABLE IF NOT EXISTS state_transitions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id           TEXT NOT NULL,
    from_state        TEXT NOT NULL,
    to_state          TEXT NOT NULL,
    worker_id         TEXT NOT NULL DEFAULT '',
    reason            TEXT NOT NULL DEFAULT '',
    occurred_iso8601  TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES fragment_tasks(task_id)
);
CREATE INDEX IF NOT EXISTS idx_transitions_task_id
    ON state_transitions (task_id);
"""

_COLUMNS = (
    "task_id", "run_id", "fragment_id", "task_type", "solver",
    "hamiltonian_sha256", "resource_family", "state", "attempt",
    "created_iso8601", "updated_iso8601", "worker_id", "lease_token",
    "lease_expires_iso8601", "serverless_job_id", "qpu_job_id", "backend",
    "last_provider_status", "result_artifact_uri", "result_payload_digest",
    "failure_reason", "recovery_notes", "dispatch_record", "extra_json",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_task_id() -> str:
    return str(uuid.uuid4())


def _new_lease_token() -> str:
    return uuid.uuid4().hex


def _row_to_record(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(**{c: row[c] for c in _COLUMNS})


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────

class FragmentTaskRegistry:
    """Persistent task registry backed by a local SQLite database.

    All mutating operations are executed inside explicit transactions so the
    database is never left in a partially-written state.

    Thread safety
    -------------
    One :class:`FragmentTaskRegistry` instance per thread.  SQLite's WAL mode
    allows concurrent readers; writers serialize via SQLite's internal locking.
    Do NOT share a single instance across threads — open one per thread.

    Fencing
    -------
    Claim operations return a *lease_token* (random UUID hex).  Callers must
    pass the token back when completing or extending a lease.  A token mismatch
    causes the operation to be rejected so an expired worker cannot commit a
    stale result.

    Outbox / dispatch
    -----------------
    When a QPU job completes and post-processing must be dispatched, the
    registry records a durable *dispatch_record* JSON blob atomically with the
    state transition to ``POSTPROCESS_QUEUED``.  The actual Serverless
    invocation is a separate, non-atomic step.  A re-entrant monitoring pass
    can re-read undispatched records and retry the Serverless call; the
    downstream worker's claim will reject duplicate execution.
    """

    # Documentation for the remote-readiness blocker list
    REMOTE_READINESS_NOTES: str = """
    BLOCKERS — shared registry across Serverless workers
    ====================================================
    1. SQLite on a local filesystem cannot be accessed by independently-
       scheduled Serverless workers on different hosts.
    2. Serverless File API (IBM Quantum / Qiskit Runtime): provides simple
       per-function file storage.  Semantics for concurrent multi-worker
       access are UNVERIFIED.  Not a substitute for a transactional DB.
    3. Shared network filesystem (NFS, CephFS): in principle accessible but
       SQLite WAL mode requires proper POSIX advisory locking, which NFS
       implementations sometimes do not honour.
    4. External managed database (PostgreSQL, CockroachDB): would satisfy the
       transactional requirement but needs authentication, network access from
       worker VMs, and connection pooling.  NOT configured in this repository.
    5. IBM Cloud Object Storage / S3-compatible: object stores lack row-level
       transactions.  Compare-and-swap / conditional PUT semantics exist but
       differ from SQL; a full re-implementation would be required.
    VERDICT: For production distributed operation, an external transactional
    database or managed task queue is required.  This SQLite backend is
    correct for local orchestration, notebook-driven workflows and all offline
    tests.
    """

    def __init__(self, conn: sqlite3.Connection, *, path: str = ":memory:"):
        self._conn = conn
        self._path = path
        self._lock = threading.Lock()

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def open(cls, path: str | Path = ":memory:") -> "FragmentTaskRegistry":
        """Open or create a registry at *path*.

        Use ``":memory:"`` for tests (each call creates a fresh in-memory DB).
        Use a ``.db`` file path for persistent operation.
        """
        db_path = str(path)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_DDL)
        # Record schema version in meta table
        conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
            ("schema_version", REGISTRY_SCHEMA),
        )
        conn.commit()
        return cls(conn, path=db_path)

    @classmethod
    def reopen(cls, path: str | Path) -> "FragmentTaskRegistry":
        """Re-open an existing database (e.g. after process restart).

        Raises :class:`RegistryError` if the file does not exist or has the
        wrong schema version.
        """
        db_path = str(path)
        if db_path != ":memory:" and not Path(db_path).is_file():
            raise RegistryError(f"Registry database not found: {db_path}")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Apply DDL idempotently (CREATE IF NOT EXISTS) for forward compat
        conn.executescript(_DDL)
        # Verify schema
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            raise RegistryError(
                f"Registry at {db_path!r} has no schema_version record."
            )
        sv = row["value"]
        if sv != REGISTRY_SCHEMA:
            raise RegistryError(
                f"Registry schema mismatch: expected {REGISTRY_SCHEMA!r}, "
                f"found {sv!r} at {db_path!r}."
            )
        conn.commit()
        return cls(conn, path=db_path)

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> "FragmentTaskRegistry":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Task creation ──────────────────────────────────────────────────────────

    def create_task(
        self,
        *,
        run_id: str,
        fragment_id: int,
        task_type: str,
        solver: str,
        hamiltonian_sha256: str,
        resource_family: str = "l",
        extra: dict | None = None,
    ) -> str:
        """Create a new task in READY state; return its task_id.

        Raises
        ------
        DuplicateTaskError
            If a task with the same (run_id, fragment_id, task_type) already
            exists (logical task identity).
        RegistryError
            If any required argument is invalid.
        """
        if not run_id:
            raise RegistryError("run_id must be a non-empty string")
        if not isinstance(fragment_id, int) or fragment_id < 0:
            raise RegistryError(f"fragment_id must be a non-negative integer, got {fragment_id!r}")
        if task_type not in TaskType.ALL:
            raise RegistryError(
                f"task_type must be one of {sorted(TaskType.ALL)}, got {task_type!r}"
            )
        from proteasome_challenge.solver_assignments import VALID_SOLVERS
        if solver not in VALID_SOLVERS:
            raise RegistryError(
                f"solver must be one of {sorted(VALID_SOLVERS)}, got {solver!r}"
            )
        if not hamiltonian_sha256:
            raise RegistryError("hamiltonian_sha256 must be a non-empty string")

        task_id = _new_task_id()
        now = _utc_iso()
        extra_json = json.dumps(extra or {})

        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO fragment_tasks
                        (task_id, run_id, fragment_id, task_type, solver,
                         hamiltonian_sha256, resource_family, state, attempt,
                         created_iso8601, updated_iso8601, extra_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'READY', 0, ?, ?, ?)
                    """,
                    (task_id, run_id, fragment_id, task_type, solver,
                     hamiltonian_sha256, resource_family, now, now, extra_json),
                )
                self._append_transition(task_id, "", "READY", "", "task created", now)
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint" in str(exc):
                raise DuplicateTaskError(
                    f"Task already exists for run_id={run_id!r}, "
                    f"fragment_id={fragment_id}, task_type={task_type!r}"
                ) from exc
            raise RegistryError(str(exc)) from exc

        return task_id

    # ── Lookup ─────────────────────────────────────────────────────────────────

    def get_task(self, task_id: str) -> TaskRecord:
        """Return the :class:`TaskRecord` for *task_id*.

        Raises :class:`TaskNotFoundError` if absent.
        """
        row = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM fragment_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            raise TaskNotFoundError(f"No task with task_id={task_id!r}")
        return _row_to_record(row)

    def find_task(
        self,
        run_id: str,
        fragment_id: int,
        task_type: str,
    ) -> TaskRecord | None:
        """Find the logical task for (run_id, fragment_id, task_type).

        Returns ``None`` if not found.
        """
        row = self._conn.execute(
            f"""
            SELECT {', '.join(_COLUMNS)} FROM fragment_tasks
            WHERE run_id = ? AND fragment_id = ? AND task_type = ?
            """,
            (run_id, fragment_id, task_type),
        ).fetchone()
        return _row_to_record(row) if row else None

    def list_tasks(
        self,
        run_id: str | None = None,
        *,
        state: str | None = None,
        task_type: str | None = None,
    ) -> list[TaskRecord]:
        """Return all matching tasks."""
        clauses, params = [], []
        if run_id is not None:
            clauses.append("run_id = ?"); params.append(run_id)
        if state is not None:
            clauses.append("state = ?"); params.append(state)
        if task_type is not None:
            clauses.append("task_type = ?"); params.append(task_type)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM fragment_tasks {where} ORDER BY fragment_id",
            params,
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    # ── State transitions ──────────────────────────────────────────────────────

    def transition(
        self,
        task_id: str,
        expected_state: str,
        new_state: str,
        *,
        worker_id: str = "",
        reason: str = "",
        lease_token: str = "",
        lease_duration_seconds: int = 300,
    ) -> TaskRecord:
        """Atomically transition *task_id* from *expected_state* to *new_state*.

        Parameters
        ----------
        task_id          : task to transition
        expected_state   : must match the current task state (CAS semantics)
        new_state        : target state
        worker_id        : identifier of the claiming worker
        reason           : human-readable explanation stored in the log
        lease_token      : if supplied, must match the stored lease_token (fencing)
        lease_duration_seconds : how long the new lease is valid (default 5 min)

        Returns
        -------
        Updated :class:`TaskRecord`

        Raises
        ------
        TaskNotFoundError       — task does not exist
        InvalidTransitionError  — transition not in the legal graph
        ClaimConflictError      — lease_token mismatch (stale worker)
        RegistryError           — state has already changed (CAS miss)
        AmbiguousSubmissionError — attempted transition out of AMBIGUOUS without
                                   explicit reconciliation
        """
        record = self.get_task(task_id)

        # Guard AMBIGUOUS — never auto-resubmit via transition().
        # The only authorised path from AMBIGUOUS → SUBMITTED is recover_ambiguous_submission().
        if record.state == TaskState.AMBIGUOUS and new_state == TaskState.SUBMITTED:
            raise AmbiguousSubmissionError(
                f"Task {task_id!r} is AMBIGUOUS (possible duplicate QPU submission). "
                "Use recover_ambiguous_submission() — not transition() — to recover."
            )

        # Validate legal transition
        TaskState.check_transition(
            record.state, new_state,
            task_type=record.task_type,
            task_id=task_id,
        )

        # Verify expected state matches
        if record.state != expected_state:
            raise RegistryError(
                f"Task {task_id!r}: expected state {expected_state!r} "
                f"but found {record.state!r} — concurrent update detected."
            )

        # Fencing: if the task is held by a lease, caller must present the token
        if record.lease_token and lease_token and record.lease_token != lease_token:
            raise ClaimConflictError(
                f"Task {task_id!r}: lease token mismatch — "
                "caller's token is stale or belongs to a different worker."
            )

        now = _utc_iso()
        # Compute new lease expiry
        lease_expires = ""
        new_lease = lease_token or record.lease_token
        if new_state not in TaskState.TERMINAL and new_state not in (
            TaskState.AMBIGUOUS, TaskState.RETRY_PENDING
        ):
            lease_expires = datetime.fromtimestamp(
                time.time() + lease_duration_seconds, tz=timezone.utc
            ).isoformat()

        # Increment attempt counter when transitioning INTO a new logical attempt.
        # One submission attempt = SUBMITTING bump.
        # One post-processing attempt = POSTPROCESSING bump.
        # RUNNING is NOT bumped — it represents QPU execution of an already-counted attempt.
        # This prevents double-counting across SUBMITTING and the later SUBMITTED→RUNNING step.
        _ATTEMPT_BUMP_TARGETS = {
            TaskState.SUBMITTING, TaskState.POSTPROCESSING
        }
        new_attempt = record.attempt + 1 if new_state in _ATTEMPT_BUMP_TARGETS else record.attempt

        # Clear failure reason if recovering
        clear_failure = new_state in (TaskState.READY, TaskState.RETRY_PENDING)

        with self._conn:
            updated = self._conn.execute(
                """
                UPDATE fragment_tasks
                SET state                 = ?,
                    updated_iso8601       = ?,
                    worker_id             = ?,
                    lease_token           = ?,
                    lease_expires_iso8601 = ?,
                    attempt               = ?,
                    failure_reason        = CASE WHEN ? THEN '' ELSE failure_reason END
                WHERE task_id = ? AND state = ?
                """,
                (
                    new_state, now,
                    worker_id or record.worker_id,
                    new_lease,
                    lease_expires,
                    new_attempt,
                    1 if clear_failure else 0,
                    task_id, expected_state,
                ),
            ).rowcount
            if updated == 0:
                raise RegistryError(
                    f"Task {task_id!r}: state transition failed — state was "
                    f"changed concurrently (expected {expected_state!r})."
                )
            self._append_transition(task_id, expected_state, new_state, worker_id, reason, now)

        return self.get_task(task_id)

    # ── QPU job ID persistence ─────────────────────────────────────────────────

    def record_qpu_job_id(
        self,
        task_id: str,
        *,
        qpu_job_id: str,
        backend: str,
        serverless_job_id: str = "",
        lease_token: str = "",
    ) -> TaskRecord:
        """Persist the QPU Job ID returned by the provider after submission.

        This is called immediately after the provider accepts the job, before
        or coincident with transitioning to SUBMITTED.  The task must be in
        SUBMITTING state.

        The QPU Job ID is written atomically so it is never lost even if the
        submission worker crashes between persisting the ID and calling
        :meth:`transition`.
        """
        record = self.get_task(task_id)
        if record.state != TaskState.SUBMITTING:
            raise RegistryError(
                f"Task {task_id!r}: can only record QPU Job ID in SUBMITTING state, "
                f"found {record.state!r}."
            )
        if not qpu_job_id:
            raise RegistryError("qpu_job_id must be a non-empty string")
        if not backend:
            raise RegistryError("backend must be a non-empty string")

        # Fencing check
        if record.lease_token and lease_token and record.lease_token != lease_token:
            raise ClaimConflictError(
                f"Task {task_id!r}: lease token mismatch in record_qpu_job_id."
            )

        now = _utc_iso()
        with self._conn:
            self._conn.execute(
                """
                UPDATE fragment_tasks
                SET qpu_job_id        = ?,
                    backend           = ?,
                    serverless_job_id = CASE WHEN ? != '' THEN ? ELSE serverless_job_id END,
                    updated_iso8601   = ?
                WHERE task_id = ? AND state = 'SUBMITTING'
                """,
                (
                    qpu_job_id, backend,
                    serverless_job_id, serverless_job_id,
                    now, task_id,
                ),
            )
        return self.get_task(task_id)

    def record_serverless_job_id(
        self,
        task_id: str,
        *,
        serverless_job_id: str,
        lease_token: str = "",
    ) -> TaskRecord:
        """Persist the Serverless Job ID returned after worker dispatch.

        Can be called while the task is claimed in SUBMITTING or RUNNING state.
        Fenced by lease_token when supplied.
        """
        record = self.get_task(task_id)
        if record.is_terminal():
            raise RegistryError(
                f"Task {task_id!r}: cannot record Serverless Job ID on terminal state {record.state!r}."
            )
        if not serverless_job_id:
            raise RegistryError("serverless_job_id must be a non-empty string")

        # Fencing check
        if record.lease_token and lease_token and record.lease_token != lease_token:
            raise ClaimConflictError(
                f"Task {task_id!r}: lease token mismatch in record_serverless_job_id."
            )

        now = _utc_iso()
        with self._conn:
            self._conn.execute(
                """
                UPDATE fragment_tasks
                SET serverless_job_id = ?,
                    updated_iso8601   = ?
                WHERE task_id = ?
                """,
                (serverless_job_id, now, task_id),
            )
        return self.get_task(task_id)

    def update_extra(
        self,
        task_id: str,
        key: str,
        value: Any,
        *,
        append: bool = False,
    ) -> TaskRecord:
        """Durably set (or append to a list) one key of the task's ``extra_json``.

        Used for append-only attempt history (Task 5.9C.1).  Existing keys are preserved; the
        read-modify-write is serialised by the registry lock.  Not fenced by lease: it only records
        facts and never changes ``state``.
        """
        if not key or not isinstance(key, str):
            raise RegistryError("extra key must be a non-empty string")
        with self._lock:
            record = self.get_task(task_id)
            extra = record.extra
            if append:
                current = extra.get(key, [])
                if not isinstance(current, list):
                    raise RegistryError(f"extra[{key!r}] is not a list; cannot append")
                extra[key] = current + [value]
            else:
                extra[key] = value
            payload = json.dumps(extra)
            with self._conn:
                self._conn.execute(
                    "UPDATE fragment_tasks SET extra_json = ?, updated_iso8601 = ? WHERE task_id = ?",
                    (payload, _utc_iso(), task_id),
                )
        return self.get_task(task_id)

    # ── Provider status ────────────────────────────────────────────────────────

    def update_provider_status(
        self,
        task_id: str,
        *,
        provider_status: str,
    ) -> TaskRecord:
        """Store the latest observed provider-side job status.

        This is a read-write operation distinct from state transitions.  It
        records the raw provider status string (e.g. "QUEUED", "RUNNING",
        "DONE") without changing the task state machine.

        **This method must not be called during Task 5.3.**
        It is declared here so the 3Q-B dispatcher can call it in Task 5.4
        without changing the registry API.
        """
        now = _utc_iso()
        with self._conn:
            self._conn.execute(
                """
                UPDATE fragment_tasks
                SET last_provider_status = ?,
                    updated_iso8601      = ?
                WHERE task_id = ?
                """,
                (provider_status, now, task_id),
            )
        return self.get_task(task_id)

    # ── Result publication ─────────────────────────────────────────────────────

    def record_result(
        self,
        task_id: str,
        *,
        artifact_uri: str,
        payload_digest: str,
        lease_token: str = "",
    ) -> TaskRecord:
        """Record the URI and payload_digest of the completed fragment result.

        The task must be in POSTPROCESSING (quantum) or RUNNING (classical)
        state.  After this call, the caller should call :meth:`transition` to
        COMPLETE, which requires the artifact_uri to be set.
        """
        record = self.get_task(task_id)
        _eligible = {TaskState.POSTPROCESSING, TaskState.RUNNING}
        if record.state not in _eligible:
            raise RegistryError(
                f"Task {task_id!r}: result can only be recorded in "
                f"{sorted(_eligible)}, found {record.state!r}."
            )
        if not artifact_uri:
            raise RegistryError("artifact_uri must be a non-empty string")
        if not payload_digest:
            raise RegistryError("payload_digest must be a non-empty string")

        if record.lease_token and lease_token and record.lease_token != lease_token:
            raise ClaimConflictError(
                f"Task {task_id!r}: lease token mismatch in record_result."
            )

        now = _utc_iso()
        with self._conn:
            self._conn.execute(
                """
                UPDATE fragment_tasks
                SET result_artifact_uri   = ?,
                    result_payload_digest = ?,
                    updated_iso8601       = ?
                WHERE task_id = ?
                """,
                (artifact_uri, payload_digest, now, task_id),
            )
        return self.get_task(task_id)

    def complete_task(
        self,
        task_id: str,
        *,
        artifact_uri: str,
        payload_digest: str,
        lease_token: str = "",
        reason: str = "result validated",
    ) -> TaskRecord:
        """Transition to COMPLETE; requires a valid result artifact.

        Idempotent: if the task is already COMPLETE with the same artifact_uri
        and payload_digest, returns the existing record without error.
        """
        record = self.get_task(task_id)

        # Idempotent: already complete with the same artifact
        if record.state == TaskState.COMPLETE:
            if (record.result_artifact_uri == artifact_uri
                    and record.result_payload_digest == payload_digest):
                return record
            raise RegistryError(
                f"Task {task_id!r} is already COMPLETE with a different artifact."
            )

        if not artifact_uri:
            raise RegistryError("complete_task requires a non-empty artifact_uri")
        if not payload_digest:
            raise RegistryError("complete_task requires a non-empty payload_digest")

        # Record result first, then transition
        self.record_result(
            task_id,
            artifact_uri=artifact_uri,
            payload_digest=payload_digest,
            lease_token=lease_token,
        )
        return self.transition(
            task_id,
            record.state,
            TaskState.COMPLETE,
            lease_token=lease_token,
            reason=reason,
        )

    # ── Failure / cancellation ─────────────────────────────────────────────────

    def fail_task(
        self,
        task_id: str,
        *,
        reason: str,
        lease_token: str = "",
    ) -> TaskRecord:
        """Transition to FAILED from any non-terminal state."""
        record = self.get_task(task_id)
        if record.state in TaskState.TERMINAL:
            raise RegistryError(
                f"Task {task_id!r} is already terminal ({record.state!r})."
            )

        if record.lease_token and lease_token and record.lease_token != lease_token:
            raise ClaimConflictError(
                f"Task {task_id!r}: lease token mismatch in fail_task."
            )

        now = _utc_iso()
        with self._conn:
            self._conn.execute(
                """
                UPDATE fragment_tasks
                SET state          = 'FAILED',
                    failure_reason = ?,
                    updated_iso8601 = ?
                WHERE task_id = ?
                """,
                (reason, now, task_id),
            )
            self._append_transition(task_id, record.state, "FAILED", "", reason, now)
        return self.get_task(task_id)

    def cancel_task(self, task_id: str, *, reason: str = "canceled") -> TaskRecord:
        """Transition to CANCELED from any non-terminal state."""
        record = self.get_task(task_id)
        if record.state in TaskState.TERMINAL:
            raise RegistryError(
                f"Task {task_id!r} is already terminal ({record.state!r})."
            )
        now = _utc_iso()
        with self._conn:
            self._conn.execute(
                """
                UPDATE fragment_tasks
                SET state           = 'CANCELED',
                    failure_reason  = ?,
                    updated_iso8601 = ?
                WHERE task_id = ?
                """,
                (reason, now, task_id),
            )
            self._append_transition(task_id, record.state, "CANCELED", "", reason, now)
        return self.get_task(task_id)

    # ── Retry ──────────────────────────────────────────────────────────────────

    def request_retry(
        self,
        task_id: str,
        *,
        reason: str = "retry requested",
        max_attempts: int = 3,
    ) -> TaskRecord:
        """Transition to RETRY_PENDING if attempt limit not reached.

        The task must be in FAILED or RUNNING (for classical) state.
        Tasks in AMBIGUOUS state cannot be retried automatically; they require
        explicit operator resolution via abandon_ambiguous_and_retry().
        If max_attempts is reached, leaves the task in FAILED state instead.
        """
        record = self.get_task(task_id)
        if record.state == TaskState.AMBIGUOUS:
            raise AmbiguousSubmissionError(
                f"Task {task_id!r} is AMBIGUOUS — provider acceptance is uncertain.  "
                "Use recover_ambiguous_submission() to recover the Job ID, or "
                "abandon_ambiguous_and_retry() to explicitly discard the uncertain attempt."
            )
        if record.state not in (TaskState.FAILED, TaskState.RUNNING, TaskState.RETRY_PENDING):
            raise InvalidTransitionError(
                f"Task {task_id!r}: retry can only be requested from FAILED/RUNNING/RETRY_PENDING, "
                f"found {record.state!r}."
            )
        if record.attempt >= max_attempts:
            # Do not retry — leave FAILED
            return self.fail_task(
                task_id,
                reason=f"max_attempts ({max_attempts}) reached after: {reason}",
            ) if record.state != TaskState.FAILED else record

        now = _utc_iso()
        with self._conn:
            self._conn.execute(
                """
                UPDATE fragment_tasks
                SET state           = 'RETRY_PENDING',
                    failure_reason  = ?,
                    updated_iso8601 = ?
                WHERE task_id = ?
                """,
                (reason, now, task_id),
            )
            self._append_transition(task_id, record.state, "RETRY_PENDING", "", reason, now)
        return self.get_task(task_id)

    def reset_for_retry(self, task_id: str) -> TaskRecord:
        """Transition from RETRY_PENDING back to READY for a new attempt."""
        return self.transition(
            task_id,
            TaskState.RETRY_PENDING,
            TaskState.READY,
            reason="reset for retry",
        )

    # ── Ambiguous submission recovery ──────────────────────────────────────────

    def recover_ambiguous_submission(
        self,
        task_id: str,
        *,
        qpu_job_id: str,
        backend: str,
        recovery_reason: str,
        lease_token: str = "",
        circuit_digest: str = "",
        submission_tags: list | None = None,
    ) -> TaskRecord:
        """Audited recovery from AMBIGUOUS state: record the verified Job ID and
        transition AMBIGUOUS → SUBMITTED.

        This is the **only** authorised path from AMBIGUOUS to SUBMITTED.
        It must only be called after the operator has:

        1. Confirmed the Job ID belongs to the correct fragment and Hamiltonian.
        2. Verified it was not a duplicate submission via the QPU provider dashboard.
        3. Confirmed the job has not already been attributed to another task.

        Parameters
        ----------
        task_id          : AMBIGUOUS task to recover
        qpu_job_id       : verified IBM Quantum Job ID (must be non-empty)
        backend          : backend name for provenance
        recovery_reason  : human-readable description of the recovery decision
        lease_token      : fencing token (recommended; verified if the task carries one)
        circuit_digest   : circuit digest for provenance (stored in extra_json)
        submission_tags  : original submission tags (stored in extra_json)

        Returns
        -------
        Updated :class:`TaskRecord` in SUBMITTED state.

        Raises
        ------
        RegistryError
            - task is not in AMBIGUOUS state
            - qpu_job_id or backend is empty
            - task already carries a different qpu_job_id (conflicting recovery)
        ClaimConflictError
            - lease token mismatch
        AmbiguousSubmissionError
            - attempt to recover without a recovery_reason (operator must document decision)
        """
        if not qpu_job_id:
            raise RegistryError("recover_ambiguous_submission: qpu_job_id must be non-empty")
        if not backend:
            raise RegistryError("recover_ambiguous_submission: backend must be non-empty")
        if not recovery_reason:
            raise AmbiguousSubmissionError(
                "recover_ambiguous_submission: recovery_reason must be provided — "
                "the operator must document why this Job ID is accepted."
            )

        record = self.get_task(task_id)

        if record.state != TaskState.AMBIGUOUS:
            raise RegistryError(
                f"Task {task_id!r}: recover_ambiguous_submission requires AMBIGUOUS state, "
                f"found {record.state!r}."
            )

        # Fencing check
        if record.lease_token and lease_token and record.lease_token != lease_token:
            raise ClaimConflictError(
                f"Task {task_id!r}: lease token mismatch in recover_ambiguous_submission."
            )

        # Conflict check: if a different Job ID is already stored, reject
        if record.qpu_job_id and record.qpu_job_id != qpu_job_id:
            raise RegistryError(
                f"Task {task_id!r}: already carries a different qpu_job_id "
                f"({record.qpu_job_id!r}); cannot overwrite with {qpu_job_id!r}. "
                "Cancel the task explicitly if the stored Job ID is invalid."
            )

        now = _utc_iso()

        # Build provenance note for recovery_notes
        note = (
            f"[RECOVERY {now}] Operator recovery: {recovery_reason}. "
            f"Job ID {qpu_job_id!r} on backend {backend!r}."
        )
        if circuit_digest:
            note += f"  circuit_digest={circuit_digest[:16]}..."

        # Build extra_json update preserving existing keys
        try:
            extra = json.loads(record.extra_json or "{}")
        except json.JSONDecodeError:
            extra = {}
        extra["recovery_circuit_digest"] = circuit_digest
        extra["recovery_tags"] = submission_tags or []
        extra["recovery_timestamp"] = now
        extra["recovery_reason"] = recovery_reason
        new_extra = json.dumps(extra)

        # Atomically write qpu_job_id, backend, recovery_notes and transition state
        with self._conn:
            updated = self._conn.execute(
                """
                UPDATE fragment_tasks
                SET state           = 'SUBMITTED',
                    qpu_job_id      = ?,
                    backend         = CASE WHEN backend = '' THEN ? ELSE backend END,
                    recovery_notes  = recovery_notes || ?,
                    extra_json      = ?,
                    updated_iso8601 = ?
                WHERE task_id = ? AND state = 'AMBIGUOUS'
                """,
                (
                    qpu_job_id,
                    backend,
                    "\n" + note,
                    new_extra,
                    now,
                    task_id,
                ),
            ).rowcount
            if updated == 0:
                raise RegistryError(
                    f"Task {task_id!r}: CAS failed in recover_ambiguous_submission — "
                    "task state changed concurrently."
                )
            self._append_transition(
                task_id, TaskState.AMBIGUOUS, TaskState.SUBMITTED,
                "", f"operator recovery: {recovery_reason}", now,
            )

        return self.get_task(task_id)

    def abandon_ambiguous_and_retry(
        self,
        task_id: str,
        *,
        abandonment_reason: str,
        max_attempts: int = 3,
    ) -> TaskRecord:
        """Explicitly abandon an AMBIGUOUS attempt and schedule a retry.

        This is a destructive operator action: it declares that the uncertain
        QPU submission attempt will NOT be recovered, and schedules a new
        submission attempt.  The operator MUST have verified that:

        1. The original submission was NOT accepted by the provider (or)
        2. The original job has been definitively cancelled on the provider side.

        A new attempt will increment the attempt counter when it enters SUBMITTING.

        Parameters
        ----------
        task_id           : task in AMBIGUOUS state
        abandonment_reason : mandatory operator justification
        max_attempts      : refuse to retry if attempt >= this limit

        Returns
        -------
        :class:`TaskRecord` in RETRY_PENDING state.

        Raises
        ------
        AmbiguousSubmissionError  — task is not AMBIGUOUS
        RegistryError             — attempt limit reached or abandonment_reason empty
        """
        if not abandonment_reason:
            raise AmbiguousSubmissionError(
                "abandon_ambiguous_and_retry: abandonment_reason must be provided — "
                "the operator must document why the uncertain attempt is discarded."
            )

        record = self.get_task(task_id)
        if record.state != TaskState.AMBIGUOUS:
            raise AmbiguousSubmissionError(
                f"Task {task_id!r}: abandon_ambiguous_and_retry requires AMBIGUOUS state, "
                f"found {record.state!r}."
            )
        if record.attempt >= max_attempts:
            return self.fail_task(
                task_id,
                reason=(
                    f"max_attempts ({max_attempts}) reached; abandonment: {abandonment_reason}"
                ),
            )

        now = _utc_iso()
        note = (
            f"[ABANDONMENT {now}] Operator abandoned uncertain attempt: {abandonment_reason}"
        )
        with self._conn:
            updated = self._conn.execute(
                """
                UPDATE fragment_tasks
                SET state           = 'RETRY_PENDING',
                    failure_reason  = ?,
                    recovery_notes  = recovery_notes || ?,
                    updated_iso8601 = ?
                WHERE task_id = ? AND state = 'AMBIGUOUS'
                """,
                (abandonment_reason, "\n" + note, now, task_id),
            ).rowcount
            if updated == 0:
                raise RegistryError(
                    f"Task {task_id!r}: CAS failed in abandon_ambiguous_and_retry — "
                    "state changed concurrently."
                )
            self._append_transition(
                task_id, TaskState.AMBIGUOUS, TaskState.RETRY_PENDING,
                "", f"abandoned: {abandonment_reason}", now,
            )
        return self.get_task(task_id)

    # ── Lease / claim management ───────────────────────────────────────────────

    def claim_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        lease_duration_seconds: int = 300,
    ) -> tuple[TaskRecord, str]:
        """Claim a READY task for exclusive execution; return (record, lease_token).

        The returned lease_token must be supplied to :meth:`transition`,
        :meth:`record_result`, etc. to prove ownership.

        Raises :class:`ClaimConflictError` if the task is already claimed.
        """
        record = self.get_task(task_id)
        if record.state != TaskState.READY:
            raise ClaimConflictError(
                f"Task {task_id!r} is in state {record.state!r}, not READY."
            )
        if record.lease_token:
            # Already claimed — check expiry
            if record.lease_expires_iso8601:
                try:
                    exp = datetime.fromisoformat(record.lease_expires_iso8601)
                    if datetime.now(timezone.utc) < exp:
                        raise ClaimConflictError(
                            f"Task {task_id!r} is already claimed by "
                            f"worker {record.worker_id!r} (lease expires "
                            f"{record.lease_expires_iso8601})."
                        )
                    # Lease expired — allow takeover
                except ValueError:
                    pass  # unparseable expiry — allow takeover

        lease_token = _new_lease_token()
        now = _utc_iso()
        now_ts = now  # ISO string used in WHERE comparison
        lease_expires = datetime.fromtimestamp(
            time.time() + lease_duration_seconds, tz=timezone.utc
        ).isoformat()

        # Determine whether the existing lease (if any) has expired
        old_lease_expired = (
            bool(record.lease_token)
            and bool(record.lease_expires_iso8601)
            and datetime.now(timezone.utc)
            >= datetime.fromisoformat(record.lease_expires_iso8601)
        )

        with self._conn:
            if not record.lease_token or old_lease_expired:
                # Fresh claim or takeover of expired lease:
                # Use the old token value (or '') as the CAS guard.
                old_token = record.lease_token if old_lease_expired else ""
                updated = self._conn.execute(
                    """
                    UPDATE fragment_tasks
                    SET worker_id             = ?,
                        lease_token           = ?,
                        lease_expires_iso8601 = ?,
                        updated_iso8601       = ?
                    WHERE task_id = ? AND state = 'READY' AND lease_token = ?
                    """,
                    (worker_id, lease_token, lease_expires, now_ts, task_id, old_token),
                ).rowcount
            else:
                updated = 0  # active lease, not expired

            if updated == 0:
                raise ClaimConflictError(
                    f"Task {task_id!r}: concurrent claim — state changed or "
                    "already claimed before commit."
                )

        return self.get_task(task_id), lease_token

    def release_lease(self, task_id: str, *, lease_token: str) -> None:
        """Release a lease (e.g. before returning a task to READY for retry)."""
        record = self.get_task(task_id)
        if record.lease_token != lease_token:
            raise ClaimConflictError(
                f"Task {task_id!r}: lease token mismatch in release_lease."
            )
        now = _utc_iso()
        with self._conn:
            self._conn.execute(
                """
                UPDATE fragment_tasks
                SET worker_id             = '',
                    lease_token           = '',
                    lease_expires_iso8601 = '',
                    updated_iso8601       = ?
                WHERE task_id = ?
                """,
                (now, task_id),
            )

    # ── Dispatch outbox ────────────────────────────────────────────────────────

    def mark_postprocess_queued(
        self,
        task_id: str,
        *,
        dispatch_record: dict,
        expected_state: str = TaskState.SAMPLES_READY,
    ) -> TaskRecord:
        """Atomically transition to POSTPROCESS_QUEUED and write the dispatch outbox record.

        The *dispatch_record* dict is stored as a JSON blob alongside the state
        transition.  A re-entrant monitoring pass can read pending dispatch
        records via :meth:`list_pending_dispatch` and re-issue Serverless
        invocations without creating duplicate logical tasks.

        Parameters
        ----------
        task_id        : quantum task transitioning to post-processing
        dispatch_record : JSON-serialisable dict describing the Serverless
                          invocation to be made (fragment_id, art_ref, etc.)
        expected_state  : CAS guard (default: SAMPLES_READY)
        """
        if not dispatch_record:
            raise RegistryError("dispatch_record must be a non-empty dict")
        dispatch_json = json.dumps(dispatch_record)
        now = _utc_iso()

        TaskState.check_transition(
            expected_state,
            TaskState.POSTPROCESS_QUEUED,
            task_type=TaskType.QUANTUM,
            task_id=task_id,
        )

        with self._conn:
            updated = self._conn.execute(
                """
                UPDATE fragment_tasks
                SET state           = 'POSTPROCESS_QUEUED',
                    dispatch_record = ?,
                    updated_iso8601 = ?
                WHERE task_id = ? AND state = ?
                """,
                (dispatch_json, now, task_id, expected_state),
            ).rowcount
            if updated == 0:
                raise RegistryError(
                    f"Task {task_id!r}: mark_postprocess_queued CAS failed — "
                    f"state is no longer {expected_state!r}."
                )
            self._append_transition(
                task_id, expected_state, TaskState.POSTPROCESS_QUEUED,
                "", "dispatch record written", now,
            )
        return self.get_task(task_id)

    def list_pending_dispatch(
        self, run_id: str | None = None
    ) -> list[tuple[str, dict]]:
        """Return (task_id, dispatch_record_dict) for all POSTPROCESS_QUEUED tasks.

        Used by the 3Q-B dispatcher to find work that needs a Serverless
        invocation issued.
        """
        clauses, params = ["state = 'POSTPROCESS_QUEUED'", "dispatch_record != ''"], []
        if run_id is not None:
            clauses.append("run_id = ?"); params.append(run_id)
        rows = self._conn.execute(
            f"SELECT task_id, dispatch_record FROM fragment_tasks WHERE "
            + " AND ".join(clauses),
            params,
        ).fetchall()
        result = []
        for row in rows:
            try:
                d = json.loads(row["dispatch_record"])
            except json.JSONDecodeError:
                d = {}
            result.append((row["task_id"], d))
        return result

    # ── Observability ──────────────────────────────────────────────────────────

    def quantum_job_status(
        self,
        run_id: str,
        *,
        refresh: bool = False,
    ) -> StatusTable:
        """Return a stored-state-only status snapshot for all quantum tasks.

        Parameters
        ----------
        run_id  : the run to inspect
        refresh : reserved for future use — must be False in Task 5.3.
                  When True (Task 5.4+), will trigger a live QPU provider poll
                  before building the snapshot.

        Returns
        -------
        :class:`StatusTable`  (never None; empty if no quantum tasks found)

        Note: 'provider_status' reflects the last stored value from a previous
        monitoring call, NOT a live query.  Live polling is out of scope here.
        """
        if refresh:
            # Future hook — the 3Q-B implementation will add live polling here.
            # For now, warn but still return stored state.
            import warnings
            warnings.warn(
                "refresh=True has no effect in Task 5.3 — live QPU polling "
                "is not yet implemented.  Returning stored state.",
                stacklevel=2,
            )

        tasks = self.list_tasks(run_id, task_type=TaskType.QUANTUM)
        snapshots: list[QuantumJobSnapshot] = []
        now_str = _utc_iso()

        for t in tasks:
            # Compute elapsed from creation to last update
            elapsed: float | None = None
            try:
                created = datetime.fromisoformat(t.created_iso8601)
                updated = datetime.fromisoformat(t.updated_iso8601)
                elapsed = (updated - created).total_seconds()
            except (ValueError, TypeError):
                pass

            # Determine post-processing state
            pp_states = {
                TaskState.POSTPROCESS_QUEUED,
                TaskState.POSTPROCESSING,
                TaskState.COMPLETE,
            }
            pp_ready = t.state == TaskState.SAMPLES_READY or t.state in pp_states
            pp_state = t.state if t.state in pp_states else ""

            snapshots.append(QuantumJobSnapshot(
                fragment_id=t.fragment_id,
                qpu_job_id=t.qpu_job_id,
                backend=t.backend,
                last_provider_status=t.last_provider_status,
                task_state=t.state,
                submission_iso8601=t.created_iso8601,   # best proxy without separate timestamp
                last_checked_iso8601=t.updated_iso8601,
                elapsed_seconds=elapsed,
                postprocess_ready=pp_ready,
                postprocess_state=pp_state,
                failure_reason=t.failure_reason,
                is_ambiguous=(t.state == TaskState.AMBIGUOUS),
            ))

        return StatusTable(
            snapshots=snapshots,
            as_of_iso8601=now_str,
            run_id=run_id,
        )

    # ── Transition history ─────────────────────────────────────────────────────

    def get_transition_history(self, task_id: str) -> list[dict]:
        """Return the ordered state-transition log for *task_id*."""
        rows = self._conn.execute(
            """
            SELECT from_state, to_state, worker_id, reason, occurred_iso8601
            FROM state_transitions
            WHERE task_id = ?
            ORDER BY id
            """,
            (task_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _append_transition(
        self,
        task_id: str,
        from_state: str,
        to_state: str,
        worker_id: str,
        reason: str,
        occurred: str,
    ) -> None:
        """Insert one row into state_transitions.  Must be called inside a transaction."""
        self._conn.execute(
            """
            INSERT INTO state_transitions
                (task_id, from_state, to_state, worker_id, reason, occurred_iso8601)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (task_id, from_state, to_state, worker_id or "", reason or "", occurred),
        )
