"""
campaign_authorization.py — Organizer Quantum Campaign Authorization & Budget Accounting.

Defines the explicit authorization model for the 11-fragment quantum campaign in Phase 5:
- 11 TrimSQD fragments: [1, 2, 3, 4, 6, 8, 9, 13, 17, 19, 22]
- Reference shot count: 100,000 shots per fragment
- Strict quota: exactly 11 submissions authorized per campaign
- Durable persistence: tracks provider Job IDs, attempts, and outcomes
- Budget isolation: preserves the historical participant ledger ({'count': 5}) without modification.
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Pinned 11 quantum fragment IDs for the hybrid experiment
FROZEN_QUANTUM_FRAGMENT_IDS: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 9, 13, 17, 19, 22)
DEFAULT_SHOTS_PER_FRAGMENT: int = 100_000
DEFAULT_MAX_SUBMISSIONS: int = 11

SCHEMA_CAMPAIGN_AUTH = "campaign_auth/1"


class CampaignAuthorizationError(Exception):
    """Raised when a quantum submission violates campaign authorization policy."""


class QuantumDuplicateSubmissionError(CampaignAuthorizationError):
    """Raised when an already-submitted fragment is submitted again."""


@dataclass
class QuantumSubmissionEntry:
    """Provenance entry for a single QPU submission within a campaign."""
    fragment_id: int
    task_id: str
    job_id: str
    backend: str
    attempt_id: str = ""
    n_shots: int = DEFAULT_SHOTS_PER_FRAGMENT
    submitted_utc: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    # RESERVED  : quota consumed BEFORE the provider call (acceptance unknown, no job_id yet)
    # ACCEPTED  : provider returned a job_id
    # ABANDONED : operator explicitly abandoned an unresolved reservation (quota stays consumed)
    state: str = "ACCEPTED"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuantumSubmissionEntry:
        return cls(
            fragment_id=int(data["fragment_id"]),
            task_id=str(data.get("task_id", "")),
            job_id=str(data.get("job_id", "")),
            backend=str(data.get("backend", "")),
            attempt_id=str(data.get("attempt_id", "")),
            n_shots=int(data.get("n_shots", DEFAULT_SHOTS_PER_FRAGMENT)),
            submitted_utc=str(data.get("submitted_utc", "")),
            state=str(data.get("state", "ACCEPTED")),
        )


def _entry_key(entry: QuantumSubmissionEntry) -> Tuple[Any, ...]:
    """Stable identity of a ledger entry (attempt_id when present, else fragment+job)."""
    if entry.attempt_id:
        return ("attempt", entry.attempt_id)
    return ("fragment_job", entry.fragment_id, entry.job_id)


@dataclass
class CampaignAuthorization:
    """Authoritative authorization for an organizer quantum campaign.

    Attributes
    ----------
    run_id : str
        Unique scientific run identifier
    authorized_backend : str
        Specific approved QPU backend (e.g. 'ibm_fez', 'ibm_torino', 'ibm_test_backend')
    authorized_fragments : Tuple[int, ...]
        Tuple of approved fragment IDs (default: 11 frozen hybrid IDs)
    shots_per_fragment : int
        Maximum allowed shots per fragment (default: 100,000)
    max_submissions : int
        Maximum number of total QPU job submissions permitted (default: 11)
    manifest_digest : str
        SHA-256 digest of the frozen RunManifest
    authorized_by : str
        Identity of authorizing organizer (default: 'organizer')
    created_utc : str
        ISO-8601 creation timestamp
    submissions : List[QuantumSubmissionEntry]
        Durable ledger of accepted submissions
    ledger_path : Optional[Path]
        On-disk location for atomic ledger synchronization
    """
    run_id: str
    authorized_backend: str
    authorized_fragments: Tuple[int, ...] = FROZEN_QUANTUM_FRAGMENT_IDS
    shots_per_fragment: int = DEFAULT_SHOTS_PER_FRAGMENT
    max_submissions: int = DEFAULT_MAX_SUBMISSIONS
    manifest_digest: str = ""
    authorized_by: str = "organizer"
    created_utc: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    submissions: List[QuantumSubmissionEntry] = field(default_factory=list)
    ledger_path: Optional[Path] = None
    # ── additive (Task 5.9E): campaign identity separate from lifetime/historical accounting ──────
    campaign_id: str = ""                      # unique per campaign; "" for legacy authorizations
    trial_profile: str = ""                    # e.g. "FULL_LIVE_TRIAL": enables the contamination guards
    historical_accounting: Dict[str, Any] = field(default_factory=dict)   # read-only snapshot, never a budget
    forbidden_job_ids: Tuple[str, ...] = ()    # historical provider Job IDs that may never be attributed here

    def __post_init__(self):
        if isinstance(self.authorized_fragments, list):
            self.authorized_fragments = tuple(self.authorized_fragments)
        if isinstance(self.forbidden_job_ids, list):
            self.forbidden_job_ids = tuple(self.forbidden_job_ids)

    def _guard_job_id(self, job_id: str) -> None:
        """Contamination guard: a historical, replayed or foreign Job ID can never enter this campaign."""
        if job_id and job_id in self.forbidden_job_ids:
            raise CampaignAuthorizationError(
                f"QPU Job ID {job_id!r} belongs to a previous campaign and cannot be recorded in campaign "
                f"{self.campaign_id or self.run_id!r}.")
        if self.trial_profile and str(job_id).startswith("replay_"):
            raise CampaignAuthorizationError(
                "replayed measurements are not permitted under the live-trial campaign profile")

    def is_authorized_fragment(self, fragment_id: int) -> bool:
        """Check if fragment_id is in the authorized fragment set."""
        return fragment_id in self.authorized_fragments

    def get_submission_count(self) -> int:
        """Return the number of accepted submissions in this campaign."""
        return len(self.submissions)

    def remaining_allocation(self) -> int:
        """Return the remaining number of allowed submissions."""
        return max(0, self.max_submissions - len(self.submissions))

    def get_submission_for_fragment(self, fragment_id: int) -> Optional[QuantumSubmissionEntry]:
        """Find existing submission entry for a fragment if present."""
        found = None
        for sub in self.submissions:
            if sub.fragment_id == fragment_id:
                if sub.state != "ABANDONED":
                    return sub
                found = sub
        return found

    def check_budget(self) -> None:
        """Verify remaining quota; raise CampaignAuthorizationError if exhausted.

        RESERVED, ACCEPTED and ABANDONED entries all count (conservative accounting).
        """
        self._refresh_from_disk()
        if len(self.submissions) >= self.max_submissions:
            raise CampaignAuthorizationError(
                f"Campaign QPU budget exhausted for run {self.run_id!r}: "
                f"{len(self.submissions)}/{self.max_submissions} submissions already used."
            )

    def validate_submission(
        self,
        fragment_id: int,
        backend: str,
        n_shots: int,
        manifest_digest: str = "",
    ) -> None:
        """Validate an intended submission against campaign authorization rules.

        Raises CampaignAuthorizationError or QuantumDuplicateSubmissionError on violation.
        """
        # 1. Fragment membership check
        if fragment_id not in self.authorized_fragments:
            raise CampaignAuthorizationError(
                f"Fragment {fragment_id} is not authorized for quantum execution in campaign {self.run_id!r}. "
                f"Authorized fragments: {sorted(self.authorized_fragments)}"
            )

        # 2. Backend check
        if backend != self.authorized_backend:
            raise CampaignAuthorizationError(
                f"Backend {backend!r} does not match authorized backend {self.authorized_backend!r} "
                f"for campaign {self.run_id!r}."
            )

        # 3. Shot count check
        if n_shots > self.shots_per_fragment:
            raise CampaignAuthorizationError(
                f"Requested shot count ({n_shots}) exceeds authorized limit ({self.shots_per_fragment}) "
                f"for fragment {fragment_id}."
            )

        # 4. Manifest digest check (if supplied)
        if manifest_digest and self.manifest_digest and manifest_digest != self.manifest_digest:
            raise CampaignAuthorizationError(
                f"Manifest digest mismatch for campaign {self.run_id!r}: "
                f"expected {self.manifest_digest[:16]}..., got {manifest_digest[:16]}..."
            )

        # 5. Duplicate submission check
        self._refresh_from_disk()
        for existing in self.submissions:
            if existing.fragment_id == fragment_id and existing.state != "ABANDONED":
                raise QuantumDuplicateSubmissionError(
                    f"Fragment {fragment_id} has already been submitted (or reserved) for QPU in campaign "
                    f"{self.run_id!r} (Job ID: {existing.job_id or 'UNKNOWN - reserved'!r}, state={existing.state}). "
                    "Automatic resubmission is prohibited."
                )

        # 6. Overall budget check
        self.check_budget()

    # ── Ledger persistence (locked, non-clobbering) ───────────────────────────

    @contextlib.contextmanager
    def _ledger_lock(self):
        """Exclusive inter-process lock around ledger read-modify-write."""
        if self.ledger_path is None:
            yield
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.ledger_path.with_name(self.ledger_path.name + ".lock")
        with open(lock_path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _refresh_from_disk(self) -> None:
        """Merge on-disk submissions that this (possibly stale) instance has not seen.

        On-disk entries missing in memory are adopted; in-memory entries win for
        identical keys.  Submissions are never dropped by a refresh.
        """
        if self.ledger_path is None or not self.ledger_path.is_file():
            return
        data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        known = {_entry_key(e) for e in self.submissions}
        for raw in data.get("submissions", []):
            entry = QuantumSubmissionEntry.from_dict(raw)
            if _entry_key(entry) not in known:
                self.submissions.append(entry)
        # A disk entry may be further advanced than ours (e.g. job_id filled in by
        # another process): adopt the disk state for entries sharing a key.
        disk = {_entry_key(QuantumSubmissionEntry.from_dict(r)): QuantumSubmissionEntry.from_dict(r)
                for r in data.get("submissions", [])}
        for i, e in enumerate(self.submissions):
            d = disk.get(_entry_key(e))
            if d is not None and (d.job_id and not e.job_id or d.state == "ABANDONED"):
                self.submissions[i] = d

    def reserve_submission(
        self,
        fragment_id: int,
        task_id: str,
        backend: str,
        attempt_id: str,
        n_shots: int = DEFAULT_SHOTS_PER_FRAGMENT,
    ) -> int:
        """Consume one unit of quota BEFORE the provider call (conservative accounting).

        Atomic with respect to other processes: the budget and duplicate checks are
        re-evaluated under the ledger lock.  Returns the total reserved+accepted count.
        """
        if not attempt_id:
            raise CampaignAuthorizationError("reserve_submission requires a non-empty attempt_id")
        with self._ledger_lock():
            self._refresh_from_disk()
            self.validate_submission(fragment_id=fragment_id, backend=backend, n_shots=n_shots)
            self.submissions.append(QuantumSubmissionEntry(
                fragment_id=fragment_id, task_id=task_id, job_id="", backend=backend,
                attempt_id=attempt_id, n_shots=n_shots, state="RESERVED",
            ))
            if self.ledger_path is not None:
                self._write_unlocked(self.ledger_path)
            return len(self.submissions)

    def confirm_submission(self, attempt_id: str, job_id: str) -> int:
        """Attach the provider Job ID to a RESERVED entry (RESERVED → ACCEPTED)."""
        if not job_id:
            raise CampaignAuthorizationError("confirm_submission requires a non-empty job_id")
        self._guard_job_id(job_id)
        with self._ledger_lock():
            self._refresh_from_disk()
            for entry in self.submissions:
                if entry.attempt_id == attempt_id:
                    if entry.job_id and entry.job_id != job_id:
                        raise CampaignAuthorizationError(
                            f"attempt {attempt_id!r} already carries job {entry.job_id!r}"
                        )
                    entry.job_id = job_id
                    entry.state = "ACCEPTED"
                    break
            else:
                raise CampaignAuthorizationError(f"No reservation for attempt {attempt_id!r}")
            if self.ledger_path is not None:
                self._write_unlocked(self.ledger_path)
            return len(self.submissions)

    def release_reservation(self, attempt_id: str) -> int:
        """Remove a RESERVED entry for which the provider was provably never contacted."""
        with self._ledger_lock():
            self._refresh_from_disk()
            keep = [e for e in self.submissions
                    if not (e.attempt_id == attempt_id and e.state == "RESERVED" and not e.job_id)]
            self.submissions[:] = keep
            if self.ledger_path is not None:
                self._write_unlocked(self.ledger_path, authoritative_remove=attempt_id)
            return len(self.submissions)

    def abandon_reservation(self, fragment_id: int, reason: str = "") -> int:
        """Operator action: mark an unresolved reservation ABANDONED.

        The quota stays consumed (conservative); the fragment becomes resubmittable.
        """
        with self._ledger_lock():
            self._refresh_from_disk()
            hit = False
            for entry in self.submissions:
                if entry.fragment_id == fragment_id and entry.state == "RESERVED" and not entry.job_id:
                    entry.state = "ABANDONED"
                    hit = True
            if not hit:
                raise CampaignAuthorizationError(
                    f"No unresolved reservation for fragment {fragment_id} to abandon"
                )
            if self.ledger_path is not None:
                self._write_unlocked(self.ledger_path)
            return len(self.submissions)

    def unresolved_reservations(self) -> List[QuantumSubmissionEntry]:
        return [e for e in self.submissions if e.state == "RESERVED" and not e.job_id]

    def record_submission(
        self,
        fragment_id: int,
        task_id: str,
        job_id: str,
        backend: str,
        attempt_id: str = "",
        n_shots: int = DEFAULT_SHOTS_PER_FRAGMENT,
    ) -> int:
        """Record an accepted QPU submission and persist the ledger atomically.

        If a reservation with the same ``attempt_id`` exists it is confirmed rather
        than duplicated.  Returns total submissions count.
        """
        self._guard_job_id(job_id)
        if attempt_id and any(e.attempt_id == attempt_id for e in self.submissions):
            return self.confirm_submission(attempt_id, job_id)
        with self._ledger_lock():
            self._refresh_from_disk()
            existing = self.get_submission_for_fragment(fragment_id)
            if existing is not None and existing.job_id == job_id:
                return len(self.submissions)
            self.submissions.append(QuantumSubmissionEntry(
                fragment_id=fragment_id, task_id=task_id, job_id=job_id, backend=backend,
                attempt_id=attempt_id, n_shots=n_shots,
            ))
            if self.ledger_path is not None:
                self._write_unlocked(self.ledger_path)
            return len(self.submissions)

    def as_dict(self) -> Dict[str, Any]:
        """Convert authorization to JSON-serializable dictionary."""
        d: Dict[str, Any] = {
            "schema": SCHEMA_CAMPAIGN_AUTH,
            "run_id": self.run_id,
            "authorized_backend": self.authorized_backend,
            "authorized_fragments": list(self.authorized_fragments),
            "shots_per_fragment": self.shots_per_fragment,
            "max_submissions": self.max_submissions,
            "manifest_digest": self.manifest_digest,
            "authorized_by": self.authorized_by,
            "created_utc": self.created_utc,
            "submissions": [s.as_dict() for s in self.submissions],
        }
        # Additive keys are written only when set, so legacy authorizations serialise exactly as before.
        if self.campaign_id:
            d["campaign_id"] = self.campaign_id
        if self.trial_profile:
            d["trial_profile"] = self.trial_profile
        if self.historical_accounting:
            d["historical_accounting"] = dict(self.historical_accounting)
        if self.forbidden_job_ids:
            d["forbidden_job_ids"] = sorted(self.forbidden_job_ids)
        return d

    def _write_unlocked(self, target_path: Path, authoritative_remove: str = "") -> None:
        """Atomic write that never drops on-disk submissions this instance has not seen."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.is_file():
            try:
                disk = json.loads(target_path.read_text(encoding="utf-8")).get("submissions", [])
            except (OSError, ValueError):
                disk = []
            known = {_entry_key(e) for e in self.submissions}
            for raw in disk:
                entry = QuantumSubmissionEntry.from_dict(raw)
                if authoritative_remove and entry.attempt_id == authoritative_remove:
                    continue
                if _entry_key(entry) not in known:
                    self.submissions.append(entry)
        data = self.as_dict()
        fd, tmp_name = tempfile.mkstemp(prefix=target_path.name + ".", suffix=".tmp", dir=str(target_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(data, indent=2))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        self.ledger_path = target_path

    def save(self, path: Optional[Path] = None) -> Path:
        """Save campaign authorization record atomically (locked, merging with on-disk entries)."""
        target_path = Path(path) if path is not None else self.ledger_path
        if target_path is None:
            raise CampaignAuthorizationError("Cannot save CampaignAuthorization: no ledger_path specified.")
        if self.ledger_path is None:
            self.ledger_path = target_path
        self.ledger_path = target_path
        with self._ledger_lock():
            self._write_unlocked(target_path)
        return target_path

    @classmethod
    def load(cls, path: Path) -> CampaignAuthorization:
        """Load campaign authorization record from disk."""
        target_path = Path(path)
        if not target_path.is_file():
            raise FileNotFoundError(f"Campaign authorization file not found: {target_path}")

        data = json.loads(target_path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA_CAMPAIGN_AUTH:
            raise CampaignAuthorizationError(
                f"Unsupported campaign authorization schema: {data.get('schema')!r}"
            )

        submissions = [QuantumSubmissionEntry.from_dict(s) for s in data.get("submissions", [])]
        return cls(
            run_id=str(data["run_id"]),
            authorized_backend=str(data["authorized_backend"]),
            authorized_fragments=tuple(data.get("authorized_fragments", FROZEN_QUANTUM_FRAGMENT_IDS)),
            shots_per_fragment=int(data.get("shots_per_fragment", DEFAULT_SHOTS_PER_FRAGMENT)),
            max_submissions=int(data.get("max_submissions", DEFAULT_MAX_SUBMISSIONS)),
            manifest_digest=str(data.get("manifest_digest", "")),
            authorized_by=str(data.get("authorized_by", "organizer")),
            created_utc=str(data.get("created_utc", "")),
            submissions=submissions,
            ledger_path=target_path,
            campaign_id=str(data.get("campaign_id", "")),
            trial_profile=str(data.get("trial_profile", "")),
            historical_accounting=dict(data.get("historical_accounting") or {}),
            forbidden_job_ids=tuple(data.get("forbidden_job_ids") or ()),
        )


def create_organizer_campaign_authorization(
    run_id: str,
    authorized_backend: str,
    *,
    manifest_digest: str = "",
    authorized_fragments: Optional[Sequence[int]] = None,
    shots_per_fragment: int = DEFAULT_SHOTS_PER_FRAGMENT,
    max_submissions: Optional[int] = None,
    ledger_path: Optional[Path] = None,
    authorized_by: str = "organizer",
    campaign_id: str = "",
    trial_profile: str = "",
    historical_accounting: Optional[Dict[str, Any]] = None,
    forbidden_job_ids: Sequence[str] = (),
) -> CampaignAuthorization:
    """Create a new CampaignAuthorization instance for an organizer run.

    Supports arbitrary manifest-defined quantum fragment subsets. When
    ``authorized_fragments`` is provided without an explicit ``max_submissions``,
    the quota defaults to ``len(authorized_fragments)``.
    """
    frags = tuple(authorized_fragments) if authorized_fragments is not None else FROZEN_QUANTUM_FRAGMENT_IDS
    max_sub = max_submissions if max_submissions is not None else (len(frags) if authorized_fragments is not None else DEFAULT_MAX_SUBMISSIONS)
    auth = CampaignAuthorization(
        run_id=run_id,
        authorized_backend=authorized_backend,
        authorized_fragments=frags,
        shots_per_fragment=shots_per_fragment,
        max_submissions=max_sub,
        manifest_digest=manifest_digest,
        authorized_by=authorized_by,
        ledger_path=ledger_path,
        campaign_id=campaign_id,
        trial_profile=trial_profile,
        historical_accounting=dict(historical_accounting or {}),
        forbidden_job_ids=tuple(forbidden_job_ids),
    )
    if ledger_path is not None:
        auth.save(ledger_path)
    return auth
