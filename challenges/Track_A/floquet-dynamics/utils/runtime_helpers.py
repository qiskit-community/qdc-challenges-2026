"""Restart-safe runtime bookkeeping for the optional Floquet hardware track.

Only a job ID and plain run settings are saved. The caller creates the Runtime
service and passes its submit/retrieve callables; importing this module never
connects to a backend. After ``run_program`` returns, the caller may wait for
``job.result()`` and then record the measured QPU charge with
``charge_qpu_time``.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar


DEFAULT_LEDGER_PATH = Path("runs/walkthrough_jobs.json")
QPU_CAP_SECONDS = 1800.0
_VERSION = 1
_SENSITIVE_NAMES = ("token", "secret", "password", "credential", "api_key", "apikey", "auth")
_Job = TypeVar("_Job")


class PendingSubmissionError(RuntimeError):
    """A prior submission may exist but its job ID has not been recorded."""


class PendingChargeError(RuntimeError):
    """A previous job's QPU use must be recorded before another submission."""


class BudgetExceededError(RuntimeError):
    """The measured QPU use has reached the challenge's 1,800-second cap."""


def _normalise_settings(settings: Mapping) -> dict:
    """Validate plain finite settings and return their canonical JSON form.

    Reject non-string keys, credential-like field names and values that cannot
    be encoded as finite JSON. This keeps ledger identities reproducible.
    """
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping of plain JSON values")

    def check_names(value) -> None:
        """Recursively reject non-string or credential-like mapping keys."""
        if isinstance(value, Mapping):
            for name, nested in value.items():
                if not isinstance(name, str):
                    raise TypeError("settings keys must be strings")
                if any(word in name.casefold() for word in _SENSITIVE_NAMES):
                    raise ValueError(f"settings field {name!r} may contain a credential")
                check_names(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                check_names(nested)

    check_names(settings)
    try:
        canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("settings must contain only finite, plain JSON values") from exc
    return json.loads(canonical)


def _identity(backend_name: str, settings: Mapping, seed: int | None) -> dict:
    """Build the validated backend/settings/seed identity for one recorded job."""
    if not isinstance(backend_name, str) or not backend_name.strip():
        raise ValueError("backend_name must be a nonempty string")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    return {
        "backend_name": backend_name,
        "settings": _normalise_settings(settings),
        "seed": seed,
    }


def _job_id(job) -> str:
    """Read a nonempty job ID from either a method or an attribute."""
    identifier = getattr(job, "job_id", None)
    if callable(identifier):
        identifier = identifier()
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("submitted job does not expose a nonempty job_id")
    return identifier


def _read_ledger(path: Path) -> dict:
    """Load and validate the ledger envelope, or return an empty ledger.

    Missing files represent fresh ledgers; malformed JSON and unsupported
    versions raise ValueError before a caller can submit or charge a job.
    """
    if not path.exists():
        return {"version": _VERSION, "jobs": {}}
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read job ledger {path}; inspect it before submitting") from exc
    if (
        not isinstance(ledger, dict)
        or ledger.get("version") != _VERSION
        or not isinstance(ledger.get("jobs"), dict)
    ):
        raise ValueError(f"Unsupported job ledger format in {path}")
    return ledger


def _write_ledger(path: Path, ledger: dict) -> None:
    """Replace the ledger atomically; do not leave a partial JSON document."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(ledger, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _locked_ledger(path: Path):
    """Yield the ledger while holding its exclusive sibling-file lock.

    Create parent directories as needed and release the lock on every exit.
    Callers perform atomic ledger updates while this context remains active.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield _read_ledger(path)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def initialise_ledger(
    ledger_path: str | Path = DEFAULT_LEDGER_PATH, *, resume: bool = False,
) -> Path:
    """Create an independent ledger, or validate and resume the specified file.

    Fresh ledgers live in unique sibling directories; the supplied file is never
    overwritten or read in fresh mode. Keep the returned path for all submissions
    and charges in this execution, and supply it with ``resume=True`` to recover
    that execution later. This records job IDs and settings, not result arrays.
    """
    path = Path(ledger_path).expanduser().resolve()
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"Cannot resume: job ledger {path} does not exist")
        with _locked_ledger(path) as ledger:
            _spent_seconds(ledger)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="walkthrough_", dir=path.parent))
    fresh_path = directory / "walkthrough_jobs.json"
    _write_ledger(fresh_path, {"version": _VERSION, "jobs": {}})
    return fresh_path


def saved_run_settings(
    key: str, *, backend_name: str, ledger_path: str | Path = DEFAULT_LEDGER_PATH,
) -> dict | None:
    """Read a run's original settings, for example to restore its physical layout.

    ``run_program`` still checks the complete identity before retrieving the job.
    A ledger for a different backend is rejected before compiling or submitting.
    """
    path = Path(ledger_path)
    if not path.is_file():
        raise FileNotFoundError(f"Job ledger {path} does not exist; initialise it first")
    with _locked_ledger(path) as ledger:
        entry = ledger["jobs"].get(key)
        if entry is None:
            return None
        if not isinstance(entry, dict) or not isinstance(entry.get("settings"), dict):
            raise ValueError(f"Invalid job ledger entry for {key!r}")
        if entry.get("backend_name") != backend_name:
            raise ValueError(
                f"Run {key!r} belongs to backend {entry.get('backend_name')!r}, "
                f"not {backend_name!r}; choose the original backend or a fresh ledger"
            )
        return _normalise_settings(entry["settings"])


def prepare_isa_circuits(
    label: str, build: Callable[[], list], *, ledger_path: str | Path,
) -> list:
    """Keep compiled Sampler circuits beside the ledger for faithful recovery.

    Readout correction needs the physical measurement mapping of the original
    circuits. Recompiling against a later calibration can change that mapping.
    Legacy ledgers without QPY files compile once with the current backend.
    """
    from qiskit import qpy

    if not label or Path(label).name != label:
        raise ValueError("circuit label must be a nonempty filename component")
    path = Path(ledger_path).parent / f"{label}.qpy"
    if path.is_file():
        with path.open("rb") as stream:
            return qpy.load(stream)
    circuits = list(build())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            qpy.dump(circuits, stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return circuits


def _spent_seconds(ledger: dict) -> float:
    """Sum measured charges once per job ID, rejecting inconsistent entries.

    Uncharged jobs contribute no seconds. Invalid IDs, negative/non-finite
    charges or conflicting charges for the same job raise ValueError.
    """
    charges: dict[str, float] = {}
    for key, entry in ledger["jobs"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid job ledger entry for {key!r}")
        charge = entry.get("qpu_seconds")
        if charge is None:
            continue
        job_id = entry.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"Charged entry {key!r} has no job ID")
        if isinstance(charge, bool) or not isinstance(charge, (int, float)):
            raise ValueError(f"Invalid QPU charge for {key!r}")
        charge = float(charge)
        if not math.isfinite(charge) or charge < 0:
            raise ValueError(f"Invalid QPU charge for {key!r}")
        if job_id in charges and not math.isclose(charges[job_id], charge, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(f"Conflicting QPU charges for job {job_id}")
        charges[job_id] = charge
    return float(sum(charges.values()))


def qpu_time_spent(ledger_path: str | Path = DEFAULT_LEDGER_PATH) -> float:
    """Return actual QPU seconds, counting each recorded job ID once."""
    path = Path(ledger_path)
    if not path.exists():
        return 0.0
    with _locked_ledger(path) as ledger:
        return _spent_seconds(ledger)


def run_program(
    key: str,
    *,
    backend_name: str,
    settings: Mapping,
    seed: int | None,
    submit: Callable[[], _Job],
    retrieve: Callable[[str], _Job],
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    estimated_qpu_seconds: float | None = None,
    allow_submit: bool = True,
) -> _Job:
    """Submit once or retrieve the job recorded for exactly these run settings.

    The new job ID is saved before this function returns and before the caller
    waits for results. A marker is saved *before* submission; if submission is
    interrupted without a job ID, a rerun refuses to guess whether it reached
    IBM. Resolve that marker against the service's job list before trying again.
    Retrieval failures also never trigger a fresh submission.

    Before starting a different job, record the previous job's measured QPU time
    with ``charge_qpu_time``. Fresh submission stops when the recorded total has
    reached 1,800 seconds. Since actual charge is known only after execution,
    callers can pass an estimate for the next job. The estimate applies only to
    a fresh submission, so an existing job remains retrievable at the cap.
    Set ``allow_submit=False`` for retrieval only: missing jobs raise before
    saving a submission marker or calling the backend.
    """
    if not isinstance(key, str) or not key.strip():
        raise ValueError("key must be a nonempty string")
    if not callable(submit) or not callable(retrieve):
        raise TypeError("submit and retrieve must be callables")
    identity = _identity(backend_name, settings, seed)
    path = Path(ledger_path)

    with _locked_ledger(path) as ledger:
        entries = ledger["jobs"]
        if key in entries:
            entry = entries[key]
            if not isinstance(entry, dict) or any(entry.get(field) != value for field, value in identity.items()):
                raise ValueError(
                    f"Run key {key!r} already belongs to different backend, settings, or seed "
                    f"in {path}. Restore the original configuration to resume, "
                    "or initialise a fresh ledger for a new experiment."
                )
            job_id = entry.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise PendingSubmissionError(
                    f"Run {key!r} may already have been submitted without a saved job ID; "
                    "inspect the service job list before retrying"
                )
        else:
            if not allow_submit:
                raise RuntimeError(
                    f"No job recorded for {key!r} in {path}; new submissions are disabled"
                )
            spent = _spent_seconds(ledger)
            uncharged = [name for name, entry in entries.items() if entry.get("qpu_seconds") is None]
            if uncharged:
                raise PendingChargeError(
                    f"Record the QPU charge for {uncharged[0]!r} before submitting another job"
                )
            if spent >= QPU_CAP_SECONDS:
                raise BudgetExceededError("Recorded QPU time has reached the 1,800-second cap")
            if estimated_qpu_seconds is not None:
                if (
                    isinstance(estimated_qpu_seconds, bool)
                    or not isinstance(estimated_qpu_seconds, (int, float))
                    or not math.isfinite(estimated_qpu_seconds)
                    or estimated_qpu_seconds < 0
                ):
                    raise ValueError("estimated_qpu_seconds must be finite and nonnegative")
                if spent + estimated_qpu_seconds > QPU_CAP_SECONDS:
                    raise BudgetExceededError(
                        "Estimated job exceeds the remaining 1,800-second QPU budget"
                    )

            entry = {**identity, "job_id": None, "qpu_seconds": None}
            entries[key] = entry
            _write_ledger(path, ledger)
            job = submit()
            job_id = _job_id(job)
            if any(other.get("job_id") == job_id for name, other in entries.items() if name != key):
                raise ValueError(f"Job ID {job_id!r} is already recorded under another key")
            entry["job_id"] = job_id
            try:
                _write_ledger(path, ledger)
            except OSError as exc:
                raise RuntimeError(
                    f"Job {job_id} was submitted, but its ID could not be saved to {path}"
                ) from exc
            return job

    return retrieve(job_id)


def attach_job_id(key: str, job_id: str, *, ledger_path: str | Path = DEFAULT_LEDGER_PATH) -> None:
    """Bind a verified service job ID to an interrupted submission marker."""
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a nonempty string")
    path = Path(ledger_path)
    with _locked_ledger(path) as ledger:
        entries = ledger["jobs"]
        if key not in entries:
            raise KeyError(f"No run marker for {key!r}")
        entry = entries[key]
        if entry.get("job_id") is not None:
            if entry["job_id"] == job_id:
                return
            raise ValueError(f"Run {key!r} already has a different job ID")
        if any(other.get("job_id") == job_id for name, other in entries.items() if name != key):
            raise ValueError(f"Job ID {job_id!r} is already recorded under another key")
        entry["job_id"] = job_id
        _write_ledger(path, ledger)


def charge_qpu_time(
    key: str,
    *,
    job_id: str,
    qpu_seconds: float,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    cap_seconds: float = QPU_CAP_SECONDS,
) -> float:
    """Record measured QPU use once and return the cumulative actual seconds.

    An over-cap charge is still saved: the QPU time was already consumed. The
    caller can compare the returned total with ``cap_seconds`` when scoring, and
    ``run_program`` will refuse further fresh submissions at the fixed cap.
    """
    if isinstance(qpu_seconds, bool) or not isinstance(qpu_seconds, (int, float)):
        raise ValueError("qpu_seconds must be finite and nonnegative")
    charge = float(qpu_seconds)
    if not math.isfinite(charge) or charge < 0:
        raise ValueError("qpu_seconds must be finite and nonnegative")
    if not math.isfinite(cap_seconds) or cap_seconds <= 0:
        raise ValueError("cap_seconds must be finite and positive")
    path = Path(ledger_path)
    with _locked_ledger(path) as ledger:
        if key not in ledger["jobs"]:
            raise KeyError(f"No job recorded for {key!r}")
        entry = ledger["jobs"][key]
        if entry.get("job_id") != job_id:
            raise ValueError(f"Job ID does not match run {key!r}")
        previous = entry.get("qpu_seconds")
        if previous is not None:
            if not math.isclose(float(previous), charge, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f"Run {key!r} already has a different QPU charge")
            return _spent_seconds(ledger)
        entry["qpu_seconds"] = charge
        _write_ledger(path, ledger)
        return _spent_seconds(ledger)
