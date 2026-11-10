"""Run a job once per tag.

``run_once(tag, sampler, pubs)`` submits ``sampler.run(pubs)`` the first
time a tag is used and keeps ``{tag: job_id}`` in ``job_cache/jobs.json``;
with a known tag it returns the earlier job instead of submitting a new
one. Re-running a notebook cell, or the whole notebook after a kernel
restart, therefore spends no QPU time twice. A known tag whose job ended
in a terminal failure (status ERROR, FAILED or CANCELLED) is submitted
again. ``forget(tag)`` drops a tag.

This is the tag cache of the prototype event client (``event.Campaign.
run_once``) without a team, a ledger or a job anchor: the cache directory
is the only state. Each submission also leaves a record in
``job_cache/job_records.json`` (tag, time, backend and a fingerprint of the
PUBs), so that a known tag used with other PUBs prints a warning.

The job of a known tag is taken from memory when this kernel submitted or
fetched it, otherwise from the IBM Quantum service of the sampler
(``sampler.backend().service``, the account the job ran on).
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = "job_cache"
JOBS_FILE = "jobs.json"              # {tag: job_id}
RECORDS_FILE = "job_records.json"    # {job_id: {tag, submitted, backend, pubs, ...}}
FAILED_STATES = ("ERROR", "FAILED", "CANCELLED")

_IN_MEMORY: dict = {}                # (cache directory, tag) -> job of this kernel


# ------------------------------------------------------------------ helpers
def _job_id(job) -> str:
    jid = getattr(job, "job_id", None)
    return str(jid() if callable(jid) else job)


def _failed_status(job):
    """``(status, reason)`` when ``job`` ended in a terminal failure
    (``FAILED_STATES``; ``reason`` is the first line of
    ``job.error_message()``, cut at about 200 characters, or ""), else None.
    None as well when the status cannot be read."""
    try:
        status = job.status()
    except Exception:                            # noqa: BLE001 - network, fakes
        return None
    status = str(getattr(status, "name", status)).upper()
    if status not in FAILED_STATES:
        return None
    reason = ""
    get = getattr(job, "error_message", None)
    if callable(get):
        try:
            reason = get() or ""
        except Exception:                        # noqa: BLE001
            reason = ""
    lines = [ln.strip() for ln in str(reason).splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    if len(first) > 200:
        return status, first[:200].rstrip() + "..."
    return status, first.rstrip(".")


def _probe_value(i: int) -> float:
    """A fixed, generic angle for parameter i (sorted by name): a circuit
    bound at these values identifies its parametrized structure."""
    return 0.1 + 0.6180339887498949 * (i + 1)


def pubs_fingerprint(pubs, shots=None) -> list | None:
    """What a ``sampler.run(pubs, shots=shots)`` call submits, per PUB:
    ``circuit`` (the canonical fingerprint of the circuit, its free
    parameters bound at fixed probe values), ``values`` (sha256 of the
    parameter names and values, rounded to 1e-9; None without parameters)
    and ``shots`` (the PUB's, else ``shots``). None if the PUBs cannot be
    read."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.primitives.containers.sampler_pub import SamplerPub

        from .fingerprint import circuit_fingerprint

        single = isinstance(pubs, (QuantumCircuit, SamplerPub)) or (
            isinstance(pubs, tuple) and pubs and isinstance(pubs[0], QuantumCircuit))
        out = []
        for item in ([pubs] if single else list(pubs)):
            pub = SamplerPub.coerce(tuple(item) if isinstance(item, list) else item)
            qc = pub.circuit
            params = sorted(qc.parameters, key=lambda p: p.name)
            values = None
            if params:
                qc = qc.assign_parameters({p: _probe_value(i)
                                           for i, p in enumerate(params)})
                arr = np.round(np.asarray(pub.parameter_values.as_array(params),
                                          dtype=float), 9) + 0.0
                h = hashlib.sha256(json.dumps([p.name for p in params]).encode())
                h.update(str(arr.shape).encode())
                h.update(arr.tobytes())
                values = h.hexdigest()
            n_shots = pub.shots if pub.shots is not None else shots
            out.append({"circuit": circuit_fingerprint(qc), "values": values,
                        "shots": None if n_shots is None else int(n_shots)})
        return out
    except Exception:                            # noqa: BLE001 - never block a run
        return None


def _same_submission(before: list, now: list) -> bool:
    """Two :func:`pubs_fingerprint` lists describe the same submission: the
    same circuits and parameter values PUB by PUB, and the same shots
    wherever both calls give them (a call that leaves the shots out, only to
    get the job back, is not a change)."""
    if len(before) != len(now):
        return False
    for b, n in zip(before, now, strict=True):
        if b.get("circuit") != n.get("circuit") or b.get("values") != n.get("values"):
            return False
        if None not in (b.get("shots"), n.get("shots")) and b["shots"] != n["shots"]:
            return False
    return True


def _read_json(path: Path) -> dict:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _write_json(path: Path, d: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1) + "\n")
    tmp.replace(path)


def _dir(cache_dir) -> Path:
    return Path(DEFAULT_CACHE_DIR if cache_dir is None else cache_dir)


def _memory_key(cache_dir, tag: str) -> tuple:
    return (str(_dir(cache_dir).resolve()), str(tag))


def _forget_hint(tag: str, cache_dir) -> str:
    extra = "" if str(cache_dir) == DEFAULT_CACHE_DIR else f", cache_dir={str(cache_dir)!r}"
    return f'jobs.forget("{tag}"{extra})'


def _backend_name(sampler) -> str | None:
    get = getattr(sampler, "backend", None)
    try:
        backend = get() if callable(get) else get
    except Exception:                            # noqa: BLE001
        return None
    name = getattr(backend, "name", None)
    name = name() if callable(name) else name
    return None if name is None else str(name)


def _service_of(sampler):
    """The service the sampler submits to (``Sampler._service``, else
    ``sampler.backend().service``), or None."""
    service = getattr(sampler, "_service", None)
    if service is None:
        get = getattr(sampler, "backend", None)
        try:
            backend = get() if callable(get) else get
        except Exception:                        # noqa: BLE001
            backend = None
        service = getattr(backend, "service", None)
    return service if callable(getattr(service, "job", None)) else None


# --------------------------------------------------------------- public API
def known_jobs(cache_dir=DEFAULT_CACHE_DIR) -> dict:
    """``{tag: job_id}`` of the cache directory."""
    return {str(tag): str(v["job_id"] if isinstance(v, dict) else v)
            for tag, v in _read_json(_dir(cache_dir) / JOBS_FILE).items()}


def run_once(tag: str, sampler, pubs, shots: int | None = None,
             cache_dir=DEFAULT_CACHE_DIR):
    """Submit ``sampler.run(pubs, shots=shots)`` the first time ``tag`` is
    used, keep ``{tag: job_id}`` in ``<cache_dir>/jobs.json`` and return the
    job. With a known tag nothing is submitted: the earlier job is returned
    (from memory, or fetched from the sampler's service with
    ``service.job(job_id)``) and one line says so.

    A job is recognized by its tag only. The cache keeps a fingerprint of
    what was submitted (per PUB the circuit, the parameter values and the
    shots); when a known tag is used with other PUBs, the earlier job is
    still returned and the line says that it ran different circuits,
    parameter values or shots (a re-transpilation after a recalibration can
    change the circuit slightly): use a new tag for new PUBs.

    A known tag whose job ended in a terminal failure (status ERROR, FAILED
    or CANCELLED) is submitted again: one line gives the status and the
    first line of ``job.error_message()``, the PUBs passed now are
    submitted as for a new tag, the tag then holds the new job, and its
    record lists the failed job ids under ``earlier_attempts``. When the
    status cannot be read, the known job is returned.

    ``shots``: passed to ``sampler.run`` when given (PUBs may carry their
    own). ``cache_dir``: the directory of the cache, created on the first
    submission (relative to the working directory)."""
    tag = str(tag)
    if not tag:
        raise ValueError("run_once needs a non-empty tag, such as \"baseline\"")
    known = known_jobs(cache_dir)
    if tag not in known:
        return _submit_new(tag, sampler, pubs, shots, cache_dir)
    job_id = known[tag]
    job = _IN_MEMORY.get(_memory_key(cache_dir, tag))
    if job is None or _job_id(job) != job_id:
        job = _fetch(tag, job_id, sampler, cache_dir)
    failed = _failed_status(job)
    if failed is not None:
        status, reason = failed
        head = (f"job {job_id} (tag {tag}) ended with status {status}"
                + (f": {reason}" if reason else ""))
        print(head + ("" if head.endswith(".") else ".")
              + " Submitting the PUBs again under the same tag.")
        return _submit_new(tag, sampler, pubs, shots, cache_dir, failed_id=job_id)
    rec = _read_json(_dir(cache_dir) / RECORDS_FILE).get(job_id)
    before = rec.get("pubs") if isinstance(rec, dict) else None
    now = pubs_fingerprint(pubs, shots) if before else None
    hint = _forget_hint(tag, cache_dir)
    if before and now is not None and not _same_submission(before, now):
        print(f"reusing job {job_id} (tag {tag}), but that job ran different "
              "circuits, parameter values or shots than the PUBs passed now: "
              f"use a new tag, or {hint} to submit again")
    else:
        print(f"reusing job {job_id} (tag {tag}); {hint} to submit again")
    return job


def _fetch(tag: str, job_id: str, sampler, cache_dir):
    """The job of a known tag from the sampler's service."""
    service = _service_of(sampler)
    hint = _forget_hint(tag, cache_dir)
    if service is None:
        raise RuntimeError(
            f"tag {tag!r} holds job {job_id}, but the sampler has no IBM Quantum "
            "service to fetch it from: pass a Sampler made with mode=backend, "
            f"or {hint} to submit again")
    try:
        job = service.job(job_id)
    except Exception as exc:                     # noqa: BLE001 - network, other account
        msg = " ".join(str(exc).split())[:200]
        raise RuntimeError(
            f"tag {tag!r} holds job {job_id}, which the IBM Quantum service does "
            f"not return ({type(exc).__name__}: {msg}). If the job belongs to "
            f"another account, {hint} to submit a new one") from exc
    _IN_MEMORY[_memory_key(cache_dir, tag)] = job
    return job


def _submit_new(tag: str, sampler, pubs, shots, cache_dir,
                failed_id: str | None = None):
    """``sampler.run(pubs[, shots=shots])``; the tag then holds the new job
    (``jobs.json``) and its record is written (``job_records.json``; with
    ``failed_id``, the failed job it replaces is appended to
    ``earlier_attempts``)."""
    job = sampler.run(pubs) if shots is None else sampler.run(pubs, shots=shots)
    job_id = _job_id(job)
    folder = _dir(cache_dir)
    known = known_jobs(cache_dir)
    known[tag] = job_id
    _write_json(folder / JOBS_FILE, known)
    record = {"tag": tag,
              "submitted": datetime.datetime.now().isoformat(timespec="seconds"),
              "backend": _backend_name(sampler),
              "pubs": pubs_fingerprint(pubs, shots)}
    records = _read_json(folder / RECORDS_FILE)
    if failed_id is not None:
        old = records.get(failed_id)
        earlier = old.get("earlier_attempts") if isinstance(old, dict) else None
        record["earlier_attempts"] = list(earlier or []) + [failed_id]
    records[job_id] = record
    _write_json(folder / RECORDS_FILE, records)
    _IN_MEMORY[_memory_key(cache_dir, tag)] = job
    print(f"submitted {job_id} (tag {tag})")
    return job


def forget(tag: str, cache_dir=DEFAULT_CACHE_DIR) -> None:
    """Remove ``tag`` from ``<cache_dir>/jobs.json``: the next
    :func:`run_once` with this tag submits a new job. The job itself is not
    touched."""
    tag = str(tag)
    known = known_jobs(cache_dir)
    job_id = known.pop(tag, None)
    _IN_MEMORY.pop(_memory_key(cache_dir, tag), None)
    if job_id is None:
        print(f"tag {tag} is not known")
        return
    _write_json(_dir(cache_dir) / JOBS_FILE, known)
    print(f"forgot tag {tag} (job {job_id})")
