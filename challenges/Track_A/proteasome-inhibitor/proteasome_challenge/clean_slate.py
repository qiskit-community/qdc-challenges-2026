"""
clean_slate.py — fresh-state guarantees for a live hardware trial (Task 5.9E, section 12).

Everything here is organizer-side bookkeeping.  Nothing contacts Serverless or a QPU, nothing recomputes
science, and the legacy participant QPU ledger (``checkpoints/.qpu_sampling_record.json*``) is *never*
modified: it is only read to produce an immutable historical-accounting snapshot.

Capabilities
------------
``inventory_active_state``      exact, read-only inventory of prior execution state (A)
``retire_active_state``         move run-specific state to ``_backup_staging/retired_<stamp>/`` with a SHA-256
                                manifest (B).  Dry run by default; refuses while possibly-running remote jobs
                                are unacknowledged (G).  Never deletes, never touches the ledger.
``historical_accounting_snapshot`` / ``write_historical_snapshot``   separate lifetime history from the
                                campaign allocation (C)
``historical_deny_list``        job / run identifiers from archives and active state that may never be
                                attributed to a new run (F)
``new_trial_run_id``            unique RUN_ID (D)
``assert_clean_slate``          the ten acceptance criteria H1-H10 for one run (H)
``guard_*``                     contamination guards (F)
``clear_notebook_outputs``      strip outputs / execution counts, keep code and cell IDs (E)
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import secrets
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "CleanSlateError",
    "PROTECTED_PREFIX",
    "RETAINED_CHECKPOINT_ENTRIES",
    "new_trial_run_id",
    "inventory_active_state",
    "retire_active_state",
    "historical_accounting_snapshot",
    "write_historical_snapshot",
    "historical_deny_list",
    "guard_task_registry",
    "guard_job_ids",
    "guard_fragment_result",
    "guard_summary_belongs_to_run",
    "assert_clean_slate",
    "clear_notebook_outputs",
]

PROTECTED_PREFIX = ".qpu_sampling_record"          # the legacy ledger, its lock and any journal
# Everything in checkpoints/ except the legacy ledger is run-specific or derived: the deployment fingerprint
# is stale after any bundle change (it forces a fresh, explicit deploy) and the final selection is
# regenerated deterministically by the notebook.  The set is kept so a deployment may retain entries explicitly.
RETAINED_CHECKPOINT_ENTRIES: frozenset = frozenset()
_JOB_KEYS = frozenset({"job_id", "qpu_job_id", "serverless_job_id", "jobid"})
_RUN_KEYS = frozenset({"run_id", "run_name"})
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_PROVIDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{6,63}$")


class CleanSlateError(RuntimeError):
    """A clean-slate precondition is violated; nothing was changed."""


def _utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_protected(path: Path) -> bool:
    return path.name.startswith(PROTECTED_PREFIX)


# ─────────────────────────────────────────────────────────────────────────────
#  Identity
# ─────────────────────────────────────────────────────────────────────────────

def new_trial_run_id(prefix: str = "ixazomib_live", *, now: Optional[datetime.datetime] = None) -> str:
    """A unique, store-safe RUN_ID: ``<prefix>_<UTC stamp>_<8 hex>``.  Never derived from a previous id."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.\-]{0,63}", prefix or ""):
        raise CleanSlateError(f"unsafe run id prefix {prefix!r}")
    when = (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{when}_{secrets.token_hex(4)}"


# ─────────────────────────────────────────────────────────────────────────────
#  Legacy ledger: read-only
# ─────────────────────────────────────────────────────────────────────────────

def historical_accounting_snapshot(project_root: str | Path) -> dict[str, Any]:
    """Read-only snapshot of the lifetime participant ledger.  The ledger itself is never opened for writing."""
    ledger = Path(project_root) / "checkpoints" / ".qpu_sampling_record.json"
    snap: dict[str, Any] = {
        "schema": "historical_qpu_accounting/1",
        "ledger_path": str(ledger),
        "ledger_present": ledger.is_file(),
        "lifetime_count": None,
        "ledger_sha256": None,
        "ledger_size_bytes": None,
        "note": "Lifetime participant sampling history. It is NOT a campaign allocation and is never reset.",
    }
    if ledger.is_file():
        raw = ledger.read_bytes()
        snap["ledger_sha256"] = hashlib.sha256(raw).hexdigest()
        snap["ledger_size_bytes"] = len(raw)
        try:
            snap["lifetime_count"] = int(json.loads(raw.decode("utf-8")).get("count"))
        except (ValueError, TypeError, AttributeError):
            snap["lifetime_count"] = None
    return snap


def write_historical_snapshot(project_root: str | Path, dest_dir: Optional[str | Path] = None) -> Path:
    """Write the snapshot outside the active namespace (idempotent: the file name embeds the ledger digest)."""
    root = Path(project_root)
    snap = historical_accounting_snapshot(root)
    dest = Path(dest_dir) if dest_dir is not None else root / "_backup_staging" / "historical_accounting"
    dest.mkdir(parents=True, exist_ok=True)
    tag = (snap["ledger_sha256"] or "noledger")[:12]
    path = dest / f"historical_qpu_accounting_{tag}.json"
    if not path.exists():
        snap_out = dict(snap, written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        path.write_text(json.dumps(snap_out, indent=2, sort_keys=True), encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Historical identifiers (deny-list)
# ─────────────────────────────────────────────────────────────────────────────

def _walk_ids(obj: Any, jobs: set[str], runs: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if isinstance(v, str) and v:
                if lk in _JOB_KEYS and (_UUID.match(v) or _PROVIDER_ID.match(v)):
                    jobs.add(v)
                elif lk in _RUN_KEYS:
                    runs.add(v)
            else:
                _walk_ids(v, jobs, runs)
    elif isinstance(obj, list):
        for item in obj:
            _walk_ids(item, jobs, runs)


def _registry_ids(db: Path) -> tuple[set[str], set[str], dict[str, int], list[dict[str, Any]]]:
    """(qpu job ids, serverless job ids, state counts, task rows) read from a registry, read-only."""
    qpu: set[str] = set()
    sls: set[str] = set()
    states: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    # Read a private copy (database + any WAL/SHM): opening a WAL-mode database in place would create
    # sidecar files next to the original, and this inventory must modify nothing.
    tmpdir = tempfile.mkdtemp(prefix="cs_registry_")
    try:
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(db) + suffix)
            if src.is_file():
                shutil.copy2(src, Path(tmpdir) / (db.name + suffix))
        con = sqlite3.connect(str(Path(tmpdir) / db.name), timeout=5)
    except (sqlite3.Error, OSError):
        shutil.rmtree(tmpdir, ignore_errors=True)
        return qpu, sls, states, rows
    try:
        cur = con.execute(
            "SELECT run_id, task_id, fragment_id, state, qpu_job_id, serverless_job_id, backend FROM fragment_tasks")
        for run_id, task_id, fid, state, q, s, backend in cur.fetchall():
            states[state] = states.get(state, 0) + 1
            if q:
                qpu.add(q)
            if s:
                sls.add(s)
            rows.append({"run_id": run_id, "task_id": task_id, "fragment_id": fid, "state": state,
                         "qpu_job_id": q, "serverless_job_id": s, "backend": backend})
    except sqlite3.Error:
        pass
    finally:
        con.close()
        shutil.rmtree(tmpdir, ignore_errors=True)
    return qpu, sls, states, rows


def historical_deny_list(
    project_root: str | Path,
    *,
    include_active: bool = True,
    extra_job_ids: Iterable[str] = (),
) -> dict[str, list[str]]:
    """Every provider / Serverless Job ID and run id found in archives (and active state) -- read-only.

    The returned identifiers may never be attributed to a new run.  ``extra_job_ids`` lets the operator add
    identifiers known from outside the repository (e.g. a Serverless job started from a notebook).
    """
    root = Path(project_root)
    jobs: set[str] = {str(j) for j in extra_job_ids if j}
    runs: set[str] = set()
    # Retired state (moved out of checkpoints/ by retire_active_state) is history by definition: its job ids
    # must stay on the deny list after the move.
    scan_roots = [root / "archive", root / "_backup_staging"]
    if include_active:
        scan_roots.append(root / "checkpoints")
    for base in scan_roots:
        if not base.is_dir():
            continue
        for p in base.rglob("*.json"):
            if _is_protected(p) or p.stat().st_size > 20_000_000:
                continue
            try:
                _walk_ids(json.loads(p.read_text(encoding="utf-8")), jobs, runs)
            except (OSError, ValueError):
                continue
        for db in base.rglob("registry.db"):
            q, s, _, rows = _registry_ids(db)
            jobs |= q | s
            runs |= {r["run_id"] for r in rows if r["run_id"]}
    return {"job_ids": sorted(jobs), "run_ids": sorted(runs)}


# ─────────────────────────────────────────────────────────────────────────────
#  A. Inventory
# ─────────────────────────────────────────────────────────────────────────────

def _notebook_output_state(nb_path: Optional[Path]) -> dict[str, Any]:
    if nb_path is None or not Path(nb_path).is_file():
        return {"path": str(nb_path) if nb_path else None, "present": False}
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    with_out = [c.get("id") for c in cells if c.get("outputs")]
    with_cnt = [c.get("id") for c in cells if c.get("execution_count") is not None]
    return {"path": str(nb_path), "present": True, "code_cells": len(cells),
            "cells_with_outputs": with_out, "cells_with_execution_count": with_cnt,
            "clean": not with_out and not with_cnt}


def inventory_active_state(
    project_root: str | Path,
    *,
    notebook_path: Optional[str | Path] = None,
    known_remote_job_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Exact read-only inventory of prior execution state under ``checkpoints/`` (and the notebook outputs)."""
    root = Path(project_root)
    ckpt = root / "checkpoints"
    entries: list[dict[str, Any]] = []
    run_dirs: list[dict[str, Any]] = []
    remote_jobs: set[str] = {str(j) for j in known_remote_job_ids if j}
    campaigns: list[dict[str, Any]] = []
    summaries: list[str] = []
    transports: set[str] = set()

    if ckpt.is_dir():
        for child in sorted(ckpt.iterdir()):
            if _is_protected(child):
                kind = "protected_ledger"
            elif child.name in RETAINED_CHECKPOINT_ENTRIES:
                kind = "retained"
            else:
                kind = "run_specific"
            size = (sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                    if child.is_dir() else child.stat().st_size)
            entries.append({"name": child.name, "kind": kind, "is_dir": child.is_dir(), "bytes": size})

        for db in sorted(ckpt.rglob("registry.db")):
            q, s, states, rows = _registry_ids(db)
            run_dir = db.parent
            remote_jobs |= s
            for r in rows:
                if r["serverless_job_id"] and r["state"] not in ("COMPLETE", "FAILED", "CANCELED"):
                    remote_jobs.add(r["serverless_job_id"])
            run_dirs.append({
                "run_dir": str(run_dir.relative_to(root)), "run_ids": sorted({r["run_id"] for r in rows}),
                "tasks": len(rows), "states": states,
                "complete_tasks": states.get("COMPLETE", 0),
                "qpu_job_ids": sorted(q), "serverless_job_ids": sorted(s),
                "manifest_frozen": (run_dir / "run_manifest.json").is_file(),
                "reconstruction_summary": (run_dir / "reconstruction_summary.json").is_file(),
                "stage4_ledger": (run_dir / "stage4_ledger.json").is_file(),
            })
            for name in ("reconstruction_summary.json", "stage4_ledger.json"):
                if (run_dir / name).is_file():
                    summaries.append(str((run_dir / name).relative_to(root)))
        for cp in sorted(ckpt.rglob("campaign_authorization.json")):
            try:
                d = json.loads(cp.read_text(encoding="utf-8"))
                campaigns.append({"path": str(cp.relative_to(root)), "run_id": d.get("run_id"),
                                  "campaign_id": d.get("campaign_id", ""),
                                  "submissions": len(d.get("submissions", [])),
                                  "job_ids": sorted(s.get("job_id") for s in d.get("submissions", []) if s.get("job_id"))})
            except (OSError, ValueError):
                campaigns.append({"path": str(cp.relative_to(root)), "error": "unreadable"})
        for jf in sorted(ckpt.rglob("*.json")):
            if _is_protected(jf) or jf.stat().st_size > 20_000_000:
                continue
            try:
                jobs: set[str] = set()
                _walk_ids(json.loads(jf.read_text(encoding="utf-8")), jobs, set())
                remote_jobs |= jobs
            except (OSError, ValueError):
                continue
        for ref in sorted(ckpt.rglob("*.ref.json")):
            transports.add(str(ref.relative_to(root)))

    return {
        "schema": "clean_slate_inventory/1",
        "project_root": str(root),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint_entries": entries,
        "run_dirs": run_dirs,
        "campaign_authorizations": campaigns,
        "reconstruction_artifacts": summaries,
        "published_references": sorted(transports),
        "possibly_running_remote_jobs": sorted(remote_jobs),
        "historical_ledger": historical_accounting_snapshot(root),
        "notebook": _notebook_output_state(Path(notebook_path) if notebook_path else None),
        "to_retire": [e["name"] for e in entries if e["kind"] == "run_specific"],
        "to_keep": [e["name"] for e in entries if e["kind"] in ("retained", "protected_ledger")],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  B/G. Retire (move, never delete)
# ─────────────────────────────────────────────────────────────────────────────

def retire_active_state(
    project_root: str | Path,
    *,
    apply: bool = False,
    acknowledged_remote_job_ids: Sequence[str] = (),
    known_remote_job_ids: Iterable[str] = (),
    stamp: Optional[str] = None,
) -> dict[str, Any]:
    """Move run-specific checkpoint state out of the active namespace.

    ``apply=False`` (default) only reports what would move.  With ``apply=True`` every possibly-running
    remote job must be listed in ``acknowledged_remote_job_ids`` (remote jobs are never cancelled here); the
    entries are *moved* (never deleted) to ``<project>/_backup_staging/retired_<stamp>/`` together with a
    SHA-256 manifest, and the legacy ledger digest is verified unchanged afterwards.
    """
    root = Path(project_root)
    inv = inventory_active_state(root, known_remote_job_ids=known_remote_job_ids)
    unacknowledged = sorted(set(inv["possibly_running_remote_jobs"]) - {str(j) for j in acknowledged_remote_job_ids})
    plan = {"stamp": stamp, "to_retire": inv["to_retire"], "to_keep": inv["to_keep"],
            "unacknowledged_remote_jobs": unacknowledged, "applied": False}
    if not apply:
        plan["note"] = "dry run: nothing was moved"
        return plan
    if unacknowledged:
        raise CleanSlateError(
            f"{len(unacknowledged)} possibly-running remote job(s) must be explicitly acknowledged before state "
            f"is retired (they are never cancelled automatically): {unacknowledged[:5]}")
    ledger_before = historical_accounting_snapshot(root)["ledger_sha256"]
    stamp = stamp or _utc_stamp()
    dest = root / "_backup_staging" / f"retired_{stamp}"
    if dest.exists():
        raise CleanSlateError(f"retire destination already exists: {dest}")
    ckpt = root / "checkpoints"
    manifest: list[dict[str, Any]] = []
    dest.mkdir(parents=True)
    for name in inv["to_retire"]:
        src = ckpt / name
        if _is_protected(src) or name in RETAINED_CHECKPOINT_ENTRIES:
            raise CleanSlateError(f"refusing to retire protected entry {name!r}")
        files = [src] if src.is_file() else [f for f in sorted(src.rglob("*")) if f.is_file()]
        for f in files:
            manifest.append({"path": str(f.relative_to(ckpt)), "bytes": f.stat().st_size, "sha256": _sha256_file(f)})
        shutil.move(str(src), str(dest / name))
    (dest / "RETIRED_MANIFEST.json").write_text(json.dumps({
        "schema": "retired_manifest/1", "stamp": stamp, "moved": inv["to_retire"], "files": manifest,
        "inventory": inv, "acknowledged_remote_job_ids": sorted(map(str, acknowledged_remote_job_ids)),
        "ledger_sha256_before": ledger_before,
    }, indent=2, sort_keys=True), encoding="utf-8")
    ledger_after = historical_accounting_snapshot(root)["ledger_sha256"]
    if ledger_after != ledger_before:  # pragma: no cover - defensive; the ledger is never a move target
        raise CleanSlateError("the legacy QPU ledger changed during retirement; investigate immediately")
    write_historical_snapshot(root)
    plan.update(applied=True, destination=str(dest), files_recorded=len(manifest), ledger_sha256=ledger_after)
    return plan


# ─────────────────────────────────────────────────────────────────────────────
#  F. Contamination guards
# ─────────────────────────────────────────────────────────────────────────────

def guard_job_ids(job_ids: Iterable[str], deny: Mapping[str, Sequence[str]]) -> list[str]:
    """Return the identifiers that belong to history (empty list == clean)."""
    banned = set(deny.get("job_ids", ()))
    return sorted({j for j in job_ids if j and (j in banned or str(j).startswith("replay_"))})


def guard_task_registry(ctx: Any, deny: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """Findings for a run's registry: pre-completed tasks, historical / replay job ids, foreign run ids."""
    tasks = ctx.registry.list_tasks(ctx.run_id)
    ids = [t.qpu_job_id for t in tasks] + [t.serverless_job_id for t in tasks]
    return {
        "completed_tasks": [t.task_id for t in tasks if t.state == "COMPLETE"],
        "historical_job_ids": guard_job_ids(ids, deny),
        "replay_tasks": [t.task_id for t in tasks if str(t.backend).lower() == "replay"
                         or str(t.qpu_job_id or "").startswith("replay_")],
        "foreign_run_tasks": [t.task_id for t in tasks if t.run_id != ctx.run_id],
        "historical_run_id": ctx.run_id in set(deny.get("run_ids", ())),
    }


def guard_fragment_result(ctx: Any, *, fragment_id: int, payload_run_id: str, payload_manifest_digest: str = "") -> None:
    """Reject a SolverResult that is not bound to this run (and manifest, when supplied)."""
    if payload_run_id != ctx.run_id:
        raise CleanSlateError(
            f"fragment {fragment_id} result belongs to run {payload_run_id!r}, not {ctx.run_id!r}; cross-run reuse is prohibited")
    if payload_manifest_digest and ctx.manifest is not None and payload_manifest_digest != ctx.manifest.manifest_digest:
        raise CleanSlateError(f"fragment {fragment_id} result is bound to a different manifest")


def guard_summary_belongs_to_run(ctx: Any, summary: Mapping[str, Any]) -> None:
    """Reject a reconstruction summary from another run or manifest."""
    if summary.get("run_id") != ctx.run_id:
        raise CleanSlateError(f"reconstruction summary belongs to run {summary.get('run_id')!r}, not {ctx.run_id!r}")
    mdig = summary.get("manifest_digest") or (summary.get("manifest") or {}).get("manifest_digest")
    if mdig and ctx.manifest is not None and mdig != ctx.manifest.manifest_digest:
        raise CleanSlateError("reconstruction summary is bound to a different manifest")


# ─────────────────────────────────────────────────────────────────────────────
#  H. Acceptance
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Check:
    code: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "passed": self.passed, "detail": self.detail}


def assert_clean_slate(
    ctx: Any,
    campaign_auth: Any,
    *,
    project_root: str | Path,
    deny: Optional[Mapping[str, Sequence[str]]] = None,
    notebook_path: Optional[str | Path] = None,
    expected_ledger_sha256: Optional[str] = None,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Evaluate acceptance criteria H1-H10 for the run, before any trial work starts."""
    root = Path(project_root)
    deny = deny if deny is not None else historical_deny_list(root, include_active=False)
    checks: list[_Check] = []
    tasks = ctx.registry.list_tasks(ctx.run_id)
    finds = guard_task_registry(ctx, deny)

    checks.append(_Check("H1_zero_completed_tasks", not finds["completed_tasks"] and not finds["foreign_run_tasks"],
                         f"{len(tasks)} tasks, {len(finds['completed_tasks'])} complete"))
    n_sub = len(campaign_auth.submissions) if campaign_auth is not None else -1
    checks.append(_Check("H2_zero_campaign_submissions", campaign_auth is not None and n_sub == 0,
                         f"{n_sub} submissions; campaign_id={getattr(campaign_auth, 'campaign_id', '')!r}"))
    stage = ctx.paths.run_dir / "fragment_result"
    stored = sorted(p.name for p in stage.glob("*.h5")) if stage.is_dir() else []
    checks.append(_Check("H3_no_historical_results_loaded", not stored and not finds["replay_tasks"],
                         f"{len(stored)} stored fragment results, {len(finds['replay_tasks'])} replay tasks"))
    checks.append(_Check("H4_no_old_serverless_job_ids", not finds["historical_job_ids"] and not finds["historical_run_id"],
                         f"{len(finds['historical_job_ids'])} historical ids attached; run id reuse={finds['historical_run_id']}"))
    checks.append(_Check("H5_no_prior_solver_results_accepted", not stored and not finds["completed_tasks"],
                         "result store empty and no task pre-completed"))
    stale = [n for n in ("reconstruction_summary.json", "stage4_ledger.json", "real_transport_evidence.json")
             if (ctx.paths.run_dir / n).exists()]
    checks.append(_Check("H6_no_prior_reconstruction_summary", not stale, f"stale artifacts: {stale}"))
    nb = _notebook_output_state(Path(notebook_path)) if notebook_path else {"present": False, "clean": True}
    checks.append(_Check("H7_notebook_has_no_outputs", bool(nb.get("clean", True)),
                         f"cells with outputs: {nb.get('cells_with_outputs', [])}"))
    bound = ctx.manifest is not None and ctx.manifest.run_id == ctx.run_id and (
        campaign_auth is None or not campaign_auth.manifest_digest
        or campaign_auth.manifest_digest == ctx.manifest.manifest_digest)
    checks.append(_Check("H8_results_bind_to_new_manifest_and_run", bound,
                         f"manifest frozen={ctx.manifest is not None}; campaign manifest_digest matches={bound}"))
    snap = historical_accounting_snapshot(root)
    ok_hist = snap["ledger_present"] and (expected_ledger_sha256 is None or snap["ledger_sha256"] == expected_ledger_sha256) \
        and (campaign_auth is None or not campaign_auth.historical_accounting
             or campaign_auth.historical_accounting.get("ledger_sha256") == snap["ledger_sha256"])
    checks.append(_Check("H9_historical_accounting_accurate", bool(ok_hist),
                         f"legacy ledger count={snap['lifetime_count']} sha={str(snap['ledger_sha256'])[:12]}"))
    ready = campaign_auth is not None and bool(campaign_auth.trial_profile) and campaign_auth.remaining_allocation() > 0 \
        and bool(campaign_auth.authorized_fragments)
    checks.append(_Check("H10_parallel_execution_ready_under_new_authorization", ready,
                         f"profile={getattr(campaign_auth, 'trial_profile', '')!r} remaining="
                         f"{campaign_auth.remaining_allocation() if campaign_auth is not None else 0}"))
    report = {"passed": all(c.passed for c in checks), "run_id": ctx.run_id, "checks": [c.as_dict() for c in checks]}
    if raise_on_failure and not report["passed"]:
        failed = [c.code for c in checks if not c.passed]
        raise CleanSlateError(f"clean-slate acceptance failed: {failed}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
#  E. Notebook
# ─────────────────────────────────────────────────────────────────────────────

def clear_notebook_outputs(notebook_path: str | Path) -> dict[str, Any]:
    """Strip outputs and execution counts in place (code, markdown, cell IDs and metadata are preserved)."""
    p = Path(notebook_path)
    nb = json.loads(p.read_text(encoding="utf-8"))
    cleared = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs") or cell.get("execution_count") is not None:
                cleared += 1
            cell["outputs"] = []
            cell["execution_count"] = None
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    return {"cells": len(nb.get("cells", [])), "cleared": cleared}
