"""Read-only, run-scoped organizer audit for Proteasome Challenge 2026.

No provider connections, submissions, mutation of the registry, or new calculations.
The only writes are explicit exports to the requested output directory.

Evidence is organized by experiment and *attempt* rather than only final task state.
Missing records are explicitly called out. This module intentionally avoids
reading private provider credentials, unrelated runs, or reference energies.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

SCHEMA = "proteasome-organizer-report/1"
_TERMINAL = {"COMPLETE", "DONE", "SUCCESS", "FAILED", "ERROR", "CANCELLED"}
_FAILURE = {"FAILED", "ERROR", "CANCELLED", "REJECTED", "RETRY_PENDING", "AMBIGUOUS"}
_SECRET = re.compile(r"token|api.?key|password|secret|credential|authorization|\.env", re.I)
_FAILED_JOB = re.compile(r"Serverless\s+3Q-C\s+job\s+([a-zA-Z0-9-]+).*provider\s+status\s+(\w+)")


def _plain(value: Any) -> Any:
    """Convert ordinary immutable scientific metadata to JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items() if not _SECRET.search(str(k))}
    if isinstance(value, (tuple, list, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _plain(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _record(obj: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {k: _plain(getattr(obj, k, None)) for k in names}


def _json(path: Path, errors: list[str], *, max_bytes: int = 8_000_000) -> Any:
    if not path.is_file():
        return None
    try:
        if path.stat().st_size > max_bytes:
            raise ValueError("JSON exceeds safe reporting size limit")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        errors.append(f"Could not read {path.name}: {type(exc).__name__}: {exc}")
        return None


def _time_sort(entry: dict[str, Any]) -> str:
    return str(entry.get("occurred_iso8601") or entry.get("at") or entry.get("timestamp") or "")


def _sha(path: Path, max_bytes: int = 8_000_000) -> str | None:
    if path.stat().st_size > max_bytes:
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _pick(obj: Mapping[str, Any] | None, names: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(obj, Mapping):
        return {}
    return {key: _plain(obj[key]) for key in names if key in obj and not _SECRET.search(key)}


def _row(kind: str, identifier: str, *, stage: str, state: str = "NOT_EVIDENCED",
         fragment: int | None = None, solver: str | None = None, **kwargs: Any) -> dict[str, Any]:
    return {"id": identifier, "stage": stage, "kind": kind, "fragment_id": fragment,
            "solver": solver, "state": state, "job_id": None, "resource": None,
            "energy_ha": None, "config": None, "evidence": [], **kwargs}


def _status(state: Any) -> str:
    return str(state or "NOT_EVIDENCED").upper()


def _to_html_table(rows: list[dict[str, Any]], cols: list[str] | None = None, limit: int = 300) -> str:
    if not rows:
        return '<p class="muted">No records.</p>'
    cols = cols or list(rows[0])
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    rendered = []
    for row in rows[:limit]:
        cells = []
        for col in cols:
            val = row.get(col)
            if isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=False, default=str)
            cells.append(f"<td>{html.escape('—' if val is None else str(val))}</td>")
        rendered.append("<tr>" + "".join(cells) + "</tr>")
    more = (f"<p class='muted'>Displaying {limit}/{len(rows)} entries; full tables in JSON/CSV.</p>"
            if len(rows) > limit else "")
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rendered)}</tbody></table></div>{more}"


@dataclass
class ExperimentAudit:
    data: dict[str, Any]

    def to_html(self) -> str:
        d = self.data
        summary = d["campaign"]
        totals = summary["experiment_states"]
        count_text = ", ".join(f"{k}: {v}" for k, v in sorted(totals.items())) or "no experiments"
        e = d.get("reconstruction", {})
        energy = e.get("e_total")
        energy_text = f"{energy:.12f} Ha" if isinstance(energy, (float, int)) else "Not available"
        title = f"Organizer experiment audit · {d['run_id']}"
        warnings = [{"issue": issue} for issue in d["evidence_gaps"]]
        css = """body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#20242b;margin:2.5rem auto;max-width:1100px;padding:0 1.5rem}h1{font-size:1.7rem}h2{font-size:1.12rem;margin-top:2.2rem;padding-top:.5rem;border-top:1px solid #ddd}p{max-width:88ch}.muted{color:#647084}.scroll{max-width:100%;overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}th{background:#f1f3f5;position:sticky;top:0}code{font-size:12px}strong{font-weight:650}.metric{font-size:1.35rem;font-weight:650}footer{margin:3rem 0;color:#647084}"""
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Generated: {html.escape(d['generated_at'])} · Schema: {html.escape(d['schema'])} · Snapshot of local, run-scoped evidence only.</p>
<p><strong>Campaign:</strong> {summary['tasks_complete']}/{summary['tasks_planned']} fragment tasks complete · {html.escape(count_text)}.<br><strong>Molecular energy:</strong> <span class="metric">{html.escape(energy_text)}</span></p>
<h2>1 · Campaign provenance</h2>{_to_html_table([d['provenance']])}
<h2>2 · Experiment inventory</h2><p>Each planned scientific experiment is a separate row. Missing evidence is not proof that an experiment was never attempted.</p>{_to_html_table(d['experiments'], ['id','stage','kind','fragment_id','solver','state','job_id','resource','energy_ha','config','evidence'])}
<h2>3 · Execution attempts and recoveries</h2>{_to_html_table(d['attempts'], ['experiment_id','attempt','outcome','job_id','occurred_at','reason','evidence'])}
<h3>Recorded campaign operations</h3>{_to_html_table(d['campaign_operations'])}
<h2>4 · Scientific results and comparisons</h2>{_to_html_table(d['scientific_comparisons'])}
<h2>5 · Molecular reconstruction</h2>{_to_html_table([e]) if e else '<p>Not available</p>'}
<h2>6 · Fragment contributions</h2>{_to_html_table(d['fragment_contributions'])}
<h2>7 · Failures and incomplete work</h2>{_to_html_table(d['failures'])}
<h2>8 · Evidence gaps and unsupported conclusions</h2>{_to_html_table(warnings)}
<h2>9 · Artifact inventory</h2><p>Run-scoped files only. SHA-256 is omitted for files above 8 MB to avoid loading large numerical artifacts.</p>{_to_html_table(d['files'])}
<footer>Generated locally. No QPU, Serverless, or network calls were made.</footer></body></html>"""

    def export(self, directory: str | Path) -> dict[str, Path]:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        exported = {"html": out / "experiment_report.html",
                    "json": out / "experiment_report.json",
                    "inventory_csv": out / "experiment_inventory.csv",
                    "attempts_csv": out / "execution_attempts.csv",
                    "files_csv": out / "artifact_inventory.csv"}
        exported["html"].write_text(self.to_html(), encoding="utf-8")
        exported["json"].write_text(json.dumps(self.data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        for key, rows in (("inventory_csv", self.data["experiments"]),
                          ("attempts_csv", self.data["attempts"]),
                          ("files_csv", self.data["files"])):
            cols = list(dict.fromkeys(k for row in rows for k in row))
            with exported[key].open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=cols)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                                     for k, v in row.items()})
        return exported


def build_experiment_report(ctx: Any, *, fragment_lab: Any = None,
                            active_fragment_id: int | None = None) -> ExperimentAudit:
    """Construct a complete local audit without contacting QPU or Serverless services.

    ctx: existing `run_ctx`, including its frozen manifest and registry.
    fragment_lab: optional run-scoped fragment lab (for planned tuning-grid labels).
    active_fragment_id: optional selected laboratory fragment from the current notebook;
        only a planning hint, never used to certify result identity.
    """
    if ctx is None or not getattr(ctx, "run_id", None) or not getattr(ctx, "paths", None):
        raise ValueError("An existing RunContext is required")
    root = Path(ctx.paths.run_dir).resolve()
    if not root.is_dir() or root.name != str(ctx.run_id):
        raise ValueError("Run directory is missing or mismatched; refusing cross-run audit")
    issues: list[str] = []
    entries: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {"run_id": ctx.run_id}

    # Inventory every run-scoped file; never inspect credentials, other runs, or huge HDF/NPY payloads.
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or "organizer_report" in path.relative_to(root).parts:
            continue
        rel = path.relative_to(root).as_posix()
        if _SECRET.search(rel):
            continue
        try:
            files.append({"path": rel, "bytes": path.stat().st_size, "sha256": _sha(path)})
        except OSError as exc:
            issues.append(f"Unable to stat/hash {rel}: {exc}")
    provenance["artifact_files"] = len(files)
    provenance["artifact_index_scope"] = "Current run directory only; no reference/global checkpoints"
    provenance["run_directory"] = str(root)
    provenance["compute_config"] = _plain(getattr(ctx, "compute_config", {}))

    manifest = getattr(ctx, "manifest", None)
    planned: dict[int, str] = {}
    if manifest is not None:
        assignments = getattr(manifest, "solver_assignments", {}) or {}
        planned = {int(fid): str(solver) for fid, solver in assignments.items()}
        provenance.update(_record(manifest, ("manifest_digest", "protocol_digest", "geometry_sha256", "ligand",
                                             "basis", "charge", "spin", "n_electrons", "n_ao", "run_id")))
        if manifest.run_id != ctx.run_id:
            raise ValueError("RunManifest belongs to another run")
        check = getattr(manifest, "verify_digest", None)
        if callable(check) and not check():
            raise ValueError("RunManifest digest validation failed")
    else:
        issues.append("No frozen RunManifest: fragment allocation and completeness cannot be established")

    protocol = _json(root / "shared_protocol.json", issues)
    if isinstance(protocol, dict):
        provenance["shared_protocol"] = _pick(protocol, ("protocol_digest", "document_digest", "solver", "target_fragment_ids"))
    else:
        issues.append("Run-scoped shared_protocol.json is missing or unreadable")
    campaign = _json(root / "campaign_authorization.json", issues)
    if isinstance(campaign, dict):
        provenance["campaign"] = _pick(campaign, ("campaign_id", "authorized_backend", "shots_per_fragment",
                                                          "n_shots", "max_submissions", "manifest_digest", "run_id"))
        if campaign.get("run_id") not in (None, ctx.run_id):
            issues.append("Campaign authorization run ID disagrees with the report run")
    ledger = _json(root / "full_trial_ledger.json", issues)
    operations: list[dict[str, Any]] = []
    if isinstance(ledger, dict):
        for event in ledger.get("events", []):
            if isinstance(event, dict):
                operations.append(_plain(event))
        provenance["full_trial_operations"] = len(operations)
        provenance["trial_profile"] = ledger.get("profile")
    else:
        issues.append("No full_trial_ledger.json: campaign operation history unavailable")

    # Published stage references are read through the authoritative local artifact store.
    stage_refs: dict[str, dict[str, Any]] = {}
    try:
        from proteasome_challenge.run_control_plane import load_stage_ref
        for role in ("meanfield", "embedding_ewf", "embedding_pkl"):
            ref = load_stage_ref(ctx, role)
            if ref is not None:
                if ref.run_id != ctx.run_id:
                    issues.append(f"Stage reference {role} belongs to another run")
                else:
                    stage_refs[role] = _record(ref, ("artifact_name", "kind", "file_sha256", "uri", "run_id"))
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        issues.append(f"Stage reference metadata could not be loaded: {type(exc).__name__}: {exc}")
    provenance["published_stage_refs"] = stage_refs

    # Scientific planning decisions are part of the run's audit even though they
    # are not separately executed solver jobs.
    selected_ids = [fid for fid, solver in sorted(planned.items()) if solver in ("SQD", "TrimSQD")]
    entries.append(_row("solver_allocation", "solver-routing", stage="3",
                        state="COMPLETE" if manifest is not None else "NOT_EVIDENCED",
                        config=dict(Counter(planned.values())),
                        evidence=["run_manifest.json"] if manifest else []))
    entries.append(_row("quantum_selection", "quantum-selection", stage="4",
                        state="COMPLETE" if manifest is not None else "NOT_EVIDENCED",
                        config={"quantum_fragment_ids": selected_ids},
                        evidence=["run_manifest.json"] if manifest else []))
    entries.append(_row("protocol_promotion", "protocol-promotion", stage="5",
                        state="COMPLETE" if isinstance(protocol, dict) else "NOT_EVIDENCED",
                        config=_pick(protocol, ("protocol_digest", "solver", "document_digest")),
                        evidence=["shared_protocol.json"] if isinstance(protocol, dict) else []))

    # Skipped submissions in the campaign ledger remain distinct from final task outcomes.
    def _skipped_in(node: Any, at: str | None = None):
        if isinstance(node, dict):
            if isinstance(node.get("skipped_tasks"), list):
                for skipped in node["skipped_tasks"]:
                    if isinstance(skipped, dict):
                        fid = skipped.get("fragment_id")
                        attempts.append({"experiment_id": f"fragment:{fid}" if fid is not None else "campaign",
                                         "attempt": None, "outcome": "SKIPPED", "job_id": skipped.get("job_id"),
                                         "occurred_at": at, "reason": skipped.get("reason"),
                                         "evidence": "full_trial_ledger.json"})
            for k, value in node.items():
                if k != "skipped_tasks":
                    _skipped_in(value, at)
        elif isinstance(node, list):
            for value in node:
                _skipped_in(value, at)
    for operation in operations:
        _skipped_in(operation, operation.get("at"))

    # Stage 1/2: planned by the participant workflow; scoped outputs or verified refs are evidence.
    for stage, name, candidates in (
        ("1", "Hartree–Fock", [root / "stage_outputs" / "hf_results.json"]),
        ("2", "EWF fragmentation", [root / "stage_outputs" / "fragment_catalog.csv"]),
    ):
        found = [p for p in candidates if p.is_file()]
        refs = ["meanfield"] if stage == "1" else ["embedding_ewf", "embedding_pkl"]
        evid = [str(p.relative_to(root)) for p in found] + [f"published ref: {r}" for r in refs if r in stage_refs]
        state = "COMPLETE" if evid else "NOT_EVIDENCED"
        rows = _row("scientific_stage", f"stage{stage}", stage=stage, state=state,
                    resource=_plain(getattr(ctx, "compute_config", {})), evidence=evid)
        if stage == "1" and found:
            hf = _json(found[0], issues)
            if isinstance(hf, dict):
                rows["scientific_metadata"] = _pick(hf, ("hf_energy", "energy", "total_energy", "basis", "method"))
        if stage == "2" and found:
            try:
                with found[0].open(newline="", encoding="utf-8") as fh:
                    rows["catalog_rows"] = sum(1 for _ in csv.DictReader(fh))
            except (OSError, UnicodeError, csv.Error) as exc:
                issues.append(f"Could not parse fragment catalog: {exc}")
        entries.append(rows)

    # One planned full-campaign fragment row per frozen manifest assignment, plus registry-only anomalies.
    tasks = list(ctx.registry.list_tasks(ctx.run_id))
    by_id: dict[int, list[Any]] = defaultdict(list)
    for task in tasks:
        by_id[int(task.fragment_id)].append(task)
    for fid in sorted(set(planned) | set(by_id)):
        registered = by_id.get(fid, [])
        if len(registered) > 1:
            issues.append(f"Fragment {fid} has {len(registered)} task records")
        task = registered[0] if registered else None
        solver = planned.get(fid) or (str(task.solver) if task else None)
        attrs = _record(task, ("task_id", "task_type", "resource_family", "backend", "provider_status",
                               "hamiltonian_sha256", "result_artifact_uri", "qpu_job_id", "serverless_job_id",
                               "failure_reason", "created_iso8601", "updated_iso8601")) if task else {}
        extra = getattr(task, "extra", {}) or {} if task else {}
        evidence = ["run_manifest.json"] if fid in planned else []
        if task:
            evidence.append("registry.db")
        if attrs.get("result_artifact_uri"):
            evidence.append(str(attrs["result_artifact_uri"]))
        row = _row("full_campaign_fragment", f"fragment:{fid}", stage="6", fragment=fid, solver=solver,
                   state=_status(getattr(task, "state", None)), job_id=attrs.get("serverless_job_id"),
                   resource=attrs.get("resource_family"), evidence=evidence)
        row.update({"qpu_job_id": attrs.get("qpu_job_id"), "hamiltonian_sha256": attrs.get("hamiltonian_sha256"),
                    "provider_status": attrs.get("provider_status"), "result_uri": attrs.get("result_artifact_uri"),
                    "failure_reason": attrs.get("failure_reason"), "n_shots": extra.get("pp_counts_shots"),
                    "counts_sha256": extra.get("pp_counts_sha256"), "postprocess_attempts": extra.get("pp_attempt"),
                    "elapsed_seconds": extra.get("worker_elapsed_seconds"),
                    "config": {"protocol_digest": getattr(manifest, "protocol_digest", None),
                               "backend": attrs.get("backend"), "solver": solver,
                               "quantum_protocol": _pick(protocol.get("transferable"), ("sqd_config", "trim_config"))
                               if solver in ("SQD", "TrimSQD") and isinstance(protocol, dict) else None}})
        entries.append(row)
        if not task:
            issues.append(f"Fragment {fid} is planned but has no registry task (not evidenced as executed)")
            continue
        if fid not in planned:
            issues.append(f"Fragment {fid} has a task but is absent from the frozen manifest")
        if solver != str(task.solver):
            issues.append(f"Fragment {fid} solver differs between manifest and registry")
        hist = ctx.registry.get_transition_history(task.task_id) or []
        transitions: list[dict[str, Any]] = []
        for tr in hist:
            tr = _plain(tr)
            if not isinstance(tr, dict):
                continue
            transitions.append(_pick(tr, ("from_state", "to_state", "occurred_iso8601", "reason", "provider_status")))
        row["transitions"] = transitions
        for k, tr in enumerate(transitions):
            to = _status(tr.get("to_state"))
            if to in _FAILURE:
                reason = str(tr.get("reason") or "")
                found_job = _FAILED_JOB.search(reason)
                attempts.append({"experiment_id": row["id"], "attempt": None,
                                 "outcome": to, "job_id": found_job.group(1) if found_job else None,
                                 "occurred_at": tr.get("occurred_iso8601"), "reason": reason,
                                 "evidence": "registry.db / transition history"})
        if int(extra.get("pp_attempt") or 0) > 1 and not any(
            h.get("to_state") == "FAILED" for h in transitions
        ):
            issues.append(f"Fragment {fid}: post-processing retry count indicates earlier attempts but failure transitions are unavailable")
        # Final task evidence must not overwrite evidence of earlier failed attempts.
        if task.serverless_job_id:
            attempts.append({"experiment_id": row["id"], "attempt": extra.get("pp_attempt") or 1,
                             "outcome": _status(task.state), "job_id": task.serverless_job_id,
                             "occurred_at": attrs.get("updated_iso8601"),
                             "reason": attrs.get("failure_reason"), "evidence": "registry.db"})
        if task.qpu_job_id:
            attempts.append({"experiment_id": row["id"] + ":QPU", "attempt": 1,
                             "outcome": "SAMPLED" if extra.get("pp_counts_sha256") else _status(task.state),
                             "job_id": task.qpu_job_id, "occurred_at": None, "reason": None,
                             "evidence": "registry.db / authenticated counts metadata"})
        if str(task.state) == "COMPLETE" and task.result_artifact_uri:
            try:
                from serverless_bundle.steps.fragment_result_store import load_fragment_result
                result = load_fragment_result(task.result_artifact_uri,
                    expected_fragment_id=fid, expected_run_id=ctx.run_id,
                    expected_solver=str(task.solver), expected_hamiltonian_sha256=task.hamiltonian_sha256)
                row["energy_ha"] = _finite(getattr(result, "energy", None))
                row["result_verified"] = row["energy_ha"] is not None
                if row["energy_ha"] is None:
                    issues.append(f"Fragment {fid}: authenticated result returned no finite energy")
                row["scientific_metadata"] = _pick(getattr(result, "meta", {}),
                    ("norb", "n_orb", "iterations", "converged", "n_shots", "elapsed_seconds", "solver_mode"))
            except Exception as exc:  # A broken artifact must not hide the remaining audit.
                row["result_verified"] = False
                issues.append(f"Fragment {fid}: could not authenticate/parse saved solver energy: {type(exc).__name__}: {exc}")
        elif _status(task.state) == "COMPLETE":
            issues.append(f"Fragment {fid} is COMPLETE but no result artifact URI is recorded")

    # Single-fragment laboratory is run-scoped; NEVER read old global checkpoints.
    labdir = root / "single_fragment_lab" / "stage3"
    lab_id = None
    if fragment_lab is not None and getattr(fragment_lab, "run_id", None) == ctx.run_id:
        selected = getattr(fragment_lab, "_selected_ids", set())
        # The participant lab selects one target, but the source checkpoint defines the observed fragment.
        if selected:
            lab_id = min(int(x) for x in selected)
    ccsd_energies: dict[int, float] = {}
    ccsd_hashes: dict[int, str] = {}
    for path in sorted(labdir.glob("*.json")) if labdir.is_dir() else []:
        rel = path.relative_to(root).as_posix()
        match = re.fullmatch(r"(.+)_frag(\d+)_(ccsd|both|sqd|trimsqd)\.json", path.name, re.I)
        if not match:
            continue
        ligand, fragment, method = match.group(1), int(match.group(2)), match.group(3).lower()
        rec = _json(path, issues)
        if not isinstance(rec, dict):
            continue
        if rec.get("fragment_id") not in (None, fragment):
            issues.append(f"Checkpoint {rel}: fragment identity mismatch")
            continue
        e = None
        if fragment_lab is not None and getattr(fragment_lab, "run_id", None) == ctx.run_id:
            try:
                e = _finite(fragment_lab.energy_of(rec, method))
            except (TypeError, ValueError, KeyError, RuntimeError):
                pass
        for key in ("energy_ha", "energy", "e_ccsd", "e_sqd", "e_trim", "ccsd_energy") if e is None else ():
            e = _finite(rec.get(key))
            if e is not None:
                break
        entries.append(_row("single_fragment_lab", f"lab:{fragment}:{method}", stage="5", fragment=fragment,
                            solver=method.upper(), state="COMPLETE", energy_ha=e,
                            job_id=rec.get("job_id"),
                            evidence=[rel], config=_pick(rec, ("sqd_config", "trim_config", "n_shots", "n_reps", "qpu_backend")),
                            hamiltonian_sha256=rec.get("hamiltonian_sha256")))
        if method == "ccsd" and e is not None:
            ccsd_energies[fragment] = e
            if rec.get("hamiltonian_sha256"):
                ccsd_hashes[fragment] = str(rec["hamiltonian_sha256"])
        if method == "both" and fragment_lab is not None and getattr(fragment_lab, "run_id", None) == ctx.run_id:
            for mode in ("sqd", "trimsqd"):
                try:
                    mode_e = _finite(fragment_lab.energy_of(rec, mode))
                except (TypeError, ValueError, KeyError, RuntimeError):
                    mode_e = None
                entries.append(_row("single_fragment_solver", f"lab:{fragment}:{mode}", stage="5",
                                    fragment=fragment, solver=mode.upper(),
                                    state="COMPLETE" if mode_e is not None else "INCOMPLETE",
                                    energy_ha=mode_e, evidence=[rel],
                                    hamiltonian_sha256=rec.get("hamiltonian_sha256")))

    # Pending and failed single-fragment worker attempts remain visible even if
    # no successful checkpoint was ever written.
    pending_dir = labdir / "pending"
    for path in sorted(pending_dir.glob("*.json")) if pending_dir.is_dir() else []:
        rec = _json(path, issues)
        if not isinstance(rec, dict):
            continue
        if rec.get("fragment_id") is None or not rec.get("method"):
            issues.append(f"Incomplete single-fragment job record: {path.name}")
            continue
        fid = int(rec["fragment_id"])
        method = str(rec["method"]).lower()
        failed = ".failed-" in path.name
        rel = path.relative_to(root).as_posix()
        entries.append(_row("single_fragment_job", f"lab_job:{fid}:{method}:{path.name}",
                            stage="5", fragment=fid, solver=method.upper(),
                            state="FAILED" if failed else "SUBMITTED",
                            job_id=rec.get("job_id"), resource=rec.get("resource_family"),
                            config=_pick(rec, ("execution", "n_shots", "n_reps", "qpu_backend",
                                               "sqd_config", "trim_config")), evidence=[rel]))
        attempts.append({"experiment_id": f"lab:{fid}:{method}", "attempt": None,
                         "outcome": "FAILED" if failed else "SUBMITTED", "job_id": rec.get("job_id"),
                         "occurred_at": None, "reason": "Worker record archived as failed" if failed else None,
                         "evidence": rel})

    grid_labels: dict[str, dict] = {}
    if fragment_lab is not None and getattr(fragment_lab, "run_id", None) == ctx.run_id:
        try:
            for definition in fragment_lab.load_tuning_grid():
                grid_labels[str(definition["label"])] = definition
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            issues.append(f"Could not load organizer grid definitions: {exc}")
    grid_files = sorted(labdir.glob("*_tuning_results.json")) if labdir.is_dir() else []
    partial_files = sorted(labdir.glob("*_tuning_results_partial.json")) if labdir.is_dir() else []
    job_files = sorted(labdir.glob("*_tuning_jobs.json")) if labdir.is_dir() else []
    grid_sets: dict[int, dict[str, dict]] = defaultdict(dict)
    for path in grid_files + partial_files:
        m = re.search(r"_frag(\d+)_tuning_results", path.name)
        if not m:
            continue
        fid = int(m.group(1))
        raw = _json(path, issues)
        if isinstance(raw, dict):
            rows = raw.get("tuning_results", []) if "tuning_results" in raw else raw
            rows = rows.values() if isinstance(rows, dict) else rows
            if isinstance(rows, (tuple, list)) or hasattr(rows, "__iter__"):
                for item in rows:
                    if isinstance(item, dict) and item.get("label"):
                        grid_sets[fid][str(item["label"])] = {**item, "_evidence": path.relative_to(root).as_posix()}
    grid_jobs: dict[int, dict] = defaultdict(dict)
    for path in job_files:
        m = re.search(r"_frag(\d+)_tuning_jobs", path.name)
        if m:
            raw = _json(path, issues)
            if isinstance(raw, dict):
                grid_jobs[int(m.group(1))].update(raw)
    lab_fragment_ids = sorted(set(grid_sets) | set(grid_jobs))
    if active_fragment_id is not None:
        lab_fragment_ids = sorted(set(lab_fragment_ids) | {int(active_fragment_id)})
    elif not lab_fragment_ids and lab_id is not None:
        lab_fragment_ids = [lab_id]
    for fid in lab_fragment_ids:
        for label in sorted(set(grid_labels) | set(grid_sets[fid]) | set(grid_jobs[fid])):
            definition = grid_labels.get(label, {})
            res = grid_sets[fid].get(label, {})
            job = grid_jobs[fid].get(label)
            solver = str(res.get("solver_method") or definition.get("solver_method") or definition.get("solver_mode") or "UNKNOWN").upper()
            energy = _finite(res.get("energy_ha"))
            entries.append(_row("tuning_grid", f"grid:{fid}:{label}", stage="5", fragment=fid,
                                solver=solver, state="COMPLETE" if energy is not None else "SUBMITTED" if job else "NOT_EVIDENCED",
                                job_id=job, energy_ha=energy,
                                config={"label": label, "sqd_config": _plain(res.get("sqd_config") or definition.get("sqd_config")),
                                        "trim_config": _plain(res.get("trim_config") or definition.get("trim_config"))},
                                evidence=[res.get("_evidence")] if res else (["tuning_jobs.json"] if job else []),
                                hamiltonian_sha256=res.get("hamiltonian_sha256")))
            if energy is not None and fid in ccsd_energies:
                grid_hash = res.get("hamiltonian_sha256")
                if grid_hash and grid_hash == ccsd_hashes.get(fid):
                    comparisons.append({"comparison": f"Grid {label} vs CCSD · fragment {fid}",
                                        "delta_energy_mha": round(1000 * (energy - ccsd_energies[fid]), 6),
                                        "identity": "Verified matching Hamiltonian SHA-256"})
                else:
                    issues.append(f"Grid {label} fragment {fid}: CCSD comparison withheld (Hamiltonian identity not proven)")

    # Stage 4: use the authoritative organizer API, never trust a provider DONE without reconciliation.
    summary = None
    stage4_jobs: list[dict] = []
    try:
        from proteasome_challenge import stage4_reconstruction as s4
        status = s4.stage4_status(ctx)
        stage4_jobs = list(status.get("jobs", []))
        summary = s4.load_authoritative_summary(ctx)
        if summary is not None:
            s4.validate_summary(summary)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        summary = None  # A malformed/unvalidated summary is NOT a completed scientific result.
        issues.append(f"Stage 4 authoritative inspection unavailable: {type(exc).__name__}: {exc}")
    if isinstance(summary, dict):
        summary_job = summary.get("execution", {}).get("serverless_job_id")
        if not summary_job or not any(
            j.get("serverless_job_id") == summary_job and str(j.get("state", "")).upper() == "COMPLETE"
            for j in stage4_jobs
        ):
            issues.append("Stage 4 summary job ID has no matching COMPLETE authoritative ledger entry")
            summary = None
    if isinstance(summary, dict):
        reconstruction = _pick(summary, ("e_hf", "e_corr_total", "e_total"))
        exe = summary.get("execution", {})
        reconstruction.update(_pick(exe, ("mode", "serverless_job_id")))
        fragment_rows = summary.get("fragments", {})
        if isinstance(fragment_rows, dict):
            for fid, rec in fragment_rows.items():
                if isinstance(rec, dict):
                    contributions.append({"fragment_id": int(fid), **_pick(rec, ("solver", "n_orb", "e_corr", "rdm_integrated"))})
        reconstruction["fragment_contributions"] = len(contributions)
        sum_corr = math.fsum(float(row["e_corr"]) for row in contributions if _finite(row.get("e_corr")) is not None)
        reconstruction["correlation_accounting_residual"] = (sum_corr - float(summary["e_corr_total"])) if contributions else None
        group = defaultdict(list)
        for row in contributions:
            if _finite(row.get("e_corr")) is not None:
                group[row.get("solver", "UNKNOWN")].append(float(row["e_corr"]))
        for solver, values in sorted(group.items()):
            comparisons.append({"comparison": f"Partitioned-cumulant E_corr: {solver}",
                                "n_fragments": len(values), "e_corr_ha": math.fsum(values)})
    else:
        reconstruction = {}
        issues.append("No validated authoritative Stage 4 summary available")
    if stage4_jobs:
        for rec in stage4_jobs:
            attempts.append({"experiment_id": "stage4:reconstruction", "attempt": rec.get("attempt"),
                             "outcome": rec.get("state"), "job_id": rec.get("serverless_job_id"),
                             "occurred_at": rec.get("updated_iso8601"), "reason": rec.get("failure_reason"),
                             "evidence": "Stage 4 authoritative ledger"})
    entries.append(_row("molecular_reconstruction", "stage4:reconstruction", stage="7",
                        state="COMPLETE" if summary is not None else "SUBMITTED" if stage4_jobs else "NOT_EVIDENCED",
                        job_id=(summary or {}).get("execution", {}).get("serverless_job_id") if isinstance(summary, dict) else None,
                        energy_ha=reconstruction.get("e_total"),
                        resource=next((j.get("function_size") for j in stage4_jobs
                                       if j.get("serverless_job_id") == (summary or {}).get("execution", {}).get("serverless_job_id")), None),
                        evidence=["Stage 4 authoritative summary"] if summary else []))

    # Files from an unregistered workload are listed but never promoted to a proven executed experiment.
    missing_files = [x["path"] for x in files if x["path"].startswith("single_fragment_lab/") and
                     x["path"].endswith(".json") and x["path"].count("/") >= 2]
    if missing_files and not lab_fragment_ids:
        issues.append("Some single-fragment checkpoints exist but no tuning-grid inventory was detected")
    if any(r["state"] == "NOT_EVIDENCED" for r in entries):
        issues.append("NOT_EVIDENCED means no supported local execution record was found; it does not prove an experiment was never executed")

    # The report shows EVERY failure transition, even when a later attempt succeeded.
    failures = [{"experiment_id": a["experiment_id"], "attempt": a.get("attempt"),
                 "outcome": a["outcome"], "job_id": a.get("job_id"),
                 "reason": a.get("reason"), "occurred_at": a.get("occurred_at")}
                for a in attempts if _status(a["outcome"]) in _FAILURE]
    failures.extend({"experiment_id": e["id"], "attempt": None, "outcome": e["state"],
                     "job_id": e.get("job_id"), "reason": e.get("failure_reason"), "occurred_at": None}
                    for e in entries if e["state"] in (_FAILURE | {"NOT_EVIDENCED", "SUBMITTED", "RUNNING", "PENDING"}))
    tasks_complete = sum(1 for r in entries if r["kind"] == "full_campaign_fragment" and r["state"] == "COMPLETE")
    tasks_planned = len(planned)
    counts = dict(Counter(r["state"] for r in entries))
    campaign_summary = {"tasks_complete": tasks_complete, "tasks_planned": tasks_planned,
                        "fully_complete": bool(tasks_planned) and tasks_complete == tasks_planned,
                        "experiment_states": counts, "recorded_failed_attempts": len([a for a in attempts if _status(a["outcome"]) in _FAILURE])}
    if any(r["kind"] == "full_campaign_fragment" and r.get("scientific_metadata", {}).get("iterations") is None
           for r in entries):
        issues.append("Per-iteration convergence traces are not available for all fragment workers in the persisted artifacts; provider logs are not fetched")
    if not files:
        issues.append("No run-scoped artifacts were found")
    payload = {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
               "run_id": ctx.run_id, "campaign": campaign_summary, "provenance": provenance,
               "experiments": entries, "campaign_operations": operations,
               "attempts": sorted(attempts, key=lambda e: str(e.get("occurred_at") or "")),
               "failures": failures, "scientific_comparisons": comparisons,
               "reconstruction": reconstruction, "fragment_contributions": sorted(contributions, key=lambda r: r["fragment_id"]),
               "files": files, "evidence_gaps": sorted(set(issues))}
    return ExperimentAudit(payload)
