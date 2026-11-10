"""
stage4_reconstruction — Organizer-side manifest-driven completion barrier and EWF reconstruction (Task 5.9D).

Runs in the organizer notebook environment only.  The organizer owns the SQLite registry and the
``LocalArtifactStore``; a Serverless worker never sees either.

What this module adds (and what it deliberately does not)
---------------------------------------------------------
* It adds **no reconstruction mathematics**.  Every energy comes from the inherited
  ``step4_reconstruct.run_reconstruct_from_refs`` reached through ``reconstruction_coordinator``;
  ``E_total = E_HF + sum(E_corr_f)`` with ``E_corr_f`` the partitioned-cumulant contribution of each
  fragment's own RDMs.  Embedded solver energies are never substituted.
* It adds **no orchestration framework**.  It follows the dispatch/reconcile protocol already used by
  ``classical_dispatcher``: durable intent before the side effect, the provider Job ID recorded
  immediately, an uncertain submission is never resubmitted automatically, and every operator action is
  explicit.  The only new persistent object is a small run-level JSON ledger (``stage4_ledger.json``),
  because the task registry is fragment-scoped and has no run-level job table.

Safety contracts
----------------
1. **One barrier for every experiment.**  The required fragment set, the solver of each fragment and the
   Hamiltonian SHA-256 of each fragment are derived from the frozen ``RunManifest`` and the authenticated
   Stage 2 dump — never from a literal.  Classical-only, single-quantum and multi-quantum runs go through
   the same code.
2. **Hamiltonian identity is mandatory.**  Production never calls the inherited barrier with optional
   hashes.
3. **Fail closed.**  The reconstruction is refused unless the barrier passes and the Stage 1/2 scientific
   inputs authenticate against the frozen manifest.  The coordinator's embedded-energy "test double"
   path is unreachable from here (``require_pipeline_data=True`` semantics are enforced by construction).
4. **Immutable authoritative result.**  ``reconstruction_summary.json`` is written once; re-importing a
   scientifically identical result is idempotent; a conflicting one raises.
5. **No automatic duplication.**  A Stage 4 job is dispatched only with ``authorize=True``, only when no
   earlier job is active or uncertain, and only when no authoritative result exists.  A dispatch whose
   outcome is unknown is ``UNCERTAIN`` and needs :func:`resolve_uncertain_stage4_submission`.
6. **Path-free remote contract.**  The job specification carries transport descriptors only.

Honest status: everything here is validated offline against mocks.  Real cross-worker transport and the
deployed Stage 4 function have never been exercised (see :func:`hardware_readiness_report`).
"""
from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional

_ROOT = Path(__file__).resolve().parent.parent
_BUNDLE = _ROOT / "serverless_bundle"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from proteasome_challenge.classical_baseline import hamiltonian_hashes_from_dumpfile  # noqa: E402
from proteasome_challenge.classical_dispatcher import (  # noqa: E402
    VALID_RESOURCE_FAMILIES,
    CatalogServerlessService,
    make_live_transport,
    _PROVIDER_CANCELED,
    _PROVIDER_FAILED,
    _PROVIDER_SUCCESS,
)
from proteasome_challenge.run_control_plane import (  # noqa: E402
    ControlPlaneError,
    RunContext,
    _software_versions,
    _write_once,
    load_stage_ref,
    manifest_identity,
)
from proteasome_challenge.solver_assignments import QUANTUM_SOLVERS  # noqa: E402
from proteasome_challenge.task_registry import TaskState, TaskType  # noqa: E402
from serverless_bundle.steps import reconstruction_coordinator as rc  # noqa: E402
from serverless_bundle.steps.artifact_store import (  # noqa: E402
    KIND_EMBEDDING,
    KIND_SUMMARY_JSON,
    ArtifactConflict,
)
from serverless_bundle.steps.artifact_transport import (  # noqa: E402
    ServerlessTransportAdapter,
    TransportDescriptor,
)
from serverless_bundle.steps.fragment_result_store import FragmentResultReference  # noqa: E402

__all__ = [
    "SUMMARY_SCHEMA", "SPEC_SCHEMA", "LEDGER_SCHEMA", "STAGE4_FUNCTION_TITLE",
    "SUMMARY_FILENAME", "LEDGER_FILENAME",
    "Stage4Error", "Stage4NotReady", "Stage4ConflictError", "Stage4State",
    "ScientificInputs", "StrictBarrier",
    "manifest_composition", "authenticate_scientific_inputs", "run_strict_barrier",
    "format_barrier_report", "readiness_summary",
    "validate_summary", "load_authoritative_summary", "persist_summary", "reconstruct_local",
    "prepare_stage4_job", "dispatch_stage4_job", "reconcile_stage4_job",
    "resolve_uncertain_stage4_submission", "stage4_status", "run_notebook_step",
    "reconciliation_view", "hardware_readiness_report", "format_hardware_readiness",
    # Researcher-facing API (Pass-2D / Task F)
    "ensure_reconstruction", "reconstruction_status",
]

SUMMARY_SCHEMA = "recon/2"
SPEC_SCHEMA = "stage4_spec/1"
RESULT_SCHEMA = "stage4_result/2"
LEDGER_SCHEMA = "stage4_ledger/1"
STAGE4_FUNCTION_TITLE = "qfm-proteasome-stage4-reconstruct"
SUMMARY_FILENAME = "reconstruction_summary.json"
LEDGER_FILENAME = "stage4_ledger.json"
REAL_TRANSPORT_EVIDENCE_FILENAME = "real_transport_evidence.json"

_E_TOL = 1e-8             # Ha; remote-vs-local agreement of E_HF and re-import comparisons
_INVARIANT_TOL = 1e-9     # Ha; E_total = E_HF + sum(E_corr) and sum(fragments) = E_corr_total
_REMOTE_ERROR_LIMIT = 500


class Stage4Error(ControlPlaneError):
    """Base error for Stage 4 operations."""


class Stage4NotReady(Stage4Error):
    """The completion barrier or the scientific-input authentication did not pass."""

    def __init__(self, message: str, barrier: "StrictBarrier | None" = None):
        super().__init__(message)
        self.barrier = barrier


class Stage4ConflictError(Stage4Error):
    """A different result/job already exists where an immutable one is recorded."""


class Stage4State:
    """States of a run-level Stage 4 job-ledger entry."""

    SUBMIT_INTENT = "SUBMIT_INTENT"    # written before the provider is contacted
    SUBMITTED = "SUBMITTED"            # provider Job ID durable
    UNCERTAIN = "UNCERTAIN"            # outcome of the submission unknown -> operator must resolve
    COMPLETE = "COMPLETE"              # result retrieved, verified and persisted
    FAILED = "FAILED"                  # provider reported failure (definite)
    REJECTED = "REJECTED"              # provider said DONE but the returned result failed verification
    ABANDONED = "ABANDONED"            # operator asserted nothing was accepted
    ACTIVE = frozenset({"SUBMIT_INTENT", "SUBMITTED", "UNCERTAIN"})


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _ensure_bundle_on_path() -> None:
    """The inherited reconstruction imports ``quantum_fragment_methods`` from the bundle directory."""
    for p in (_BUNDLE, _BUNDLE / "vendor" / "qiskit-addon-sqd"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.append(str(p))


def _require_manifest(ctx: RunContext):
    if ctx.manifest is None:
        raise Stage4Error(f"Run {ctx.run_id!r} has no frozen manifest; Stage 4 cannot derive its inputs.")
    return ctx.manifest


# ─────────────────────────────────────────────────────────────────────────────
#  Manifest-derived facts
# ─────────────────────────────────────────────────────────────────────────────

def manifest_composition(manifest: Any) -> dict[str, Any]:
    """Solver composition planned by the frozen manifest (general: any solver mix, any fragment count)."""
    assign = {int(f): manifest.solver_for(int(f)) for f in manifest.fragment_ids}
    counts = Counter(assign.values())
    return {
        "total": len(assign),
        "counts": dict(sorted(counts.items())),
        "fragments_by_solver": {s: sorted(f for f, v in assign.items() if v == s) for s in sorted(counts)},
        "quantum_fragment_ids": sorted(f for f, v in assign.items() if v in QUANTUM_SOLVERS),
        "is_classical_only": not any(v in QUANTUM_SOLVERS for v in assign.values()),
    }


@dataclass
class ScientificInputs:
    """Authenticated Stage 1/2 scientific inputs of the frozen experiment."""

    mf_ref: Any
    ewf_ref: Any
    mf_path: Path
    ewf_path: Path
    e_hf: float
    hashes: dict[int, str]
    fragment_ids: list[int]


def authenticate_scientific_inputs(ctx: RunContext) -> ScientificInputs:
    """Authenticate Stage 1 (E_HF) and Stage 2 (EWF data) against the frozen manifest; fail closed."""
    manifest = _require_manifest(ctx)
    mf_ref = load_stage_ref(ctx, "meanfield")
    ewf_ref = load_stage_ref(ctx, "embedding_ewf")
    if mf_ref is None:
        raise Stage4Error("Stage 1 'meanfield' reference is not published for this run")
    if ewf_ref is None:
        raise Stage4Error("Stage 2 'embedding_ewf' reference is not published for this run")
    if mf_ref.file_sha256 != manifest.hf_artifact_ref.get("file_sha256"):
        raise Stage4Error("published Stage 1 artifact differs from the frozen manifest")
    if ewf_ref.file_sha256 != manifest.embedding_artifact_ref.get("file_sha256"):
        raise Stage4Error("published Stage 2 artifact differs from the frozen manifest")
    for label, ref in (("Stage 1", mf_ref), ("Stage 2", ewf_ref)):
        verification = ctx.store.verify(ref)
        if not verification.ok:
            raise Stage4Error(f"{label} artifact failed local verification: {verification.reason}")
    try:
        e_hf = float(mf_ref.meta["hf_energy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage4Error("authenticated Stage 1 reference carries no usable hf_energy") from exc
    if not math.isfinite(e_hf):
        raise Stage4Error("authenticated Stage 1 hf_energy is not finite")
    try:
        hashes = hamiltonian_hashes_from_dumpfile(Path(ewf_ref.uri))
    except Exception as exc:  # noqa: BLE001
        raise Stage4Error(f"Stage 2 dump could not be read for Hamiltonian identities: {exc}") from exc
    ids = sorted(hashes)
    if ids != sorted(int(f) for f in manifest.fragment_ids):
        raise Stage4Error(
            "Stage 2 fragment set does not belong to the frozen experiment "
            f"(dump has {len(ids)} fragments, manifest {len(manifest.fragment_ids)})")
    return ScientificInputs(mf_ref=mf_ref, ewf_ref=ewf_ref, mf_path=Path(mf_ref.uri),
                            ewf_path=Path(ewf_ref.uri), e_hf=e_hf, hashes=hashes, fragment_ids=ids)


# ─────────────────────────────────────────────────────────────────────────────
#  Strict completion barrier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrictBarrier:
    """Outcome of the manifest-derived completion barrier (extends the inherited ``BarrierReport``)."""

    report: Any                                   # reconstruction_coordinator.BarrierReport
    refs: dict[int, FragmentResultReference]      # verified references (only complete & clean fragments)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.report.passed)

    def as_dict(self) -> dict[str, Any]:
        out = self.report.as_dict()
        out["strict"] = True
        out.update({k: v for k, v in self.details.items()})
        return out


def _is_replay(task: Any) -> bool:
    return str(getattr(task, "backend", "") or "") == "replay" or \
        str(getattr(task, "qpu_job_id", "") or "").startswith("replay_")


def run_strict_barrier(ctx: RunContext, inputs: ScientificInputs | None = None) -> StrictBarrier:
    """Validate that every manifest fragment has exactly one authoritative, verified COMPLETE result.

    Enforced per required fragment: exactly one task of the right type; COMPLETE state; solver equal to the
    frozen assignment; task Hamiltonian SHA-256 equal to the authenticated Stage 2 hash; artifact inside the
    organizer store; immutable frs/1 file loaded with run/fragment/solver/Hamiltonian identity, file SHA-256
    and recomputed payload digest; finite complete dimensionally consistent 1-/2-RDMs; required scientific
    metadata; for quantum fragments ``qpu_job_id``, ``measurement_sha256``, TrimSQD policy and a valid
    frozen protocol digest.  Fragments outside the manifest are rejected.  Nothing is repaired or skipped.
    """
    manifest = _require_manifest(ctx)
    inputs = inputs or authenticate_scientific_inputs(ctx)
    required = sorted(int(f) for f in manifest.fragment_ids)
    assign = {f: manifest.solver_for(f) for f in required}
    store_root = Path(ctx.store.root).resolve()

    by_frag: dict[int, list[Any]] = {}
    for t in ctx.registry.list_tasks(ctx.run_id):
        by_frag.setdefault(int(t.fragment_id), []).append(t)

    missing: list[int] = []
    incomplete: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    duplicates: list[int] = []
    wrong_solver: list[int] = []
    invalid_rdms: list[int] = []
    checksum_failures: list[int] = []
    metadata_failures: list[int] = []
    quantum_failures: list[int] = []
    errors: list[str] = []
    refs: dict[int, FragmentResultReference] = {}
    provenance: dict[str, dict[str, Any]] = {}
    actual_composition: Counter = Counter()
    clean: set[int] = set()

    protocol_digest = str(manifest.protocol_digest or "")
    has_quantum = any(s in QUANTUM_SOLVERS for s in assign.values())
    protocol_ok = (not has_quantum) or bool(rc._HEX64_RE.match(protocol_digest))
    if not protocol_ok:
        errors.append("manifest has quantum fragments but no valid promoted-protocol digest")

    def _row(fid: int, task: Any, reason: str) -> dict[str, Any]:
        return {"fragment_id": fid, "solver": assign[fid], "state": getattr(task, "state", None),
                "reason": reason}

    for fid in required:
        solver = assign[fid]
        tasks = by_frag.get(fid, [])
        if not tasks:
            missing.append(fid)
            errors.append(f"Fragment {fid}: no task registered for run {ctx.run_id}")
            continue
        if len(tasks) > 1:
            duplicates.append(fid)
            errors.append(f"Fragment {fid}: {len(tasks)} tasks exist ({[t.task_type for t in tasks]}); "
                          "exactly one authoritative task is required")
            continue
        task = tasks[0]
        expected_type = TaskType.QUANTUM if solver in QUANTUM_SOLVERS else TaskType.CLASSICAL

        if task.state != TaskState.COMPLETE:
            reason = (task.failure_reason or "").strip() or f"state {task.state}"
            if task.state in (TaskState.FAILED, TaskState.CANCELED, TaskState.RETRY_PENDING):
                failed.append(_row(fid, task, reason))
            elif task.state == TaskState.AMBIGUOUS or (
                    task.state == TaskState.SUBMITTING and not task.serverless_job_id and not task.qpu_job_id):
                ambiguous.append(_row(fid, task, "submission outcome unknown; operator resolution required"))
            else:
                incomplete.append(_row(fid, task, f"state {task.state}"))
            errors.append(f"Fragment {fid}: not COMPLETE (state {task.state})")
            continue

        ok = True
        if task.solver != solver or task.task_type != expected_type:
            wrong_solver.append(fid)
            ok = False
            errors.append(f"Fragment {fid}: task is {task.task_type}/{task.solver}, manifest requires "
                          f"{expected_type}/{solver}")
        if task.hamiltonian_sha256 != inputs.hashes.get(fid):
            checksum_failures.append(fid)
            ok = False
            errors.append(f"Fragment {fid}: task Hamiltonian SHA-256 differs from the authenticated Stage 2 hash")
        if not task.result_artifact_uri or not task.result_payload_digest:
            checksum_failures.append(fid)
            errors.append(f"Fragment {fid}: no recorded result artifact / payload digest")
            continue
        path = Path(task.result_artifact_uri)
        try:
            inside = path.resolve().is_relative_to(store_root)
        except OSError:
            inside = False
        if not inside or not path.is_file():
            checksum_failures.append(fid)
            errors.append(f"Fragment {fid}: result artifact is not an existing file inside the organizer store")
            continue

        if solver in QUANTUM_SOLVERS:
            if not str(task.qpu_job_id or "").strip():
                quantum_failures.append(fid)
                ok = False
                errors.append(f"Fragment {fid}: quantum task has no qpu_job_id provenance")
            declared = task.extra.get("protocol_digest")
            if declared and declared != protocol_digest:
                quantum_failures.append(fid)
                ok = False
                errors.append(f"Fragment {fid}: task protocol digest differs from the frozen manifest protocol")
            if not protocol_ok:
                quantum_failures.append(fid)
                ok = False

        file_sha = _sha256_file(path)
        result, problems = rc.inspect_fragment_result(
            path, run_id=ctx.run_id, fragment_id=fid, solver=solver,
            hamiltonian_sha256=inputs.hashes[fid], expected_payload_digest=task.result_payload_digest,
            expected_file_sha256=file_sha)
        for category, bucket in (("checksum", checksum_failures), ("invalid_rdm", invalid_rdms),
                                 ("metadata", metadata_failures), ("quantum_provenance", quantum_failures)):
            if problems.get(category):
                ok = False
                bucket.append(fid)
                errors.extend(problems[category])

        if result is not None:
            md = result.metadata or {}
            actual_composition[str(md.get("solver_label", "unknown"))] += 1
            if solver in QUANTUM_SOLVERS:
                provenance[str(fid)] = {
                    "solver": solver, "qpu_job_id": str(task.qpu_job_id or ""), "backend": str(task.backend or ""),
                    "is_replay": _is_replay(task),
                    "measurement_sha256": md.get("measurement_sha256"), "policy": md.get("policy"),
                    "trim_ratio": md.get("trim_ratio"), "sqd_config_hash": md.get("sqd_config_hash"),
                    "protocol_digest": protocol_digest,
                }
        if ok and result is not None:
            clean.add(fid)
            refs[fid] = FragmentResultReference(
                path=path, file_sha256=file_sha, payload_digest=task.result_payload_digest, fragment_id=fid,
                run_id=ctx.run_id, solver=solver, hamiltonian_sha256=inputs.hashes[fid],
                extra={"artifact_name": path.name})

    stray = sorted(f for f in by_frag if f not in set(required))
    if stray:
        errors.append(f"registry holds tasks for fragments {stray} outside the frozen manifest")

    completed = len(clean)
    not_complete = sorted({r["fragment_id"] for r in incomplete + failed + ambiguous} | set(missing))
    passed = (completed == len(required) and not errors and not stray)
    report = rc.BarrierReport(
        run_id=ctx.run_id, passed=passed, total_required=len(required), completed=completed,
        missing=not_complete, wrong_solver=sorted(set(wrong_solver)), invalid_rdms=sorted(set(invalid_rdms)),
        checksum_failures=sorted(set(checksum_failures)),
        quantum_provenance_failures=sorted(set(quantum_failures)), errors=errors, checked_utc=rc._utc_now())
    planned = manifest_composition(manifest)
    details = {
        "manifest_digest": manifest.manifest_digest,
        "required_fragments": required,
        "no_task": sorted(missing), "incomplete": incomplete, "failed": failed, "ambiguous": ambiguous,
        "duplicate_tasks": sorted(duplicates), "unexpected_fragments": stray,
        "metadata_failures": sorted(set(metadata_failures)),
        "planned_composition": planned["counts"], "actual_composition": dict(sorted(actual_composition.items())),
        "quantum_provenance": provenance,
        "subset_complete": bool(completed) and completed < len(required),
    }
    return StrictBarrier(report=report, refs=refs, details=details)


def format_barrier_report(barrier: StrictBarrier) -> str:
    """Human-readable incomplete / failed / ambiguous summary (what blocks reconstruction)."""
    r, d = barrier.report, barrier.details
    lines = [f"Completion barrier: {'PASSED' if barrier.passed else 'NOT PASSED'} — "
             f"{r.completed}/{r.total_required} fragments complete and verified"]
    if d.get("subset_complete"):
        lines.append("  NOTE: a subset-complete campaign is not a complete molecular experiment.")
    for title, key in (("INCOMPLETE", "incomplete"), ("FAILED / RETRY-PENDING", "failed"),
                       ("AMBIGUOUS (operator action required)", "ambiguous")):
        rows = d.get(key) or []
        if rows:
            lines.append(f"  {title}: {len(rows)}")
            for row in rows[:20]:
                lines.append(f"    fragment {row['fragment_id']:>3} {row['solver']:<8} {row['state']}: {row['reason']}")
            if len(rows) > 20:
                lines.append(f"    ... {len(rows) - 20} more")
    for title, ids in (("no task", d.get("no_task")), ("duplicate tasks", d.get("duplicate_tasks")),
                       ("wrong solver/type", r.wrong_solver), ("invalid RDMs", r.invalid_rdms),
                       ("checksum/identity failures", r.checksum_failures),
                       ("metadata failures", d.get("metadata_failures")),
                       ("quantum provenance failures", r.quantum_provenance_failures),
                       ("outside manifest", d.get("unexpected_fragments"))):
        if ids:
            lines.append(f"  {title}: {sorted(ids)}")
    lines.append(f"  planned composition: {d.get('planned_composition')}")
    return "\n".join(lines)


def readiness_summary(ctx: RunContext) -> dict[str, Any]:
    """Read-only readiness for reconstruction (no network, no mutation).  Never raises on a failed gate."""
    out: dict[str, Any] = {"run_id": ctx.run_id, "ready": False, "barrier": None, "text": "", "error": None}
    if ctx.manifest is None:
        out["text"] = "Run manifest is not frozen yet; Stage 4 readiness cannot be evaluated."
        return out
    out["composition"] = manifest_composition(ctx.manifest)
    try:
        barrier = run_strict_barrier(ctx)
    except Stage4Error as exc:
        out["error"] = str(exc)
        out["text"] = f"Scientific inputs could not be authenticated: {exc}"
        return out
    out.update(ready=barrier.passed, barrier=barrier.as_dict(), text=format_barrier_report(barrier))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  ReconstructionSummary (recon/2)
# ─────────────────────────────────────────────────────────────────────────────

def _finite(value: Any, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage4Error(f"{name} is not a number: {value!r}") from exc
    if not math.isfinite(x):
        raise Stage4Error(f"{name} is not finite: {x!r}")
    return x


def validate_summary(summary: Mapping[str, Any]) -> None:
    """Check schema and the energy invariants of a ``recon/2`` summary; raises :class:`Stage4Error`."""
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise Stage4Error(f"unsupported reconstruction summary schema {summary.get('schema')!r}")
    for key in ("run_id", "manifest", "e_hf", "e_corr_total", "e_total", "fragment_energies", "fragments",
                "solver_used", "composition", "hamiltonian_sha256s", "result_refs", "quantum",
                "barrier_report", "scientific_inputs", "execution"):
        if key not in summary:
            raise Stage4Error(f"reconstruction summary lacks {key!r}")
    e_hf = _finite(summary["e_hf"], "e_hf")
    e_corr = _finite(summary["e_corr_total"], "e_corr_total")
    e_total = _finite(summary["e_total"], "e_total")
    energies = {str(k): _finite(v, f"fragment {k} e_corr") for k, v in summary["fragment_energies"].items()}
    keys = set(energies)
    for key in ("fragments", "solver_used", "hamiltonian_sha256s", "result_refs"):
        if set(map(str, summary[key])) != keys:
            raise Stage4Error(f"reconstruction summary {key!r} does not cover exactly the reconstructed fragments")
    total = 0.0
    for k in sorted(energies, key=int):
        total += energies[k]
    if abs(total - e_corr) > _INVARIANT_TOL:
        raise Stage4Error(f"sum of fragment contributions ({total!r}) != E_corr_total ({e_corr!r})")
    if abs((e_hf + e_corr) - e_total) > _INVARIANT_TOL:
        raise Stage4Error(f"E_total ({e_total!r}) != E_HF + E_corr ({e_hf + e_corr!r})")
    if not summary["barrier_report"].get("passed"):
        raise Stage4Error("reconstruction summary embeds a barrier report that did not pass")


def _build_summary(ctx: RunContext, inputs: ScientificInputs, barrier: StrictBarrier,
                   res: Mapping[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    manifest = _require_manifest(ctx)
    required = barrier.details["required_fragments"]
    assign = {f: manifest.solver_for(f) for f in required}
    frags_in = {int(k): v for k, v in res["fragments"].items()}
    if sorted(frags_in) != required:
        raise Stage4Error(f"reconstruction covered fragments {sorted(frags_in)}, manifest requires {required}")
    fragments: dict[str, dict[str, Any]] = {}
    fragment_energies: dict[str, float] = {}
    for fid in required:
        v = frags_in[fid]
        if str(v.get("solver", "")) != assign[fid]:
            raise Stage4Error(f"fragment {fid}: reconstructed under solver {v.get('solver')!r}, "
                              f"manifest requires {assign[fid]!r}")
        e = _finite(v["e_corr"], f"fragment {fid} e_corr")
        fragment_energies[str(fid)] = e
        fragments[str(fid)] = {"e_corr": e, "solver": assign[fid], "n_orb": int(v.get("n_orb", 0)),
                               "converged": bool(v.get("converged", True)),
                               "rdm_integrated": bool(v.get("rdm_integrated", False))}
    e_hf = _finite(res["e_hf"], "e_hf")
    if abs(e_hf - inputs.e_hf) > _E_TOL:
        raise Stage4Error(f"E_HF of the reconstruction ({e_hf!r}) differs from the authenticated Stage 1 "
                          f"value ({inputs.e_hf!r})")
    e_corr_total = float(res.get("e_corr_total", res.get("e_corr")))
    e_total = float(res["e_total"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "run_id": ctx.run_id,
        "manifest": {"manifest_digest": manifest.manifest_digest, "protocol_digest": manifest.protocol_digest,
                     "ligand": manifest.ligand, "basis": manifest.basis, "n_fragments": len(required),
                     "identity_sha256": hashlib.sha256(_canonical(manifest_identity(manifest)).encode()).hexdigest()},
        "e_hf": e_hf, "e_corr_total": e_corr_total, "e_total": e_total,
        "fragment_energies": fragment_energies, "fragments": fragments,
        "solver_used": {str(f): assign[f] for f in required},
        "composition": {"planned": barrier.details["planned_composition"],
                        "actual": barrier.details["actual_composition"]},
        "hamiltonian_sha256s": {str(f): inputs.hashes[f] for f in required},
        "result_refs": {str(f): {k: v for k, v in barrier.refs[f].as_dict().items() if k != "path"}
                        for f in required},
        "quantum": {"protocol_digest": manifest.protocol_digest,
                    "fragments": barrier.details["quantum_provenance"]},
        "barrier_report": barrier.as_dict(),
        "scientific_inputs": {"mf_file_sha256": inputs.mf_ref.file_sha256, "ewf_file_sha256": inputs.ewf_ref.file_sha256,
                              "hf_energy_artifact": inputs.e_hf,
                              "e_hf_source": "authenticated Stage 1 artifact (mf_data.pkl)"},
        "execution": {**execution, "delegate": "step4_reconstruct.run_reconstruct_from_refs",
                      "reconstruction_utc": _now()},
    }
    validate_summary(summary)
    return summary


def _scientific_identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {"run_id": summary["run_id"], "manifest_digest": summary["manifest"]["manifest_digest"],
            "solver_used": dict(summary["solver_used"]), "hamiltonian_sha256s": dict(summary["hamiltonian_sha256s"]),
            "payload_digests": {k: v.get("payload_digest") for k, v in summary["result_refs"].items()}}


def _summaries_agree(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if _scientific_identity(a) != _scientific_identity(b):
        return False
    if abs(a["e_hf"] - b["e_hf"]) > _E_TOL or abs(a["e_total"] - b["e_total"]) > _E_TOL:
        return False
    return all(abs(a["fragment_energies"][k] - b["fragment_energies"][k]) <= _E_TOL for k in a["fragment_energies"])


def _summary_path(ctx: RunContext) -> Path:
    return ctx.paths.run_dir / SUMMARY_FILENAME


def load_authoritative_summary(ctx: RunContext) -> dict[str, Any] | None:
    """The persisted authoritative reconstruction summary, validated; ``None`` if none exists yet."""
    path = _summary_path(ctx)
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise Stage4Error(f"authoritative reconstruction summary {path.name} is not valid JSON") from exc
    validate_summary(summary)
    if summary["run_id"] != ctx.run_id:
        raise Stage4Error("authoritative reconstruction summary belongs to a different run")
    return summary


def persist_summary(ctx: RunContext, summary: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Write the authoritative summary once.  Returns ``(authoritative_summary, created)``.

    A scientifically identical re-import is idempotent (the first execution provenance is kept); a
    conflicting result raises :class:`Stage4ConflictError`.  The file is also published to the artifact
    store (idempotent) so that it is addressable like every other run artifact.
    """
    validate_summary(summary)
    path = _summary_path(ctx)
    created = _write_once(path, json.dumps(summary, indent=2, sort_keys=True))
    authoritative = summary
    if not created:
        authoritative = load_authoritative_summary(ctx)
        if not _summaries_agree(authoritative, summary):
            raise Stage4ConflictError(
                "an authoritative reconstruction summary already exists and differs from the new result; "
                "immutable results are never overwritten")
    identity = hashlib.sha256(_canonical(_scientific_identity(authoritative)).encode()).hexdigest()
    try:
        ctx.store.publish(path, KIND_SUMMARY_JSON, ctx.run_id, artifact_name=SUMMARY_FILENAME,
                          payload_digest=identity,
                          meta={"schema": SUMMARY_SCHEMA, "e_total": authoritative["e_total"]})
    except ArtifactConflict as exc:
        raise Stage4ConflictError(f"summary artifact conflict in the store: {exc}") from exc
    return authoritative, created


# ─────────────────────────────────────────────────────────────────────────────
#  Local reconstruction (organizer CPU) — reference path and fallback
# ─────────────────────────────────────────────────────────────────────────────

def _stage_data_dir(inputs: ScientificInputs, scratch: Path) -> Path:
    """``run_reconstruct_from_refs`` writes a temp pickle into ``data_dir``; never point it at the store."""
    data = scratch / "data"
    data.mkdir(parents=True, exist_ok=True)
    for src, name in ((inputs.mf_path, "mf_data.pkl"), (inputs.ewf_path, "ewf_dumpfile.h5")):
        dest = data / name
        try:
            os.symlink(src.resolve(), dest)
        except OSError:
            shutil.copyfile(src, dest)
    return data


def reconstruct_local(ctx: RunContext, *, persist: bool = True) -> dict[str, Any]:
    """Validate the barrier and reconstruct on the organizer CPU through the inherited EWF code.

    Idempotent: when an authoritative summary already exists and still matches the current verified
    fragment results, it is returned without recomputation; a mismatch raises.
    """
    t0 = time.time()
    inputs = authenticate_scientific_inputs(ctx)
    barrier = run_strict_barrier(ctx, inputs)
    if not barrier.passed:
        raise Stage4NotReady(format_barrier_report(barrier), barrier)

    existing = load_authoritative_summary(ctx)
    required = barrier.details["required_fragments"]
    if existing is not None:
        probe = {"run_id": ctx.run_id, "manifest": existing["manifest"], "solver_used": existing["solver_used"],
                 "hamiltonian_sha256s": {str(f): inputs.hashes[f] for f in required},
                 "result_refs": {str(f): barrier.refs[f].as_dict() for f in required}}
        if _scientific_identity(probe) != _scientific_identity(existing):
            raise Stage4ConflictError("the authoritative reconstruction no longer matches the verified "
                                      "fragment results; refusing to overwrite an immutable result")
        return {"summary": existing, "reused": True, "created": False, "barrier": barrier.as_dict()}

    _ensure_bundle_on_path()
    scratch = Path(tempfile.mkdtemp(prefix="qfm_stage4_local_"))
    try:
        data_dir = _stage_data_dir(inputs, scratch)
        res = rc.run_reconstruct_from_refs(
            data_dir=str(data_dir), run_name=ctx.run_id, fragment_result_refs=dict(barrier.refs),
            results_dir=str(scratch / "results"), expected_fragment_ids=required)
        execution = {"mode": "local", "serverless_job_id": None, "attempt": None, "software": _software_versions(),
                     "elapsed_s": time.time() - t0}
        summary = _build_summary(ctx, inputs, barrier, res, execution)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    created = False
    if persist:
        summary, created = persist_summary(ctx, summary)
    return {"summary": summary, "reused": False, "created": created, "barrier": barrier.as_dict()}


# ─────────────────────────────────────────────────────────────────────────────
#  Run-level Stage 4 job ledger
# ─────────────────────────────────────────────────────────────────────────────

def _ledger_path(ctx: RunContext) -> Path:
    return ctx.paths.run_dir / LEDGER_FILENAME


def _read_ledger(ctx: RunContext) -> dict[str, Any]:
    path = _ledger_path(ctx)
    if not path.is_file():
        return {"schema": LEDGER_SCHEMA, "run_id": ctx.run_id, "jobs": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise Stage4Error(f"Stage 4 ledger {path.name} is corrupt; operator inspection required") from exc
    if data.get("schema") != LEDGER_SCHEMA or data.get("run_id") != ctx.run_id:
        raise Stage4Error(f"Stage 4 ledger {path.name} has an unexpected schema or run id")
    return data


def _write_ledger(ctx: RunContext, data: dict[str, Any]) -> None:
    """Atomic replace (temp file + fsync + rename): a crash leaves the previous or the new ledger, never half."""
    path = _ledger_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".s4_", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _event(entry: dict[str, Any], kind: str, **fields: Any) -> None:
    entry.setdefault("events", []).append({"event": kind, "at": _now(), **fields})
    entry["updated"] = _now()


def _effective_state(entry: Mapping[str, Any]) -> str:
    """A crash between intent and Job-ID leaves SUBMIT_INTENT without an id: that is an uncertain submission."""
    if entry["state"] == Stage4State.SUBMIT_INTENT and not entry.get("serverless_job_id"):
        return Stage4State.UNCERTAIN
    return entry["state"]


def stage4_status(ctx: RunContext) -> dict[str, Any]:
    """Read-only status of the Stage 4 ledger and the authoritative result (no provider contact)."""
    ledger = _read_ledger(ctx)
    rows = [{"attempt": e["attempt"], "state": _effective_state(e), "serverless_job_id": e.get("serverless_job_id", ""),
             "function_size": e.get("function_size"), "last_event": (e.get("events") or [{}])[-1].get("event", ""),
             "error": e.get("error", "")} for e in ledger["jobs"]]
    summary = load_authoritative_summary(ctx)
    return {"run_id": ctx.run_id, "jobs": rows,
            "active": [r for r in rows if r["state"] in Stage4State.ACTIVE],
            "uncertain": [r for r in rows if r["state"] == Stage4State.UNCERTAIN],
            "authoritative_summary": None if summary is None else {
                "e_total": summary["e_total"], "mode": summary["execution"].get("mode"),
                "serverless_job_id": summary["execution"].get("serverless_job_id")}}


# ─────────────────────────────────────────────────────────────────────────────
#  Job preparation (publishes inputs through the existing transport adapter)
# ─────────────────────────────────────────────────────────────────────────────

_PATH_KEYS = ("data_dir", "results_dir", "ewf_dumpfile_path", "result_dir", "result_path", "fragment_result_refs",
              "mf_data_path")


def _spec_identity_digest(spec: Mapping[str, Any]) -> str:
    """Digest of everything scientifically meaningful in a spec (descriptor timestamps excluded)."""
    def _d(desc: Mapping[str, Any]) -> list[Any]:
        return [desc["remote_name"], desc["file_sha256"], desc["file_size_bytes"], desc.get("payload_digest")]
    return hashlib.sha256(_canonical({
        "run_id": spec["run_id"], "attempt": spec["attempt"], "manifest_digest": spec["manifest_digest"],
        "required": spec["required_fragments"], "assign": spec["solver_assignments"],
        "hashes": spec["hamiltonian_hashes"], "mf": _d(spec["mf_input"]), "ewf": _d(spec["ewf_input"]),
        "frags": {k: _d(v) for k, v in spec["fragment_inputs"].items()},
        "expected_e_hf": spec["expected_e_hf"]}).encode()).hexdigest()


def prepare_stage4_job(ctx: RunContext, adapter: ServerlessTransportAdapter, *, attempt: int,
                       inputs: ScientificInputs | None = None, barrier: StrictBarrier | None = None
                       ) -> tuple[dict[str, Any], str]:
    """Validate readiness, publish every scientific input and return ``(spec, spec_identity_digest)``.

    The specification is JSON-serialisable and path-free: transport descriptors only.  Uploads are
    content-addressed (``upload_input``), so repeating this is harmless.  The spec is persisted write-once per
    attempt so a restart re-uses the identical specification.
    """
    manifest = _require_manifest(ctx)
    inputs = inputs or authenticate_scientific_inputs(ctx)
    barrier = barrier or run_strict_barrier(ctx, inputs)
    if not barrier.passed:
        raise Stage4NotReady(format_barrier_report(barrier), barrier)
    required = barrier.details["required_fragments"]

    mf_desc = adapter.upload_input(inputs.mf_path, run_id=ctx.run_id, file_role="mf_data",
                                   expected_sha256=inputs.mf_ref.file_sha256)
    ewf_desc = adapter.upload_input(inputs.ewf_path, run_id=ctx.run_id, file_role="ewf_dumpfile",
                                    expected_sha256=inputs.ewf_ref.file_sha256)
    frag_inputs: dict[str, dict[str, Any]] = {}
    for fid in required:
        ref = barrier.refs[fid]
        d = adapter.upload_input(ref.path, run_id=ctx.run_id, file_role=f"frag{fid}",
                                 expected_sha256=ref.file_sha256)
        d = replace(d, artifact_kind="fragment_result", fragment_id=fid, solver=ref.solver,
                    payload_digest=ref.payload_digest, hamiltonian_sha256=ref.hamiltonian_sha256)
        frag_inputs[str(fid)] = d.as_dict()

    spec = {
        "schema": SPEC_SCHEMA, "run_id": ctx.run_id, "attempt": int(attempt),
        "manifest_digest": manifest.manifest_digest, "protocol_digest": manifest.protocol_digest,
        "required_fragments": list(required),
        "solver_assignments": {str(f): manifest.solver_for(f) for f in required},
        "hamiltonian_hashes": {str(f): inputs.hashes[f] for f in required},
        "mf_input": mf_desc.as_dict(), "ewf_input": ewf_desc.as_dict(), "fragment_inputs": frag_inputs,
        "expected_e_hf": inputs.e_hf,
    }
    if any(k in spec for k in _PATH_KEYS):
        raise Stage4Error("internal error: a Stage 4 specification must not carry filesystem paths")
    json.dumps(spec)
    digest = _spec_identity_digest(spec)

    spec_file = ctx.paths.run_dir / f"stage4_spec_a{int(attempt):02d}.json"
    if not _write_once(spec_file, json.dumps(spec, indent=2, sort_keys=True)):
        stored = json.loads(spec_file.read_text(encoding="utf-8"))
        if _spec_identity_digest(stored) != digest:
            raise Stage4ConflictError(f"{spec_file.name} already exists with a different specification")
        spec = stored          # the persisted (immutable) specification is the one that is submitted
    return spec, digest


# ─────────────────────────────────────────────────────────────────────────────
#  Dispatch (explicitly authorized; never duplicates)
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_stage4_job(ctx: RunContext, service: Any, adapter: ServerlessTransportAdapter, *,
                        authorize: bool = False, function_title: str = STAGE4_FUNCTION_TITLE,
                        function_size: str = "l") -> dict[str, Any]:
    """Submit one CPU Stage 4 reconstruction job.  Requires ``authorize=True``.

    Refuses (without contacting the provider) when an authoritative result exists or when an earlier job
    is still active or uncertain: reconcile it, or resolve the uncertainty explicitly, first.
    """
    if authorize is not True:
        raise Stage4Error("remote Stage 4 dispatch requires explicit authorization (authorize=True)")
    if function_size not in VALID_RESOURCE_FAMILIES:
        raise Stage4Error(f"function_size must be one of {sorted(VALID_RESOURCE_FAMILIES)} (CPU families only)")
    if load_authoritative_summary(ctx) is not None:
        return {"status": "ALREADY_RECONSTRUCTED", "dispatched": False}
    ledger = _read_ledger(ctx)
    active = [e for e in ledger["jobs"] if _effective_state(e) in Stage4State.ACTIVE]
    if active:
        e = active[-1]
        return {"status": "ACTIVE_JOB_EXISTS", "dispatched": False, "attempt": e["attempt"],
                "state": _effective_state(e), "serverless_job_id": e.get("serverless_job_id", ""),
                "action": "reconcile the existing job, or resolve an uncertain submission explicitly"}

    attempt = len(ledger["jobs"]) + 1
    spec, digest = prepare_stage4_job(ctx, adapter, attempt=attempt)
    entry = {"attempt": attempt, "state": Stage4State.SUBMIT_INTENT, "spec_digest": digest,
             "spec_file": f"stage4_spec_a{attempt:02d}.json", "function": function_title,
             "function_size": function_size, "serverless_job_id": "", "created": _now(), "updated": _now(),
             "events": [], "error": ""}
    _event(entry, "SUBMIT_INTENT", function=function_title, function_size=function_size, spec_digest=digest)
    ledger["jobs"].append(entry)
    _write_ledger(ctx, ledger)                      # durable intent BEFORE the provider is contacted

    try:
        job = service.run(function_title, arguments=spec, function_size=function_size)
        job_id = str(getattr(job, "job_id", "") or "")
        if not job_id:
            raise Stage4Error("provider returned a job handle without a job_id")
    except Exception as exc:  # noqa: BLE001 — the request may or may not have been accepted: UNCERTAIN
        entry["state"] = Stage4State.UNCERTAIN
        entry["error"] = f"{type(exc).__name__}: {exc}"[:_REMOTE_ERROR_LIMIT]
        _event(entry, "SUBMIT_UNCERTAIN", error=entry["error"])
        _write_ledger(ctx, ledger)
        return {"status": "SUBMISSION_UNCERTAIN", "dispatched": False, "attempt": attempt,
                "error": entry["error"], "action": "resolve_uncertain_stage4_submission(...)"}

    entry["serverless_job_id"] = job_id            # Job ID durable immediately
    entry["state"] = Stage4State.SUBMITTED
    _event(entry, "SUBMIT_ACCEPTED", serverless_job_id=job_id)
    _write_ledger(ctx, ledger)
    return {"status": "SUBMITTED", "dispatched": True, "attempt": attempt, "serverless_job_id": job_id,
            "function_size": function_size}


# ─────────────────────────────────────────────────────────────────────────────
#  Reconciliation: retrieve, verify, persist
# ─────────────────────────────────────────────────────────────────────────────

def _verify_remote_payload(ctx: RunContext, entry: Mapping[str, Any], payload: Any) -> dict[str, Any]:
    """Authenticate a returned worker result against locally re-verified state; return a ``recon/2`` summary."""
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise Stage4Error(f"result payload is {type(payload).__name__}, not a dictionary")
    if payload.get("status") != "COMPLETE":
        raise Stage4Error(f"worker reported {payload.get('status')!r}: "
                          f"{str(payload.get('error', 'unknown error'))[:_REMOTE_ERROR_LIMIT]}")
    if payload.get("schema") != RESULT_SCHEMA:
        raise Stage4Error(f"unexpected result schema {payload.get('schema')!r}")
    manifest = _require_manifest(ctx)
    inputs = authenticate_scientific_inputs(ctx)
    barrier = run_strict_barrier(ctx, inputs)               # local state is re-verified, not trusted
    if not barrier.passed:
        raise Stage4NotReady("local completion barrier no longer passes; remote result not accepted", barrier)
    required = barrier.details["required_fragments"]

    def _same(key: str, want: Any) -> None:
        if payload.get(key) != want:
            raise Stage4Error(f"worker result {key} does not match the dispatched job")
    _same("run_id", ctx.run_id)
    _same("manifest_digest", manifest.manifest_digest)
    _same("attempt", entry["attempt"])
    _same("mf_file_sha256", inputs.mf_ref.file_sha256)
    _same("ewf_file_sha256", inputs.ewf_ref.file_sha256)
    _same("solver_used", {str(f): manifest.solver_for(f) for f in required})
    _same("hamiltonian_sha256s", {str(f): inputs.hashes[f] for f in required})
    _same("payload_digests", {str(f): barrier.refs[f].payload_digest for f in required})
    if sorted(int(k) for k in payload.get("fragments", {})) != required:
        raise Stage4Error("worker result does not cover exactly the required fragments")
    res = {"e_hf": payload["e_hf"], "e_corr_total": payload["e_corr_total"], "e_total": payload["e_total"],
           "fragments": payload["fragments"]}
    execution = {"mode": "serverless", "serverless_job_id": entry["serverless_job_id"], "attempt": entry["attempt"],
                 "function": entry.get("function"), "function_size": entry.get("function_size"),
                 "spec_digest": entry.get("spec_digest"), "software": payload.get("software", {}),
                 "elapsed_s": payload.get("elapsed_s"), "result_schema": payload.get("schema")}
    return _build_summary(ctx, inputs, barrier, res, execution)


def reconcile_stage4_job(ctx: RunContext, service: Any) -> dict[str, Any]:
    """Observe the provider job(s), then retrieve, verify and persist a finished result.  Never submits."""
    report: dict[str, Any] = {"completed": [], "failed": [], "running": [], "uncertain": [], "rejected": []}
    ledger = _read_ledger(ctx)
    dirty = False
    for entry in ledger["jobs"]:
        state = _effective_state(entry)
        if state == Stage4State.UNCERTAIN:
            if entry["state"] != Stage4State.UNCERTAIN:
                entry["state"] = Stage4State.UNCERTAIN
                _event(entry, "RECOVERED_AS_UNCERTAIN", note="intent without a provider Job ID")
                dirty = True
            report["uncertain"].append({"attempt": entry["attempt"], "error": entry.get("error", "")})
            continue
        if state != Stage4State.SUBMITTED:
            continue
        job_id = entry["serverless_job_id"]
        try:
            job = service.job(job_id)
            if job is None:
                raise Stage4Error("provider reports no such job")
            status = str(job.status()).upper()
        except Exception as exc:  # noqa: BLE001 — a failed query is never evidence of failure
            report["running"].append({"attempt": entry["attempt"], "serverless_job_id": job_id,
                                      "status": "PROVIDER_QUERY_ERROR", "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        entry["last_provider_status"] = status
        if status in _PROVIDER_SUCCESS:
            try:
                summary = _verify_remote_payload(ctx, entry, job.result())
                _, created = persist_summary(ctx, summary)
            except Stage4Error as exc:
                entry["state"] = Stage4State.REJECTED
                entry["error"] = str(exc)[:_REMOTE_ERROR_LIMIT]
                _event(entry, "RESULT_REJECTED", error=entry["error"])
                report["rejected"].append({"attempt": entry["attempt"], "serverless_job_id": job_id,
                                           "reason": entry["error"]})
            except Exception as exc:  # noqa: BLE001 — retrieval trouble: stay SUBMITTED, retry next pass
                report["running"].append({"attempt": entry["attempt"], "serverless_job_id": job_id,
                                          "status": "RESULT_RETRIEVAL_ERROR",
                                          "error": f"{type(exc).__name__}: {exc}"[:200]})
            else:
                entry["state"] = Stage4State.COMPLETE
                _event(entry, "RESULT_IMPORTED", created=created)
                report["completed"].append({"attempt": entry["attempt"], "serverless_job_id": job_id,
                                            "e_total": summary["e_total"], "newly_persisted": created})
        elif status in _PROVIDER_FAILED or status in _PROVIDER_CANCELED:
            entry["state"] = Stage4State.FAILED
            entry["error"] = f"Serverless job {job_id} ended with provider status {status}"
            _event(entry, "PROVIDER_TERMINAL_FAILURE", provider_status=status)
            report["failed"].append({"attempt": entry["attempt"], "serverless_job_id": job_id, "status": status})
        else:
            report["running"].append({"attempt": entry["attempt"], "serverless_job_id": job_id, "status": status})
        dirty = True
    if dirty:
        _write_ledger(ctx, ledger)
    return report


def resolve_uncertain_stage4_submission(ctx: RunContext, *, reason: str, serverless_job_id: Optional[str] = None,
                                        abandon: bool = False, serverless_service: Any = None) -> dict[str, Any]:
    """Operator resolution of an uncertain Stage 4 submission: attach the real Job ID, or abandon the attempt."""
    if not reason or not reason.strip():
        raise Stage4Error("an operator reason is required")
    if bool(serverless_job_id) == bool(abandon):
        raise Stage4Error("provide exactly one of serverless_job_id or abandon=True")
    ledger = _read_ledger(ctx)
    uncertain = [e for e in ledger["jobs"] if _effective_state(e) == Stage4State.UNCERTAIN]
    if not uncertain:
        raise Stage4Error("there is no uncertain Stage 4 submission to resolve")
    entry = uncertain[-1]
    if abandon:
        entry["state"] = Stage4State.ABANDONED
        _event(entry, "SUBMIT_ABANDONED_BY_OPERATOR", reason=reason)
    else:
        if serverless_service is not None:
            try:
                if serverless_service.job(serverless_job_id) is None:
                    raise Stage4Error("provider does not know this job id")
            except Stage4Error:
                raise
            except Exception as exc:  # noqa: BLE001
                raise Stage4Error(f"could not confirm job {serverless_job_id!r} with the provider: {exc}") from exc
        entry["serverless_job_id"] = serverless_job_id
        entry["state"] = Stage4State.SUBMITTED
        _event(entry, "SUBMIT_RECOVERED_BY_OPERATOR", serverless_job_id=serverless_job_id, reason=reason)
    _write_ledger(ctx, ledger)
    return {"attempt": entry["attempt"], "state": entry["state"], "serverless_job_id": entry.get("serverless_job_id", "")}


# ─────────────────────────────────────────────────────────────────────────────
#  Notebook entry point (thin; the cell contains no orchestration)
# ─────────────────────────────────────────────────────────────────────────────

def run_notebook_step(ctx: RunContext, client: Any = None, *, prepare: bool = False, local_reconstruct: bool = False,
                      reconcile: bool = False, dispatch: bool = False, authorize_remote: bool = False,
                      function_size: str = "l", service: Any = None,
                      adapter: ServerlessTransportAdapter | None = None) -> dict[str, Any]:
    """One explicit operator step.  With every flag false nothing is read, written or contacted.

    ``prepare``            local only: authenticate inputs and evaluate the completion barrier.
    ``local_reconstruct``  local only: reconstruct on the organizer CPU and persist the authoritative result.
    ``reconcile``          contacts Serverless (read-only queries) and imports a finished verified result.
    ``dispatch``           submits ONE job — only with ``authorize_remote=True``, and only after reconciliation.
    ``service``/``adapter`` allow injecting doubles; otherwise they are built from ``client``.
    """
    out: dict[str, Any] = {}
    if not (prepare or local_reconstruct or reconcile or dispatch):
        return out
    if prepare:
        out["readiness"] = readiness_summary(ctx)
    if local_reconstruct:
        out["local"] = reconstruct_local(ctx)
    if reconcile or dispatch:
        if service is None:
            if client is None:
                raise Stage4Error("a Serverless client is required for reconcile/dispatch")
            service = CatalogServerlessService(client)
        if reconcile:
            out["reconciliation"] = reconcile_stage4_job(ctx, service)
        if dispatch:
            if adapter is None:
                if client is None:
                    raise Stage4Error("a Serverless client or transport adapter is required for dispatch")
                adapter = make_live_transport(client, STAGE4_FUNCTION_TITLE)
            out["dispatch"] = dispatch_stage4_job(ctx, service, adapter, authorize=authorize_remote,
                                                  function_size=function_size)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def reconciliation_view(summary: Mapping[str, Any], classical_baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compare a reconstruction with a persisted classical baseline.  Reporting only: nothing is injected."""
    out = {"e_total": summary["e_total"], "e_hf": summary["e_hf"], "e_corr_total": summary["e_corr_total"],
           "composition": summary["composition"]["actual"], "baseline_available": classical_baseline is not None}
    if classical_baseline:
        ref = classical_baseline.get("energies") or classical_baseline      # archived file or loaded dict
        if "e_total" in ref:
            out["baseline_e_total"] = float(ref["e_total"])
            out["delta_vs_baseline"] = summary["e_total"] - float(ref["e_total"])
            out["hf_agreement"] = summary["e_hf"] - float(ref.get("e_hf", summary["e_hf"]))
    return out


def _read_json_safe(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except ValueError:
        return None


def hardware_readiness_report(ctx: RunContext | None = None, lab: Any = None) -> dict[str, Any]:
    """Organizer-facing preflight for the first real-hardware milestone.  Offline, read-only.

    Each row separates *what has been proven* from *what is merely coded*.  A mock adapter that works never
    marks real-hardware transport as passing; real transport counts only when an operator-written evidence
    record (``real_transport_evidence.json``, produced by a manual real smoke test) exists.
    """
    from proteasome_challenge import STAGES
    from proteasome_challenge import quantum_coordinator as qc
    from proteasome_challenge.run_control_plane import SUPPORTED_COMPUTE_BACKENDS

    rows: dict[str, dict[str, Any]] = {}

    rows["offline_validated_capabilities"] = {
        "status": "VALIDATED_OFFLINE",
        "detail": ["task registry + frozen manifest + artifact store", "classical dispatch/reconcile (mock Serverless)",
                   "quantum campaign state machine incl. replay (mock)", "3Q-C postprocess worker logic (mock)",
                   "manifest-derived completion barrier", "Stage 4 reconstruction (local + mock-remote)",
                   "idempotent persistence and uncertain-submission handling"],
        "note": "All evidence is offline/mocked; it does not demonstrate behaviour on real Serverless or a QPU."}

    functions = {}
    for n in sorted(STAGES):
        if n >= 4:
            functions[STAGES[n]["title"]] = "REGISTERED_NOT_EXECUTED_ON_REAL_SERVERLESS"
    deploy_note = "no lab supplied; deployment status unknown"
    if lab is not None:
        fp = Path(lab.ckpt_dir) / ".deployed_bundle_sha256"
        if not fp.is_file():
            deploy_note = "bundle never deployed from this checkout"
        elif fp.read_text().strip() == lab.bundle_fingerprint():
            deploy_note = "deployed fingerprint matches the local bundle (functions still never executed)"
        else:
            deploy_note = "STALE: local bundle differs from the last deployed fingerprint; redeploy required"
    rows["deployed_untested_serverless_functions"] = {
        "status": "UNTESTED", "functions": functions, "detail": deploy_note}

    evidence = None
    imports = 0
    if ctx is not None:
        evidence = _read_json_safe(ctx.paths.run_dir / REAL_TRANSPORT_EVIDENCE_FILENAME)
        for t in ctx.registry.list_tasks(ctx.run_id):
            imports += sum(1 for e in t.extra.get("dispatch_events", []) if e.get("event") == "RESULT_IMPORTED")
    verified = bool(evidence and evidence.get("worker_published_file_downloaded_and_authenticated") is True)
    rows["authenticated_remote_artifact_transfers"] = {
        "status": "VERIFIED_ON_REAL_SERVERLESS" if verified else "NEVER_VERIFIED_ON_REAL_SERVERLESS",
        "worker_published_file_ever_downloaded_and_authenticated_by_organizer": verified,
        "detail": ("operator evidence record present" if verified else
                   f"no {REAL_TRANSPORT_EVIDENCE_FILENAME}; {imports} RESULT_IMPORTED event(s) exist in this "
                   "registry but mock and live imports are indistinguishable, so they are not accepted as evidence"),
        "note": "Stage 4 returns its summary in the job result and publishes no worker file; its inputs, "
                "however, depend on organizer->worker File API download, which is equally unproven."}
    rows["real_classical_worker_execution"] = {
        "status": "EXECUTED_ON_REAL_SERVERLESS" if verified else "NEVER_EXECUTED_ON_REAL_SERVERLESS",
        "detail": "classical worker entrypoint is deployed-or-deployable but has no real-run evidence"}

    qpu_state = "READY_PENDING_EXPLICIT_AUTHORIZATION"
    qpu_detail = "submission is gated by RUN_*/ALLOW_* flags and the organizer campaign authorization"
    if ctx is not None:
        real = [t for t in ctx.registry.list_tasks(ctx.run_id, task_type=TaskType.QUANTUM)
                if t.qpu_job_id and not _is_replay(t)]
        if real:
            qpu_state, qpu_detail = "QPU_JOBS_RECORDED", f"{len(real)} non-replay QPU job id(s) recorded"
    rows["real_qpu_measurement_readiness"] = {"status": qpu_state, "detail": qpu_detail,
                                              "note": "no QPU is contacted by this report or by Task 5.9D"}

    try:
        src = inspect.getsource(qc.dispatch_postprocessing_campaign)
    except (OSError, TypeError):
        src = ""
    blocked = "ewf_dumpfile_path" in src and "ewf_transport_name" not in src
    rows["real_3qc_postprocessing_readiness"] = {
        "status": "BLOCKED_FOR_REMOTE_EXECUTION" if blocked else "NOT_VERIFIED",
        "detail": ("dispatch_postprocessing_campaign sends an organizer-local ewf_dumpfile_path/result_dir to the "
                   "worker instead of transport descriptors; a real remote worker cannot reach them") if blocked
                  else "transport-based input handoff present but never exercised on real Serverless"}

    barrier_state = "UNKNOWN"
    if ctx is not None and ctx.manifest is not None:
        try:
            barrier_state = "BARRIER_PASSES" if run_strict_barrier(ctx).passed else "BARRIER_NOT_PASSED"
        except Stage4Error as exc:
            barrier_state = f"INPUTS_NOT_AUTHENTICATED ({exc})"[:200]
    rows["full_reconstruction_readiness"] = {
        "status": "LOGIC_VALIDATED_OFFLINE_REMOTE_UNTESTED", "completion_barrier": barrier_state,
        "detail": "local CPU reconstruction is available as an organizer-side fallback; the remote Stage 4 "
                  "function has never executed on real Serverless"}
    rows["gpu_availability"] = {
        "status": "NOT_AVAILABLE_FUTURE_UPGRADE", "supported_compute_backends": sorted(SUPPORTED_COMPUTE_BACKENDS),
        "detail": "GPU integration is out of scope; every Stage 4 dispatch is restricted to CPU function sizes"}
    return {"schema": "hardware_readiness/1", "generated_utc": _now(), "rows": rows}


def format_hardware_readiness(report: Mapping[str, Any]) -> str:
    lines = ["HARDWARE-TEST READINESS (organizer preflight)", "-" * 60]
    for key, row in report["rows"].items():
        lines.append(f"{key.replace('_', ' ').upper()}: {row['status']}")
        detail = row.get("detail")
        for d in ([detail] if isinstance(detail, str) else detail or []):
            lines.append(f"    - {d}")
        if row.get("note"):
            lines.append(f"    * {row['note']}")
        if key == "authenticated_remote_artifact_transfers":
            lines.append("    ever downloaded+authenticated from a real worker: "
                         f"{row['worker_published_file_ever_downloaded_and_authenticated_by_organizer']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Researcher-facing Stage 4 API  (Task Pass-2D / Section F)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_reconstruction(ctx: RunContext, *, allow_remote: bool = False,
                          function_size: str = "l") -> dict[str, Any]:
    """Validate the completion barrier and reconstruct the EWF energy.

    This is the single researcher-facing operation for Stage 4.  Internally it:
    1. Authenticates Stage 1/2 scientific inputs.
    2. Runs the strict completion barrier across all required fragments.
    3. Reconstructs on the organizer CPU (always; remote dispatch requires
       the separate low-level ``dispatch_stage4_job`` API).
    4. Persists and returns the authoritative reconstruction summary.

    The operation is idempotent: when an authoritative summary already exists
    and still matches the current verified fragment results it is returned
    without recomputation.

    Parameters
    ----------
    ctx : RunContext
        The active run context (must have a frozen RunManifest).
    allow_remote : bool, optional
        Reserved for future remote dispatch authorisation.  Currently unused;
        pass ``True`` only under explicit authorisation.  Defaults to ``False``.
    function_size : str, optional
        Serverless CPU-family for future remote dispatch (``"s"`` | ``"m"`` | ``"l"``).

    Returns
    -------
    dict
        The authoritative ``recon/2`` reconstruction summary.  Key fields:

        ``e_hf``            RHF reference energy (Ha).
        ``e_corr_total``    Reconstructed EWF correlation energy (Ha).
        ``e_total``         Reconstructed total molecular energy (Ha).
        ``composition``     Solver composition (planned vs. actual).
        ``fragments``       Per-fragment ``e_corr``, solver, ``rdm_integrated``.
        ``quantum``         Quantum measurement provenance for quantum fragments.
        ``barrier_report``  Completion-barrier status record.

    Raises
    ------
    Stage4NotReady
        When the completion barrier fails (fragments missing, invalid RDMs, …).
    Stage4Error
        On any other scientific or infrastructure failure.
    """
    result = reconstruct_local(ctx, persist=True)
    return result["summary"]


def reconstruction_status(ctx: RunContext | None = None) -> dict[str, Any]:
    """Return a read-only status dictionary for Stage 4.

    **Never submits, writes, or contacts any provider.**  Safe to call at any time.

    Parameters
    ----------
    ctx : RunContext or None
        When ``None``, reports that no run context is active.

    Returns
    -------
    dict with keys:

    ``ready``
        ``True`` when a complete authoritative summary exists (``ensure_reconstruction``
        has already succeeded).
    ``summary``
        The authoritative ``recon/2`` summary dict, or ``None`` if not yet
        reconstructed.
    ``e_hf``              RHF reference energy (Ha), or ``None``.
    ``e_corr_total``      EWF correlation energy (Ha), or ``None``.
    ``e_total``           Total molecular energy (Ha), or ``None``.
    ``composition``       Actual solver composition dict, or ``None``.
    ``fragments``         Per-fragment summary dict, or ``None``.
    ``barrier``           Current completion-barrier status (read-only evaluation).
    ``replay_count``      Number of quantum fragments from replayed measurements.
    ``serverless_job``    Serverless job id if one exists, else ``None``.
    ``message``           Human-readable status message.
    """
    if ctx is None or ctx.manifest is None:
        return {
            "ready": False, "summary": None,
            "e_hf": None, "e_corr_total": None, "e_total": None,
            "composition": None, "fragments": None,
            "barrier": None, "replay_count": 0,
            "serverless_job": None,
            "message": "No active run context or frozen manifest.",
        }

    summary = load_authoritative_summary(ctx)
    barrier_info = None
    try:
        rd = readiness_summary(ctx)
        barrier_info = rd.get("barrier")
    except Exception:
        pass

    ledger = _read_ledger(ctx)
    sj_id = None
    for entry in ledger.get("entries", []):
        if _effective_state(entry) == Stage4State.COMPLETE:
            sj_id = entry.get("serverless_job_id")
            break

    if summary is not None:
        q = summary.get("quantum", {})
        replay_count = sum(
            1 for v in q.get("fragments", {}).values() if v.get("is_replay")
        )
        comp = summary.get("composition", {}).get("actual", {})
        msg = (
            f"Reconstruction complete.  "
            f"E_total = {summary['e_total']:.8f} Ha  "
            f"({'+'.join(f'{n} {s}' for s,n in comp.items())})"
        )
        return {
            "ready": True, "summary": summary,
            "e_hf": summary["e_hf"],
            "e_corr_total": summary["e_corr_total"],
            "e_total": summary["e_total"],
            "composition": comp,
            "fragments": summary.get("fragments"),
            "barrier": barrier_info,
            "replay_count": replay_count,
            "serverless_job": sj_id,
            "message": msg,
        }

    stage_status = stage4_status(ctx)
    if stage_status.get("uncertain"):
        msg = "UNCERTAIN Stage 4 submission — operator action required."
    elif any(stage_status.get(k) for k in ("jobs",)):
        msg = f"Stage 4 job in progress: {stage_status['jobs']}"
    else:
        msg = "Reconstruction pending — set RUN_STAGE4 = True to reconstruct."

    return {
        "ready": False, "summary": None,
        "e_hf": None, "e_corr_total": None, "e_total": None,
        "composition": None, "fragments": None,
        "barrier": barrier_info,
        "replay_count": 0,
        "serverless_job": sj_id,
        "message": msg,
    }
