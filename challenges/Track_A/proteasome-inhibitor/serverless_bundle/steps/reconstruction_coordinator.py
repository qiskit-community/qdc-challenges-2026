"""
reconstruction_coordinator.py — Stage 4 Independent EWF Reconstruction and Completion Barrier.

Run-level coordinator that:
1. Validates the strict completion barrier across all required fragments
   (task states == COMPLETE, matching solvers, valid finite RDMs, checksum integrity,
   and quantum provenance).
2. Retrieves and verifies complete SolverResults from frs/1 store.
3. Invokes the authoritative step4_reconstruct.run_reconstruct_from_refs() cumulant
   energy assembly (inherited, unmodified).
4. Validates numerical consistency and publishes a versioned ReconstructionSummary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Ensure vendored dependencies are resolvable
_bundle_dir = Path(__file__).resolve().parent.parent
_vendor_sqd = _bundle_dir / "vendor" / "qiskit-addon-sqd"
if _vendor_sqd.is_dir() and str(_vendor_sqd) not in sys.path:
    sys.path.insert(0, str(_vendor_sqd))

try:
    from .fragment_result_store import (
        FragmentResultReference,
        FragmentStoreError,
        load_fragment_result,
        load_fragment_result_from_ref,
    )
    from .step4_reconstruct import QuantumReconstructionError, run_reconstruct_from_refs
except ImportError:
    from fragment_result_store import (  # type: ignore[no-redef]
        FragmentResultReference,
        FragmentStoreError,
        load_fragment_result,
        load_fragment_result_from_ref,
    )
    from step4_reconstruct import (  # type: ignore[no-redef]
        QuantumReconstructionError,
        run_reconstruct_from_refs,
    )

RECONSTRUCTION_SCHEMA = "recon/1"
BARRIER_SCHEMA = "barrier/1"

QUANTUM_SOLVERS = frozenset({"SQD", "TrimSQD", "sqd", "trimsqd"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class BarrierReport:
    """Structured report produced by validate_completion_barrier."""

    run_id: str
    passed: bool
    total_required: int
    completed: int
    missing: list[int]
    wrong_solver: list[int]
    invalid_rdms: list[int]
    checksum_failures: list[int]
    quantum_provenance_failures: list[int]
    errors: list[str]
    checked_utc: str

    def as_dict(self) -> dict:
        return {
            "schema": BARRIER_SCHEMA,
            "run_id": self.run_id,
            "passed": self.passed,
            "total_required": self.total_required,
            "completed": self.completed,
            "missing": list(self.missing),
            "wrong_solver": list(self.wrong_solver),
            "invalid_rdms": list(self.invalid_rdms),
            "checksum_failures": list(self.checksum_failures),
            "quantum_provenance_failures": list(self.quantum_provenance_failures),
            "errors": list(self.errors),
            "checked_utc": self.checked_utc,
        }


@dataclass
class ReconstructionSummary:
    """Authoritative result of an EWF energy reconstruction."""

    run_id: str
    e_hf: float
    e_corr_total: float
    e_total: float
    fragment_energies: dict[int, float]  # frag_id -> e_corr
    solver_used: dict[int, str]
    hamiltonian_sha256s: dict[int, str]
    barrier_report: BarrierReport
    result_refs: dict[int, dict]  # frag_id -> FragmentResultReference.as_dict()
    reconstruction_utc: str
    schema: str = RECONSTRUCTION_SCHEMA

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "e_hf": float(self.e_hf),
            "e_corr_total": float(self.e_corr_total),
            "e_total": float(self.e_total),
            "fragment_energies": {str(k): float(v) for k, v in self.fragment_energies.items()},
            "solver_used": {str(k): str(v) for k, v in self.solver_used.items()},
            "hamiltonian_sha256s": {str(k): str(v) for k, v in self.hamiltonian_sha256s.items()},
            "barrier_report": self.barrier_report.as_dict(),
            "result_refs": {str(k): dict(v) for k, v in self.result_refs.items()},
            "reconstruction_utc": self.reconstruction_utc,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)


def validate_completion_barrier(
    registry: Any,
    run_id: str,
    *,
    required_fragments: Sequence[int],
    solver_assignments: Mapping[int, str],
    hamiltonian_hashes: Mapping[int, str] | None = None,
    strict_quantum_provenance: bool = True,
    result_root: str | Path | None = None,
) -> BarrierReport:
    """Validate that all required fragments are COMPLETE and chemically/structurally valid.

    Checks performed per required fragment:
    1. Exactly one task exists and is in TaskState.COMPLETE state.
    2. Task solver matches the solver assigned in solver_assignments.
    3. Task has non-empty result_artifact_uri and result_payload_digest.
    4. If hamiltonian_hashes is provided, task.hamiltonian_sha256 matches.
    5. If quantum solver (SQD/TrimSQD) and strict_quantum_provenance=True:
       task.qpu_job_id must be non-empty.
    6. Result artifact file exists on disk and is loaded/verified via frs/1 store:
       - file SHA-256 matches
       - array checksums match
       - rdm1 is 2D and rdm2 is 4D
       - all RDM elements are finite (no NaN or Inf)

    Fails closed: any error or mismatch sets passed=False.
    """
    checked_utc = _utc_now()
    req_frags = sorted({int(fid) for fid in required_fragments})
    total_required = len(req_frags)

    missing: list[int] = []
    wrong_solver: list[int] = []
    invalid_rdms: list[int] = []
    checksum_failures: list[int] = []
    quantum_provenance_failures: list[int] = []
    errors: list[str] = []
    completed = 0

    # Obtain tasks from registry or list
    if hasattr(registry, "list_tasks"):
        tasks = registry.list_tasks(run_id, task_type=None)
    elif isinstance(registry, (list, tuple)):
        tasks = [t for t in registry if getattr(t, "run_id", "") == run_id]
    else:
        raise TypeError(f"registry must have list_tasks() or be an iterable of TaskRecords, got {type(registry)}")

    tasks_by_frag: dict[int, list[Any]] = {}
    for t in tasks:
        fid = int(getattr(t, "fragment_id", -1))
        tasks_by_frag.setdefault(fid, []).append(t)

    for fid in req_frags:
        frag_tasks = tasks_by_frag.get(fid, [])
        if not frag_tasks:
            missing.append(fid)
            errors.append(f"Fragment {fid}: no task registered for run {run_id}")
            continue

        # Look for COMPLETE tasks
        complete_tasks = [t for t in frag_tasks if getattr(t, "state", "") == "COMPLETE"]
        if not complete_tasks:
            missing.append(fid)
            states = [getattr(t, "state", "UNKNOWN") for t in frag_tasks]
            errors.append(f"Fragment {fid}: not COMPLETE (found states: {states})")
            continue

        task = complete_tasks[-1]

        # 1. Solver check
        expected_solver = solver_assignments.get(fid)
        if expected_solver is not None and getattr(task, "solver", "") != expected_solver:
            wrong_solver.append(fid)
            errors.append(
                f"Fragment {fid}: solver mismatch (expected {expected_solver}, got {getattr(task, 'solver', '')})"
            )

        # 2. URI and payload digest check
        artifact_uri = getattr(task, "result_artifact_uri", "")
        payload_digest = getattr(task, "result_payload_digest", "")
        if not artifact_uri or not payload_digest:
            checksum_failures.append(fid)
            errors.append(f"Fragment {fid}: missing result_artifact_uri or result_payload_digest")
            continue

        # 3. Hamiltonian hash check
        if hamiltonian_hashes and fid in hamiltonian_hashes:
            expected_hsha = hamiltonian_hashes[fid]
            actual_hsha = getattr(task, "hamiltonian_sha256", "")
            if actual_hsha != expected_hsha:
                checksum_failures.append(fid)
                errors.append(
                    f"Fragment {fid}: hamiltonian_sha256 mismatch (expected {expected_hsha[:12]}, got {actual_hsha[:12]})"
                )

        # 4. Quantum provenance check
        task_solver = getattr(task, "solver", "")
        task_type = getattr(task, "task_type", "")
        is_quantum = task_solver in QUANTUM_SOLVERS or task_type == "quantum"
        if is_quantum and strict_quantum_provenance:
            qpu_job_id = getattr(task, "qpu_job_id", "")
            if not qpu_job_id or not str(qpu_job_id).strip():
                quantum_provenance_failures.append(fid)
                errors.append(f"Fragment {fid}: quantum task missing qpu_job_id provenance")

        # 5. Resolve file path and load/validate result
        art_path = Path(artifact_uri)
        if result_root and not art_path.is_absolute():
            art_path = Path(result_root) / art_path

        if not art_path.is_file():
            checksum_failures.append(fid)
            errors.append(f"Fragment {fid}: artifact file not found at {art_path}")
            continue

        try:
            file_sha = _sha256_file(art_path)
            ref = FragmentResultReference(
                path=art_path,
                file_sha256=file_sha,
                payload_digest=payload_digest,
                fragment_id=fid,
                run_id=run_id,
                solver=task_solver,
                hamiltonian_sha256=getattr(task, "hamiltonian_sha256", ""),
            )
            result = load_fragment_result_from_ref(ref)
        except Exception as exc:
            checksum_failures.append(fid)
            errors.append(f"Fragment {fid}: failed to load/verify artifact ({type(exc).__name__}: {exc})")
            continue

        # Verify payload digest if recorded
        if payload_digest:
            try:
                try:
                    from .fragment_result_store import _split_metadata, compute_payload_digest
                except ImportError:
                    from fragment_result_store import (  # type: ignore[no-redef]
                        _split_metadata,
                        compute_payload_digest,
                    )

                jm, am, nm = _split_metadata(getattr(result, "metadata", {}))
                actual_payload_digest = compute_payload_digest(
                    energy=getattr(result, "energy", 0.0),
                    rdm1=getattr(result, "rdm1", None),
                    rdm2=getattr(result, "rdm2", None),
                    wavefunction=getattr(result, "wavefunction", None),
                    json_meta=jm,
                    array_meta=am,
                    nonser_meta=nm,
                    fragment_id=fid,
                    run_id=run_id,
                    solver=task_solver,
                    hamiltonian_sha256=getattr(task, "hamiltonian_sha256", ""),
                )
                if actual_payload_digest != payload_digest:
                    checksum_failures.append(fid)
                    errors.append(
                        f"Fragment {fid}: payload_digest mismatch (expected {payload_digest[:12]}, got {actual_payload_digest[:12]})"
                    )
                    continue
            except Exception as exc:
                checksum_failures.append(fid)
                errors.append(f"Fragment {fid}: payload_digest verification error ({type(exc).__name__}: {exc})")
                continue

        # 6. RDM validity check
        rdm1 = getattr(result, "rdm1", None)
        rdm2 = getattr(result, "rdm2", None)
        if rdm1 is None or rdm2 is None:
            invalid_rdms.append(fid)
            errors.append(f"Fragment {fid}: missing rdm1 or rdm2")
            continue

        if getattr(rdm1, "ndim", 0) != 2 or getattr(rdm2, "ndim", 0) != 4:
            invalid_rdms.append(fid)
            errors.append(
                f"Fragment {fid}: invalid RDM dimensions: rdm1.ndim={getattr(rdm1, 'ndim', None)}, rdm2.ndim={getattr(rdm2, 'ndim', None)}"
            )
            continue

        if not (bool(np.all(np.isfinite(rdm1))) and bool(np.all(np.isfinite(rdm2)))):
            invalid_rdms.append(fid)
            errors.append(f"Fragment {fid}: RDMs contain non-finite values (NaN/inf)")
            continue

        # Check if fragment is cleanly completed
        if (
            fid not in wrong_solver
            and fid not in invalid_rdms
            and fid not in checksum_failures
            and fid not in quantum_provenance_failures
        ):
            completed += 1

    passed = (
        len(missing) == 0
        and len(wrong_solver) == 0
        and len(invalid_rdms) == 0
        and len(checksum_failures) == 0
        and len(quantum_provenance_failures) == 0
        and len(errors) == 0
        and completed == total_required
    )

    return BarrierReport(
        run_id=run_id,
        passed=passed,
        total_required=total_required,
        completed=completed,
        missing=missing,
        wrong_solver=wrong_solver,
        invalid_rdms=invalid_rdms,
        checksum_failures=checksum_failures,
        quantum_provenance_failures=quantum_provenance_failures,
        errors=errors,
        checked_utc=checked_utc,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Per-fragment content inspection (Task 5.9D) — shared by the organizer-side strict barrier and the
#  Serverless Stage 4 worker, so that both enforce exactly the same rules on a loaded frs/1 result.
# ─────────────────────────────────────────────────────────────────────────────

FRAGMENT_CHECK_CATEGORIES = ("checksum", "invalid_rdm", "metadata", "quantum_provenance")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def recompute_payload_digest(result: Any, *, fragment_id: int, run_id: str, solver: str,
                             hamiltonian_sha256: str) -> str:
    """Recompute the canonical frs/1 payload digest of a loaded result (independent of the stored attr)."""
    try:
        from .fragment_result_store import _split_metadata, compute_payload_digest
    except ImportError:
        from fragment_result_store import _split_metadata, compute_payload_digest  # type: ignore[no-redef]
    jm, am, nm = _split_metadata(getattr(result, "metadata", {}))
    return compute_payload_digest(
        energy=getattr(result, "energy", 0.0), rdm1=getattr(result, "rdm1", None),
        rdm2=getattr(result, "rdm2", None), wavefunction=getattr(result, "wavefunction", None),
        json_meta=jm, array_meta=am, nonser_meta=nm, fragment_id=fragment_id, run_id=run_id,
        solver=solver, hamiltonian_sha256=hamiltonian_sha256)


def inspect_fragment_result(
    path: str | Path,
    *,
    run_id: str,
    fragment_id: int,
    solver: str,
    hamiltonian_sha256: str,
    expected_payload_digest: str,
    expected_file_sha256: str | None = None,
) -> tuple[Any, dict[str, list[str]]]:
    """Load one frs/1 result and classify every problem found; ``(result_or_None, problems)``.

    Enforced: identity (run, fragment, solver, Hamiltonian SHA-256 — mandatory), file SHA-256 (when
    given), recomputed payload digest equal to ``expected_payload_digest`` (mandatory), finite energy
    (zero is legitimate), complete finite RDMs with ``rdm1`` square ``n x n`` and ``rdm2`` ``n^4``,
    required scientific metadata, and — for quantum solvers — measurement/protocol provenance
    markers (``measurement_sha256``; TrimSQD additionally ``policy`` and ``trim_ratio``).
    """
    problems: dict[str, list[str]] = {k: [] for k in FRAGMENT_CHECK_CATEGORIES}
    fid = int(fragment_id)
    if not isinstance(hamiltonian_sha256, str) or not _HEX64_RE.match(hamiltonian_sha256):
        problems["checksum"].append(f"Fragment {fid}: authoritative Hamiltonian SHA-256 is missing or malformed")
        return None, problems
    if not expected_payload_digest:
        problems["checksum"].append(f"Fragment {fid}: no recorded payload digest to authenticate against")
        return None, problems
    try:
        result = load_fragment_result(
            path, expected_fragment_id=fid, expected_run_id=run_id, expected_solver=solver,
            expected_hamiltonian_sha256=hamiltonian_sha256, file_sha256=expected_file_sha256)
    except Exception as exc:  # noqa: BLE001 — any load failure is a verification failure
        problems["checksum"].append(f"Fragment {fid}: failed to load/verify frs/1 result ({type(exc).__name__}: {exc})")
        return None, problems
    try:
        actual = recompute_payload_digest(result, fragment_id=fid, run_id=run_id, solver=solver,
                                          hamiltonian_sha256=hamiltonian_sha256)
        if actual != expected_payload_digest:
            problems["checksum"].append(
                f"Fragment {fid}: payload_digest mismatch (recorded {expected_payload_digest[:12]}, "
                f"recomputed {actual[:12]})")
    except Exception as exc:  # noqa: BLE001
        problems["checksum"].append(f"Fragment {fid}: payload digest verification error ({type(exc).__name__}: {exc})")

    if not _is_finite_number(getattr(result, "energy", None)):
        problems["invalid_rdm"].append(f"Fragment {fid}: solver energy is missing or non-finite")
    r1, r2 = getattr(result, "rdm1", None), getattr(result, "rdm2", None)
    n_orb = None
    if r1 is None or r2 is None:
        problems["invalid_rdm"].append(f"Fragment {fid}: missing rdm1 or rdm2")
    else:
        a1, a2 = np.asarray(r1), np.asarray(r2)
        if a1.ndim != 2 or a1.shape[0] != a1.shape[1] or a1.size == 0:
            problems["invalid_rdm"].append(f"Fragment {fid}: rdm1 must be a non-empty square matrix, got shape {a1.shape}")
        else:
            n_orb = int(a1.shape[0])
            if a2.shape != (n_orb,) * 4:
                problems["invalid_rdm"].append(
                    f"Fragment {fid}: rdm2 shape {a2.shape} incompatible with rdm1 dimension {n_orb}")
        if not (bool(np.all(np.isfinite(a1))) and bool(np.all(np.isfinite(a2)))):
            problems["invalid_rdm"].append(f"Fragment {fid}: RDMs contain non-finite values (NaN/inf)")

    md = getattr(result, "metadata", None) or {}
    if str(md.get("solver_label", "")) != solver:
        problems["metadata"].append(
            f"Fragment {fid}: metadata solver_label {md.get('solver_label')!r} != assigned solver {solver!r}")
    nocc = md.get("nocc")
    if isinstance(nocc, bool) or not isinstance(nocc, (int, np.integer)) or int(nocc) < 0 \
            or (n_orb is not None and int(nocc) > n_orb):
        problems["metadata"].append(f"Fragment {fid}: metadata nocc {nocc!r} is missing or invalid")
    if "converged" in md and md["converged"] is False:
        problems["metadata"].append(f"Fragment {fid}: solver reports converged=False")
    if "e_corr" in md and not _is_finite_number(md["e_corr"]):
        problems["metadata"].append(f"Fragment {fid}: metadata e_corr is non-finite")

    if solver in QUANTUM_SOLVERS:
        msha = md.get("measurement_sha256")
        if not isinstance(msha, str) or not _HEX64_RE.match(msha):
            problems["quantum_provenance"].append(f"Fragment {fid}: missing/invalid measurement_sha256 provenance")
        if solver.lower() == "trimsqd":
            if not str(md.get("policy", "")).strip():
                problems["quantum_provenance"].append(f"Fragment {fid}: TrimSQD result lacks the trim policy")
            if not _is_finite_number(md.get("trim_ratio")):
                problems["quantum_provenance"].append(f"Fragment {fid}: TrimSQD result lacks a finite trim_ratio")
    return result, problems


def reconstruct_from_registry(
    registry: Any,
    run_id: str,
    mf: Any = None,
    ewf: Any = None,
    *,
    required_fragments: Sequence[int],
    solver_assignments: Mapping[int, str],
    result_root: str | Path | None = None,
    hamiltonian_hashes: Mapping[int, str] | None = None,
    strict_quantum_provenance: bool = True,
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    require_pipeline_data: bool = False,
) -> ReconstructionSummary:
    """Coordinate the Stage 4 independent energy reconstruction.

    1. Validates the completion barrier across all required fragments.
    2. Builds FragmentResultReference objects for all results.
    3. Invokes run_reconstruct_from_refs if data_dir is provided and contains
       Stage 1/2 artifacts (mf_data.pkl and ewf_dumpfile.h5).
       Otherwise, accumulates partitioned correlation energies from verified results.
    4. Validates numerical consistency.
    5. Writes and returns a versioned ReconstructionSummary.
    """
    barrier = validate_completion_barrier(
        registry,
        run_id,
        required_fragments=required_fragments,
        solver_assignments=solver_assignments,
        hamiltonian_hashes=hamiltonian_hashes,
        strict_quantum_provenance=strict_quantum_provenance,
        result_root=result_root,
    )

    if not barrier.passed:
        err_msg = "; ".join(barrier.errors) if barrier.errors else "incomplete fragments"
        raise RuntimeError(f"Completion barrier failed for run {run_id}: {err_msg}")

    # Build reference mapping
    if hasattr(registry, "list_tasks"):
        tasks = registry.list_tasks(run_id, task_type=None)
    elif isinstance(registry, (list, tuple)):
        tasks = [t for t in registry if getattr(t, "run_id", "") == run_id]
    else:
        tasks = []

    tasks_by_frag = {
        int(t.fragment_id): t for t in tasks if getattr(t, "state", "") == "COMPLETE"
    }

    fragment_result_refs: dict[int, FragmentResultReference] = {}
    solver_used: dict[int, str] = {}
    hamiltonian_sha256s: dict[int, str] = {}
    result_refs_dict: dict[int, dict] = {}

    for fid in required_fragments:
        fid = int(fid)
        t = tasks_by_frag[fid]
        art_path = Path(t.result_artifact_uri)
        if result_root and not art_path.is_absolute():
            art_path = Path(result_root) / art_path

        file_sha = _sha256_file(art_path)
        ref = FragmentResultReference(
            path=art_path,
            file_sha256=file_sha,
            payload_digest=t.result_payload_digest,
            fragment_id=fid,
            run_id=run_id,
            solver=t.solver,
            hamiltonian_sha256=t.hamiltonian_sha256,
        )
        fragment_result_refs[fid] = ref
        solver_used[fid] = t.solver
        hamiltonian_sha256s[fid] = t.hamiltonian_sha256
        result_refs_dict[fid] = ref.as_dict()

    # Determine execution path
    has_full_pipeline_data = False
    if data_dir is not None:
        data_path = Path(data_dir)
        if (data_path / "mf_data.pkl").is_file() and (data_path / "ewf_dumpfile.h5").is_file():
            has_full_pipeline_data = True

    if require_pipeline_data and not has_full_pipeline_data:
        # Production callers must never fall through to the embedded-energy "test double" path.
        raise RuntimeError(
            f"Stage 1/2 scientific inputs (mf_data.pkl, ewf_dumpfile.h5) are unavailable for run {run_id}; "
            "refusing the embedded-energy fallback")

    if has_full_pipeline_data:
        res_dict = run_reconstruct_from_refs(
            data_dir=str(data_dir),
            run_name=run_id,
            fragment_result_refs=fragment_result_refs,
            results_dir=str(results_dir) if results_dir else (str(result_root) if result_root else None),
            expected_fragment_ids=list(required_fragments),
        )
        e_hf = float(res_dict["e_hf"])
        e_corr_total = float(res_dict["e_corr"])
        e_total = float(res_dict["e_total"])
        fragment_energies: dict[int, float] = {}
        for fid_k, frag_info in res_dict.get("fragments", {}).items():
            fid_int = int(fid_k)
            if isinstance(frag_info, dict):
                fragment_energies[fid_int] = float(frag_info.get("e_corr", 0.0))
            else:
                fragment_energies[fid_int] = float(frag_info)
    else:
        # No Stage 1/2 pipeline data available.
        # Production callers must always supply pipeline data (mf_data.pkl + ewf_dumpfile.h5)
        # so that the partitioned-cumulant contraction is performed.  This path is only
        # reached in offline / test-double contexts where require_pipeline_data=False and
        # no data_dir was provided; it falls back to metadata-reported energies for
        # classical fragments and raises for quantum fragments (which never carry e_corr).
        if mf is not None and hasattr(mf, "e_tot"):
            e_hf = float(mf.e_tot)
        else:
            e_hf = 0.0

        fragment_energies = {}
        for fid, ref in fragment_result_refs.items():
            loaded = load_fragment_result_from_ref(ref)
            solver = getattr(ref, "solver", "unknown")
            if solver in QUANTUM_SOLVERS:
                raise RuntimeError(
                    f"reconstruct_from_registry: quantum fragment {fid} ({solver}) requires "
                    "full pipeline data (mf_data.pkl + ewf_dumpfile.h5) for cumulant reconstruction; "
                    "metadata.e_corr is not available for quantum solvers")
            # Classical offline path: use metadata-reported e_corr or solver energy.
            frag_e_corr = float(loaded.metadata.get("e_corr", loaded.energy))
            fragment_energies[fid] = frag_e_corr

        e_corr_total = float(sum(fragment_energies.values()))
        e_total = float(e_hf + e_corr_total)

    recon_summary = ReconstructionSummary(
        run_id=run_id,
        e_hf=e_hf,
        e_corr_total=e_corr_total,
        e_total=e_total,
        fragment_energies=fragment_energies,
        solver_used=solver_used,
        hamiltonian_sha256s=hamiltonian_sha256s,
        barrier_report=barrier,
        result_refs=result_refs_dict,
        reconstruction_utc=_utc_now(),
        schema=RECONSTRUCTION_SCHEMA,
    )

    if result_root is not None:
        out_dir = Path(result_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "summary.json"
        with open(summary_path, "w") as fh:
            fh.write(recon_summary.to_json())

    return recon_summary


def reconstruct_from_manifest(
    registry: Any,
    manifest: Any,
    mf: Any = None,
    ewf: Any = None,
    *,
    result_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    strict_quantum_provenance: bool = True,
    hamiltonian_hashes: Mapping[int, str] | None = None,
    require_pipeline_data: bool = False,
) -> ReconstructionSummary:
    """Coordinate reconstruction using metadata from a RunManifest.

    A real :class:`RunManifest` exposes ``fragment_ids`` and string-keyed ``solver_assignments``; the
    legacy ``fragment_list`` attribute is still honoured for duck-typed manifests.
    """
    fragment_ids = getattr(manifest, "fragment_ids", None)
    if not isinstance(fragment_ids, (list, tuple)):
        fragment_ids = manifest.fragment_list
    return reconstruct_from_registry(
        registry=registry,
        run_id=manifest.run_id,
        mf=mf,
        ewf=ewf,
        required_fragments=fragment_ids,
        solver_assignments={int(k): v for k, v in dict(manifest.solver_assignments).items()},
        result_root=result_root,
        hamiltonian_hashes=hamiltonian_hashes,
        strict_quantum_provenance=strict_quantum_provenance,
        data_dir=data_dir,
        require_pipeline_data=require_pipeline_data,
    )
