"""
Organizer run control plane (Task 5.9B).

Notebook-side helpers that connect the organizer notebook to the Phase 5 persistence
infrastructure (``FragmentTaskRegistry``, ``LocalArtifactStore``, ``RunManifest``,
``stage_artifacts``).  Nothing here runs chemistry, contacts Serverless, or touches a QPU:
every function reads or writes small local files only.  The notebook machine is the single
authoritative owner of the SQLite registry and the local artifact store.

On-disk layout
--------------
``LocalArtifactStore(root)`` itself appends ``runs/<run_id>/<kind>/<name>``.  The control plane
therefore hands it ``artifact_root`` unchanged and derives every other run path from
``store.root / "runs" / run_id`` so that ``runs/runs/<run_id>`` can never occur::

    <artifact_root>/runs/<run_id>/registry.db            SQLite task registry
    <artifact_root>/runs/<run_id>/run_manifest.json      frozen RunManifest (write-once)
    <artifact_root>/runs/<run_id>/refs/<name>.ref.json   ArtifactReference of published inputs
    <artifact_root>/runs/<run_id>/meanfield/...          published Stage 1 artifact
    <artifact_root>/runs/<run_id>/embedding/...          published Stage 2 artifacts

Compute configuration
---------------------
``COMPUTE_CONFIG`` is a resource-orchestration boundary (which kind of hardware a stage may
use).  It is distinct from SQD's internal ``classical_backend="python"`` solver option.  Only
``"cpu"`` is supported until the GPU backend task; any other value is rejected.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from serverless_bundle.steps.artifact_store import (  # noqa: E402
    ArtifactReference,
    LocalArtifactStore,
    KIND_EMBEDDING,
    KIND_MEANFIELD,
)
from serverless_bundle.steps.run_manifests import ManifestError, RunManifest  # noqa: E402
from serverless_bundle.steps.stage1_spec import RESULT_SCHEMA  # noqa: E402
from serverless_bundle.steps.stage_artifacts import (  # noqa: E402
    publish_embedding_artifact,
    publish_meanfield_artifact,
)
from proteasome_challenge.solver_assignments import (  # noqa: E402
    QUANTUM_SOLVERS,
    VALID_SOLVERS,
)
from proteasome_challenge.task_registry import (  # noqa: E402
    DuplicateTaskError,
    FragmentTaskRegistry,
    TaskType,
)

__all__ = [
    "COMPUTE_CONFIG_KEYS", "SUPPORTED_COMPUTE_BACKENDS",
    "ControlPlaneError", "ComputeConfigError", "RunNotFoundError",
    "ManifestConflictError", "ManifestInputError", "TaskConflictError", "StageSourceError",
    "validate_compute_config",
    "RunPaths", "RunContext", "init_run", "load_run_manifest",
    "validate_stage1_source", "publish_stage1", "validate_stage2_source", "publish_stage2",
    "load_stage_ref",
    "derive_solver_assignments", "collect_manifest_inputs",
    "collect_manifest_inputs_generic", "freeze_run_manifest",
    "manifest_identity", "ensure_fragment_tasks", "inspect_run_status",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class ControlPlaneError(RuntimeError):
    """Base class for control-plane errors."""


class ComputeConfigError(ControlPlaneError, ValueError):
    """COMPUTE_CONFIG has an unknown key, a missing key or an unsupported backend."""


class RunNotFoundError(ControlPlaneError):
    """The run does not exist and creation was not requested."""


class ManifestConflictError(ControlPlaneError):
    """A frozen manifest already exists and differs from the one being frozen."""


class ManifestInputError(ControlPlaneError):
    """Manifest inputs are incomplete, inconsistent, or carry a placeholder identity."""


class TaskConflictError(ControlPlaneError):
    """An existing registry task contradicts the frozen manifest."""


class StageSourceError(ControlPlaneError):
    """A Stage 1/2 source artifact is missing, incomplete or has the wrong identity."""


# ─────────────────────────────────────────────────────────────────────────────
#  Compute configuration (CPU-first resource boundary)
# ─────────────────────────────────────────────────────────────────────────────

COMPUTE_CONFIG_KEYS = ("meanfield_backend", "classical_backend", "quantum_postprocess_backend")
SUPPORTED_COMPUTE_BACKENDS = frozenset({"cpu"})   # GPU belongs to a later task


def validate_compute_config(config: Mapping[str, Any]) -> dict:
    """Return a validated copy of ``config`` or raise :class:`ComputeConfigError`."""
    if not isinstance(config, Mapping):
        raise ComputeConfigError("COMPUTE_CONFIG must be a dict")
    unknown = sorted(set(config) - set(COMPUTE_CONFIG_KEYS), key=str)
    if unknown:
        raise ComputeConfigError(
            f"unsupported COMPUTE_CONFIG keys {unknown}; allowed keys: {list(COMPUTE_CONFIG_KEYS)}")
    missing = [k for k in COMPUTE_CONFIG_KEYS if k not in config]
    if missing:
        raise ComputeConfigError(f"COMPUTE_CONFIG is missing keys {missing}")
    for key in COMPUTE_CONFIG_KEYS:
        value = config[key]
        if not isinstance(value, str) or value not in SUPPORTED_COMPUTE_BACKENDS:
            raise ComputeConfigError(
                f"COMPUTE_CONFIG[{key!r}] = {value!r} is not supported; "
                f"supported backends: {sorted(SUPPORTED_COMPUTE_BACKENDS)}")
    return {k: config[k] for k in COMPUTE_CONFIG_KEYS}


# ─────────────────────────────────────────────────────────────────────────────
#  Small file helpers
# ─────────────────────────────────────────────────────────────────────────────

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_once(path: Path, text: str) -> bool:
    """Atomically create ``path`` with ``text``; never overwrite.

    Returns True when this call created the file, False when it already existed.  The content
    is written to a temp file in the same directory, fsynced, then hard-linked into place
    (``link`` fails with EEXIST instead of replacing).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".cp_", suffix=".part", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(str(tmp), str(path))
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                return False
            raise
        return True
    finally:
        tmp.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Run initialisation / reopening
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunPaths:
    artifact_root: Path
    run_id: str
    run_dir: Path
    registry_path: Path
    manifest_path: Path
    refs_dir: Path


@dataclass
class RunContext:
    run_id: str
    paths: RunPaths
    store: LocalArtifactStore
    registry: FragmentTaskRegistry
    compute_config: dict
    created: bool                        # True when this call created the registry
    manifest: RunManifest | None = None  # frozen manifest if one is already on disk
    refs: dict = field(default_factory=dict)

    def close(self) -> None:
        self.registry.close()


def _run_paths(store: LocalArtifactStore, run_id: str) -> RunPaths:
    run_dir = store.root / "runs" / run_id          # same layout the store itself uses
    return RunPaths(
        artifact_root=store.root, run_id=run_id, run_dir=run_dir,
        registry_path=run_dir / "registry.db",
        manifest_path=run_dir / "run_manifest.json",
        refs_dir=run_dir / "refs",
    )


def init_run(
    artifact_root: str | Path,
    run_id: str,
    *,
    compute_config: Mapping[str, Any] | None = None,
    create: bool = True,
) -> RunContext:
    """Create a new run or reopen an existing one (idempotent).

    * registry absent  -> ``FragmentTaskRegistry.open`` (only when ``create`` is True)
    * registry present -> ``FragmentTaskRegistry.reopen``
    * an existing frozen manifest is loaded and digest-verified; a corrupt one raises.

    ``artifact_root`` is the store root.  It must not itself be named ``runs`` (the store adds
    that component, so such a root would produce ``runs/runs/<run_id>``).
    """
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise ControlPlaneError(
            f"RUN_ID {run_id!r} is invalid; use 1-128 characters from [A-Za-z0-9_.-], "
            "starting with a letter or digit")
    cfg = validate_compute_config(compute_config if compute_config is not None
                                  else {k: "cpu" for k in COMPUTE_CONFIG_KEYS})
    root = Path(artifact_root)
    if root.name == "runs":
        raise ControlPlaneError(
            f"artifact_root {str(root)!r} ends in 'runs'; the artifact store already adds a "
            "'runs/<run_id>' level, so pass its parent")
    store = LocalArtifactStore(root)
    paths = _run_paths(store, run_id)

    if paths.registry_path.is_file():
        registry = FragmentTaskRegistry.reopen(paths.registry_path)
        created = False
    elif create:
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        registry = FragmentTaskRegistry.open(paths.registry_path)
        created = True
    else:
        raise RunNotFoundError(f"no registry for run {run_id!r} at {paths.registry_path}")

    ctx = RunContext(run_id=run_id, paths=paths, store=store, registry=registry,
                     compute_config=cfg, created=created)
    try:
        ctx.manifest = load_run_manifest(paths)
    except Exception:
        registry.close()
        raise
    return ctx


def load_run_manifest(paths: RunPaths) -> RunManifest | None:
    """Load the frozen manifest, verify its digest and run id; ``None`` if not frozen yet."""
    if not paths.manifest_path.is_file():
        return None
    try:
        manifest = RunManifest.from_json(paths.manifest_path.read_text(encoding="utf-8"))
    except (ManifestError, ValueError) as exc:
        raise ManifestConflictError(f"frozen manifest {paths.manifest_path} is invalid: {exc}") from exc
    if not manifest.verify_digest():
        raise ManifestConflictError(f"frozen manifest {paths.manifest_path} fails digest verification")
    if manifest.run_id != paths.run_id:
        raise ManifestConflictError(
            f"frozen manifest belongs to run {manifest.run_id!r}, not {paths.run_id!r}")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 / Stage 2 artifact publication
# ─────────────────────────────────────────────────────────────────────────────

_FLOAT_TOL = 1e-8


def _read_json(path: Path, label: str) -> dict:
    if not Path(path).is_file():
        raise StageSourceError(f"{label} not found: {path}")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise StageSourceError(f"{label} is not valid JSON: {path}") from exc


def validate_stage1_source(
    mf_pkl_path: str | Path,
    meta_json_path: str | Path,
    *,
    ligand: str,
    basis: str,
    expected_hf: Mapping[str, Any],
) -> dict:
    """Check that a completed Stage 1 output matches the expected HF record.

    ``expected_hf`` is the verified record of the checkpoint (``Challenge.hf_record``) or a ligand entry of
    ``challenge_data/hf_results.json``.  The pickle is never loaded here; identity comes from the Stage 1 JSON sidecar
    and the file hash.  A ``stage1-hf-result/1`` sidecar is bound by its ``stage1_param_digest`` and by the SHA-256 it
    records for the pickle; a legacy sidecar (no digest) is bound by ``system``/``basis``.
    """
    mf = Path(mf_pkl_path)
    if not mf.is_file() or mf.stat().st_size == 0:
        raise StageSourceError(f"Stage 1 mean-field pickle missing or empty: {mf}")
    meta = _read_json(Path(meta_json_path), "Stage 1 metadata")
    if meta.get("stage") != "hf":
        raise StageSourceError(f"Stage 1 metadata has stage={meta.get('stage')!r}, expected 'hf'")
    if meta.get("basis") != basis:
        raise StageSourceError(f"Stage 1 identity mismatch: basis={meta.get('basis')!r}, expected {basis!r}")
    if meta.get("schema") == RESULT_SCHEMA:
        want = expected_hf.get("stage1_param_digest")
        if not want or meta.get("stage1_param_digest") != want:
            raise StageSourceError("Stage 1 specification digest differs from the expected record")
        if (meta.get("artifact") or {}).get("mf_data_pkl_sha256") != _sha256_file(mf):
            raise StageSourceError("Stage 1 pickle SHA-256 differs from the one recorded in its sidecar")
    elif meta.get("system") != ligand:
        raise StageSourceError(f"Stage 1 identity mismatch: system={meta.get('system')!r}, expected {ligand!r}")
    if meta.get("hf_converged") is not True:
        raise StageSourceError("Stage 1 reports an unconverged RHF")
    for key in ("n_ao", "n_electrons"):
        if int(meta.get(key, -1)) != int(expected_hf[key]):
            raise StageSourceError(
                f"Stage 1 {key}={meta.get(key)!r} differs from hf_results {expected_hf[key]!r}")
    if abs(float(meta["hf_energy"]) - float(expected_hf["hf_energy"])) > _FLOAT_TOL:
        raise StageSourceError(
            f"Stage 1 hf_energy {meta['hf_energy']!r} differs from hf_results "
            f"{expected_hf['hf_energy']!r}")
    return {"hf_energy": float(meta["hf_energy"]), "n_ao": int(meta["n_ao"]),
            "n_electrons": int(meta["n_electrons"]), "mf_sha256": _sha256_file(mf),
            "stage1_param_digest": meta.get("stage1_param_digest")}


def validate_stage2_source(
    ewf_dumpfile_path: str | Path,
    embedding_pkl_path: str | Path | None = None,
    meta_json_path: str | Path | None = None,
    *,
    ligand: str = "",
    basis: str = "",
    expected_fragment_ids: Sequence[int] | None = None,
) -> dict:
    """Check that a completed Stage 2 output is internally consistent.

    The fragment groups of the HDF5 dump must equal the sidecar's ``n_fragments`` (and
    ``expected_fragment_ids`` when given), and each must carry the ``heff`` and ``eris``
    datasets that define its Hamiltonian identity.
    """
    import h5py

    h5 = Path(ewf_dumpfile_path)
    if not h5.is_file() or h5.stat().st_size == 0:
        raise StageSourceError(f"Stage 2 EWF dumpfile missing or empty: {h5}")

    emb = Path(embedding_pkl_path) if embedding_pkl_path is not None else None
    if emb is not None and (not emb.is_file() or emb.stat().st_size == 0):
        raise StageSourceError(f"Stage 2 embedding pickle missing or empty: {emb}")

    meta = _read_json(Path(meta_json_path), "Stage 2 metadata") if meta_json_path is not None else {}
    if meta:
        if meta.get("stage") != "fragmentation":
            raise StageSourceError(f"Stage 2 metadata has stage={meta.get('stage')!r}")
        if ligand and basis and meta.get("run_name") and meta.get("run_name") != f"{ligand}_ligand_{basis}":
            raise StageSourceError(
                f"Stage 2 run_name {meta.get('run_name')!r} does not match {ligand}/{basis}")

    try:
        with h5py.File(str(h5), "r") as hf:
            ids = sorted(int(k.split("_", 1)[1]) for k in hf if k.startswith("fragment_"))
            for fid in ids:
                grp = hf[f"fragment_{fid}"]
                if "heff" not in grp or "eris" not in grp:
                    raise StageSourceError(f"Stage 2 dump fragment {fid} lacks heff/eris")
    except OSError as exc:
        raise StageSourceError(f"Stage 2 dump is not a readable HDF5 file: {h5}") from exc
    if not ids:
        raise StageSourceError(f"Stage 2 dump has no fragment_* groups: {h5}")
    if meta and int(meta.get("n_fragments", -1)) != len(ids):
        raise StageSourceError(
            f"Stage 2 n_fragments={meta.get('n_fragments')!r} but the dump holds {len(ids)}")
    if expected_fragment_ids is not None and ids != sorted(int(i) for i in expected_fragment_ids):
        raise StageSourceError("Stage 2 fragment ids differ from the authoritative assignment map")

    out = {
        "n_fragments": len(ids),
        "fragment_ids": ids,
        "ewf_sha256": _sha256_file(h5),
        "stage2_param_digest": meta.get("stage2_param_digest"),
    }
    if emb is not None:
        out["embedding_sha256"] = _sha256_file(emb)
    return out


def _persist_ref(ctx: RunContext, name: str, ref: ArtifactReference) -> ArtifactReference:
    """Keep the first ArtifactReference published under ``name`` (stable across restarts)."""
    existing = load_stage_ref(ctx, name)
    if existing is not None:
        if existing.file_sha256 != ref.file_sha256:
            raise StageSourceError(
                f"stored reference {name!r} has file_sha256 {existing.file_sha256[:12]}..., "
                f"but the source now hashes to {ref.file_sha256[:12]}...; refusing to replace it")
        ctx.refs[name] = existing
        return existing
    _write_once(ctx.paths.refs_dir / f"{name}.ref.json", ref.to_json())
    ctx.refs[name] = ref
    return ref


def load_stage_ref(ctx: RunContext, name: str) -> ArtifactReference | None:
    """Load a previously persisted reference (``meanfield``, ``embedding_ewf``, ``embedding_pkl``)."""
    path = ctx.paths.refs_dir / f"{name}.ref.json"
    if not path.is_file():
        return None
    ref = ArtifactReference.from_json(path.read_text(encoding="utf-8"))
    if ref.run_id != ctx.run_id:
        raise StageSourceError(f"stored reference {name!r} belongs to run {ref.run_id!r}")
    return ref


def _verified(ctx: RunContext, ref: ArtifactReference, source: Path, expected_sha: str) -> None:
    if ref.file_sha256 != expected_sha:
        raise StageSourceError(f"published {ref.artifact_name} hash differs from its source")
    result = ctx.store.verify(ref)
    if not result.ok:
        raise StageSourceError(f"published {ref.artifact_name} failed verification: {result.reason}")


def publish_stage1(
    ctx: RunContext,
    mf_pkl_path: str | Path,
    meta_json_path: str | Path,
    *,
    ligand: str,
    basis: str,
    expected_hf: Mapping[str, Any],
) -> ArtifactReference:
    """Validate a completed Stage 1 output and publish it (idempotent, restart-safe)."""
    info = validate_stage1_source(mf_pkl_path, meta_json_path, ligand=ligand, basis=basis,
                                  expected_hf=expected_hf)
    ref = publish_meanfield_artifact(
        ctx.store, mf_pkl_path, ctx.run_id,
        hf_energy=info["hf_energy"], n_ao=info["n_ao"], n_electrons=info["n_electrons"],
        extra_meta=({"stage1_param_digest": info["stage1_param_digest"]} if info["stage1_param_digest"] else None))
    _verified(ctx, ref, Path(mf_pkl_path), info["mf_sha256"])
    return _persist_ref(ctx, "meanfield", ref)


def publish_stage2(
    ctx: RunContext,
    ewf_dumpfile_path: str | Path,
    embedding_pkl_path: str | Path | None = None,
    meta_json_path: str | Path | None = None,
    *,
    ligand: str = "",
    basis: str = "",
    expected_fragment_ids: Sequence[int] | None = None,
    stage2_param_digest: str | None = None,
) -> ArtifactReference | tuple[ArtifactReference, ArtifactReference]:
    """Validate a completed Stage 2 output and publish it; returns ``ewf_ref`` or ``(ewf_ref, embedding_ref)``."""
    info = validate_stage2_source(ewf_dumpfile_path, embedding_pkl_path, meta_json_path,
                                  ligand=ligand, basis=basis,
                                  expected_fragment_ids=expected_fragment_ids)
    digest = stage2_param_digest or info.get("stage2_param_digest")
    extra_meta = {"stage2_param_digest": digest} if digest else None

    if embedding_pkl_path is None:
        ewf_ref = publish_embedding_artifact(
            ctx.store, ewf_dumpfile_path, None, ctx.run_id,
            n_fragments=info["n_fragments"], extra_meta=extra_meta)
        _verified(ctx, ewf_ref, Path(ewf_dumpfile_path), info["ewf_sha256"])
        return _persist_ref(ctx, "embedding_ewf", ewf_ref)

    ewf_ref, emb_ref = publish_embedding_artifact(
        ctx.store, ewf_dumpfile_path, embedding_pkl_path, ctx.run_id,
        n_fragments=info["n_fragments"], extra_meta=extra_meta)
    _verified(ctx, ewf_ref, Path(ewf_dumpfile_path), info["ewf_sha256"])
    _verified(ctx, emb_ref, Path(embedding_pkl_path), info["embedding_sha256"])
    return (_persist_ref(ctx, "embedding_ewf", ewf_ref),
            _persist_ref(ctx, "embedding_pkl", emb_ref))


# ─────────────────────────────────────────────────────────────────────────────
#  Manifest lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def derive_solver_assignments(
    classical_assignments: Mapping[int, str],
    quantum_fragment_ids: Sequence[int],
    quantum_solver: str = "TrimSQD",
) -> dict[int, str]:
    """Frozen FCI/CCSD map with the selected quantum fragments upgraded to ``quantum_solver``."""
    if quantum_solver not in QUANTUM_SOLVERS:
        raise ManifestInputError(f"{quantum_solver!r} is not a quantum solver")
    base = {int(k): str(v) for k, v in classical_assignments.items()}
    bad = {k: v for k, v in base.items() if v not in VALID_SOLVERS or v in QUANTUM_SOLVERS}
    if bad:
        raise ManifestInputError(f"baseline assignments must be FCI/CCSD only, found {bad}")
    quantum = [int(f) for f in quantum_fragment_ids]
    if len(set(quantum)) != len(quantum):
        raise ManifestInputError("quantum fragment ids contain duplicates")
    unknown = sorted(set(quantum) - set(base))
    if unknown:
        raise ManifestInputError(f"quantum fragment ids {unknown} are not in the assignment map")
    out = dict(base)
    for fid in quantum:
        out[fid] = quantum_solver
    return dict(sorted(out.items()))


def collect_manifest_inputs(
    lab: Any,
    ligand: str,
    shared_protocol: Mapping[str, Any] | None,
    final_fragment_ids: Sequence[int],
) -> dict:
    """Gather RunManifest fields from verified Stage 1/2 outputs and the frozen protocol.

    ``lab`` needs ``hf_record(ligand)``, ``baseline_assignments(ligand)`` and ``bundle_dir``.
    Raises :class:`ManifestInputError` unless the shared protocol has been frozen, is complete
    and covers exactly ``final_fragment_ids``: a placeholder protocol digest is never accepted.
    """
    if not shared_protocol:
        raise ManifestInputError(
            "the shared protocol is not frozen yet; freeze the Stage E protocol before the manifest")
    digest = shared_protocol.get("protocol_digest", "")
    if not isinstance(digest, str) or not _HEX64.match(digest):
        raise ManifestInputError("shared protocol has no valid protocol_digest")
    if shared_protocol.get("complete") is not True:
        raise ManifestInputError("shared protocol is not complete for the selected fragments")
    targets = sorted(int(f) for f in shared_protocol.get("target_fragment_ids", []))
    quantum = sorted(int(f) for f in final_fragment_ids)
    if targets != quantum:
        raise ManifestInputError(
            f"protocol targets {targets} differ from the frozen quantum selection {quantum}")
    solver = shared_protocol.get("solver", "TrimSQD")

    hf = lab.hf_record(ligand)
    if hf.get("hf_converged") is not True:
        raise ManifestInputError(f"HF record for {ligand} is not converged")
    if hf.get("stage2_qualified") is not True:
        raise ManifestInputError(f"the Stage 1 result for {ligand} is not qualified for downstream stages")
    if "charge" not in hf or "spin" not in hf:
        raise ManifestInputError(f"the HF record for {ligand} does not state its charge and spin")
    xyz = Path(lab.bundle_dir) / "configs" / f"system_{ligand}.xyz"
    if not xyz.is_file():
        raise ManifestInputError(f"geometry file not found: {xyz}")
    assignments = derive_solver_assignments(lab.baseline_assignments(ligand), quantum, solver)
    return {
        "ligand": ligand,
        "geometry_sha256": _sha256_file(xyz),
        "basis": str(hf["basis"]),
        "charge": int(hf["charge"]),
        "spin": int(hf["spin"]),
        "n_electrons": int(hf["n_electrons"]),
        "n_ao": int(hf["n_ao"]),
        "fragment_ids": sorted(assignments),
        "solver_assignments": assignments,
        "protocol_digest": digest,
    }


def collect_manifest_inputs_generic(
    hf_status_record: Mapping[str, Any],
    ewf_result: Any,
    shared_protocol: Mapping[str, Any] | None,
    final_fragment_ids: Sequence[int],
    *,
    geometry_path: Any | None = None,
    solver_assignments: Mapping[int, str] | None = None,
) -> dict:
    """Gather RunManifest fields for an *arbitrary* molecular system.

    Unlike :func:`collect_manifest_inputs`, this function does not look up
    challenge-catalog files or the frozen ``solver_assignments.json``.  It
    accepts pre-resolved inputs from the notebook's Stage 1/2 variables:

    Parameters
    ----------
    hf_status_record : dict
        The verified HF record returned by
        :meth:`~proteasome_challenge.Challenge.hf_status` (accessed via
        ``hf_status.record``).  Must contain ``charge``, ``spin``,
        ``n_electrons``, ``n_ao``, and ``basis``.
    ewf_result : EWFResult
        The verified Stage 2 result (from ``ensure_ewf`` or ``ewf_status``
        when ``ok`` is True).
    shared_protocol : dict | None
        The frozen shared TrimSQD protocol from Stage E (same object the
        canonical path uses).
    final_fragment_ids : sequence[int]
        The quantum fragment IDs selected in Stage D (``FINAL_FRAGMENT_IDS``).
    geometry_path : path-like or None
        Path to the XYZ geometry file.  When ``None``, the geometry SHA-256 is
        omitted from the manifest (recorded as the empty string).  A missing
        file is also acceptable — the manifest will carry an empty SHA-256
        rather than raising, so the notebook does not block on this field.
    solver_assignments : dict[int, str] | None
        Full solver map ``{fragment_id: solver}``.  When ``None``, the map is
        derived automatically from the EWF orbital counts using the standard
        :class:`~proteasome_challenge.solver_assignments.SolverAllocationPolicy`
        with ``final_fragment_ids`` as the quantum selection.

    Returns
    -------
    dict
        Same schema as :func:`collect_manifest_inputs`.
    """
    from proteasome_challenge.ewf_stage import EWFResult
    from proteasome_challenge.solver_assignments import (
        SolverAllocationPolicy, assign_solvers_by_size,
        QUANTUM_SOLVERS,
    )

    # ── Validate protocol ────────────────────────────────────────────────────
    if not shared_protocol:
        raise ManifestInputError(
            "the shared protocol is not frozen yet; freeze the Stage E protocol before the manifest")
    digest = shared_protocol.get("protocol_digest", "")
    if not isinstance(digest, str) or not _HEX64.match(digest):
        raise ManifestInputError("shared protocol has no valid protocol_digest")
    if shared_protocol.get("complete") is not True:
        raise ManifestInputError("shared protocol is not complete for the selected fragments")
    targets = sorted(int(f) for f in shared_protocol.get("target_fragment_ids", []))
    quantum = sorted(int(f) for f in final_fragment_ids)
    if targets != quantum:
        raise ManifestInputError(
            f"protocol targets {targets} differ from the frozen quantum selection {quantum}")
    qsolver = shared_protocol.get("solver", "TrimSQD")
    if qsolver not in QUANTUM_SOLVERS:
        raise ManifestInputError(f"protocol solver {qsolver!r} is not a quantum solver")

    # ── Validate HF record ───────────────────────────────────────────────────
    rec = dict(hf_status_record) if hf_status_record else {}
    for field_name in ("charge", "spin", "n_electrons", "n_ao", "basis"):
        if field_name not in rec:
            raise ManifestInputError(
                f"hf_status_record is missing required field {field_name!r}; "
                "use hf_status.record from an OK hf_status object")
    if rec.get("hf_converged") is False:
        raise ManifestInputError("HF record reports hf_converged=False")

    # ── Validate EWF result ──────────────────────────────────────────────────
    if ewf_result is None or not hasattr(ewf_result, "norb_by_fragment"):
        raise ManifestInputError(
            "ewf_result must be a verified EWFResult with norb_by_fragment; "
            "obtain it from ewf_status (ok=True) after running Stage 2")

    mol_name = ewf_result.spec.stage1.molecule.name if ewf_result.spec is not None else "molecule"

    # ── Geometry SHA-256 ─────────────────────────────────────────────────────
    geom_sha = ""
    if geometry_path is not None:
        from pathlib import Path as _Path
        _gp = _Path(geometry_path)
        if _gp.is_file():
            geom_sha = _sha256_file(_gp)
        # Missing geometry file → empty SHA; notebook is not blocked

    # ── Solver assignments ────────────────────────────────────────────────────
    if solver_assignments is not None:
        assignments = {int(k): str(v) for k, v in solver_assignments.items()}
    else:
        policy = SolverAllocationPolicy()
        base = assign_solvers_by_size(ewf_result.norb_by_fragment, policy=policy)
        assignments = dict(base)
        for fid in quantum:
            assignments[fid] = qsolver
    assignments = dict(sorted(assignments.items()))

    return {
        "ligand": mol_name,
        "geometry_sha256": geom_sha,
        "basis": str(rec["basis"]),
        "charge": int(rec["charge"]),
        "spin": int(rec["spin"]),
        "n_electrons": int(rec["n_electrons"]),
        "n_ao": int(rec["n_ao"]),
        "fragment_ids": sorted(assignments),
        "solver_assignments": assignments,
        "protocol_digest": digest,
    }


def _ref_dict(ref: Any) -> dict:
    return ref if isinstance(ref, dict) else ref.as_dict()


def manifest_identity(manifest: RunManifest) -> dict:
    """Scientific identity of a manifest (excludes timestamps and the self-digest)."""
    return {
        "run_id": manifest.run_id, "ligand": manifest.ligand,
        "geometry_sha256": manifest.geometry_sha256, "basis": manifest.basis,
        "charge": manifest.charge, "spin": manifest.spin,
        "n_electrons": manifest.n_electrons, "n_ao": manifest.n_ao,
        "hf_file_sha256": manifest.hf_artifact_ref.get("file_sha256"),
        "embedding_file_sha256": manifest.embedding_artifact_ref.get("file_sha256"),
        "fragment_ids": list(manifest.fragment_ids),
        "solver_assignments": dict(manifest.solver_assignments),
        "protocol_digest": manifest.protocol_digest,
    }


def _software_versions() -> dict:
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "pyscf", "h5py"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def freeze_run_manifest(
    ctx: RunContext,
    inputs: Mapping[str, Any],
    hf_ref: Any,
    embedding_ref: Any,
) -> RunManifest:
    """Build, persist (write-once, atomic) and re-verify the run manifest.

    * Refuses an empty/placeholder ``protocol_digest``.
    * Refuses source references that are not from this run or fail store verification.
    * If a manifest is already frozen, returns it when the scientific identity is equal and
      raises :class:`ManifestConflictError` otherwise; it is never overwritten.
    """
    digest = inputs.get("protocol_digest", "")
    if not isinstance(digest, str) or not _HEX64.match(digest):
        raise ManifestInputError("cannot freeze a manifest without a real protocol_digest")
    hf_d, emb_d = _ref_dict(hf_ref), _ref_dict(embedding_ref)
    for label, d, kind in (("hf", hf_d, KIND_MEANFIELD), ("embedding", emb_d, KIND_EMBEDDING)):
        if d.get("kind") != kind:
            raise ManifestInputError(f"{label} reference has kind {d.get('kind')!r}, expected {kind!r}")
        if d.get("run_id") != ctx.run_id:
            raise ManifestInputError(f"{label} reference belongs to run {d.get('run_id')!r}")
        if not ctx.store.verify(ArtifactReference.from_dict(d)).ok:
            raise ManifestInputError(f"{label} reference does not verify against the artifact store")
    if emb_d.get("meta", {}).get("file_role") != "ewf_dumpfile":
        raise ManifestInputError("embedding reference must be the EWF dumpfile artifact")

    candidate = RunManifest.build(
        run_id=ctx.run_id, ligand=inputs["ligand"], geometry_sha256=inputs["geometry_sha256"],
        basis=inputs["basis"], charge=inputs["charge"], spin=inputs["spin"],
        n_electrons=inputs["n_electrons"], n_ao=inputs["n_ao"],
        hf_artifact_ref=hf_d, embedding_artifact_ref=emb_d,
        fragment_ids=inputs["fragment_ids"], solver_assignments=inputs["solver_assignments"],
        protocol_digest=digest, software_versions=_software_versions())
    if sorted(int(k) for k in candidate.solver_assignments) != candidate.fragment_ids:
        raise ManifestInputError("solver assignments do not cover exactly the manifest fragment ids")

    existing = load_run_manifest(ctx.paths)
    if existing is None:
        created = _write_once(ctx.paths.manifest_path, candidate.to_json())
        if created:
            reloaded = load_run_manifest(ctx.paths)
            if reloaded is None or reloaded.manifest_digest != candidate.manifest_digest:
                raise ManifestConflictError("manifest read-back differs from what was written")
            ctx.manifest = reloaded
            return reloaded
        existing = load_run_manifest(ctx.paths)      # lost a creation race: compare below
    want, have = manifest_identity(candidate), manifest_identity(existing)
    diff = sorted(k for k in want if want[k] != have[k])
    if diff:
        raise ManifestConflictError(
            f"a frozen manifest already exists for run {ctx.run_id!r} and differs in {diff}; "
            "choose a new RUN_ID instead of overwriting it")
    ctx.manifest = existing
    return existing


# ─────────────────────────────────────────────────────────────────────────────
#  Idempotent task creation
# ─────────────────────────────────────────────────────────────────────────────

def ensure_fragment_tasks(
    registry: FragmentTaskRegistry,
    manifest: RunManifest,
    hamiltonian_hashes: Mapping[int, str],
    *,
    resource_family: str = "l",
) -> dict:
    """Create one task per manifest fragment; safe to call repeatedly.

    Quantum solvers map to ``TaskType.QUANTUM``, FCI/CCSD to ``TaskType.CLASSICAL``.  A
    ``DuplicateTaskError`` is treated as "already exists" after checking the stored task still
    matches the manifest; a mismatching task, or a task for a fragment outside the manifest,
    raises :class:`TaskConflictError`.  Returns ``{"created": [...], "existing": [...]}`` of
    ``(fragment_id, task_type)`` pairs.
    """
    if not manifest.verify_digest():
        raise ManifestInputError("manifest fails digest verification")
    hashes = {int(k): v for k, v in hamiltonian_hashes.items()}
    missing = [f for f in manifest.fragment_ids if f not in hashes]
    if missing:
        raise ManifestInputError(f"no Hamiltonian hash for fragments {missing}")

    existing_tasks = registry.list_tasks(manifest.run_id)
    known = set(manifest.fragment_ids)
    stray = sorted(t.fragment_id for t in existing_tasks if t.fragment_id not in known)
    if stray:
        raise TaskConflictError(f"registry holds tasks for fragments {stray} outside the manifest")

    tasks_by_fid: dict[int, list[Any]] = {}
    for t in existing_tasks:
        tasks_by_fid.setdefault(t.fragment_id, []).append(t)

    for fid, task_list in tasks_by_fid.items():
        solver = manifest.solver_for(fid)
        expected_type = TaskType.QUANTUM if solver in QUANTUM_SOLVERS else TaskType.CLASSICAL
        for t in task_list:
            if t.task_type != expected_type or t.solver != solver or t.hamiltonian_sha256 != hashes[fid]:
                raise TaskConflictError(
                    f"existing task for fragment {fid} ({t.task_type}/{t.solver}) does not match "
                    f"the manifest ({expected_type}/{solver})")

    created: list[tuple[int, str]] = []
    existing: list[tuple[int, str]] = []
    for fid in manifest.fragment_ids:
        solver = manifest.solver_for(fid)
        task_type = TaskType.QUANTUM if solver in QUANTUM_SOLVERS else TaskType.CLASSICAL
        try:
            registry.create_task(
                run_id=manifest.run_id, fragment_id=fid, task_type=task_type, solver=solver,
                hamiltonian_sha256=hashes[fid], resource_family=resource_family)
            created.append((fid, task_type))
        except DuplicateTaskError:
            rec = registry.find_task(manifest.run_id, fid, task_type)
            if rec is None or rec.solver != solver or rec.hamiltonian_sha256 != hashes[fid]:
                raise TaskConflictError(
                    f"existing {task_type} task for fragment {fid} does not match the manifest "
                    f"(solver {solver!r})")
            existing.append((fid, task_type))
    return {"created": created, "existing": existing}


# ─────────────────────────────────────────────────────────────────────────────
#  Read-only status inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_run_status(ctx: RunContext) -> dict[str, Any]:
    """Inspect persisted run state in a read-only manner (Task 5.9B).

    Safe to call across kernel restarts. Does not create tasks, launch workers,
    retrieve counts, or contact the QPU. Unimplemented capabilities are explicitly
    marked as deferred or pending.
    """
    stage1_ref = load_stage_ref(ctx, "meanfield")
    ewf_ref = load_stage_ref(ctx, "embedding_ewf")
    emb_ref = load_stage_ref(ctx, "embedding_pkl")

    manifest_info: dict[str, Any] = {
        "status": "FROZEN" if ctx.manifest is not None else "PENDING",
        "manifest_digest": ctx.manifest.manifest_digest if ctx.manifest else None,
        "protocol_digest": ctx.manifest.protocol_digest if ctx.manifest else None,
        "solver_counts": None,
    }
    if ctx.manifest is not None:
        solvers = list(ctx.manifest.solver_assignments.values())
        manifest_info["solver_counts"] = {
            "FCI": solvers.count("FCI"),
            "CCSD": solvers.count("CCSD"),
            "TrimSQD": solvers.count("TrimSQD"),
            "total": len(solvers),
        }

    tasks = ctx.registry.list_tasks(ctx.run_id)
    by_type = {"classical": 0, "quantum": 0}
    by_state: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    persisted_jobs: dict[str, dict[str, Any]] = {}

    for t in tasks:
        by_type[t.task_type] = by_type.get(t.task_type, 0) + 1
        by_state[t.state] = by_state.get(t.state, 0) + 1
        if t.state in ("FAILED", "RETRY_PENDING"):
            failures.append({
                "task_id": t.task_id,
                "fragment_id": t.fragment_id,
                "solver": t.solver,
                "state": t.state,
                "failure_reason": t.failure_reason,
            })
        if t.state == "AMBIGUOUS":
            ambiguities.append({
                "task_id": t.task_id,
                "fragment_id": t.fragment_id,
                "solver": t.solver,
                "qpu_job_id": t.qpu_job_id,
            })
        if t.qpu_job_id or t.serverless_job_id:
            persisted_jobs[t.task_id] = {
                "fragment_id": t.fragment_id,
                "qpu_job_id": t.qpu_job_id,
                "serverless_job_id": t.serverless_job_id,
            }

    total_tasks = len(tasks)
    complete_count = by_state.get("COMPLETE", 0)
    # The required fragment population comes from the frozen manifest (never a literal).
    required_count = len(ctx.manifest.fragment_ids) if ctx.manifest is not None else total_tasks

    if total_tasks == 0:
        completion_readiness = "PENDING_REGISTRATION"
    elif complete_count == total_tasks and total_tasks == required_count:
        completion_readiness = "READY_FOR_STAGE4"
    else:
        completion_readiness = f"IN_PROGRESS ({complete_count}/{total_tasks} complete)"

    return {
        "run_id": ctx.run_id,
        "created": ctx.created,
        "compute_config": dict(ctx.compute_config),
        "stage1": {
            "status": "PUBLISHED" if stage1_ref is not None else "PENDING",
            "artifact_name": stage1_ref.artifact_name if stage1_ref else None,
            "sha256": stage1_ref.file_sha256 if stage1_ref else None,
        },
        "stage2": {
            "ewf_status": "PUBLISHED" if ewf_ref is not None else "PENDING",
            "ewf_sha256": ewf_ref.file_sha256 if ewf_ref else None,
            "embedding_status": "PUBLISHED" if emb_ref is not None else "PENDING",
            "embedding_sha256": emb_ref.file_sha256 if emb_ref else None,
        },
        "manifest": manifest_info,
        "tasks": {
            "total": total_tasks,
            "by_type": by_type,
            "by_state": by_state,
            "failures": failures,
            "ambiguities": ambiguities,
            "persisted_jobs": persisted_jobs,
        },
        "completion_readiness": completion_readiness,
        "remote_dispatch": "CLASSICAL: available (classical_dispatcher; explicit flags only) | QUANTUM: deferred to 5.9C.2",
    }
