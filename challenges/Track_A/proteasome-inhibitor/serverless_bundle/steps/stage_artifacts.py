"""
stage_artifacts — Publication and retrieval helpers for Stage 1/2/3 artifacts.

Bridges the HDF5-based fragment result store (``fragment_result_store``) and the
immutable artifact store (``artifact_store``) so that:

  Stage 1 (RHF) output → mf_data.pkl → ArtifactReference
  Stage 2 (EWF) outputs → ewf_dumpfile.h5 + embedding_data.pkl → ArtifactReference
  Stage 3 (solve) output → fragment SolverResult HDF5 → ArtifactReference

Each function publishes a local file to the store and returns an ArtifactReference
that can be:
  - Embedded in a RunManifest or FragmentArtifactManifest.
  - Passed to an independent worker as a small JSON-safe reference.
  - Verified and retrieved by that worker without relying on the original
    worker's /tmp directory.

Stage 1 / Stage 2 artifacts contain PySCF objects and Vayesta embedding objects
that are pickled.  These are treated as trusted-code artifacts.  Loaders must
only unpickle files from sources they control.  This module never loads
untrusted pickle data.

The persistent QPU sampling ledger is NEVER touched by this module.

Backward compatibility
----------------------
All existing Stage 1–3 entrypoints (base64 JSON transport) are unmodified.
These helpers are additive: they can optionally be called alongside the existing
workflow to start building durable artifact references.

Usage
-----
  >>> from serverless_bundle.steps.stage_artifacts import (
  ...     publish_meanfield_artifact, publish_embedding_artifact,
  ...     publish_solver_result_artifact, retrieve_meanfield_artifact,
  ...     retrieve_embedding_artifact,
  ... )
  >>> store = LocalArtifactStore(run_root)
  >>> mf_ref = publish_meanfield_artifact(store, mf_pkl_path, run_id="ixazomib_sto-3g")
  >>> emb_ref = publish_embedding_artifact(store, ewf_h5_path, emb_pkl_path, run_id="…")
  >>> result_ref = publish_solver_result_artifact(
  ...     store, solver_result, frag_h5_path,
  ...     run_id="…", fragment_id=0, solver="FCI", hamiltonian_sha256="abc…",
  ... )
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from artifact_store import (
        ArtifactReference,
        ArtifactStoreError,
        LocalArtifactStore,
        KIND_MEANFIELD,
        KIND_EMBEDDING,
        KIND_FRAGMENT_H5,
    )
except ImportError:
    from .artifact_store import (  # type: ignore[no-redef]
        ArtifactReference,
        ArtifactStoreError,
        LocalArtifactStore,
        KIND_MEANFIELD,
        KIND_EMBEDDING,
        KIND_FRAGMENT_H5,
    )

try:
    from fragment_result_store import (
        FragmentResultReference,
        FragmentStoreError,
        save_fragment_result,
        load_fragment_result,
        load_fragment_result_from_ref,
        compute_payload_digest,
    )
except ImportError:
    from .fragment_result_store import (  # type: ignore[no-redef]
        FragmentResultReference,
        FragmentStoreError,
        save_fragment_result,
        load_fragment_result,
        load_fragment_result_from_ref,
        compute_payload_digest,
    )


class StageArtifactError(RuntimeError):
    """Publication or retrieval of a stage artifact failed."""


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 — Mean-field (mf_data.pkl)
# ─────────────────────────────────────────────────────────────────────────────

def publish_meanfield_artifact(
    store: LocalArtifactStore,
    mf_pkl_path: str | Path,
    run_id: str,
    *,
    artifact_name: str | None = None,
    producer_version: str = "",
    hf_energy: float | None = None,
    n_ao: int | None = None,
    n_electrons: int | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> ArtifactReference:
    """Publish a mean-field pickle to the artifact store.

    Parameters
    ----------
    store          : target LocalArtifactStore
    mf_pkl_path    : path to mf_data.pkl produced by step1_meanfield
    run_id         : run scoping namespace
    artifact_name  : storage name (default: ``mf_data.pkl``)
    producer_version : optional code version string
    hf_energy, n_ao, n_electrons : optional provenance fields stored in ``meta``
    extra_meta     : additional JSON-serialisable identity fields (e.g. ``stage1_param_digest``)

    Returns
    -------
    ArtifactReference
    """
    src = Path(mf_pkl_path)
    if not src.is_file():
        raise StageArtifactError(f"Mean-field pickle not found: {src}")
    if artifact_name is None:
        artifact_name = src.name

    meta: dict[str, Any] = {"kind": KIND_MEANFIELD, "run_id": run_id}
    if hf_energy is not None:
        meta["hf_energy"] = float(hf_energy)
    if n_ao is not None:
        meta["n_ao"] = int(n_ao)
    if n_electrons is not None:
        meta["n_electrons"] = int(n_electrons)
    if extra_meta:
        meta.update(extra_meta)

    return store.publish(
        src,
        kind=KIND_MEANFIELD,
        run_id=run_id,
        artifact_name=artifact_name,
        producer_version=producer_version,
        meta=meta,
    )


def retrieve_meanfield_artifact(
    store: LocalArtifactStore,
    ref: ArtifactReference,
    dest_dir: str | Path,
) -> Path:
    """Retrieve and verify a mean-field pickle from the store.

    Returns the path to the retrieved mf_data.pkl file.
    Caller is responsible for safe unpickling (trusted source only).
    """
    if ref.kind != KIND_MEANFIELD:
        raise StageArtifactError(
            f"Expected artifact of kind {KIND_MEANFIELD!r}, got {ref.kind!r}"
        )
    return store.fetch(ref, dest_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 — EWF embedding (ewf_dumpfile.h5 + embedding_data.pkl)
# ─────────────────────────────────────────────────────────────────────────────

def publish_embedding_artifact(
    store: LocalArtifactStore,
    ewf_dumpfile_path: str | Path,
    embedding_pkl_path: str | Path | None = None,
    run_id: str = "",
    *,
    ewf_artifact_name: str | None = None,
    emb_artifact_name: str | None = None,
    producer_version: str = "",
    n_fragments: int | None = None,
    bath_type: str | None = None,
    truncation: float | None = None,
    extra_meta: Mapping[str, Any] | None = None,
) -> ArtifactReference | tuple[ArtifactReference, ArtifactReference]:
    """Publish Stage 2 outputs to the artifact store.

    Parameters
    ----------
    store              : target LocalArtifactStore
    ewf_dumpfile_path  : path to ewf_dumpfile.h5 produced by step2_fragments
    embedding_pkl_path : optional legacy path to embedding_data.pkl
    run_id             : run scoping namespace

    Returns
    -------
    ArtifactReference if embedding_pkl_path is None, else tuple (ewf_ref, emb_ref)
    """
    ewf_path = Path(ewf_dumpfile_path)
    if not ewf_path.is_file():
        raise StageArtifactError(f"EWF dumpfile not found: {ewf_path}")

    if ewf_artifact_name is None:
        ewf_artifact_name = ewf_path.name

    meta: dict[str, Any] = {"kind": KIND_EMBEDDING, "run_id": run_id}
    if n_fragments is not None:
        meta["n_fragments"] = int(n_fragments)
    if bath_type is not None:
        meta["bath_type"] = bath_type
    if truncation is not None:
        meta["truncation"] = float(truncation)
    if extra_meta:
        meta.update(extra_meta)

    ewf_ref = store.publish(
        ewf_path,
        kind=KIND_EMBEDDING,
        run_id=run_id,
        artifact_name=ewf_artifact_name,
        producer_version=producer_version,
        meta=dict(meta, file_role="ewf_dumpfile"),
    )

    if embedding_pkl_path is None:
        return ewf_ref

    emb_path = Path(embedding_pkl_path)
    if not emb_path.is_file():
        raise StageArtifactError(f"embedding pickle not found: {emb_path}")
    if emb_artifact_name is None:
        emb_artifact_name = emb_path.name

    emb_ref = store.publish(
        emb_path,
        kind=KIND_EMBEDDING,
        run_id=run_id,
        artifact_name=emb_artifact_name,
        producer_version=producer_version,
        meta=dict(meta, file_role="embedding_pkl"),
    )
    return ewf_ref, emb_ref


def retrieve_embedding_artifact(
    store: LocalArtifactStore,
    ewf_ref: ArtifactReference,
    emb_ref: ArtifactReference,
    dest_dir: str | Path,
) -> tuple[Path, Path]:
    """Retrieve and verify both Stage 2 artifacts.

    Returns (ewf_dumpfile_path, embedding_pkl_path).
    Caller is responsible for safe unpickling (trusted source only).
    """
    for ref in (ewf_ref, emb_ref):
        if ref.kind != KIND_EMBEDDING:
            raise StageArtifactError(
                f"Expected artifact of kind {KIND_EMBEDDING!r}, got {ref.kind!r}"
            )
    # Ensure dest_dir exists as a directory so fetch places files inside it
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    ewf_dest = store.fetch(ewf_ref, dest_path)
    emb_dest = store.fetch(emb_ref, dest_path)
    return ewf_dest, emb_dest


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3 — SolverResult (fragment_result HDF5)
# ─────────────────────────────────────────────────────────────────────────────

def publish_solver_result_artifact(
    store: LocalArtifactStore,
    solver_result,                      # SolverResult
    h5_dest_path: str | Path,
    run_id: str,
    fragment_id: int,
    solver: str,
    hamiltonian_sha256: str,
    *,
    artifact_name: str | None = None,
    producer_version: str = "",
    extra: dict | None = None,
) -> tuple[FragmentResultReference, ArtifactReference]:
    """Persist a SolverResult to HDF5 and publish it to the artifact store.

    Steps:
    1. Write HDF5 via ``fragment_result_store.save_fragment_result``.
    2. Publish the resulting .h5 file to the artifact store.
    3. Return both the FragmentResultReference (for SolverResult loading) and
       the ArtifactReference (for cross-worker transport).

    The ArtifactReference ``payload_digest`` field is set to the canonical
    scientific payload digest computed during the HDF5 write, so downstream
    consumers can validate scientific identity without opening the HDF5.

    Parameters
    ----------
    store              : target LocalArtifactStore
    solver_result      : SolverResult object from the solver
    h5_dest_path       : where to write the local HDF5 file
    run_id, fragment_id, solver, hamiltonian_sha256 : scientific identity
    artifact_name      : storage name in the store (default: h5_dest_path.name)
    extra              : optional dict stored in FragmentResultReference.extra

    Returns
    -------
    (fragment_ref, artifact_ref) : FragmentResultReference + ArtifactReference
    """
    h5_path = Path(h5_dest_path)
    if h5_path.suffix.lower() != ".h5":
        raise StageArtifactError(
            f"h5_dest_path must end in .h5, got {h5_path}"
        )
    if artifact_name is None:
        artifact_name = h5_path.name

    # 1. Write HDF5 (fragment_result_store handles atomicity + checksums)
    try:
        frag_ref = save_fragment_result(
            solver_result,
            h5_path,
            fragment_id=fragment_id,
            run_id=run_id,
            solver=solver,
            hamiltonian_sha256=hamiltonian_sha256,
            extra=extra or {},
        )
    except FragmentStoreError as exc:
        raise StageArtifactError(
            f"Failed to write SolverResult HDF5 for fragment {fragment_id}: {exc}"
        ) from exc

    # 2. Publish to artifact store
    meta = {
        "fragment_id":         fragment_id,
        "run_id":              run_id,
        "solver":              solver,
        "hamiltonian_sha256":  hamiltonian_sha256,
        "payload_digest":      frag_ref.payload_digest,
        "frs_schema":          "frs/1",
    }
    try:
        art_ref = store.publish(
            h5_path,
            kind=KIND_FRAGMENT_H5,
            run_id=run_id,
            artifact_name=artifact_name,
            payload_digest=frag_ref.payload_digest,
            producer_version=producer_version,
            meta=meta,
        )
    except Exception as exc:
        raise StageArtifactError(
            f"Failed to publish fragment {fragment_id} HDF5 to artifact store: {exc}"
        ) from exc

    return frag_ref, art_ref


def retrieve_solver_result_artifact(
    store: LocalArtifactStore,
    art_ref: ArtifactReference,
    dest_dir: str | Path,
    *,
    expected_fragment_id: int | None = None,
    expected_run_id: str | None = None,
    expected_solver: str | None = None,
    expected_hamiltonian_sha256: str | None = None,
    expected_payload_digest: str | None = None,
):
    """Retrieve, verify, and load a SolverResult from a stored artifact.

    Performs:
    1. File-level integrity check (file_sha256).
    2. Optional scientific identity checks (fragment_id, run_id, solver,
       hamiltonian_sha256).
    3. Optional payload_digest validation.
    4. Returns the SolverResult.

    Parameters
    ----------
    store                       : LocalArtifactStore
    art_ref                     : ArtifactReference returned by publish_solver_result_artifact
    dest_dir                    : directory to retrieve the file into
    expected_*                  : optional identity assertions (fail closed if mismatch)
    expected_payload_digest     : if given, must match art_ref.payload_digest

    Returns
    -------
    SolverResult
    """
    if art_ref.kind != KIND_FRAGMENT_H5:
        raise StageArtifactError(
            f"Expected artifact of kind {KIND_FRAGMENT_H5!r}, got {art_ref.kind!r}"
        )

    # Validate payload_digest before fetching if the caller supplies an expected value
    if expected_payload_digest and art_ref.payload_digest:
        if art_ref.payload_digest != expected_payload_digest:
            raise StageArtifactError(
                f"Fragment {expected_fragment_id or '?'}: payload_digest mismatch "
                f"(expected {expected_payload_digest[:12]}…, "
                f"got {art_ref.payload_digest[:12]}…)"
            )

    # Retrieve and verify the .h5 file
    h5_path = store.fetch(art_ref, dest_dir)

    # Load via fragment_result_store (performs array-level integrity checks)
    try:
        result = load_fragment_result(
            h5_path,
            expected_fragment_id=expected_fragment_id,
            expected_run_id=expected_run_id,
            expected_solver=expected_solver,
            expected_hamiltonian_sha256=expected_hamiltonian_sha256,
            file_sha256=art_ref.file_sha256,
        )
    except FragmentStoreError as exc:
        raise StageArtifactError(
            f"Integrity check failed for fragment {expected_fragment_id or '?'} "
            f"artifact: {exc}"
        ) from exc

    return result
