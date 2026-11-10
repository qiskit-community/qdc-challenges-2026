"""
artifact_store — Durable, immutable artifact storage for Proteasome Challenge.

Provides three distinct identity concepts:

  Scientific payload identity (``payload_digest``)
      A canonical SHA-256 digest of the scientifically meaningful content of
      an artifact, computed from stable logical fields.  Stable across
      equivalent results even when file bytes differ (e.g. different write
      timestamps).  Used for cache-hit validation and cross-worker identity.

  Physical artifact identity (``file_sha256``)
      A SHA-256 digest of the exact bytes of the finalized file on disk.
      Detects corruption, file replacement, or incomplete writes.

  Scientific input identity (``hamiltonian_sha256``, ``embedding_digest``, etc.)
      Versioned descriptions of the computational inputs required to establish
      scientific compatibility.  A cache hit requires matching input identity.

Public API
----------
  ``ArtifactReference``     — small serialisable reference (never contains bulk data)
  ``ArtifactStoreError``    — base exception class
  ``LocalArtifactStore``    — filesystem-backed implementation for offline testing

  ``LocalArtifactStore.publish(local_path, kind, run_id, *, meta) -> ArtifactReference``
      Publish a local file as an immutable artifact.  Atomic, collision-safe.
  ``LocalArtifactStore.fetch(ref, dest) -> Path``
      Retrieve and verify an artifact to *dest*.
  ``LocalArtifactStore.verify(ref) -> VerificationResult``
      Verify a stored artifact without copying.
  ``LocalArtifactStore.exists(ref) -> bool``
      Check whether a reference resolves to a valid, verifiable artifact.

Storage layout
--------------
  ``<root>/runs/<run_id>/<kind>/<artifact_name>``

Protected paths
---------------
  The global QPU sampling ledger at
  ``<project_root>/checkpoints/.qpu_sampling_record.json`` is NEVER treated as
  a run artifact and is NEVER modified by this module.

Concurrency
-----------
  Publication uses an exclusive ``O_CREAT | O_EXCL`` strategy: only the first
  writer succeeds; subsequent identical writers are idempotent; conflicting
  writers raise ``ArtifactConflict``.  ``os.replace`` is intentionally NOT
  used alone because it overwrites silently on POSIX.

Cross-worker access
-------------------
  A ``LocalArtifactStore`` writes to a local filesystem path.  It is suitable
  for offline development, unit testing, and single-machine orchestration.  It
  is NOT accessible across separate Serverless workers unless the workers share
  a mounted network filesystem.  Remote storage requirements are documented in
  ``RemoteStorageCapabilityMatrix`` below; a remote adapter is not implemented
  until the Serverless File API durability guarantees are independently verified.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Schema ────────────────────────────────────────────────────────────────────
ARTIFACT_STORE_SCHEMA = "ast/1"

# ── Well-known artifact kinds ─────────────────────────────────────────────────
KIND_MEANFIELD    = "meanfield"      # mf_data.pkl
KIND_EMBEDDING    = "embedding"      # embedding_data.pkl + ewf_dumpfile.h5
KIND_FRAGMENT_H5  = "fragment_result"  # fragment SolverResult HDF5 (frs/1)
KIND_SOLVER_PKL   = "solver_results"   # solver_results.pkl (legacy transport)
KIND_SUMMARY_JSON = "summary_json"     # reconstruction summary.json

_VALID_KINDS = frozenset([
    KIND_MEANFIELD, KIND_EMBEDDING, KIND_FRAGMENT_H5,
    KIND_SOLVER_PKL, KIND_SUMMARY_JSON,
])

# ── Path-traversal guard ──────────────────────────────────────────────────────
import re as _re
_SAFE_COMPONENT = _re.compile(r"^[\w.\-]{1,128}$")


# ─────────────────────────────────────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class ArtifactStoreError(RuntimeError):
    """Base class for all artifact-store errors."""


class ArtifactConflict(ArtifactStoreError):
    """A conflicting artifact already exists at the given logical address."""


class ArtifactNotFound(ArtifactStoreError):
    """No artifact exists for the given reference."""


class ArtifactCorrupt(ArtifactStoreError):
    """The on-disk artifact does not match its expected checksum."""


class ArtifactIdentityMismatch(ArtifactStoreError):
    """The artifact's stored scientific identity does not match the expected values."""


class ArtifactSchemaError(ArtifactStoreError):
    """The artifact reference carries an unsupported or malformed schema."""


# ─────────────────────────────────────────────────────────────────────────────
#  ArtifactReference
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ArtifactReference:
    """Small, serialisable reference to an immutable artifact.

    Never contains HDF5 files, pickles, measurement arrays or density matrices.
    Safe to embed in JSON task records and orchestration messages.

    Fields
    ------
    schema_version   : always ``"ast/1"``
    kind             : artifact type — one of the KIND_* constants
    run_id           : run-scoped namespace (e.g. ``"ixazomib_ligand_sto-3g"``)
    artifact_name    : unique artifact filename within (run_id, kind)
    uri              : absolute or relative path to the artifact file
    file_sha256      : SHA-256 of the exact file bytes (physical identity)
    payload_digest   : SHA-256 of canonical scientific payload, when applicable
                       (empty string ``""`` for opaque artifacts like pickles)
    size_bytes       : file size in bytes
    created_iso8601  : UTC creation timestamp
    producer_version : optional provenance / code version string
    meta             : optional JSON-serialisable dict of scientific identity
                       fields (e.g. hamiltonian_sha256, fragment_id, run_id)
    """

    schema_version:   str
    kind:             str
    run_id:           str
    artifact_name:    str
    uri:              str          # absolute path on the local FS (or object-store URI)
    file_sha256:      str          # 64-char hex
    payload_digest:   str          # 64-char hex or ""
    size_bytes:       int
    created_iso8601:  str
    producer_version: str = ""
    meta:             dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "schema_version":   self.schema_version,
            "kind":             self.kind,
            "run_id":           self.run_id,
            "artifact_name":    self.artifact_name,
            "uri":              self.uri,
            "file_sha256":      self.file_sha256,
            "payload_digest":   self.payload_digest,
            "size_bytes":       self.size_bytes,
            "created_iso8601":  self.created_iso8601,
            "producer_version": self.producer_version,
            "meta":             dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArtifactReference":
        sv = d.get("schema_version", "")
        if sv != ARTIFACT_STORE_SCHEMA:
            raise ArtifactSchemaError(
                f"Unsupported ArtifactReference schema_version {sv!r}; "
                f"expected {ARTIFACT_STORE_SCHEMA!r}"
            )
        required = ("kind", "run_id", "artifact_name", "uri",
                    "file_sha256", "payload_digest", "size_bytes", "created_iso8601")
        missing = [k for k in required if k not in d]
        if missing:
            raise ArtifactSchemaError(
                f"ArtifactReference is missing required fields: {missing}"
            )
        return cls(
            schema_version=sv,
            kind=d["kind"],
            run_id=d["run_id"],
            artifact_name=d["artifact_name"],
            uri=d["uri"],
            file_sha256=d["file_sha256"],
            payload_digest=d["payload_digest"],
            size_bytes=int(d["size_bytes"]),
            created_iso8601=d["created_iso8601"],
            producer_version=d.get("producer_version", ""),
            meta=dict(d.get("meta") or {}),
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "ArtifactReference":
        return cls.from_dict(json.loads(s))


# ─────────────────────────────────────────────────────────────────────────────
#  VerificationResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """Outcome of a single artifact verification pass."""
    ok:             bool
    path:           str
    expected_sha:   str
    actual_sha:     str
    size_bytes:     int
    reason:         str = ""

    def __bool__(self) -> bool:
        return self.ok


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _utc_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _safe_component(value: str, label: str) -> str:
    """Validate a path component is safe (no traversal, no special chars)."""
    if not isinstance(value, str) or not value.strip():
        raise ArtifactStoreError(f"{label} must be a non-empty string, got {value!r}")
    v = value.strip()
    if not _SAFE_COMPONENT.match(v):
        raise ArtifactStoreError(
            f"{label} {v!r} contains characters not allowed in artifact paths "
            "(allowed: alphanumeric, dot, hyphen, underscore; max 128 chars)"
        )
    return v


def _validate_kind(kind: str) -> str:
    if kind not in _VALID_KINDS:
        raise ArtifactStoreError(
            f"Unknown artifact kind {kind!r}; valid kinds: {sorted(_VALID_KINDS)}"
        )
    return kind


# ─────────────────────────────────────────────────────────────────────────────
#  LocalArtifactStore
# ─────────────────────────────────────────────────────────────────────────────

class LocalArtifactStore:
    """Filesystem-backed artifact store for offline development and testing.

    Storage layout::

        <root>/runs/<run_id>/<kind>/<artifact_name>

    Publication semantics
    ---------------------
    ``publish`` copies ``local_path`` into the store using an exclusive atomic
    strategy:
      1. Write to a temp file in the same directory.
      2. Attempt ``os.link(tmp, dest)`` — succeeds only if *dest* does not exist
         (O_CREAT|O_EXCL semantics on POSIX filesystems).
      3. If *dest* already exists:
         a. If its content matches the incoming file → idempotent (return existing ref).
         b. Otherwise → raise ``ArtifactConflict``.
      4. Delete the temp file.

    Cross-worker access
    -------------------
    This store reads and writes a local filesystem path.  It is NOT accessible
    across distinct Serverless workers unless the workers share a network mount.
    Use this store for offline testing and single-machine orchestration only.
    See ``RemoteStorageCapabilityMatrix`` in this module for the verified/unverified
    status of cross-worker storage options.

    Protected paths
    ---------------
    The persistent QPU ledger at
    ``<project_root>/checkpoints/.qpu_sampling_record.json``
    is never written, moved, or deleted by this store.
    """

    def __init__(self, root: str | Path):
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _artifact_path(self, run_id: str, kind: str, artifact_name: str) -> Path:
        """Compute the absolute storage path for an artifact."""
        run_id        = _safe_component(run_id, "run_id")
        kind          = _validate_kind(kind)
        artifact_name = _safe_component(artifact_name, "artifact_name")
        p = self._root / "runs" / run_id / kind / artifact_name
        # Final traversal guard: resolved path must be under root
        try:
            p.resolve().relative_to(self._root)
        except ValueError:
            raise ArtifactStoreError(
                f"Computed artifact path {p} escapes store root {self._root}"
            )
        return p

    # ── publish ────────────────────────────────────────────────────────────────

    def publish(
        self,
        local_path: str | Path,
        kind: str,
        run_id: str,
        *,
        artifact_name: str | None = None,
        payload_digest: str = "",
        producer_version: str = "",
        meta: dict | None = None,
    ) -> ArtifactReference:
        """Publish *local_path* as an immutable artifact.

        Parameters
        ----------
        local_path       : source file to publish
        kind             : artifact kind (one of KIND_* constants)
        run_id           : run scoping namespace
        artifact_name    : storage filename; defaults to ``local_path.name``
        payload_digest   : optional canonical scientific payload digest
        producer_version : optional provenance string
        meta             : optional JSON-serialisable scientific identity fields

        Returns
        -------
        ArtifactReference

        Raises
        ------
        ArtifactConflict
            If an artifact already exists at the logical address with
            different content.
        ArtifactStoreError
            For I/O failures or validation errors.
        """
        src = Path(local_path)
        if not src.is_file():
            raise ArtifactStoreError(f"Source file does not exist: {src}")

        _validate_kind(kind)
        if artifact_name is None:
            artifact_name = src.name
        dest = self._artifact_path(run_id, kind, artifact_name)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Compute source checksum BEFORE any copy
        src_sha = _sha256_file(src)
        src_size = src.stat().st_size

        if dest.exists():
            # Already published: check for identity or conflict
            existing_sha = _sha256_file(dest)
            if existing_sha == src_sha:
                # Idempotent retry — return reference for existing artifact
                return self._make_ref(
                    dest, kind, run_id, artifact_name,
                    file_sha256=existing_sha,
                    payload_digest=payload_digest,
                    producer_version=producer_version,
                    meta=meta or {},
                )
            raise ArtifactConflict(
                f"Artifact already exists at {dest} with different content "
                f"(existing={existing_sha[:12]}…, incoming={src_sha[:12]}…). "
                "Immutable artifacts cannot be overwritten."
            )

        # Exclusive atomic publication
        tmp_fd, tmp_str = tempfile.mkstemp(
            suffix=".part",
            prefix=f".ast_{os.getpid()}_",
            dir=dest.parent,
        )
        tmp = Path(tmp_str)
        try:
            os.close(tmp_fd)
            shutil.copy2(str(src), str(tmp))
            # Verify the copy is intact
            copied_sha = _sha256_file(tmp)
            if copied_sha != src_sha:
                raise ArtifactStoreError(
                    f"Copy verification failed for {src.name}: "
                    f"expected {src_sha[:12]}…, got {copied_sha[:12]}…"
                )
            # Exclusive link — succeeds only if dest does not exist
            try:
                os.link(str(tmp), str(dest))
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    # Race: another writer published concurrently
                    existing_sha = _sha256_file(dest)
                    if existing_sha == src_sha:
                        # Same content — idempotent
                        pass
                    else:
                        raise ArtifactConflict(
                            f"Concurrent writer published a conflicting artifact at {dest}. "
                            f"Existing content ({existing_sha[:12]}…) differs from "
                            f"incoming ({src_sha[:12]}…)."
                        ) from exc
                else:
                    raise ArtifactStoreError(
                        f"Could not publish artifact to {dest}: {exc}"
                    ) from exc
        finally:
            tmp.unlink(missing_ok=True)

        return self._make_ref(
            dest, kind, run_id, artifact_name,
            file_sha256=src_sha,
            payload_digest=payload_digest,
            producer_version=producer_version,
            meta=meta or {},
        )

    # ── fetch ──────────────────────────────────────────────────────────────────

    def fetch(self, ref: ArtifactReference, dest: str | Path) -> Path:
        """Retrieve and verify an artifact to *dest*.

        Parameters
        ----------
        ref  : ArtifactReference returned by ``publish``
        dest : destination file path (or directory; filename from ref is used)

        Returns
        -------
        Path to the retrieved file.

        Raises
        ------
        ArtifactNotFound  — artifact file does not exist
        ArtifactCorrupt   — file exists but checksum does not match reference
        """
        dest = Path(dest)
        if dest.is_dir():
            dest = dest / ref.artifact_name

        src = Path(ref.uri)
        if not src.is_file():
            raise ArtifactNotFound(f"Artifact not found at {src}")

        # Verify source before copy
        actual_sha = _sha256_file(src)
        if actual_sha != ref.file_sha256:
            raise ArtifactCorrupt(
                f"Artifact {src.name} is corrupt: "
                f"expected {ref.file_sha256[:12]}…, got {actual_sha[:12]}…"
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        # Atomic copy to temp then rename
        tmp_fd, tmp_str = tempfile.mkstemp(
            suffix=".fetch",
            prefix=f".ast_fetch_{os.getpid()}_",
            dir=dest.parent,
        )
        tmp = Path(tmp_str)
        try:
            os.close(tmp_fd)
            shutil.copy2(str(src), str(tmp))
            copied_sha = _sha256_file(tmp)
            if copied_sha != ref.file_sha256:
                raise ArtifactCorrupt(
                    f"Copy of {src.name} is corrupt after transfer: "
                    f"expected {ref.file_sha256[:12]}…, got {copied_sha[:12]}…"
                )
            os.replace(str(tmp), str(dest))
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        return dest

    # ── verify ─────────────────────────────────────────────────────────────────

    def verify(self, ref: ArtifactReference) -> VerificationResult:
        """Verify a stored artifact without copying it.

        Returns
        -------
        VerificationResult  — ``.ok`` is True iff the artifact exists and matches.
        """
        p = Path(ref.uri)
        if not p.is_file():
            return VerificationResult(
                ok=False, path=str(p),
                expected_sha=ref.file_sha256, actual_sha="",
                size_bytes=0, reason="file not found",
            )
        actual = _sha256_file(p)
        size   = p.stat().st_size
        if actual != ref.file_sha256:
            return VerificationResult(
                ok=False, path=str(p),
                expected_sha=ref.file_sha256, actual_sha=actual,
                size_bytes=size, reason="checksum mismatch",
            )
        return VerificationResult(
            ok=True, path=str(p),
            expected_sha=ref.file_sha256, actual_sha=actual,
            size_bytes=size,
        )

    # ── exists ─────────────────────────────────────────────────────────────────

    def exists(self, ref: ArtifactReference) -> bool:
        """Return True iff the artifact exists and its checksum matches."""
        return bool(self.verify(ref))

    # ── internal ───────────────────────────────────────────────────────────────

    def _make_ref(
        self,
        path: Path,
        kind: str,
        run_id: str,
        artifact_name: str,
        *,
        file_sha256: str,
        payload_digest: str,
        producer_version: str,
        meta: dict,
    ) -> ArtifactReference:
        return ArtifactReference(
            schema_version=ARTIFACT_STORE_SCHEMA,
            kind=kind,
            run_id=run_id,
            artifact_name=artifact_name,
            uri=str(path),
            file_sha256=file_sha256,
            payload_digest=payload_digest,
            size_bytes=path.stat().st_size,
            created_iso8601=_utc_iso(),
            producer_version=producer_version,
            meta=dict(meta),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Remote storage capability matrix
# ─────────────────────────────────────────────────────────────────────────────

REMOTE_STORAGE_CAPABILITY_MATRIX = """
Remote Storage Capability Matrix — Task 5.2
============================================
Investigated: IBM Serverless File API (Qiskit Serverless / IBM Quantum Platform)

Source evidence
---------------
- serverless_bundle/proteasome_entrypoint_stage2_fragments.py
  (artifact upload: save_result() base64 payload in JSON result blob)
- serverless_bundle/proteasome_entrypoint_stage3_solve_reconstruct.py
  (artifact download: get_files() / extract tar.gz from File API)
- Qiskit Serverless documentation (https://qiskit-extensions.github.io/qiskit-serverless/)

VERIFIED (code evidence in codebase)
-------------------------------------
1. Upload mechanism: artifacts are base64-encoded and embedded in the
   save_result() JSON payload.  The result blob (including the base64 payload)
   is retrievable via the Serverless result API.

2. Download mechanism: entrypoint_stage3 downloads artifacts as
   `{run_name}__{filename}.tar.gz` using get_files() → extract to /tmp.

3. File size: base64(gzip(ewf_dumpfile.h5)) + base64(gzip(embedding_data.pkl))
   are currently embedded in the JSON result.  The codebase includes no
   explicit size-limit enforcement; the practical limit is the Serverless
   result-blob size limit (not publicly documented in code).

4. Authentication: governed by IBM Quantum credentials (API token).
   Scope: per-user account.

5. Artifact availability after originating worker exits: YES — results are
   stored by the Serverless service and retrievable by subsequent job submissions
   from the same account.

6. Compatibility across s/l/g resource families: the upload/download API
   (save_result/get_files) is the same across all families.

UNVERIFIED (not confirmed by code or documentation in this codebase)
----------------------------------------------------------------------
1. Artifact lifetime and expiration: no documented TTL found in the codebase
   or Qiskit Serverless SDK.  Expiration policy is not confirmed.

2. Immutable / conditional-write support: no O_CREAT|O_EXCL equivalent found
   in the Serverless File API.  Concurrent writers can overwrite each other.

3. Checksum guarantees: no built-in integrity verification found in the
   Serverless result/file API.  Checksums must be verified application-level
   (as done in this module).

4. Retry semantics: the Serverless job retry behavior (if a job fails mid-upload)
   is not documented in the codebase.

5. Whether artifacts can be accessed after notebook restart from a different
   kernel: presumed yes (server-side storage) but not verified by test.

6. Whether the File API supports object-store semantics (e.g. listing) or is
   limited to named-file access: not confirmed.

CONCLUSION
----------
The current implementation (base64 in JSON result blobs) is confirmed to work
for Stage 1→2→3 transport.  It does NOT provide:
- Immutable collision-safe publication.
- Durability guarantees beyond the Serverless result TTL.
- Artifact-level integrity verification.

A production-grade remote adapter requires:
a. Confirmation of result blob TTL (minimum: run duration + post-processing window).
b. An idempotent upload primitive (PUT-if-not-exists or content-addressed).
c. Application-level checksum verification on download (already designed here).

Until (a) and (b) are verified, LocalArtifactStore is the authoritative
adapter for offline development.  The remote adapter contract is specified
by the ArtifactReference schema and LocalArtifactStore interface above.
"""
