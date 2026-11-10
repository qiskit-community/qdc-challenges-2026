"""
artifact_transport — Artifact transport across the organizer / Serverless boundary (Task 5.9C.1).

Mechanism
---------
The smallest operational mechanism is the **Qiskit Serverless File API** already used by the
project (``ServerlessClient.file_upload`` / ``file_download``; the gateway stores files in a
per-function folder and the project wraps every file as ``<name>.tar.gz`` because the gateway
rejects ``.h5``/``.pkl`` directly).  No object-storage service is introduced.

A compact, size-capped *inline* fallback (base64 inside the job result) exists for results,
because it is **not yet verified** that a Serverless *worker* may publish files to the file
folder that the organizer can later download.  The mode actually used is recorded in the
descriptor (``transport_mode``) so that operators can see which path was exercised.

Five distinct identities (never conflated)
------------------------------------------
1. **Scientific artifact identity** — ``payload_digest`` / ``hamiltonian_sha256`` / run, fragment,
   solver.  Lives *inside* the frs/1 file and in the descriptor.
2. **Local organizer artifact path** — ``LocalArtifactStore`` URI.  Organizer-only, never sent
   to a worker.
3. **Remote transport identifier** — ``remote_name`` (logical file name in the Serverless file
   folder).  Deterministic and run/fragment/attempt scoped.
4. **Worker-local temporary file** — exists only inside the worker process; it is never placed in
   a descriptor and is deleted by the worker.
5. **Final verified organizer-side artifact reference** — the ``FragmentResultReference`` returned by
   :func:`verify_and_import_fragment_result` after the file has been downloaded, size/SHA-256
   verified, schema/identity validated and published into the ``LocalArtifactStore``.

Honest status
-------------
Everything here is validated **offline** against mock/fake clients and against the installed
``qiskit-serverless`` 0.36 method signatures.  Real cross-worker transfer has **not** been
demonstrated.
"""

from __future__ import annotations

import base64
import binascii
import datetime
import hashlib
import math
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .artifact_store import (
    KIND_EMBEDDING,
    KIND_FRAGMENT_H5,
    ArtifactConflict,
    LocalArtifactStore,
)
from .fragment_result_store import (
    FragmentResultReference,
    FragmentStoreError,
    load_fragment_result,
)

__all__ = [
    "TRANSPORT_SCHEMA",
    "MODE_FILE_API",
    "MODE_INLINE",
    "DEFAULT_MAX_INLINE_BYTES",
    "TransportError",
    "TransportChecksumError",
    "TransportIdentityError",
    "TransportNotFoundError",
    "TransportDescriptor",
    "input_transport_name",
    "result_transport_name",
    "MockServerlessFileClient",
    "ServerlessFileApiClient",
    "build_worker_file_client",
    "ServerlessTransportAdapter",
    "validate_fragment_result_file",
    "verify_and_import_fragment_result",
]

TRANSPORT_SCHEMA = "trn/1"
MODE_FILE_API = "file_api"
MODE_INLINE = "inline_b64"
DEFAULT_MAX_INLINE_BYTES = 16 * 1024 * 1024

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,200}$")
_TOKEN = re.compile(r"^[A-Za-z0-9]{1,32}$")


class TransportError(RuntimeError):
    """Base error for artifact transport operations."""


class TransportChecksumError(TransportError):
    """File SHA-256 / size verification failed at a transport boundary."""


class TransportIdentityError(TransportError):
    """Scientific identity (run / fragment / solver / Hamiltonian / digest) mismatch."""


class TransportNotFoundError(TransportError):
    """The remote transport object does not exist (possibly not yet / never published)."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
        raise TransportError(f"run_id is not a safe transport component: {run_id!r}")
    return run_id


def _check_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name) or ".." in name:
        raise TransportError(f"unsafe transport name: {name!r}")
    return name


# ─────────────────────────────────────────────────────────────────────────────
#  Deterministic run-scoped transport naming
# ─────────────────────────────────────────────────────────────────────────────

def input_transport_name(run_id: str, file_role: str, file_sha256: str) -> str:
    """Remote name of an immutable input.  Run-scoped and content-addressed."""
    _check_run_id(run_id)
    if not _TOKEN.match(file_role.replace("_", "")):
        raise TransportError(f"unsafe file_role: {file_role!r}")
    if not _HEX64.match(file_sha256):
        raise TransportError("file_sha256 must be 64 lowercase hex characters")
    return f"{run_id}__in__{file_role}__{file_sha256[:16]}.h5"


def result_transport_name(run_id: str, fragment_id: int, solver: str, attempt: int = 1) -> str:
    """Remote name of one fragment result for one attempt.

    Distinct for every (run, fragment, solver, attempt), so repeated attempts and separate
    classical/hybrid runs can never collide.
    """
    _check_run_id(run_id)
    if int(fragment_id) < 0 or int(attempt) < 1:
        raise TransportError("fragment_id must be >= 0 and attempt >= 1")
    if not _TOKEN.match(str(solver)):
        raise TransportError(f"unsafe solver label: {solver!r}")
    return f"{run_id}__res__f{int(fragment_id):03d}__{str(solver).lower()}__a{int(attempt):02d}.h5"


# ─────────────────────────────────────────────────────────────────────────────
#  Transport descriptor (organizer <-> worker contract)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TransportDescriptor:
    """Verifiable, lightweight description of one transported file.

    Deliberately contains **no filesystem path** of any machine.
    """

    remote_name: str
    file_sha256: str
    file_size_bytes: int
    artifact_kind: str
    run_id: str
    fragment_id: Optional[int] = None
    solver: Optional[str] = None
    attempt: Optional[int] = None
    payload_digest: Optional[str] = None
    hamiltonian_sha256: Optional[str] = None
    transport_mode: str = MODE_FILE_API
    inline_payload_b64: Optional[str] = None
    created_iso8601: str = ""
    schema: str = TRANSPORT_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def without_payload(self) -> dict[str, Any]:
        """Dictionary form safe for logs / registry (drops the inline payload)."""
        d = asdict(self)
        d.pop("inline_payload_b64", None)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransportDescriptor":
        try:
            sha = str(data["file_sha256"])
            size = data["file_size_bytes"]
            if not _HEX64.match(sha):
                raise TransportError("descriptor file_sha256 is not 64 hex characters")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise TransportError("descriptor file_size_bytes must be a non-negative int")
            mode = str(data.get("transport_mode", MODE_FILE_API))
            if mode not in (MODE_FILE_API, MODE_INLINE):
                raise TransportError(f"unknown transport_mode {mode!r}")
            inline = data.get("inline_payload_b64")
            if (mode == MODE_INLINE) != (inline is not None):
                raise TransportError("inline_payload_b64 must be present iff transport_mode is inline")
            return cls(
                remote_name=_check_name(str(data["remote_name"])),
                file_sha256=sha,
                file_size_bytes=size,
                artifact_kind=str(data["artifact_kind"]),
                run_id=_check_run_id(str(data["run_id"])),
                fragment_id=int(data["fragment_id"]) if data.get("fragment_id") is not None else None,
                solver=str(data["solver"]) if data.get("solver") is not None else None,
                attempt=int(data["attempt"]) if data.get("attempt") is not None else None,
                payload_digest=str(data["payload_digest"]) if data.get("payload_digest") else None,
                hamiltonian_sha256=(str(data["hamiltonian_sha256"])
                                    if data.get("hamiltonian_sha256") else None),
                transport_mode=mode,
                inline_payload_b64=str(inline) if inline is not None else None,
                created_iso8601=str(data.get("created_iso8601", "")),
            )
        except KeyError as exc:
            raise TransportError(f"descriptor is missing key {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise TransportError(f"malformed descriptor: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
#  File clients.  Common protocol (logical names; no gateway details leak out):
#      file_upload(file_path, remote_name=None) -> remote_name
#      file_download(file_name, target_path)    -> target_path
#      file_delete(file_name) -> bool
#      files() -> list[str]
# ─────────────────────────────────────────────────────────────────────────────

class MockServerlessFileClient:
    """In-memory (or directory-backed) file client for offline tests.  Never touches a network."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self._dir = Path(storage_dir) if storage_dir is not None else None
        self._memory_files: dict[str, bytes] = {}

    def file_upload(self, file_path: str, remote_name: Optional[str] = None) -> str:
        src = Path(file_path)
        if not src.is_file():
            raise TransportError(f"Upload source file not found: {src}")
        name = _check_name(remote_name or src.name)
        content = src.read_bytes()
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
            (self._dir / name).write_bytes(content)
        else:
            self._memory_files[name] = content
        return name

    def file_download(self, file_name: str, target_path: str) -> str:
        dest = Path(target_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self._dir is not None:
            src = self._dir / file_name
            if not src.is_file():
                raise TransportNotFoundError(f"Remote file not found in mock store: {file_name}")
            shutil.copyfile(src, dest)
        else:
            if file_name not in self._memory_files:
                raise TransportNotFoundError(f"Remote file not found in mock memory store: {file_name}")
            dest.write_bytes(self._memory_files[file_name])
        return str(dest)

    def file_delete(self, file_name: str) -> bool:
        if self._dir is not None:
            target = self._dir / file_name
            if target.is_file():
                target.unlink()
                return True
            return False
        return self._memory_files.pop(file_name, None) is not None

    def files(self) -> list[str]:
        if self._dir is not None:
            return sorted(p.name for p in self._dir.iterdir() if p.is_file()) if self._dir.is_dir() else []
        return sorted(self._memory_files)


def _gateway_wraps_uploads() -> bool:
    """True when the project's ``_install_upload_patch`` is active (it tar.gz-wraps every upload)."""
    try:
        from qiskit_serverless.core import files as _f
        return getattr(_f.GatewayFilesClient.upload, "__name__", "") == "_targz_upload"
    except Exception:
        return False


class ServerlessFileApiClient:
    """Adapter from the real Qiskit Serverless File API to the logical-name client protocol.

    ``client`` is any object exposing the 0.36 ``ServerlessClient`` / ``QiskitServerless`` methods
    ``file_upload(file, function)``, ``file_download(file, function, target_name, download_location)``,
    ``file_delete(file, function)`` and ``files(function)``.  ``function`` is the function handle whose
    file folder is used (files are scoped per function by the gateway).

    The gateway only accepts archive-like names, so a logical file ``X.h5`` is stored as
    ``X.h5.tar.gz`` containing the single member ``X.h5`` — identical to the project's existing
    convention (see ``Challenge._install_upload_patch``).  When that patch is already active in this
    process it performs the wrapping itself; otherwise this adapter wraps explicitly, so uploads are
    never double-wrapped.
    """

    def __init__(self, client: Any, function: Any, *, gateway_wraps_uploads: Optional[bool] = None):
        self._client = client
        self._function = function
        self._wraps = gateway_wraps_uploads

    def _gateway_wraps(self) -> bool:
        return _gateway_wraps_uploads() if self._wraps is None else bool(self._wraps)

    def file_upload(self, file_path: str, remote_name: Optional[str] = None) -> str:
        src = Path(file_path)
        if not src.is_file():
            raise TransportError(f"Upload source file not found: {src}")
        name = _check_name(remote_name or src.name)
        with tempfile.TemporaryDirectory(prefix="qfm_up_") as td:
            staged = Path(td) / name
            shutil.copyfile(src, staged)
            if self._gateway_wraps():
                payload = staged
            else:
                payload = Path(td) / f"{name}.tar.gz"
                with tarfile.open(payload, "w:gz") as tar:
                    tar.add(staged, arcname=name)
            try:
                self._client.file_upload(str(payload), self._function)
            except Exception as exc:
                raise TransportError(f"File API upload of {name!r} failed: {type(exc).__name__}: {exc}") from exc
        return name

    def file_download(self, file_name: str, target_path: str) -> str:
        name = _check_name(file_name)
        tgz = f"{name}.tar.gz"
        dest = Path(target_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="qfm_dl_") as td:
            try:
                got = self._client.file_download(tgz, self._function, target_name=tgz, download_location=td)
            except Exception as exc:
                raise TransportNotFoundError(
                    f"File API download of {tgz!r} failed: {type(exc).__name__}: {exc}") from exc
            tgz_path = Path(td) / os.path.basename(got or tgz)
            if not tgz_path.is_file():
                raise TransportNotFoundError(f"File API did not produce {tgz!r}")
            try:
                with tarfile.open(tgz_path, "r:gz") as tar:
                    member = tar.getmember(name)
                    if not member.isfile():
                        raise TransportError(f"archive member {name!r} is not a regular file")
                    src = tar.extractfile(member)
                    if src is None:
                        raise TransportError(f"archive member {name!r} cannot be read")
                    tmp = dest.with_name(dest.name + ".part")
                    with src, tmp.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    os.replace(tmp, dest)
            except (tarfile.TarError, KeyError, EOFError, OSError) as exc:
                raise TransportError(f"could not extract {name!r} from {tgz!r}: {exc}") from exc
        return str(dest)

    def file_delete(self, file_name: str) -> bool:
        try:
            self._client.file_delete(f"{_check_name(file_name)}.tar.gz", self._function)
            return True
        except Exception:
            return False

    def files(self) -> list[str]:
        listing = self._client.files(self._function) or []
        return sorted(n[:-len(".tar.gz")] if n.endswith(".tar.gz") else n for n in listing)


def build_worker_file_client(function_title: str) -> Optional[ServerlessFileApiClient]:
    """Build a file client inside a Serverless worker from the gateway environment variables.

    Returns ``None`` when the gateway environment is not present (e.g. unit tests).  Credentials are
    read from the worker environment only and are never logged or returned.
    """
    try:
        from qiskit_serverless import ServerlessClient
        from qiskit_serverless.core.constants import (
            ENV_JOB_GATEWAY_HOST,
            ENV_JOB_GATEWAY_INSTANCE,
            ENV_JOB_GATEWAY_TOKEN,
        )
        from qiskit_serverless.core.function import QiskitFunction
    except ImportError:
        return None
    host = os.environ.get(ENV_JOB_GATEWAY_HOST)
    token = os.environ.get(ENV_JOB_GATEWAY_TOKEN)
    instance = os.environ.get(ENV_JOB_GATEWAY_INSTANCE)
    if not (host and token):
        return None
    client = ServerlessClient(host=host, token=token, instance=instance)
    return ServerlessFileApiClient(client, QiskitFunction(title=function_title))


# ─────────────────────────────────────────────────────────────────────────────
#  Transport adapter
# ─────────────────────────────────────────────────────────────────────────────

class ServerlessTransportAdapter:
    """Uploads, downloads and verifies files across the organizer–Serverless boundary."""

    def __init__(self, client: Any, *, max_inline_bytes: int = DEFAULT_MAX_INLINE_BYTES):
        if client is None:
            raise TransportError("a file client is required (use MockServerlessFileClient for offline tests)")
        self._client = client
        self._max_inline = int(max_inline_bytes)

    @property
    def client(self) -> Any:
        return self._client

    # ── inputs (organizer -> worker) ─────────────────────────────────────────

    def upload_input(
        self,
        local_path: str | Path,
        *,
        run_id: str,
        file_role: str = "ewf_dumpfile",
        expected_sha256: Optional[str] = None,
        skip_if_present: bool = True,
    ) -> TransportDescriptor:
        src = Path(local_path)
        if not src.is_file():
            raise TransportError(f"Local input artifact does not exist: {src}")
        sha = _sha256_file(src)
        if expected_sha256 is not None and sha != expected_sha256:
            raise TransportChecksumError(
                f"Local input checksum mismatch before upload: expected {expected_sha256[:12]}…, got {sha[:12]}…")
        remote = input_transport_name(run_id, file_role, sha)
        present = False
        if skip_if_present and hasattr(self._client, "files"):
            try:
                present = remote in set(self._client.files())
            except Exception:
                present = False
        if not present:
            self._client.file_upload(str(src), remote)
        return TransportDescriptor(
            remote_name=remote, file_sha256=sha, file_size_bytes=src.stat().st_size,
            artifact_kind=KIND_EMBEDDING, run_id=run_id, created_iso8601=_utc_iso())

    # ── results (worker -> organizer) ────────────────────────────────────────

    def publish_result(
        self,
        local_path: str | Path,
        *,
        run_id: str,
        fragment_id: int,
        solver: str,
        attempt: int,
        payload_digest: str,
        hamiltonian_sha256: str,
        mode: str = "auto",
    ) -> TransportDescriptor:
        """Publish a worker-local result file and return a path-free descriptor.

        ``mode``: ``"file_api"`` (no fallback), ``"inline"`` (base64 in the descriptor) or ``"auto"``
        (File API first, inline fallback if the upload fails and the file is within the size cap).
        """
        if mode not in ("auto", "file_api", "inline"):
            raise TransportError(f"unknown result transport mode {mode!r}")
        src = Path(local_path)
        if not src.is_file():
            raise TransportError("Result artifact to publish does not exist")
        sha = _sha256_file(src)
        size = src.stat().st_size
        remote = result_transport_name(run_id, fragment_id, solver, attempt)
        base = dict(
            remote_name=remote, file_sha256=sha, file_size_bytes=size, artifact_kind=KIND_FRAGMENT_H5,
            run_id=run_id, fragment_id=int(fragment_id), solver=solver, attempt=int(attempt),
            payload_digest=payload_digest or None, hamiltonian_sha256=hamiltonian_sha256 or None,
            created_iso8601=_utc_iso())

        file_api_error: Optional[str] = None
        if mode in ("auto", "file_api"):
            try:
                self._client.file_upload(str(src), remote)
                return TransportDescriptor(transport_mode=MODE_FILE_API, **base)
            except Exception as exc:
                file_api_error = f"{type(exc).__name__}: {exc}"
                if mode == "file_api":
                    raise TransportError(f"result upload failed: {file_api_error}") from exc

        if size > self._max_inline:
            raise TransportError(
                f"result is {size} bytes, above the inline cap {self._max_inline}"
                + (f"; File API upload failed first ({file_api_error})" if file_api_error else ""))
        payload = base64.b64encode(src.read_bytes()).decode("ascii")
        return TransportDescriptor(transport_mode=MODE_INLINE, inline_payload_b64=payload, **base)

    # ── retrieval + verification ─────────────────────────────────────────────

    def download_file(
        self,
        remote_name: str,
        destination_path: str | Path,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> Path:
        """Download ``remote_name`` and verify BOTH size and SHA-256 before exposing the file.

        The data is written to ``<dest>.part`` and renamed only after verification, so an unverified
        file is never present at ``destination_path``.
        """
        if not expected_sha256 or expected_size is None:
            raise TransportError("expected_sha256 and expected_size are mandatory for downloads")
        dest = Path(destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".dl")
        part.unlink(missing_ok=True)
        try:
            self._client.file_download(_check_name(remote_name), str(part))
            return self._verify_and_place(part, dest, expected_sha256, expected_size)
        finally:
            part.unlink(missing_ok=True)

    def fetch_descriptor(self, desc: TransportDescriptor, destination_path: str | Path) -> Path:
        """Materialise the file described by ``desc`` (File API or inline) with full verification."""
        dest = Path(destination_path)
        if desc.transport_mode == MODE_INLINE:
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".dl")
            try:
                try:
                    part.write_bytes(base64.b64decode(desc.inline_payload_b64 or "", validate=True))
                except (binascii.Error, ValueError) as exc:
                    raise TransportChecksumError(f"inline payload is not valid base64: {exc}") from exc
                return self._verify_and_place(part, dest, desc.file_sha256, desc.file_size_bytes)
            finally:
                part.unlink(missing_ok=True)
        return self.download_file(desc.remote_name, dest,
                                  expected_sha256=desc.file_sha256, expected_size=desc.file_size_bytes)

    @staticmethod
    def _verify_and_place(part: Path, dest: Path, expected_sha: str, expected_size: int) -> Path:
        if not part.is_file():
            raise TransportNotFoundError("download produced no file")
        size = part.stat().st_size
        if size != int(expected_size):
            raise TransportChecksumError(f"size mismatch: expected {expected_size}, got {size}")
        sha = _sha256_file(part)
        if sha != expected_sha:
            raise TransportChecksumError(
                f"SHA-256 mismatch: expected {expected_sha[:12]}…, got {sha[:12]}…")
        os.replace(part, dest)
        return dest


# ─────────────────────────────────────────────────────────────────────────────
#  Organizer-side result validation and import
# ─────────────────────────────────────────────────────────────────────────────

def validate_fragment_result_file(
    path: str | Path,
    *,
    run_id: str,
    fragment_id: int,
    expected_solver: str,
    expected_hamiltonian_sha256: str,
    expected_file_sha256: Optional[str] = None,
    expected_payload_digest: Optional[str] = None,
) -> str:
    """Validate a complete frs/1 result; return its stored ``payload_digest``.

    Checks: file SHA-256 (optional), frs/1 schema, run/fragment/solver/Hamiltonian identity, stored
    ``payload_digest`` equals the one announced by the worker (optional), energy finite, RDMs present,
    finite, correctly shaped (1-RDM ``n×n``; 2-RDM ``n×n×n×n``).
    """
    path = Path(path)
    if not path.is_file():
        raise TransportError("Result file to validate does not exist")
    try:
        import h5py
        with h5py.File(path, "r") as hf:
            stored_digest = str(hf.attrs.get("payload_digest", ""))
        loaded = load_fragment_result(
            path,
            expected_fragment_id=fragment_id,
            expected_run_id=run_id,
            expected_solver=expected_solver,
            expected_hamiltonian_sha256=expected_hamiltonian_sha256,
            file_sha256=expected_file_sha256,
        )
    except FragmentStoreError as exc:
        msg = str(exc)
        cls = TransportChecksumError if "checksum" in msg else TransportIdentityError
        raise cls(f"Result failed frs/1 validation: {msg}") from exc
    except OSError as exc:
        raise TransportError(f"Result file is unreadable: {exc}") from exc

    if not stored_digest:
        raise TransportIdentityError("Result carries no payload_digest")
    if expected_payload_digest and stored_digest != expected_payload_digest:
        raise TransportIdentityError("Result payload_digest differs from the digest announced by the worker")
    if not math.isfinite(float(loaded.energy)):
        raise TransportError(f"Result energy is non-finite: {loaded.energy}")
    if loaded.rdm1 is None or loaded.rdm2 is None:
        raise TransportError("Result is missing required RDMs")
    import numpy as np
    r1, r2 = np.asarray(loaded.rdm1), np.asarray(loaded.rdm2)
    if r1.ndim != 2 or r1.shape[0] != r1.shape[1] or r1.size == 0:
        raise TransportError(f"1-RDM must be a non-empty square matrix, got shape {r1.shape}")
    n = r1.shape[0]
    if r2.shape != (n, n, n, n):
        raise TransportError(f"2-RDM shape {r2.shape} is incompatible with 1-RDM dimension {n}")
    if not (np.all(np.isfinite(r1.real)) and np.all(np.isfinite(r2.real))):
        raise TransportError("Result RDMs contain non-finite values")
    return stored_digest


def verify_and_import_fragment_result(
    store: LocalArtifactStore,
    downloaded_h5_path: str | Path,
    *,
    run_id: str,
    fragment_id: int,
    expected_solver: str,
    expected_hamiltonian_sha256: str,
    extra: Optional[dict[str, Any]] = None,
    expected_file_sha256: Optional[str] = None,
    expected_payload_digest: Optional[str] = None,
) -> FragmentResultReference:
    """Validate a downloaded result and import it into the authoritative ``LocalArtifactStore``.

    The returned reference points into the store (organizer-side, durable) — never into the download
    staging area.  Re-importing a scientifically identical result (same ``payload_digest``; e.g. after a
    crash between import and registry completion, or on a retry) is idempotent and returns the existing
    stored artifact; a different result for the same slot raises.
    """
    path = Path(downloaded_h5_path)
    digest = validate_fragment_result_file(
        path, run_id=run_id, fragment_id=fragment_id, expected_solver=expected_solver,
        expected_hamiltonian_sha256=expected_hamiltonian_sha256,
        expected_file_sha256=expected_file_sha256, expected_payload_digest=expected_payload_digest)

    name = f"frag_{int(fragment_id)}_{expected_solver.lower()}.h5"
    meta = {"fragment_id": int(fragment_id), "solver": expected_solver,
            "hamiltonian_sha256": expected_hamiltonian_sha256, "payload_digest": digest,
            "extra": extra or {}}
    try:
        ref = store.publish(path, kind=KIND_FRAGMENT_H5, run_id=run_id, artifact_name=name,
                            payload_digest=digest, meta=meta)
    except ArtifactConflict:
        existing = store._artifact_path(run_id, KIND_FRAGMENT_H5, name)
        existing_digest = validate_fragment_result_file(
            existing, run_id=run_id, fragment_id=fragment_id, expected_solver=expected_solver,
            expected_hamiltonian_sha256=expected_hamiltonian_sha256)
        if existing_digest != digest:
            raise TransportIdentityError(
                f"{name} already exists in the store with a different scientific payload") from None
        ref = store.publish(existing, kind=KIND_FRAGMENT_H5, run_id=run_id, artifact_name=name,
                            payload_digest=digest, meta=meta)

    return FragmentResultReference(
        path=Path(ref.uri), file_sha256=ref.file_sha256, payload_digest=digest,
        fragment_id=int(fragment_id), run_id=run_id, solver=expected_solver,
        hamiltonian_sha256=expected_hamiltonian_sha256, extra=extra or {})
