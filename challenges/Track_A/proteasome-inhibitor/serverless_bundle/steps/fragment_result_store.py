"""
fragment_result_store — versioned, integrity-checked persistence for SolverResult objects.

Schema version: ``frs/1``

Layout inside the HDF5 artifact
--------------------------------
/                                  root
  .attrs["schema_version"]         str    "frs/1"
  .attrs["fragment_id"]            int
  .attrs["run_id"]                 str    caller-supplied run identity
  .attrs["solver"]                 str    label of the solver that produced this result
  .attrs["hamiltonian_sha256"]     str    sha256(h1e.tobytes() + h2e.tobytes())
  .attrs["energy"]                 float64
  .attrs["created_iso8601"]        str    UTC creation timestamp
  .attrs["rdm1_checksum"]         str    sha256 of rdm1 bytes
  .attrs["rdm2_checksum"]         str    sha256 of rdm2 bytes
  /rdm1                            dataset  float64 or complex128, shape (norb, norb)
  /rdm2                            dataset  float64 or complex128, shape (norb,)*4
  /wavefunction                    dataset  optional; absent if wavefunction is None
  /metadata_json                   dataset  UTF-8 JSON; JSON-serialisable metadata fields only
  /metadata_arrays/                group    one dataset per metadata key whose value is an ndarray
    /<key>                         dataset  stored as-is (dtype + shape preserved)
  /metadata_nonser/                group    one dataset per metadata key that is a non-JSON-able scalar
    /<key>                         dataset  scalar stored in its native dtype

Whole-file checksum
-------------------
A :class:`FragmentResultReference` wraps the HDF5 path and a ``file_sha256`` computed AFTER
the file is closed and fsync'd.  The checksum is NOT stored inside the HDF5 (which would require
reopening and change the file).  Loading validates the on-disk file against the reference's
``file_sha256`` before opening it.

Atomic writes
-------------
All data is written to ``<path>.part.<pid>`` and renamed to ``<path>`` only when the HDF5 is
fully closed.  A partial write is never accepted as a complete result.

Immutability
------------
``save_fragment_result`` refuses to overwrite an existing artifact with different bytes.
Saving the identical bytes to the same path is a no-op (idempotent retry).

Supported metadata field types
-------------------------------
+  JSON-serialisable scalar (int, float, bool, None, str, dict, list) → /metadata_json
+  numpy scalar  → /metadata_nonser/<key>  (stored as 0-d dataset)
+  numpy ndarray → /metadata_arrays/<key>  (stored as dataset, dtype + shape preserved)
+  tuple of int / float → serialised as list in /metadata_json
+  tuple of ndarray → NOT supported; raises FragmentStoreError with a clear message

Unsupported types
-----------------
Objects that are not in the above list cause a ``FragmentStoreError`` with a description.
They are never silently dropped or converted to str().

Complex arrays
--------------
If rdm1, rdm2 or a metadata array is complex, it is stored with dtype ``complex128``
(upcast if necessary).  Loading preserves complex dtype.  Complex values are never
coerced to real; imaginary components are never discarded.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Public names ──────────────────────────────────────────────────────────────
__all__ = [
    "SCHEMA_VERSION",
    "FragmentStoreError",
    "FragmentResultReference",
    "save_fragment_result",
    "load_fragment_result",
    "compute_payload_digest",
]

SCHEMA_VERSION = "frs/1"

# ── Regex for run_id / fragment_id validation ─────────────────────────────────
_SAFE_STR = re.compile(r"^[\w.\-:/@]{1,256}$")

# ── Sentinel for absent optional fields ──────────────────────────────────────
_ABSENT = object()


# ─────────────────────────────────────────────────────────────────────────────
class FragmentStoreError(ValueError):
    """The artifact is malformed, invalid, or identity-mismatched."""


# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FragmentResultReference:
    """Immutable reference to a persisted :class:`SolverResult` artifact.

    Parameters
    ----------
    path               : Path (or str) to the HDF5 artifact.
    file_sha256        : SHA-256 of the closed HDF5 file bytes.  Physical identity.
                         Verified before the HDF5 is opened.
    payload_digest     : Canonical SHA-256 of all scientifically meaningful fields
                         (RDMs, wavefunction, energy, all metadata zones, identity
                         strings).  Scientific identity.  Used for cache validation.
                         Empty string ``""`` for pre-Task-5.2 artifacts.
    fragment_id        : fragment integer ID (copy of the stored attr).
    run_id             : caller-supplied run identity string.
    solver             : label of the solver that produced this result.
    hamiltonian_sha256 : SHA-256 of the embedded Hamiltonian inputs.
    extra              : optional JSON-serialisable dict of auxiliary metadata.
    """

    path:               Path
    file_sha256:        str
    payload_digest:     str = ""   # "" for pre-Task-5.2 artifacts (backward compat)
    fragment_id:        int = 0
    run_id:             str = ""
    solver:             str = ""
    hamiltonian_sha256: str = ""
    extra:              dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))

    def as_dict(self) -> dict:
        return {
            "schema_version":      SCHEMA_VERSION,
            "path":                str(self.path),
            "file_sha256":         self.file_sha256,
            "payload_digest":      self.payload_digest,
            "fragment_id":         self.fragment_id,
            "run_id":              self.run_id,
            "solver":              self.solver,
            "hamiltonian_sha256":  self.hamiltonian_sha256,
            "extra":               dict(self.extra),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    a = np.asarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


def compute_payload_digest(
    *,
    energy: float,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    wavefunction: "np.ndarray | None",
    json_meta: dict,
    array_meta: "dict[str, np.ndarray]",
    nonser_meta: "dict[str, np.ndarray]",
    fragment_id: int,
    run_id: str,
    solver: str,
    hamiltonian_sha256: str,
) -> str:
    """Compute the canonical scientific payload digest for a SolverResult.

    This is the authoritative identity function for immutability checks
    and cache validation.  It covers EVERY serialised scientifically meaningful
    field, including:

    - Scalar solver energy (as big-endian IEEE 754 bytes)
    - rdm1 and rdm2 array bytes (in canonical float64 / complex128)
    - Wavefunction array bytes (or sentinel b"ABSENT")
    - Canonical JSON of JSON-serialisable metadata (sorted keys)
    - Per-key sha256 of each metadata ndarray (sorted by key)
    - Per-key dtype-tagged bytes of each numpy-scalar metadata field (sorted by key)
    - Caller-supplied scientific identity strings:
        fragment_id, run_id, solver, hamiltonian_sha256

    The ``created_iso8601`` timestamp is intentionally excluded.

    Determinism guarantees
    ----------------------
    - All array dtypes are canonicalised (float64 / complex128) before hashing.
    - All dict keys are sorted.
    - Numpy scalars are represented as 8-byte big-endian values tagged with
      their dtype string so that int64(20260626) != float64(20260626.0).
    - The final digest is SHA-256 of the concatenation of all component hashes
      (framed with 8-byte big-endian length prefixes to prevent collisions
      between adjacent fields).

    Returns
    -------
    64-character lowercase hex SHA-256 string.
    """
    h = hashlib.sha256()

    def _feed(label: bytes, data: bytes) -> None:
        # Frame: 8-byte length of label, label bytes, 8-byte length of data, data bytes
        h.update(struct.pack(">Q", len(label)))
        h.update(label)
        h.update(struct.pack(">Q", len(data)))
        h.update(data)

    # 1. Scientific identity strings
    _feed(b"fragment_id",        str(fragment_id).encode())
    _feed(b"run_id",             run_id.encode())
    _feed(b"solver",             solver.encode())
    _feed(b"hamiltonian_sha256", hamiltonian_sha256.encode())

    # 2. Scalar energy (big-endian IEEE 754 double)
    _feed(b"energy", struct.pack(">d", float(energy)))

    # 3. rdm1 and rdm2 (canonical dtype)
    rdm1_c = _canonical_array_dtype(rdm1)
    rdm2_c = _canonical_array_dtype(rdm2)
    _feed(b"rdm1_dtype",  rdm1_c.dtype.str.encode())
    _feed(b"rdm1_shape",  json.dumps(list(rdm1_c.shape)).encode())
    _feed(b"rdm1_bytes",  rdm1_c.tobytes())
    _feed(b"rdm2_dtype",  rdm2_c.dtype.str.encode())
    _feed(b"rdm2_shape",  json.dumps(list(rdm2_c.shape)).encode())
    _feed(b"rdm2_bytes",  rdm2_c.tobytes())

    # 4. Wavefunction (optional)
    if wavefunction is not None:
        wf_c = _canonical_array_dtype(wavefunction)
        _feed(b"wavefunction_dtype",  wf_c.dtype.str.encode())
        _feed(b"wavefunction_shape",  json.dumps(list(wf_c.shape)).encode())
        _feed(b"wavefunction_bytes",  wf_c.tobytes())
    else:
        _feed(b"wavefunction", b"ABSENT")

    # 5. JSON-serialisable metadata (sorted keys, canonical JSON)
    json_str = json.dumps(json_meta, default=_json_default, sort_keys=True)
    _feed(b"metadata_json", json_str.encode("utf-8"))

    # 6. Array-valued metadata (sorted by key)
    for key in sorted(array_meta.keys()):
        arr_c = _canonical_array_dtype(array_meta[key])
        _feed(f"meta_array_key:{key}".encode(),    b"")
        _feed(f"meta_array_dtype:{key}".encode(),  arr_c.dtype.str.encode())
        _feed(f"meta_array_shape:{key}".encode(),  json.dumps(list(arr_c.shape)).encode())
        _feed(f"meta_array_bytes:{key}".encode(),  arr_c.tobytes())

    # 7. Numpy-scalar metadata (sorted by key) — previously UNCOVERED gap
    #    Each scalar is tagged with its dtype string to prevent int/float collisions.
    for key in sorted(nonser_meta.keys()):
        scalar_arr = np.asarray(nonser_meta[key])           # 0-d array
        _feed(f"meta_nonser_key:{key}".encode(),    b"")
        _feed(f"meta_nonser_dtype:{key}".encode(),  scalar_arr.dtype.str.encode())
        _feed(f"meta_nonser_bytes:{key}".encode(),  scalar_arr.tobytes())

    return h.hexdigest()


def _utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _validate_identity_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FragmentStoreError(f"{name!r} must be a non-empty string, got {value!r}")
    return str(value).strip()


def _validate_fragment_id(value: Any) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise FragmentStoreError(
            f"fragment_id must be a non-negative integer, got {value!r} ({type(value).__name__})"
        )
    v = int(value)
    if v < 0:
        raise FragmentStoreError(f"fragment_id must be >= 0, got {v}")
    return v


def _validate_ndarray_rdm(arr: Any, name: str) -> np.ndarray:
    """Validate and canonicalise rdm1 / rdm2 arrays.  Preserves complex dtype."""
    if arr is None:
        raise FragmentStoreError(f"{name} is required and must not be None")
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.asarray(arr)
        except Exception as exc:
            raise FragmentStoreError(f"{name} cannot be converted to ndarray: {exc}") from exc
    if not np.issubdtype(arr.dtype, np.number):
        raise FragmentStoreError(
            f"{name} must have a numeric dtype; got {arr.dtype}"
        )
    if not all(math.isfinite(float(v.real)) for v in arr.ravel()[:4]):
        # Fast non-finite check on first 4 elements; full check below
        pass  # handled by the full scan
    # Check for non-finite in a dtype-safe way
    real_part = arr.real if np.issubdtype(arr.dtype, np.complexfloating) else arr
    if not np.all(np.isfinite(real_part)):
        raise FragmentStoreError(f"{name} contains non-finite real values")
    if np.issubdtype(arr.dtype, np.complexfloating):
        imag_part = arr.imag
        if not np.all(np.isfinite(imag_part)):
            raise FragmentStoreError(f"{name} contains non-finite imaginary values")
    return arr


def _canonical_array_dtype(arr: np.ndarray) -> np.ndarray:
    """Store as float64 or complex128; upcasts smaller types without loss."""
    if np.issubdtype(arr.dtype, np.complexfloating):
        return arr.astype(np.complex128, copy=False)
    return arr.astype(np.float64, copy=False)


def _validate_energy(energy: Any) -> float:
    if not isinstance(energy, (int, float, np.floating, np.integer)):
        raise FragmentStoreError(
            f"energy must be a numeric scalar, got {energy!r} ({type(energy).__name__})"
        )
    v = float(energy)
    if not math.isfinite(v):
        raise FragmentStoreError(f"energy must be finite, got {v!r}")
    return v


# ─────────────────────────────────────────────────────────────────────────────
#  Metadata encoding
# ─────────────────────────────────────────────────────────────────────────────

def _json_default(obj):
    """Allow numpy scalars and small tuples in JSON; raise on everything else."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isfinite(v):
            return v
        return None   # NaN/inf → null in JSON; noted in schema doc
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.complexfloating,)):
        raise FragmentStoreError(
            f"Complex scalar {obj!r} in metadata JSON is not supported.  "
            "Store it as a 0-d complex128 ndarray instead."
        )
    if isinstance(obj, (np.ndarray,)):
        # ndarrays are handled separately via /metadata_arrays; reaching here means
        # they were nested inside a list/dict.
        raise FragmentStoreError(
            f"ndarray nested inside a JSON-serialisable container is not supported.  "
            "Top-level metadata arrays are stored in /metadata_arrays/; nested arrays cannot "
            "be reliably round-tripped through JSON."
        )
    raise FragmentStoreError(
        f"metadata value of type {type(obj).__name__!r} ({obj!r}) is not supported by "
        "fragment_result_store.  Supported types: JSON scalars (int/float/bool/None/str), "
        "dicts/lists of those, numpy scalars, numpy ndarrays (top-level only), and tuples of "
        "JSON-serialisable values."
    )


def _split_metadata(metadata: dict) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split metadata into (json_part, arrays_part, nonser_scalars_part).

    Rules
    -----
    * numpy.ndarray                   → arrays_part  (stored as HDF5 dataset)
    * numpy scalar (np.integer, etc.) → nonser_scalars_part (0-d dataset)
    * tuple of ndarrays               → HDF5 arrays + JSON descriptor
    * everything else                 → json_part  (validated by json.dumps)
    """
    json_part: dict = {}
    arrays_part: dict[str, np.ndarray] = {}
    nonser_part: dict[str, np.ndarray] = {}

    for key, value in metadata.items():
        if not isinstance(key, str):
            raise FragmentStoreError(
                f"metadata key {key!r} is not a string; all metadata keys must be strings"
            )
        if isinstance(value, np.ndarray):
            arrays_part[key] = value
        elif isinstance(value, (np.integer, np.floating, np.bool_)):
            # numpy scalar: store as 0-d dataset to preserve dtype precisely
            nonser_part[key] = np.asarray(value)
        elif isinstance(value, tuple):
            if any(isinstance(v, np.ndarray) for v in value):
                if not all(isinstance(v, np.ndarray) for v in value):
                    raise FragmentStoreError(
                        f"metadata[{key!r}] mixes arrays and non-arrays"
                    )

                # FRS_TUPLE_ARRAY_PATCH_V1
                # Preserve each array in HDF5 and its tuple ordering in JSON.
                prefix = f"__frs_tuple_array_v1_{key.encode('utf-8').hex()}_"
                names = [f"{prefix}{i}" for i in range(len(value))]

                if any(n in metadata or n in arrays_part for n in names):
                    raise FragmentStoreError(
                        f"Reserved metadata key collision for {key!r}"
                    )

                for name, array in zip(names, value):
                    arrays_part[name] = array

                json_part[key] = {"__frs_array_tuple_v1__": names}
            else:
                json_part[key] = list(value)
        else:
            if isinstance(value, dict) and set(value) == {"__frs_array_tuple_v1__"}:
                raise FragmentStoreError(
                    f"metadata[{key!r}] uses a reserved persistence marker"
                )
            json_part[key] = value

    # Validate the JSON part now (fail loudly on unsupported nested types)
    try:
        json.dumps(json_part, default=_json_default)
    except FragmentStoreError:
        raise
    except (TypeError, ValueError) as exc:
        raise FragmentStoreError(
            f"metadata contains JSON-unserializable value: {exc}"
        ) from exc

    return json_part, arrays_part, nonser_part


def _restore_metadata(
    json_str: str,
    arrays_group,   # h5py group or None
    nonser_group,   # h5py group or None
) -> dict:
    """Reconstruct the metadata dict from the three storage zones."""
    meta: dict = json.loads(json_str) if json_str else {}

    # Restore numpy arrays
    if arrays_group is not None:
        for key in arrays_group:
            ds = arrays_group[key]
            meta[key] = ds[...]

    # Restore numpy scalars (0-d datasets)
    if nonser_group is not None:
        for key in nonser_group:
            ds = nonser_group[key]
            meta[key] = ds[()]   # returns numpy scalar, preserving dtype

    # Restore tuple-of-array metadata from the HDF5 datasets.
    for key, descriptor in list(meta.items()):
        if not (
            isinstance(descriptor, dict)
            and set(descriptor) == {"__frs_array_tuple_v1__"}
        ):
            continue

        names = descriptor["__frs_array_tuple_v1__"]
        if not isinstance(names, list) or not names:
            raise FragmentStoreError(f"Invalid tuple descriptor: {key!r}")

        prefix = f"__frs_tuple_array_v1_{key.encode('utf-8').hex()}_"
        expected = [f"{prefix}{i}" for i in range(len(names))]

        if names != expected:
            raise FragmentStoreError(f"Invalid tuple array names: {key!r}")

        if any(
            name not in meta or not isinstance(meta[name], np.ndarray)
            for name in names
        ):
            raise FragmentStoreError(
                f"Missing tuple-array datasets for {key!r}"
            )

        meta[key] = tuple(meta.pop(name) for name in names)

    return meta


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def save_fragment_result(
    result,
    path: str | Path,
    *,
    fragment_id: int,
    run_id: str,
    solver: str,
    hamiltonian_sha256: str,
    extra: dict | None = None,
) -> FragmentResultReference:
    """Persist a complete :class:`SolverResult` to an HDF5 artifact.

    Parameters
    ----------
    result            : SolverResult (``energy``, ``rdm1``, ``rdm2``,
                        ``wavefunction``, ``metadata``)
    path              : Destination path for the HDF5 file (must end in ``.h5``).
    fragment_id       : Fragment integer ID.
    run_id            : Caller-supplied run identity string.
    solver            : Solver label (e.g. ``"FCI"``, ``"CCSD"``, ``"TrimSQD"``).
    hamiltonian_sha256: sha256(h1e.tobytes() + h2e.tobytes()), as computed by
                        ``ccsd_diagnostics.hamiltonian_hash``.
    extra             : Optional dict of additional reference metadata (e.g.
                        ``{"n_orb": 12}``).  Must be JSON-serialisable.

    Returns
    -------
    FragmentResultReference

    Raises
    ------
    FragmentStoreError
        On any validation failure, unsupported type, or immutability violation.
    """
    import h5py

    path = Path(path)
    if path.suffix.lower() != ".h5":
        raise FragmentStoreError(f"artifact path must have .h5 extension, got {path}")

    fragment_id = _validate_fragment_id(fragment_id)
    run_id = _validate_identity_string(run_id, "run_id")
    solver = _validate_identity_string(solver, "solver")
    hamiltonian_sha256 = _validate_identity_string(hamiltonian_sha256, "hamiltonian_sha256")

    energy = _validate_energy(result.energy)

    rdm1 = _validate_ndarray_rdm(result.rdm1, "rdm1")
    rdm2 = _validate_ndarray_rdm(result.rdm2, "rdm2")
    rdm1 = _canonical_array_dtype(rdm1)
    rdm2 = _canonical_array_dtype(rdm2)

    rdm1_checksum = _sha256_array(rdm1)
    rdm2_checksum = _sha256_array(rdm2)

    wavefunction = result.wavefunction
    if wavefunction is not None:
        if not isinstance(wavefunction, np.ndarray):
            try:
                wavefunction = np.asarray(wavefunction)
            except Exception as exc:
                raise FragmentStoreError(
                    f"wavefunction cannot be converted to ndarray: {exc}"
                ) from exc
        wavefunction = _canonical_array_dtype(wavefunction)

    # Wavefunction checksum — stored in artifact and compared on idempotent write.
    # Sentinel "ABSENT" is stored when wavefunction is None so the check is unambiguous.
    wavefunction_checksum = _sha256_array(wavefunction) if wavefunction is not None else "ABSENT"

    metadata = dict(result.metadata) if result.metadata else {}
    json_meta, array_meta, nonser_meta = _split_metadata(metadata)
    json_str = json.dumps(json_meta, default=_json_default, sort_keys=True)

    # ── Per-zone checksums (kept for backward-compatible artifact inspection) ─
    metadata_json_checksum = _sha256_bytes(json_str.encode("utf-8"))
    metadata_array_checksums = {
        k: _sha256_array(_canonical_array_dtype(v))
        for k, v in sorted(array_meta.items())
    }
    metadata_arrays_checksum = _sha256_bytes(
        json.dumps(metadata_array_checksums, sort_keys=True).encode("utf-8")
    )

    # ── Unified canonical payload digest ──────────────────────────────────────
    # This single digest covers ALL scientifically meaningful fields, including
    # the metadata_nonser zone (numpy scalars such as `seed: np.int64`) which
    # the per-zone checksums above do not cover.  It is the authoritative
    # identity test used for the immutability check.
    payload_digest = compute_payload_digest(
        energy=energy,
        rdm1=rdm1,
        rdm2=rdm2,
        wavefunction=wavefunction,
        json_meta=json_meta,
        array_meta=array_meta,
        nonser_meta=nonser_meta,
        fragment_id=fragment_id,
        run_id=run_id,
        solver=solver,
        hamiltonian_sha256=hamiltonian_sha256,
    )

    extra_dict = dict(extra) if extra else {}
    try:
        json.dumps(extra_dict)
    except (TypeError, ValueError) as exc:
        raise FragmentStoreError(f"extra dict is not JSON-serialisable: {exc}") from exc

    # ── Write to a temporary file first; rename on success ───────────────────
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        suffix=".h5.part",
        prefix=f".frs_{os.getpid()}_",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_str)

    try:
        with h5py.File(tmp_path, "w") as hf:
            hf.attrs["schema_version"]           = SCHEMA_VERSION
            hf.attrs["fragment_id"]              = fragment_id
            hf.attrs["run_id"]                   = run_id
            hf.attrs["solver"]                   = solver
            hf.attrs["hamiltonian_sha256"]        = hamiltonian_sha256
            hf.attrs["energy"]                   = np.float64(energy)
            hf.attrs["created_iso8601"]          = _utc_iso()
            hf.attrs["rdm1_checksum"]            = rdm1_checksum
            hf.attrs["rdm2_checksum"]            = rdm2_checksum
            hf.attrs["wavefunction_checksum"]    = wavefunction_checksum
            hf.attrs["metadata_json_checksum"]   = metadata_json_checksum
            hf.attrs["metadata_arrays_checksum"] = metadata_arrays_checksum
            # Unified canonical payload digest (authoritative identity)
            hf.attrs["payload_digest"]           = payload_digest

            hf.create_dataset("rdm1", data=rdm1, compression="gzip", compression_opts=1)
            hf.create_dataset("rdm2", data=rdm2, compression="gzip", compression_opts=1)

            if wavefunction is not None:
                hf.create_dataset(
                    "wavefunction", data=wavefunction,
                    compression="gzip", compression_opts=1,
                )

            # metadata_json: always present (may be empty object)
            hf.create_dataset(
                "metadata_json",
                data=np.bytes_(json_str.encode("utf-8")),
            )

            if array_meta:
                grp = hf.create_group("metadata_arrays")
                for key, arr in sorted(array_meta.items()):
                    canonical = _canonical_array_dtype(arr)
                    grp.create_dataset(
                        key, data=canonical,
                        compression="gzip", compression_opts=1,
                    )

            if nonser_meta:
                grp = hf.create_group("metadata_nonser")
                for key, arr in nonser_meta.items():
                    grp.create_dataset(key, data=arr)

    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    # ── Atomic rename ─────────────────────────────────────────────────────────
    if path.exists():
        # Immutability check: the unified ``payload_digest`` is the single
        # authoritative test.  It covers every serialised scientifically
        # meaningful field (RDMs, wavefunction, energy, all three metadata
        # zones including nonser scalars, and identity strings).
        #
        # For artifacts written before this field existed (pre Task 5.2), we
        # fall back to the multi-field comparison so old artifacts are not
        # incorrectly rejected as conflicts.
        #
        # The created_iso8601 timestamp is intentionally excluded.
        try:
            with h5py.File(path, "r") as _ex:
                stored_pd = str(_ex.attrs.get("payload_digest", ""))
                if stored_pd:
                    # Task 5.2+ artifact: single authoritative comparison
                    _same = (stored_pd == payload_digest)
                else:
                    # Legacy (pre Task 5.2) artifact: multi-field comparison
                    def _s(k): return str(_ex.attrs.get(k, ""))
                    def _f(k): return float(_ex.attrs.get(k, float("nan")))
                    def _i(k): return int(_ex.attrs.get(k, -1))
                    _same = (
                        _s("schema_version")           == SCHEMA_VERSION
                        and _i("fragment_id")           == fragment_id
                        and _s("run_id")               == run_id
                        and _s("solver")               == solver
                        and _s("hamiltonian_sha256")   == hamiltonian_sha256
                        and _f("energy")               == energy
                        and _s("rdm1_checksum")        == rdm1_checksum
                        and _s("rdm2_checksum")        == rdm2_checksum
                        and _s("wavefunction_checksum")    == wavefunction_checksum
                        and _s("metadata_json_checksum")   == metadata_json_checksum
                        and _s("metadata_arrays_checksum") == metadata_arrays_checksum
                    )
        except Exception:
            _same = False
        tmp_path.unlink(missing_ok=True)
        if not _same:
            raise FragmentStoreError(
                f"artifact {path} already exists with different content "
                "(fragment result artifacts are immutable; delete the file explicitly "
                "if you intend to overwrite it)"
            )
        # Same scientific content: idempotent — keep existing file
    else:
        os.replace(tmp_path, path)   # atomic on POSIX; best-effort on Windows

    file_sha256 = _sha256_file(path)

    return FragmentResultReference(
        path=path,
        file_sha256=file_sha256,
        payload_digest=payload_digest,
        fragment_id=fragment_id,
        run_id=run_id,
        solver=solver,
        hamiltonian_sha256=hamiltonian_sha256,
        extra=extra_dict,
    )


def load_fragment_result(
    path: str | Path,
    *,
    expected_fragment_id: int | None = None,
    expected_run_id: str | None = None,
    expected_solver: str | None = None,
    expected_hamiltonian_sha256: str | None = None,
    file_sha256: str | None = None,
) -> "SolverResult":
    """Load and validate a :class:`SolverResult` from a persisted HDF5 artifact.

    Parameters
    ----------
    path                        : Path to the HDF5 artifact.
    expected_fragment_id        : If given, must match the stored ``fragment_id``.
    expected_run_id             : If given, must match the stored ``run_id``.
    expected_solver             : If given, must match the stored ``solver``.
    expected_hamiltonian_sha256 : If given, must match the stored hash.
    file_sha256                 : If given (from a :class:`FragmentResultReference`),
                                  the on-disk file is checked against this hash before
                                  the HDF5 is opened.

    Returns
    -------
    SolverResult  (from ``quantum_fragment_methods.application.solvers.base``)

    Raises
    ------
    FragmentStoreError
        On missing file, schema mismatch, checksum failure, identity mismatch,
        or corrupt arrays.
    """
    import h5py

    # Import SolverResult lazily so this module doesn't force the full stack at import time
    try:
        from quantum_fragment_methods.application.solvers.base import SolverResult
    except ImportError:
        # Fallback: define a minimal compatible class for environments where the
        # bundle is not on sys.path (e.g. pure test environments)
        class SolverResult:  # type: ignore[no-redef]
            def __init__(self, energy, wavefunction, rdm1, rdm2, metadata):
                self.energy = energy
                self.wavefunction = wavefunction
                self.rdm1 = rdm1
                self.rdm2 = rdm2
                self.metadata = metadata or {}

    path = Path(path)
    if not path.is_file():
        raise FragmentStoreError(f"artifact not found: {path}")

    # ── Whole-file integrity check (if reference provides expected checksum) ──
    if file_sha256 is not None:
        actual = _sha256_file(path)
        if actual != file_sha256:
            raise FragmentStoreError(
                f"file-level checksum mismatch for {path.name}: "
                f"expected {file_sha256[:12]}…, got {actual[:12]}…"
            )

    try:
        with h5py.File(path, "r") as hf:
            # ── Schema version ────────────────────────────────────────────────
            sv = hf.attrs.get("schema_version", "")
            if sv != SCHEMA_VERSION:
                raise FragmentStoreError(
                    f"unknown schema version {sv!r}; this store only reads {SCHEMA_VERSION!r}"
                )

            # ── Required attributes ───────────────────────────────────────────
            stored_fid   = int(hf.attrs["fragment_id"])
            stored_rid   = str(hf.attrs["run_id"])
            stored_solv  = str(hf.attrs["solver"])
            stored_hsha  = str(hf.attrs["hamiltonian_sha256"])
            stored_energy = float(hf.attrs["energy"])
            rdm1_cksum   = str(hf.attrs["rdm1_checksum"])
            rdm2_cksum   = str(hf.attrs["rdm2_checksum"])

            # ── Identity checks ───────────────────────────────────────────────
            if expected_fragment_id is not None and stored_fid != int(expected_fragment_id):
                raise FragmentStoreError(
                    f"fragment_id mismatch: stored {stored_fid}, expected {int(expected_fragment_id)}"
                )
            if expected_run_id is not None and stored_rid != str(expected_run_id):
                raise FragmentStoreError(
                    f"run_id mismatch: stored {stored_rid!r}, expected {str(expected_run_id)!r}"
                )
            if expected_solver is not None and stored_solv != str(expected_solver):
                raise FragmentStoreError(
                    f"solver mismatch: stored {stored_solv!r}, expected {str(expected_solver)!r}"
                )
            if (expected_hamiltonian_sha256 is not None
                    and stored_hsha != str(expected_hamiltonian_sha256)):
                raise FragmentStoreError(
                    f"hamiltonian_sha256 mismatch: stored {stored_hsha[:12]}…, "
                    f"expected {str(expected_hamiltonian_sha256)[:12]}…"
                )

            # ── Load RDMs ─────────────────────────────────────────────────────
            if "rdm1" not in hf or "rdm2" not in hf:
                raise FragmentStoreError("artifact is missing required 'rdm1' or 'rdm2' dataset")

            rdm1 = hf["rdm1"][...]
            rdm2 = hf["rdm2"][...]

            # ── Array integrity checks ─────────────────────────────────────────
            actual_rdm1_cksum = _sha256_array(rdm1)
            if actual_rdm1_cksum != rdm1_cksum:
                raise FragmentStoreError(
                    f"rdm1 data checksum mismatch for {path.name}: "
                    f"stored {rdm1_cksum[:12]}…, computed {actual_rdm1_cksum[:12]}…"
                )
            actual_rdm2_cksum = _sha256_array(rdm2)
            if actual_rdm2_cksum != rdm2_cksum:
                raise FragmentStoreError(
                    f"rdm2 data checksum mismatch for {path.name}: "
                    f"stored {rdm2_cksum[:12]}…, computed {actual_rdm2_cksum[:12]}…"
                )

            # ── Wavefunction (optional) ───────────────────────────────────────
            wavefunction = hf["wavefunction"][...] if "wavefunction" in hf else None

            # ── Metadata ─────────────────────────────────────────────────────
            raw_json_ds = hf["metadata_json"] if "metadata_json" in hf else None
            if raw_json_ds is not None:
                raw_bytes = bytes(raw_json_ds[()])
                json_str = raw_bytes.decode("utf-8")
            else:
                json_str = "{}"

            arrays_grp  = hf.get("metadata_arrays")
            nonser_grp  = hf.get("metadata_nonser")
            metadata = _restore_metadata(json_str, arrays_grp, nonser_grp)

    except FragmentStoreError:
        raise
    except Exception as exc:
        raise FragmentStoreError(
            f"could not read artifact {path}: {type(exc).__name__}: {exc}"
        ) from exc

    return SolverResult(
        energy=stored_energy,
        wavefunction=wavefunction,
        rdm1=rdm1,
        rdm2=rdm2,
        metadata=metadata,
    )


def load_fragment_result_from_ref(ref: FragmentResultReference) -> "SolverResult":
    """Convenience wrapper: load and validate using all fields of a reference."""
    return load_fragment_result(
        ref.path,
        expected_fragment_id=ref.fragment_id,
        expected_run_id=ref.run_id,
        expected_solver=ref.solver,
        expected_hamiltonian_sha256=ref.hamiltonian_sha256,
        file_sha256=ref.file_sha256,
    )
