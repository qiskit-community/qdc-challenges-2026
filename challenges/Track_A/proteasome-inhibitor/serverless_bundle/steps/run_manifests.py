"""
run_manifests — Versioned, validated data contracts for Proteasome Challenge runs.

Defines two schemas:

  ``RunManifest`` (``rm/1``)
      Top-level record for a complete scientific run.  Created once per run;
      never mutated.  Contains molecular identity, HF/embedding artifact
      references, fragment list, frozen solver assignments, promoted protocol
      digest, and dependency provenance.

  ``FragmentArtifactManifest`` (``fm/1``)
      Per-fragment record linking a fragment's scientific identity (Hamiltonian,
      embedding) to its input artifact references and, once available, its
      result artifact reference.  Created by the fragment solver worker;
      updated once (idempotently) when the result is published.

These contracts are intentionally separate from the mutable task-state registry
(Task 5.3).  A manifest records *what was computed* and *where the artifacts live*.
It does not record task scheduling state, retries, or QPU job lifecycle.

The persistent QPU sampling ledger is NEVER modified by this module.

Usage
-----
  >>> manifest = RunManifest.build(
  ...     run_id="ixazomib_ligand_sto-3g",
  ...     ligand="ixazomib", geometry_sha256="abc...", basis="sto-3g",
  ...     charge=0, spin=0, n_electrons=142, n_ao=142,
  ...     hf_artifact_ref=hf_ref,
  ...     embedding_artifact_ref=emb_ref,
  ...     fragment_ids=[0, 1, ..., 41],
  ...     solver_assignments={0: "FCI", 1: "CCSD", ...},
  ... )
  >>> manifest.as_dict()   # → JSON-safe dict
  >>> RunManifest.from_dict(d)  # → validate and reconstruct
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# ── Schema versions ───────────────────────────────────────────────────────────
RUN_MANIFEST_SCHEMA    = "rm/1"
FRAGMENT_MANIFEST_SCHEMA = "fm/1"

# ── Supported solver labels ───────────────────────────────────────────────────
VALID_SOLVERS = frozenset(["FCI", "CCSD", "SQD", "TrimSQD"])


# ─────────────────────────────────────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class ManifestError(ValueError):
    """A manifest is malformed, incomplete, or scientifically inconsistent."""


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

_HEX64 = None  # lazy compiled regex


def _hex64_re():
    global _HEX64
    if _HEX64 is None:
        import re
        _HEX64 = re.compile(r"^[0-9a-f]{64}$")
    return _HEX64


def _require_hex64(value: str, name: str) -> str:
    if not isinstance(value, str) or not _hex64_re().match(value):
        raise ManifestError(
            f"{name} must be a 64-character lowercase hex string, got {value!r}"
        )
    return value


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty string, got {value!r}")
    return value.strip()


def _require_int(value: Any, name: str, *, min_val: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < min_val:
        raise ManifestError(
            f"{name} must be an integer >= {min_val}, got {value!r}"
        )
    return value


def _manifest_digest(d: dict) -> str:
    """Canonical SHA-256 of a manifest dict (sorted keys, UTF-8 JSON)."""
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _utc_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  RunManifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunManifest:
    """Top-level scientific run record.

    Fields
    ------
    schema_version           : always ``"rm/1"``
    run_id                   : unique run identifier
    ligand                   : ligand name (e.g. ``"ixazomib"``)
    geometry_sha256          : SHA-256 of the XYZ geometry file bytes
    basis                    : basis set string (e.g. ``"sto-3g"``)
    charge                   : total molecular charge
    spin                     : spin multiplicity - 1
    n_electrons              : number of electrons (from HF)
    n_ao                     : number of atomic orbitals (from HF)
    hf_artifact_ref          : ArtifactReference for mf_data.pkl (serialised as dict)
    embedding_artifact_ref   : ArtifactReference for ewf_dumpfile.h5 (serialised as dict)
    fragment_ids             : sorted list of fragment integer IDs
    solver_assignments       : frozen {fragment_id: solver_label} mapping
    protocol_digest          : SHA-256 of the promoted quantum protocol (or ``""``)
    software_versions        : {package: version} provenance dict
    schema_versions          : {"frs": "frs/1", "ast": "ast/1", ...}
    created_iso8601          : UTC creation timestamp
    manifest_digest          : SHA-256 of the serialised manifest (self-verifying)
    """

    schema_version:         str
    run_id:                 str
    ligand:                 str
    geometry_sha256:        str
    basis:                  str
    charge:                 int
    spin:                   int
    n_electrons:            int
    n_ao:                   int
    hf_artifact_ref:        dict         # ArtifactReference.as_dict()
    embedding_artifact_ref: dict         # ArtifactReference.as_dict()
    fragment_ids:           list[int]
    solver_assignments:     dict[str, str]   # str(frag_id) → solver label
    protocol_digest:        str          # "" when no quantum protocol is active
    software_versions:      dict[str, str]
    schema_versions:        dict[str, str]
    created_iso8601:        str
    manifest_digest:        str

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        ligand: str,
        geometry_sha256: str,
        basis: str,
        charge: int,
        spin: int,
        n_electrons: int,
        n_ao: int,
        hf_artifact_ref,          # ArtifactReference or dict
        embedding_artifact_ref,   # ArtifactReference or dict
        fragment_ids: list[int],
        solver_assignments: dict,
        protocol_digest: str = "",
        software_versions: dict | None = None,
        schema_versions: dict | None = None,
    ) -> "RunManifest":
        """Create and self-sign a new RunManifest."""
        _require_str(run_id, "run_id")
        _require_str(ligand, "ligand")
        if geometry_sha256:
            _require_hex64(geometry_sha256, "geometry_sha256")
        _require_str(basis, "basis")
        _require_int(n_electrons, "n_electrons", min_val=1)
        _require_int(n_ao, "n_ao", min_val=1)
        if protocol_digest:
            _require_hex64(protocol_digest, "protocol_digest")

        # Validate solver assignments
        sa_clean: dict[str, str] = {}
        for fid, solver in solver_assignments.items():
            fid_int = int(fid)
            if solver not in VALID_SOLVERS:
                raise ManifestError(
                    f"Fragment {fid_int}: unknown solver {solver!r}; "
                    f"valid solvers: {sorted(VALID_SOLVERS)}"
                )
            sa_clean[str(fid_int)] = solver

        hf_dict  = hf_artifact_ref  if isinstance(hf_artifact_ref, dict) \
                   else hf_artifact_ref.as_dict()
        emb_dict = embedding_artifact_ref if isinstance(embedding_artifact_ref, dict) \
                   else embedding_artifact_ref.as_dict()

        sw = dict(software_versions or {})
        sv = dict(schema_versions or {"frs": "frs/1", "ast": "ast/1", "rm": "rm/1"})
        now = _utc_iso()

        # Build without digest first to compute it
        body: dict[str, Any] = {
            "schema_version":         RUN_MANIFEST_SCHEMA,
            "run_id":                 run_id,
            "ligand":                 ligand,
            "geometry_sha256":        geometry_sha256,
            "basis":                  basis,
            "charge":                 charge,
            "spin":                   spin,
            "n_electrons":            n_electrons,
            "n_ao":                   n_ao,
            "hf_artifact_ref":        hf_dict,
            "embedding_artifact_ref": emb_dict,
            "fragment_ids":           sorted(int(i) for i in fragment_ids),
            "solver_assignments":     sa_clean,
            "protocol_digest":        protocol_digest,
            "software_versions":      sw,
            "schema_versions":        sv,
            "created_iso8601":        now,
        }
        digest = _manifest_digest(body)

        return cls(
            schema_version=RUN_MANIFEST_SCHEMA,
            run_id=run_id,
            ligand=ligand,
            geometry_sha256=geometry_sha256,
            basis=basis,
            charge=charge,
            spin=spin,
            n_electrons=n_electrons,
            n_ao=n_ao,
            hf_artifact_ref=hf_dict,
            embedding_artifact_ref=emb_dict,
            fragment_ids=sorted(int(i) for i in fragment_ids),
            solver_assignments=sa_clean,
            protocol_digest=protocol_digest,
            software_versions=sw,
            schema_versions=sv,
            created_iso8601=now,
            manifest_digest=digest,
        )

    # ── serialisation ─────────────────────────────────────────────────────────

    def as_dict(self) -> dict:
        return {
            "schema_version":         self.schema_version,
            "run_id":                 self.run_id,
            "ligand":                 self.ligand,
            "geometry_sha256":        self.geometry_sha256,
            "basis":                  self.basis,
            "charge":                 self.charge,
            "spin":                   self.spin,
            "n_electrons":            self.n_electrons,
            "n_ao":                   self.n_ao,
            "hf_artifact_ref":        self.hf_artifact_ref,
            "embedding_artifact_ref": self.embedding_artifact_ref,
            "fragment_ids":           self.fragment_ids,
            "solver_assignments":     self.solver_assignments,
            "protocol_digest":        self.protocol_digest,
            "software_versions":      self.software_versions,
            "schema_versions":        self.schema_versions,
            "created_iso8601":        self.created_iso8601,
            "manifest_digest":        self.manifest_digest,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "RunManifest":
        """Validate and reconstruct a RunManifest from a dict."""
        sv = d.get("schema_version", "")
        if sv != RUN_MANIFEST_SCHEMA:
            raise ManifestError(
                f"Unsupported RunManifest schema_version {sv!r}; "
                f"expected {RUN_MANIFEST_SCHEMA!r}"
            )
        required = (
            "run_id", "ligand", "basis", "n_electrons", "n_ao",
            "hf_artifact_ref", "embedding_artifact_ref",
            "fragment_ids", "solver_assignments", "manifest_digest",
        )
        missing = [k for k in required if k not in d]
        if missing:
            raise ManifestError(
                f"RunManifest is missing required fields: {missing}"
            )
        # Verify self-digest
        body = {k: v for k, v in d.items() if k != "manifest_digest"}
        expected_digest = _manifest_digest(body)
        if d["manifest_digest"] != expected_digest:
            raise ManifestError(
                "RunManifest manifest_digest does not match content "
                f"(stored={d['manifest_digest'][:12]}…, "
                f"computed={expected_digest[:12]}…)"
            )
        # Validate solver assignments
        for fid, solver in d["solver_assignments"].items():
            if solver not in VALID_SOLVERS:
                raise ManifestError(
                    f"Fragment {fid}: unknown solver {solver!r} in manifest"
                )
        return cls(
            schema_version=sv,
            run_id=d["run_id"],
            ligand=d["ligand"],
            geometry_sha256=d.get("geometry_sha256", ""),
            basis=d["basis"],
            charge=int(d.get("charge", 0)),
            spin=int(d.get("spin", 0)),
            n_electrons=int(d["n_electrons"]),
            n_ao=int(d["n_ao"]),
            hf_artifact_ref=dict(d["hf_artifact_ref"]),
            embedding_artifact_ref=dict(d["embedding_artifact_ref"]),
            fragment_ids=sorted(int(i) for i in d["fragment_ids"]),
            solver_assignments={str(k): str(v) for k, v in d["solver_assignments"].items()},
            protocol_digest=d.get("protocol_digest", ""),
            software_versions=dict(d.get("software_versions") or {}),
            schema_versions=dict(d.get("schema_versions") or {}),
            created_iso8601=d.get("created_iso8601", ""),
            manifest_digest=d["manifest_digest"],
        )

    @classmethod
    def from_json(cls, s: str) -> "RunManifest":
        return cls.from_dict(json.loads(s))

    # ── helpers ───────────────────────────────────────────────────────────────

    def verify_digest(self) -> bool:
        body = {k: v for k, v in self.as_dict().items() if k != "manifest_digest"}
        return _manifest_digest(body) == self.manifest_digest

    def solver_for(self, fragment_id: int) -> str:
        """Return the solver assigned to *fragment_id* or raise ManifestError."""
        key = str(int(fragment_id))
        if key not in self.solver_assignments:
            raise ManifestError(
                f"Fragment {fragment_id} has no solver assignment in this manifest"
            )
        return self.solver_assignments[key]


# ─────────────────────────────────────────────────────────────────────────────
#  FragmentArtifactManifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FragmentArtifactManifest:
    """Per-fragment artifact record linking scientific identity to storage.

    Fields
    ------
    schema_version        : always ``"fm/1"``
    run_id                : parent run identifier
    fragment_id           : integer fragment ID
    hamiltonian_sha256    : SHA-256 of (h1e‖h2e) — embedding identity
    embedding_digest      : SHA-256 of the embedding artifact file (from ArtifactReference)
    solver                : solver label assigned to this fragment
    n_orb                 : number of cluster orbitals
    n_occ                 : number of occupied orbitals
    input_refs            : {name: ArtifactReference.as_dict()} for all inputs
    result_ref            : ArtifactReference.as_dict() or None (before completion)
    payload_digest        : canonical scientific payload digest of the result (or "")
    producer_version      : optional provenance string
    created_iso8601       : UTC creation timestamp
    manifest_digest       : SHA-256 of the serialised manifest (self-verifying)
    """

    schema_version:     str
    run_id:             str
    fragment_id:        int
    hamiltonian_sha256: str
    embedding_digest:   str
    solver:             str
    n_orb:              int
    n_occ:              int
    input_refs:         dict[str, dict]   # name → ArtifactReference.as_dict()
    result_ref:         dict | None       # ArtifactReference.as_dict() or None
    payload_digest:     str
    producer_version:   str
    created_iso8601:    str
    manifest_digest:    str

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        fragment_id: int,
        hamiltonian_sha256: str,
        embedding_artifact_ref,   # ArtifactReference or dict
        solver: str,
        n_orb: int,
        n_occ: int,
        input_refs: dict | None = None,
        result_ref=None,
        payload_digest: str = "",
        producer_version: str = "",
    ) -> "FragmentArtifactManifest":
        """Create and self-sign a new FragmentArtifactManifest."""
        _require_str(run_id, "run_id")
        _require_int(fragment_id, "fragment_id")
        _require_hex64(hamiltonian_sha256, "hamiltonian_sha256")
        if solver not in VALID_SOLVERS:
            raise ManifestError(
                f"Unknown solver {solver!r}; valid: {sorted(VALID_SOLVERS)}"
            )
        _require_int(n_orb, "n_orb", min_val=1)
        _require_int(n_occ, "n_occ", min_val=1)

        emb_dict = embedding_artifact_ref if isinstance(embedding_artifact_ref, dict) \
                   else embedding_artifact_ref.as_dict()
        embedding_digest = emb_dict.get("file_sha256", "")

        irefs: dict[str, dict] = {}
        for name, ref in (input_refs or {}).items():
            irefs[name] = ref if isinstance(ref, dict) else ref.as_dict()

        rref_dict: dict | None = None
        if result_ref is not None:
            rref_dict = result_ref if isinstance(result_ref, dict) else result_ref.as_dict()

        if payload_digest:
            _require_hex64(payload_digest, "payload_digest")

        now = _utc_iso()
        body: dict[str, Any] = {
            "schema_version":     FRAGMENT_MANIFEST_SCHEMA,
            "run_id":             run_id,
            "fragment_id":        fragment_id,
            "hamiltonian_sha256": hamiltonian_sha256,
            "embedding_digest":   embedding_digest,
            "solver":             solver,
            "n_orb":              n_orb,
            "n_occ":              n_occ,
            "input_refs":         irefs,
            "result_ref":         rref_dict,
            "payload_digest":     payload_digest,
            "producer_version":   producer_version,
            "created_iso8601":    now,
        }
        digest = _manifest_digest(body)

        return cls(
            schema_version=FRAGMENT_MANIFEST_SCHEMA,
            run_id=run_id,
            fragment_id=fragment_id,
            hamiltonian_sha256=hamiltonian_sha256,
            embedding_digest=embedding_digest,
            solver=solver,
            n_orb=n_orb,
            n_occ=n_occ,
            input_refs=irefs,
            result_ref=rref_dict,
            payload_digest=payload_digest,
            producer_version=producer_version,
            created_iso8601=now,
            manifest_digest=digest,
        )

    # ── serialisation ─────────────────────────────────────────────────────────

    def as_dict(self) -> dict:
        return {
            "schema_version":     self.schema_version,
            "run_id":             self.run_id,
            "fragment_id":        self.fragment_id,
            "hamiltonian_sha256": self.hamiltonian_sha256,
            "embedding_digest":   self.embedding_digest,
            "solver":             self.solver,
            "n_orb":              self.n_orb,
            "n_occ":              self.n_occ,
            "input_refs":         self.input_refs,
            "result_ref":         self.result_ref,
            "payload_digest":     self.payload_digest,
            "producer_version":   self.producer_version,
            "created_iso8601":    self.created_iso8601,
            "manifest_digest":    self.manifest_digest,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "FragmentArtifactManifest":
        sv = d.get("schema_version", "")
        if sv != FRAGMENT_MANIFEST_SCHEMA:
            raise ManifestError(
                f"Unsupported FragmentArtifactManifest schema_version {sv!r}; "
                f"expected {FRAGMENT_MANIFEST_SCHEMA!r}"
            )
        required = (
            "run_id", "fragment_id", "hamiltonian_sha256", "embedding_digest",
            "solver", "n_orb", "n_occ", "manifest_digest",
        )
        missing = [k for k in required if k not in d]
        if missing:
            raise ManifestError(
                f"FragmentArtifactManifest is missing required fields: {missing}"
            )
        if d["solver"] not in VALID_SOLVERS:
            raise ManifestError(
                f"Unknown solver {d['solver']!r} in manifest"
            )
        body = {k: v for k, v in d.items() if k != "manifest_digest"}
        expected_digest = _manifest_digest(body)
        if d["manifest_digest"] != expected_digest:
            raise ManifestError(
                "FragmentArtifactManifest manifest_digest does not match content"
            )
        return cls(
            schema_version=sv,
            run_id=d["run_id"],
            fragment_id=int(d["fragment_id"]),
            hamiltonian_sha256=d["hamiltonian_sha256"],
            embedding_digest=d["embedding_digest"],
            solver=d["solver"],
            n_orb=int(d["n_orb"]),
            n_occ=int(d["n_occ"]),
            input_refs=dict(d.get("input_refs") or {}),
            result_ref=d.get("result_ref"),
            payload_digest=d.get("payload_digest", ""),
            producer_version=d.get("producer_version", ""),
            created_iso8601=d.get("created_iso8601", ""),
            manifest_digest=d["manifest_digest"],
        )

    @classmethod
    def from_json(cls, s: str) -> "FragmentArtifactManifest":
        return cls.from_dict(json.loads(s))

    def verify_digest(self) -> bool:
        body = {k: v for k, v in self.as_dict().items() if k != "manifest_digest"}
        return _manifest_digest(body) == self.manifest_digest

    def with_result(
        self,
        result_ref,             # ArtifactReference or dict
        payload_digest: str,
    ) -> "FragmentArtifactManifest":
        """Return a new manifest with the result artifact reference attached."""
        rref = result_ref if isinstance(result_ref, dict) else result_ref.as_dict()
        if payload_digest:
            _require_hex64(payload_digest, "payload_digest")
        return FragmentArtifactManifest.build(
            run_id=self.run_id,
            fragment_id=self.fragment_id,
            hamiltonian_sha256=self.hamiltonian_sha256,
            embedding_artifact_ref=self.input_refs.get(
                "embedding", {"file_sha256": self.embedding_digest}
            ),
            solver=self.solver,
            n_orb=self.n_orb,
            n_occ=self.n_occ,
            input_refs=self.input_refs,
            result_ref=rref,
            payload_digest=payload_digest,
            producer_version=self.producer_version,
        )
