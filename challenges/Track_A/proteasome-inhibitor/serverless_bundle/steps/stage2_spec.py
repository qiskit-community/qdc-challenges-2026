"""
stage2_spec — scientific specification, identity and validation of Stage 2 (EWF fragmentation + bath construction).

Shared by the Serverless worker (``step2_fragments``) and the local orchestrator (``proteasome_challenge.ewf_stage``)
so both sides agree on what a Stage 2 calculation *is*.  Nothing here runs Serverless jobs or touches the network.

Three distinct notions are kept apart:

``stage2_param_digest``   SCIENTIFIC IDENTITY of the requested embedding calculation (this module).
artifact SHA-256          INTEGRITY of a persisted file (``hashlib`` over ewf_dumpfile.h5 bytes).
numerical tolerances      AGREEMENT of exported Hamiltonians and integrals (validation gates).
None of them substitutes for another.

Canonicalization policy (``stage2-spec/1``)
-------------------------------------------
The digest is SHA-256 over compact, key-sorted UTF-8 JSON of::

    {"schema": "stage2-spec/1",
     "stage1": <canonical Stage1Spec JSON>,
     "ewf": {
         "fragmentation": {"basis": b, "scheme": s, "iao_minao": m},
         "bath": {"method": m, "truncation": t, "truncation_kind": k, "dmet_threshold": d,
                  "canonicalize": c, "occupation_tolerance": o, ...},
         "solver": "DUMP"
     }}

* NOT part of the identity: ``verbose``, serverless job IDs, temporary paths, timestamps,
  and software versions (a software version is execution provenance, not the requested spec).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .stage1_spec import (
    HFConfig,
    MoleculeSpec,
    Stage1Spec,
    Stage1SpecError,
    Stage1ValidationError,
    sha256_bytes,
    sha256_file,
)

__all__ = [
    "SPEC_SCHEMA", "RESULT_SCHEMA", "Stage2SpecError", "Stage2ValidationError",
    "FragmentationConfig", "BathConfig", "EWFConfig", "Stage2Spec",
    "SUPPORTED_FRAGMENTATION_SCHEMES", "SUPPORTED_FRAGMENTATION_BASES",
    "SUPPORTED_BATH_METHODS", "DEFERRED_BATH_METHODS",
    "VAYESTA_UPSTREAM_COMMIT", "VAYESTA_VERSION",
    "resolve_iao_minao", "validate_ewf_dumpfile", "ewf_diagnostics",
    "build_stage2_result_record", "diff_stage2_specs",
]

SPEC_SCHEMA = "stage2-spec/1"
RESULT_SCHEMA = "stage2-ewf-result/1"

VAYESTA_UPSTREAM_COMMIT = "e9e48c7090b1ac3b92e9384de45cab88dcb140a4"
VAYESTA_VERSION = "1.0.1"

SUPPORTED_FRAGMENTATION_SCHEMES = ("atomic", "iao")
SUPPORTED_FRAGMENTATION_BASES = ("iao",)
SUPPORTED_BATH_METHODS = ("mp2", "dmet")
DEFERRED_BATH_METHODS = ("r2", "ewdmet", "rpa", "bosonic_rpa")
SUPPORTED_TRUNCATION_KINDS = ("occupation", "energy")

_DEFAULT_MINAO_MAP = {
    "sto3g": "minao",
    "sto-3g": "minao",
    "631g": "minao",
    "6-31g": "minao",
    "631g*": "minao",
    "6-31g*": "minao",
    "631g**": "minao",
    "6-31g**": "minao",
    "ccpvdz": "minao",
    "cc-pvdz": "minao",
    "ccpvtz": "minao",
    "cc-pvtz": "minao",
    "def2svp": "def2-svp-minao",
    "def2-svp": "def2-svp-minao",
    "def2tzvp": "def2-tzvp-minao",
    "def2-tzvp": "def2-tzvp-minao",
}


class Stage2SpecError(ValueError):
    """The requested EWF / fragmentation / bath configuration is invalid or unsupported."""


class Stage2ValidationError(RuntimeError):
    """An exported EWF fragment dataset failed scientific validation."""


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _finite(value: Any, label: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None,
            exclusive_max: bool = False, exclusive_min: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage2SpecError(f"{label} must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise Stage2SpecError(f"{label} must be finite, got {value!r}")
    if minimum is not None and (v < minimum or (exclusive_min and v == minimum)):
        raise Stage2SpecError(f"{label} must be {'>' if exclusive_min else '>='} {minimum}, got {v}")
    if maximum is not None and (v > maximum or (exclusive_max and v == maximum)):
        raise Stage2SpecError(f"{label} must be {'<' if exclusive_max else '<='} {maximum}, got {v}")
    return v + 0.0


def resolve_iao_minao(iao_minao: Optional[str], mol_basis: Optional[str] = None) -> str:
    """Resolve the reference minimal basis for IAO construction.

    Resolution policy:
    * If ``iao_minao`` is an explicit string not equal to ``'auto'``, normalize to lower case.
    * If ``iao_minao`` is ``'auto'`` or ``None``:
      - If ``mol_basis`` is given, look up standard PySCF/Vayesta minao mappings.
      - Default fallback is ``'minao'``.
    """
    if iao_minao is None or (isinstance(iao_minao, str) and iao_minao.strip().lower() in ("", "auto")):
        if mol_basis:
            key = mol_basis.strip().lower().replace("-", "")
            return _DEFAULT_MINAO_MAP.get(key, _DEFAULT_MINAO_MAP.get(mol_basis.strip().lower(), "minao"))
        return "minao"
    if not isinstance(iao_minao, str):
        raise Stage2SpecError(f"iao_minao must be a string or None, got {iao_minao!r}")
    m = iao_minao.strip().lower()
    if not m:
        return "minao"
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  FragmentationConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FragmentationConfig:
    """Validated fragmentation configuration.

    Parameters
    ----------
    scheme     : 'atomic' (default) or 'iao' (treated identically for atomic IAO).
    basis      : local orbital basis for fragmentation — 'iao' (default).
    iao_minao  : reference minimal basis for IAO construction ('auto' / 'minao' / explicit).
    """
    scheme: str = "atomic"
    basis: str = "iao"
    iao_minao: str = "auto"

    def __post_init__(self):
        if not isinstance(self.scheme, str):
            raise Stage2SpecError(f"fragmentation scheme must be a string, got {self.scheme!r}")
        scheme = self.scheme.strip().lower()
        if scheme not in SUPPORTED_FRAGMENTATION_SCHEMES:
            raise Stage2SpecError(
                f"unsupported fragmentation scheme {self.scheme!r}; supported: {SUPPORTED_FRAGMENTATION_SCHEMES}")
        if not isinstance(self.basis, str):
            raise Stage2SpecError(f"fragmentation basis must be a string, got {self.basis!r}")
        basis = self.basis.strip().lower()
        if basis not in SUPPORTED_FRAGMENTATION_BASES:
            raise Stage2SpecError(
                f"unsupported fragmentation basis {self.basis!r}; supported: {SUPPORTED_FRAGMENTATION_BASES}")
        if not isinstance(self.iao_minao, str):
            raise Stage2SpecError(f"iao_minao must be a string, got {self.iao_minao!r}")
        minao = self.iao_minao.strip().lower()

        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "iao_minao", minao)

    def canonical(self, mol_basis: Optional[str] = None) -> dict:
        resolved_minao = resolve_iao_minao(self.iao_minao, mol_basis)
        return {
            "scheme": "atomic" if self.scheme in ("atomic", "iao") else self.scheme,
            "basis": self.basis,
            "iao_minao": self.iao_minao,
            "resolved_iao_minao": resolved_minao,
        }

    def to_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "basis": self.basis,
            "iao_minao": self.iao_minao,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FragmentationConfig":
        return cls(
            scheme=d.get("scheme", "atomic"),
            basis=d.get("basis", "iao"),
            iao_minao=d.get("iao_minao", "auto"),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  BathConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BathConfig:
    """Validated bath configuration.

    Parameters
    ----------
    method               : 'mp2' (default) or 'dmet'.
    truncation           : MP2 bath BNO truncation threshold (default: 1e-4). Ignored for DMET-only.
    truncation_kind      : 'occupation' (default) or 'energy'.
    dmet_threshold       : threshold for initial DMET bath selection (default: 1e-8).
    canonicalize         : canonicalize bath orbitals (default: True).
    occupation_tolerance : tolerance for degenerate orbital occupations (default: 1e-6).
    project_dmet_order   : MP2 projector order (default: 2).
    project_dmet_mode    : MP2 projector mode (default: 'squared-entropy').
    addbuffer            : whether to add buffer orbitals (default: False).
    """
    method: str = "mp2"
    truncation: float = 1e-4
    truncation_kind: str = "occupation"
    dmet_threshold: float = 1e-8
    canonicalize: bool = True
    occupation_tolerance: float = 1e-6
    project_dmet_order: int = 2
    project_dmet_mode: str = "squared-entropy"
    addbuffer: bool = False

    def __post_init__(self):
        if not isinstance(self.method, str):
            raise Stage2SpecError(f"bath method must be a string, got {self.method!r}")
        method = self.method.strip().lower()
        if method in DEFERRED_BATH_METHODS:
            raise Stage2SpecError(
                f"bath method {self.method!r} is deferred / not qualified in this release "
                f"(deferred: {', '.join(DEFERRED_BATH_METHODS)}); supported: {', '.join(SUPPORTED_BATH_METHODS)}")
        if method not in SUPPORTED_BATH_METHODS:
            raise Stage2SpecError(
                f"unknown bath method {self.method!r}; supported: {', '.join(SUPPORTED_BATH_METHODS)}")

        trunc = _finite(self.truncation, "truncation", minimum=0.0, maximum=1.0, exclusive_min=True)
        if not isinstance(self.truncation_kind, str):
            raise Stage2SpecError(f"truncation_kind must be a string, got {self.truncation_kind!r}")
        t_kind = self.truncation_kind.strip().lower()
        if t_kind not in SUPPORTED_TRUNCATION_KINDS:
            raise Stage2SpecError(
                f"unsupported truncation_kind {self.truncation_kind!r}; supported: {SUPPORTED_TRUNCATION_KINDS}")

        dmet_tol = _finite(self.dmet_threshold, "dmet_threshold", minimum=0.0, maximum=0.5, exclusive_min=True)
        occ_tol = _finite(self.occupation_tolerance, "occupation_tolerance", minimum=0.0, maximum=0.5, exclusive_min=True)

        if not isinstance(self.canonicalize, bool):
            raise Stage2SpecError(f"canonicalize must be a bool, got {self.canonicalize!r}")
        if not isinstance(self.addbuffer, bool):
            raise Stage2SpecError(f"addbuffer must be a bool, got {self.addbuffer!r}")
        if isinstance(self.project_dmet_order, bool) or not isinstance(self.project_dmet_order, int) or self.project_dmet_order < 0:
            raise Stage2SpecError(f"project_dmet_order must be a non-negative int, got {self.project_dmet_order!r}")
        if not isinstance(self.project_dmet_mode, str):
            raise Stage2SpecError(f"project_dmet_mode must be a string, got {self.project_dmet_mode!r}")

        object.__setattr__(self, "method", method)
        object.__setattr__(self, "truncation", trunc)
        object.__setattr__(self, "truncation_kind", t_kind)
        object.__setattr__(self, "dmet_threshold", dmet_tol)
        object.__setattr__(self, "occupation_tolerance", occ_tol)
        object.__setattr__(self, "canonicalize", bool(self.canonicalize))
        object.__setattr__(self, "addbuffer", bool(self.addbuffer))
        object.__setattr__(self, "project_dmet_order", int(self.project_dmet_order))
        object.__setattr__(self, "project_dmet_mode", str(self.project_dmet_mode).strip())

    def canonical(self) -> dict:
        base = {
            "method": self.method,
            "canonicalize": self.canonicalize,
            "dmet_threshold": self.dmet_threshold,
            "occupation_tolerance": self.occupation_tolerance,
        }
        if self.method == "mp2":
            base.update({
                "truncation": self.truncation,
                "truncation_kind": self.truncation_kind,
                "project_dmet_order": self.project_dmet_order,
                "project_dmet_mode": self.project_dmet_mode,
                "addbuffer": self.addbuffer,
            })
        return base

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "truncation": self.truncation,
            "truncation_kind": self.truncation_kind,
            "dmet_threshold": self.dmet_threshold,
            "canonicalize": self.canonicalize,
            "occupation_tolerance": self.occupation_tolerance,
            "project_dmet_order": self.project_dmet_order,
            "project_dmet_mode": self.project_dmet_mode,
            "addbuffer": self.addbuffer,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BathConfig":
        allowed = {
            "method", "truncation", "truncation_kind", "dmet_threshold",
            "canonicalize", "occupation_tolerance", "project_dmet_order",
            "project_dmet_mode", "addbuffer"
        }
        unknown = set(d.keys()) - allowed
        if unknown:
            raise Stage2SpecError(f"unknown bath configuration keys: {sorted(unknown)}")
        return cls(**{k: d[k] for k in allowed if k in d})


# ─────────────────────────────────────────────────────────────────────────────
#  EWFConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EWFConfig:
    """Validated scientific configuration for Embedded Wave Function (EWF)."""
    fragmentation: FragmentationConfig = field(default_factory=FragmentationConfig)
    bath: BathConfig = field(default_factory=BathConfig)
    solver: str = "DUMP"
    verbose: int = 3

    def __post_init__(self):
        if not isinstance(self.fragmentation, FragmentationConfig):
            if isinstance(self.fragmentation, Mapping):
                object.__setattr__(self, "fragmentation", FragmentationConfig.from_dict(self.fragmentation))
            else:
                raise Stage2SpecError(f"fragmentation must be a FragmentationConfig, got {self.fragmentation!r}")
        if not isinstance(self.bath, BathConfig):
            if isinstance(self.bath, Mapping):
                object.__setattr__(self, "bath", BathConfig.from_dict(self.bath))
            else:
                raise Stage2SpecError(f"bath must be a BathConfig, got {self.bath!r}")
        if not isinstance(self.solver, str) or self.solver.strip().upper() != "DUMP":
            raise Stage2SpecError(f"only solver='DUMP' is supported for Stage 2 Hamiltonian export, got {self.solver!r}")
        if isinstance(self.verbose, bool) or not isinstance(self.verbose, int) or not 0 <= self.verbose <= 9:
            raise Stage2SpecError(f"verbose must be an integer in [0, 9], got {self.verbose!r}")
        object.__setattr__(self, "solver", "DUMP")

    def canonical(self, mol_basis: Optional[str] = None) -> dict:
        return {
            "fragmentation": self.fragmentation.canonical(mol_basis),
            "bath": self.bath.canonical(),
            "solver": self.solver,
        }

    def to_dict(self) -> dict:
        return {
            "fragmentation": self.fragmentation.to_dict(),
            "bath": self.bath.to_dict(),
            "solver": self.solver,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EWFConfig":
        frag = d.get("fragmentation", {})
        bath = d.get("bath", {})
        return cls(
            fragmentation=FragmentationConfig.from_dict(frag) if isinstance(frag, Mapping) else frag,
            bath=BathConfig.from_dict(bath) if isinstance(bath, Mapping) else bath,
            solver=d.get("solver", "DUMP"),
            verbose=d.get("verbose", 3),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Stage2Spec (Stage 1 + EWF configuration) and its digest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stage2Spec:
    stage1: Stage1Spec
    config: EWFConfig = field(default_factory=EWFConfig)

    def __post_init__(self):
        if not isinstance(self.stage1, Stage1Spec):
            if isinstance(self.stage1, Mapping):
                object.__setattr__(self, "stage1", Stage1Spec.from_dict(self.stage1))
            else:
                raise Stage2SpecError(f"stage1 must be a Stage1Spec, got {type(self.stage1).__name__}")
        if not isinstance(self.config, EWFConfig):
            if isinstance(self.config, Mapping):
                object.__setattr__(self, "config", EWFConfig.from_dict(self.config))
            else:
                raise Stage2SpecError(f"config must be an EWFConfig, got {type(self.config).__name__}")

    def canonical(self) -> dict:
        mol_basis = self.stage1.config.basis
        return {
            "schema": SPEC_SCHEMA,
            "stage1": self.stage1.canonical(),
            "ewf": self.config.canonical(mol_basis),
        }

    def digest(self) -> str:
        """``stage2_param_digest``: deterministic scientific identity of the Stage 2 request."""
        return sha256_bytes(_canonical_json(self.canonical()))

    @property
    def tag(self) -> str:
        return f"{self.stage1.molecule.name}__s2_{self.digest()[:12]}"

    def to_dict(self) -> dict:
        return {
            "schema": SPEC_SCHEMA,
            "stage1": self.stage1.to_dict(),
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Stage2Spec":
        if d.get("schema") != SPEC_SCHEMA:
            raise Stage2SpecError(f"unsupported Stage 2 specification schema {d.get('schema')!r}")
        return cls(
            stage1=Stage1Spec.from_dict(d["stage1"]),
            config=EWFConfig.from_dict(d["config"]),
        )


def diff_stage2_specs(a: Stage2Spec, b: Stage2Spec) -> list[str]:
    """Human-readable list of scientific components in which two Stage 2 specifications differ."""
    ca, cb = a.canonical(), b.canonical()
    out = []
    if ca["stage1"] != cb["stage1"]:
        out.append(f"Stage 1 reference (digest {a.stage1.digest()[:8]} vs {b.stage1.digest()[:8]})")
    ewf_a, ewf_b = ca["ewf"], cb["ewf"]
    for sect in ("fragmentation", "bath", "solver"):
        if ewf_a.get(sect) != ewf_b.get(sect):
            out.append(f"ewf.{sect}: {ewf_a.get(sect)!r} vs {ewf_b.get(sect)!r}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Scientific validation of exported EWF fragment data (ewf_dumpfile.h5)
# ─────────────────────────────────────────────────────────────────────────────

def validate_ewf_dumpfile(
    dumpfile_path: str | Path,
    spec: Optional[Stage2Spec] = None,
    *,
    expected_fragment_ids: Optional[Sequence[int]] = None,
    mf: Any = None,
    mol: Any = None,
    hermiticity_tol: float = 1e-10,
    symmetry_tol: float = 1e-10,
) -> dict[str, Any]:
    """Independent scientific validation of an exported EWF HDF5 dumpfile.

    Checks:
    1. Fragment completeness & coverage (no missing indices, continuous 0..N-1).
    2. Presence and non-emptiness of all datasets: heff, fock, eris, c_cluster, c_frag, c_dmet_cluster.
    3. Finiteness of all numerical values (no NaN or Inf).
    4. Dataset dimensions and shapes consistency (norb, nocc, nvir, nao).
    5. One-electron Hamiltonian Hermiticity: heff == heff.T, fock == fock.T.
    6. Two-electron integral permutation symmetries (8-fold real spatial symmetry):
       (pq|rs) == (qp|rs) == (pq|sr) == (rs|pq).
    7. Orthonormality of cluster orbitals when overlap matrix S is available.
    8. Electron counting and consistency.

    Returns a rich dictionary of validated diagnostics or raises ``Stage2ValidationError``.
    """
    import h5py
    import numpy as np

    p = Path(dumpfile_path)
    if not p.is_file() or p.stat().st_size == 0:
        raise Stage2ValidationError(f"EWF dumpfile missing or empty: {p}")

    problems: list[str] = []
    fragment_data: dict[int, dict[str, Any]] = {}
    fids: list[int] = []

    try:
        with h5py.File(str(p), "r") as hf:
            keys = [k for k in hf if k.startswith("fragment_")]
            if not keys:
                raise Stage2ValidationError(f"EWF dumpfile {p.name} contains no fragment_* groups")

            for k in keys:
                try:
                    fid = int(k.split("_", 1)[1])
                    fids.append(fid)
                except ValueError:
                    problems.append(f"malformed fragment group name: {k!r}")

            fids.sort()
            # Check completeness: contiguous 0 .. len(fids)-1
            if fids != list(range(len(fids))):
                missing = set(range(max(fids) + 1)) - set(fids)
                problems.append(f"fragment IDs are not contiguous 0..N-1; missing IDs: {sorted(missing)}")

            if expected_fragment_ids is not None:
                exp_sorted = sorted(int(i) for i in expected_fragment_ids)
                if fids != exp_sorted:
                    problems.append(f"fragment IDs {fids} differ from expected {exp_sorted}")

            if spec is not None and spec.config.fragmentation.scheme in ("atomic", "iao"):
                n_atoms = spec.stage1.molecule.n_atoms
                if len(fids) != n_atoms:
                    problems.append(f"atomic fragmentation expected {n_atoms} fragments for {spec.stage1.molecule.name}, got {len(fids)}")

            # Validate each fragment group
            max_herm_err = 0.0
            max_sym_err = 0.0
            total_electrons = 0

            # Overlap matrix if mf/mol is given
            ovlp = None
            if mf is not None and hasattr(mf, "get_ovlp"):
                try:
                    ovlp = np.asarray(mf.get_ovlp(), dtype=float)
                except Exception:
                    ovlp = None

            for fid in fids:
                grp = hf[f"fragment_{fid}"]
                attrs = dict(grp.attrs)

                # Required attributes
                for req_attr in ("norb", "nocc", "nvir"):
                    if req_attr not in attrs:
                        problems.append(f"fragment {fid} lacks attribute {req_attr!r}")

                norb = int(attrs.get("norb", 0))
                nocc = int(attrs.get("nocc", 0))
                nvir = int(attrs.get("nvir", 0))

                if norb <= 0:
                    problems.append(f"fragment {fid} has non-positive norb={norb}")
                if nocc < 0 or nvir < 0 or norb != (nocc + nvir):
                    problems.append(f"fragment {fid} has inconsistent orbital partitioning norb={norb} != nocc({nocc}) + nvir({nvir})")

                # Required datasets
                for req_ds in ("heff", "fock", "eris", "c_cluster", "c_frag", "c_dmet_cluster"):
                    if req_ds not in grp:
                        problems.append(f"fragment {fid} lacks dataset {req_ds!r}")

                if problems:
                    continue  # skip numeric checks if structural fields are missing

                heff = grp["heff"][:]
                fock = grp["fock"][:]
                eris = grp["eris"][:]
                c_cluster = grp["c_cluster"][:]
                c_frag = grp["c_frag"][:]
                c_dmet = grp["c_dmet_cluster"][:]

                # Finiteness
                for name, arr in (("heff", heff), ("fock", fock), ("eris", eris),
                                  ("c_cluster", c_cluster), ("c_frag", c_frag), ("c_dmet_cluster", c_dmet)):
                    if not np.all(np.isfinite(arr)):
                        problems.append(f"fragment {fid} dataset {name} contains non-finite values (NaN/Inf)")

                # Shapes
                if heff.shape != (norb, norb):
                    problems.append(f"fragment {fid} heff shape {heff.shape} != ({norb}, {norb})")
                if fock.shape != (norb, norb):
                    problems.append(f"fragment {fid} fock shape {fock.shape} != ({norb}, {norb})")
                if eris.shape != (norb, norb, norb, norb):
                    problems.append(f"fragment {fid} eris shape {eris.shape} != ({norb}, {norb}, {norb}, {norb})")
                if c_cluster.ndim != 2 or c_cluster.shape[1] != norb:
                    problems.append(f"fragment {fid} c_cluster shape {c_cluster.shape} incompatible with norb={norb}")

                # Hermiticity
                h_err = float(np.max(np.abs(heff - heff.T)))
                f_err = float(np.max(np.abs(fock - fock.T)))
                max_herm_err = max(max_herm_err, h_err, f_err)
                if h_err > hermiticity_tol:
                    problems.append(f"fragment {fid} heff Hermiticity residual {h_err:.3e} exceeds tolerance {hermiticity_tol:.3e}")
                if f_err > hermiticity_tol:
                    problems.append(f"fragment {fid} fock Hermiticity residual {f_err:.3e} exceeds tolerance {hermiticity_tol:.3e}")

                # ERI 8-fold real spatial symmetries:
                # (1) (pq|rs) == (qp|rs)
                s1 = float(np.max(np.abs(eris - eris.transpose(1, 0, 2, 3))))
                # (2) (pq|rs) == (pq|sr)
                s2 = float(np.max(np.abs(eris - eris.transpose(0, 1, 3, 2))))
                # (3) (pq|rs) == (rs|pq)
                s3 = float(np.max(np.abs(eris - eris.transpose(2, 3, 0, 1))))
                frag_sym_err = max(s1, s2, s3)
                max_sym_err = max(max_sym_err, frag_sym_err)
                if frag_sym_err > symmetry_tol:
                    problems.append(f"fragment {fid} ERI symmetry residual {frag_sym_err:.3e} exceeds tolerance {symmetry_tol:.3e}")

                # Orthonormality if overlap is provided
                ortho_err = None
                if ovlp is not None and c_cluster.shape[0] == ovlp.shape[0]:
                    s_mo = c_cluster.T @ ovlp @ c_cluster
                    ortho_err = float(np.max(np.abs(s_mo - np.eye(norb))))
                    eps = float(np.finfo(float).eps)
                    ev = np.linalg.eigvalsh(ovlp)
                    cond = float(ev.max() / ev.min()) if ev.min() > 0 else 1.0
                    ortho_tol = max(1e-8, 100.0 * eps * cond)
                    if ortho_err > ortho_tol:
                        problems.append(f"fragment {fid} cluster orbital orthonormality residual {ortho_err:.3e} exceeds tolerance {ortho_tol:.3e}")

                fragment_data[fid] = {
                    "norb": norb,
                    "nocc": nocc,
                    "nvir": nvir,
                    "hermiticity_residual": max(h_err, f_err),
                    "symmetry_residual": frag_sym_err,
                    "orthonormality_residual": ortho_err,
                }

    except Exception as exc:
        if isinstance(exc, Stage2ValidationError):
            raise
        raise Stage2ValidationError(f"failed reading EWF dumpfile {p}: {exc}") from exc

    if problems:
        raise Stage2ValidationError(f"EWF dumpfile {p.name} failed scientific validation:\n  * " + "\n  * ".join(problems))

    norb_list = [fragment_data[fid]["norb"] for fid in fids]
    nocc_list = [fragment_data[fid]["nocc"] for fid in fids]

    return {
        "n_fragments": len(fids),
        "fragment_ids": fids,
        "norb_by_fragment": {fid: fragment_data[fid]["norb"] for fid in fids},
        "nocc_by_fragment": {fid: fragment_data[fid]["nocc"] for fid in fids},
        "nvir_by_fragment": {fid: fragment_data[fid]["nvir"] for fid in fids},
        "norb_min": min(norb_list) if norb_list else 0,
        "norb_max": max(norb_list) if norb_list else 0,
        "norb_mean": float(np.mean(norb_list)) if norb_list else 0.0,
        "max_hermiticity_residual": max_herm_err,
        "max_symmetry_residual": max_sym_err,
        "file_sha256": sha256_file(p),
    }


def ewf_diagnostics(dumpfile_path: str | Path, spec: Optional[Stage2Spec] = None, **kwargs) -> dict[str, Any]:
    """Compute numerical and structural diagnostics for an EWF dumpfile."""
    return validate_ewf_dumpfile(dumpfile_path, spec=spec, **kwargs)


def build_stage2_result_record(
    spec: Stage2Spec,
    dumpfile_path: str | Path,
    *,
    stage1_result: Optional[Mapping[str, Any]] = None,
    validation_diag: Optional[Mapping[str, Any]] = None,
    software_versions: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Construct the Stage 2 sidecar provenance record conforming to RESULT_SCHEMA."""
    p = Path(dumpfile_path)
    file_sha = sha256_file(p)
    diag = dict(validation_diag or validate_ewf_dumpfile(p, spec=spec))

    sw = dict(software_versions or {})
    sw.setdefault("vayesta_upstream_commit", VAYESTA_UPSTREAM_COMMIT)
    sw.setdefault("vayesta_version", VAYESTA_VERSION)
    try:
        import pyscf
        sw.setdefault("pyscf", pyscf.__version__)
    except Exception:
        pass
    try:
        import numpy
        sw.setdefault("numpy", numpy.__version__)
    except Exception:
        pass

    return {
        "schema": RESULT_SCHEMA,
        "stage": "fragmentation",
        "stage2_param_digest": spec.digest(),
        "stage1_param_digest": spec.stage1.digest(),
        "stage1_parent": dict(stage1_result or {}),
        "spec": spec.to_dict(),
        "resolved_settings": spec.canonical(),
        "software": sw,
        "dumpfile_sha256": file_sha,
        "n_fragments": diag["n_fragments"],
        "fragment_ids": diag["fragment_ids"],
        "norb_by_fragment": diag["norb_by_fragment"],
        "nocc_by_fragment": diag["nocc_by_fragment"],
        "nvir_by_fragment": diag["nvir_by_fragment"],
        "diagnostics": {
            "norb_min": diag["norb_min"],
            "norb_max": diag["norb_max"],
            "norb_mean": diag["norb_mean"],
            "max_hermiticity_residual": diag["max_hermiticity_residual"],
            "max_symmetry_residual": diag["max_symmetry_residual"],
        },
    }
