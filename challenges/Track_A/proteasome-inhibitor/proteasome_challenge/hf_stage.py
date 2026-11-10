"""
Stage 0/1 — molecular specification and Hartree-Fock as ONE operation.

The ``Challenge`` mixin below provides

* ``hf_status(molecule, config)``  read-only: what exists for this exact specification and is it trustworthy?
* ``ensure_hf(molecule, config)``  validate -> reuse a verified result, else (if authorized) submit or reconnect ->
                                   reject an invalid result -> persist -> publish to the run (only if a run exists)
* ``hf_record(ligand)``            the verified Stage 1 record of a Challenge ligand, for downstream consumers

Identity, integrity and agreement are three different things and are never conflated:

* identity   ``stage1_param_digest``  SHA-256 of the canonical scientific specification (``stage1_spec``)
* integrity  SHA-256 of ``mf_data.pkl``, recorded in the sidecar and in the published artifact reference
* agreement  numerical tolerances (orthonormality, electron trace, recomputed energy, SCF stationarity)

A checkpoint is reused only if ALL of them hold.  Nothing here ever deletes, rewrites or "upgrades" an existing
checkpoint, sidecar, published reference or manifest.

Namespaces
----------
``canonical``  ``checkpoints/stage1/<ligand>_ligand_sto-3g[__mf_data.pkl|.json]`` — the file names Stage 2/3 read.  Used
               only for a specification whose digest equals a Challenge preset.  Checkpoints written before the digest
               existed live here; they carry no digest and are admitted ONLY by independent reconstruction (below).
``digest``     ``checkpoints/stage1/<name>__<digest12>[__mf_data.pkl|.json]`` — every other specification.

Legacy checkpoints
------------------
A legacy sidecar has no charge, spin, digest or software version.  It is accepted solely because the *pickled PySCF
objects themselves* reconstruct (``spec_from_pyscf``) to exactly the requested specification and pass the same numerical
validation as a new result, and — when the run already published the artifact — the published SHA-256 equals the file's.
The result is labelled ``legacy_reconstructed``: the missing provenance (PySCF version, creation time) is NOT claimed.

Downstream qualification
------------------------
Stage 2 onward is qualified for the Challenge presets only.  ``stage2_qualification`` states, per specification, whether
it is qualified and why not; ``collect_manifest_inputs`` refuses an unqualified one.
"""
from __future__ import annotations

import base64
import binascii
import errno
import hashlib
import json
import math
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from proteasome_challenge import BASIS, LIGANDS, ChallengeError  # noqa: E402
from serverless_bundle.steps.stage1_spec import (  # noqa: E402
    RESULT_SCHEMA,
    HFConfig,
    MoleculeSpec,
    Stage1Spec,
    Stage1SpecError,
    Stage1ValidationError,
    diff_specs,
    sha256_bytes,
    sha256_file,
    validate_scf_state,
)

__all__ = [
    "MoleculeSpec", "HFConfig", "Stage1Spec", "Stage1SpecError",
    "HFError", "HFNotAvailable", "HFStatus", "HartreeFockMixin",
    "challenge_preset", "stage2_qualification",
    "VERIFIED", "VERIFIED_LEGACY", "PENDING", "ABSENT", "INVALID", "STALE",
]

VERIFIED = "verified"                  # new-schema sidecar, digest + SHA-256 + numerics all pass
VERIFIED_LEGACY = "verified_legacy"    # pre-digest checkpoint, independently reconstructed + numerics pass
PENDING = "pending"                    # a submitted job is recorded; no result yet
ABSENT = "absent"                      # nothing for this specification
INVALID = "invalid"                    # files exist but fail a check; never reused, never overwritten
STALE = "stale"                        # the canonical files belong to a DIFFERENT specification

_ENERGY_AGREEMENT = 1e-8               # Ha: sidecar energy vs the stored mean-field state (same as _FLOAT_TOL)
_TERMINAL_FAILURE = frozenset({"ERROR", "FAILED", "CANCELED", "CANCELLED"})


class HFError(ChallengeError):
    """A Stage 1 result is missing, inconsistent or unsafe to reuse."""


class HFNotAvailable(HFError):
    """No verified result exists and submitting paid work was not authorized."""


# ─────────────────────────────────────────────────────────────────────────────
#  Challenge preset and downstream qualification
# ─────────────────────────────────────────────────────────────────────────────

def challenge_preset(bundle_dir: str | Path, ligand: str = "ixazomib") -> Stage1Spec:
    """The Proteasome Challenge Stage 0/1 specification for ``ligand`` (default ixazomib).

    The only place a ligand name is tied to a geometry: ``<bundle>/configs/system_<ligand>.xyz``, Angstrom, neutral
    singlet, RHF/sto-3g with the default SCF settings.  Researchers use their own ``MoleculeSpec``/``HFConfig``.
    """
    if ligand not in LIGANDS:
        raise HFError(f"{ligand!r} is not a Challenge ligand; the preset covers {', '.join(LIGANDS)}")
    xyz = Path(bundle_dir) / "configs" / f"system_{ligand}.xyz"
    return Stage1Spec(MoleculeSpec(name=ligand, path=str(xyz), units="angstrom", charge=0, spin=0),
                      HFConfig(method="RHF", basis=BASIS))


def _presets(bundle_dir: Path) -> dict[str, Stage1Spec]:
    out: dict[str, Stage1Spec] = {}
    for lig in LIGANDS:
        try:
            out[lig] = challenge_preset(bundle_dir, lig)
        except (Stage1SpecError, HFError):
            continue                                   # a missing geometry file just means that preset is unavailable
    return out


def _canonical_ligand(spec: Stage1Spec, bundle_dir: Path) -> Optional[str]:
    digest = spec.digest()
    return next((lig for lig, p in _presets(bundle_dir).items() if p.digest() == digest), None)


def stage2_qualification(spec: Stage1Spec, bundle_dir: str | Path) -> tuple[bool, list[str]]:
    """``(qualified, reasons)``: may Stage 2 and everything after it consume a result for ``spec``?

    Qualified iff ``spec`` is IDENTICAL (same digest) to a Challenge preset.  Anything else is a valid Stage 1
    calculation that no downstream stage has been qualified for; the differences are listed.
    """
    bundle_dir = Path(bundle_dir)
    presets = _presets(bundle_dir)
    digest = spec.digest()
    if any(p.digest() == digest for p in presets.values()):
        return True, []
    gdig = spec.molecule.geometry_digest()
    near = next((p for p in presets.values() if p.molecule.geometry_digest() == gdig), None)
    if near is None:
        return False, [
            "the molecule is not a Challenge ligand: Stage 2 fragmentation, the fragment-count and solver contracts "
            "and the reference protocol are qualified only for ixazomib and bortezomib"]
    return False, ["differs from the qualified preset in " + "; ".join(diff_specs(spec, near)) +
                   " (valid at Stage 1, not qualified downstream)"]


# ─────────────────────────────────────────────────────────────────────────────
#  Status object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HFStatus:
    spec: Stage1Spec
    state: str
    namespace: str                                   # "canonical" | "digest"
    sidecar_path: Path
    pkl_path: Path
    provenance: Optional[str] = None                 # "full" | "legacy_reconstructed"
    record: Optional[dict] = None                    # flat verified record (see _record)
    diagnostics: Optional[dict] = None
    problems: list = field(default_factory=list)
    pending: Optional[dict] = None
    qualified: bool = False
    qualification_reasons: list = field(default_factory=list)
    canonical_ligand: Optional[str] = None           # set when the specification IS a Challenge preset
    artifact_ref: Any = None                         # ArtifactReference published for the run, if it matches
    run_checked: bool = False                        # a run context was supplied
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.state in (VERIFIED, VERIFIED_LEGACY)

    @property
    def digest(self) -> str:
        return self.spec.digest()

    def summary(self) -> str:
        s, m, c = self.spec, self.spec.molecule, self.spec.config
        lines = [f"Stage 1 · {c.method}/{c.basis} · {m.describe()}",
                 f"  spec digest {self.digest}",
                 f"  state: {self._state_text()}"]
        if self.record and self.diagnostics:
            r, d = self.record, self.diagnostics
            lines += [
                f"  E_HF = {r['hf_energy']:.10f} Ha   converged = {r['hf_converged']} "
                f"({d.get('scf_cycles')} cycles)   conv_tol {c.conv_tol:g}   init_guess {c.init_guess}",
                f"  electrons {r['n_electrons']} ({d['n_alpha']} alpha, {d['n_beta']} beta)   charge {r['charge']:+d}   "
                f"2S {r['spin']}   n_AO {r['n_ao']}   occupied {d['n_occupied']} / virtual {d['n_virtual']}   "
                f"occupations {d['occupation_values']}",
            ]
            if d.get("homo_energy") is not None and d.get("lumo_energy") is not None:
                lines.append(f"  HOMO {d['homo_energy']:.6f} Ha   LUMO {d['lumo_energy']:.6f} Ha   "
                             f"gap {d['homo_lumo_gap']:.6f} Ha")
            checks = [f"orthonormality {d['orthonormality_residual']:.1e} (tol {d['orthonormality_tolerance']:.1e})",
                      f"tr(DS)-N {d['electron_trace_error']:.1e}"]
            if "orbital_gradient_norm" in d:
                checks.append(f"orbital gradient {d['orbital_gradient_norm']:.1e} (tol {d['thresholds']['stationarity']:.1e})")
                checks.append(f"energy recomputed from density: {d['energy_recomputed_error']:.1e} Ha")
            lines.append("  checks: " + "; ".join(checks))
            lines.append(f"  geometry digest {r.get('geometry_digest')}")
            if r.get("pyscf_version"):
                lines.append(f"  produced with PySCF {r['pyscf_version']} at {r.get('completed_utc')}")
        for p in self.problems:
            lines.append(f"  PROBLEM: {p}")
        if self.pending:
            lines.append(f"  pending job {self.pending.get('job_id')} (submitted {self.pending.get('submitted_utc')})")
        lines.append("  downstream (Stage 2+): " + ("QUALIFIED" if self.qualified else
                                                    "NOT QUALIFIED — " + " | ".join(self.qualification_reasons)))
        if self.run_checked:
            if self.artifact_ref is not None:
                lines.append(f"  run artifact: published {self.artifact_ref.artifact_name} "
                             f"sha256 {self.artifact_ref.file_sha256[:16]}...")
            else:
                lines.append("  run artifact: not published to the run" if self.ok else "  run artifact: none")
        else:
            lines.append("  run artifact: no run context (validated local result only)")
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)

    def _state_text(self) -> str:
        return {
            VERIFIED: "VERIFIED (full provenance: digest, SHA-256 and numerical validation)",
            VERIFIED_LEGACY: ("VERIFIED (legacy checkpoint: spec reconstructed from the stored PySCF objects and "
                              "numerically validated; digest, PySCF version and creation time were never recorded)"),
            PENDING: "PENDING (a submitted job has not been collected)",
            ABSENT: "ABSENT (no result for this specification)",
            INVALID: "INVALID (files exist but fail validation; not reused, not overwritten)",
            STALE: "STALE (the checkpoint files belong to a different specification; not reused, not overwritten)",
        }[self.state]


# ─────────────────────────────────────────────────────────────────────────────
#  File helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_once_bytes(path: Path, data: bytes) -> bool:
    """Create ``path`` with ``data`` atomically; never replace.  An identical existing file is fine (False)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".hf_", suffix=".part", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(str(tmp), str(path))
            return True
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
        if sha256_file(path) != sha256_bytes(data):
            raise HFError(f"refusing to replace {path.name}: it exists with different content")
        return False
    finally:
        tmp.unlink(missing_ok=True)


def _json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_sidecar(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HFError(f"Stage 1 sidecar {path.name} is unreadable or not valid JSON") from exc
    if not isinstance(data, dict):
        raise HFError(f"Stage 1 sidecar {path.name} is not a JSON object")
    return data


def _load_state(pkl_path: Path):
    """Unpickle a mean-field checkpoint (trusted local file; SHA-256 is verified first where one is recorded)."""
    try:
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        mol, mf = data["mol"], data["mf"]
    except Exception as exc:                                   # noqa: BLE001 - any failure means "not usable"
        raise HFError(f"{pkl_path.name} cannot be loaded as a mean-field checkpoint ({type(exc).__name__})") from exc
    for obj in (mol, mf):
        try:
            obj.verbose = 0                                    # quiet the loaded copy only
        except Exception:                                      # noqa: BLE001
            pass
    return mol, mf


# ─────────────────────────────────────────────────────────────────────────────
#  Verification of one candidate (sidecar + pickle) against a specification
# ─────────────────────────────────────────────────────────────────────────────

def _verify_candidate(spec: Stage1Spec, sidecar: Mapping[str, Any], pkl_path: Path, *,
                      expect_system: Optional[str], stationarity: bool) -> tuple[str, dict, dict, list]:
    """Returns ``(provenance, record, diagnostics, problems)``; a non-empty ``problems`` means: do not use."""
    problems: list[str] = []
    full = sidecar.get("schema") == RESULT_SCHEMA
    if sidecar.get("stage") != "hf":
        problems.append(f"sidecar stage is {sidecar.get('stage')!r}, expected 'hf'")
    if sidecar.get("hf_converged") is not True:
        problems.append("the sidecar does not report a converged SCF")
    if not pkl_path.is_file() or pkl_path.stat().st_size == 0:
        problems.append("the mean-field pickle is missing or empty")
    if full:
        digest = spec.digest()
        if sidecar.get("stage1_param_digest") != digest:
            problems.append("the sidecar digest differs from the requested specification")
        try:
            if Stage1Spec.from_dict(sidecar["spec"]).digest() != sidecar.get("stage1_param_digest"):
                problems.append("the sidecar's embedded specification does not reproduce its own digest")
        except (KeyError, TypeError, ValueError):
            problems.append("the sidecar's embedded specification is malformed")
        recorded = (sidecar.get("artifact") or {}).get("mf_data_pkl_sha256")
        if not problems and (not isinstance(recorded, str) or sha256_file(pkl_path) != recorded):
            problems.append("the pickle SHA-256 differs from the one recorded in the sidecar (integrity)")
    else:
        if expect_system is None or sidecar.get("system") != expect_system:
            problems.append(f"legacy sidecar belongs to system {sidecar.get('system')!r}, not {expect_system!r}")
        if sidecar.get("basis") != spec.config.basis:
            problems.append(f"legacy sidecar basis {sidecar.get('basis')!r} differs from {spec.config.basis!r}")
    if problems:
        return ("full" if full else "legacy_reconstructed"), {}, {}, problems

    mol, mf = _load_state(pkl_path)
    try:
        diag = validate_scf_state(mf, mol, spec, stationarity=stationarity)
    except Stage1ValidationError as exc:
        return ("full" if full else "legacy_reconstructed"), {}, {}, [str(exc)]
    try:
        if abs(float(sidecar["hf_energy"]) - float(mf.e_tot)) > _ENERGY_AGREEMENT:
            problems.append("the sidecar energy disagrees with the stored mean-field state")
        if int(sidecar["n_ao"]) != int(mol.nao) or int(sidecar["n_electrons"]) != int(mol.nelectron):
            problems.append("the sidecar n_ao / n_electrons disagree with the stored mean-field state")
    except (KeyError, TypeError, ValueError):
        problems.append("the sidecar lacks energy / n_ao / n_electrons")
    execution = sidecar.get("execution") or {}
    record = {
        "hf_energy": float(mf.e_tot), "hf_converged": bool(mf.converged), "basis": spec.config.basis,
        "n_ao": int(mol.nao), "n_electrons": int(mol.nelectron), "charge": int(mol.charge), "spin": int(mol.spin),
        "reference": spec.config.method, "source": "stage1_checkpoint",
        "provenance": "full" if full else "legacy_reconstructed",
        "stage1_param_digest": spec.digest(), "geometry_digest": spec.molecule.geometry_digest(),
        "pyscf_version": execution.get("pyscf_version") if full else None,
        "completed_utc": execution.get("completed_utc") if full else None,
        "mf_data_pkl_sha256": sha256_file(pkl_path),
    }
    return ("full" if full else "legacy_reconstructed"), record, diag, problems


def _decode_remote(spec: Stage1Spec, result: Any) -> tuple[dict, bytes]:
    """Check a Serverless result structurally and return ``(sidecar, pickle_bytes)``; raises ``HFError``."""
    if not isinstance(result, dict):
        raise HFError("the Stage 1 job returned no result record")
    sidecar = {k: v for k, v in result.items() if k != "mf_pkl_b64"}
    problems = []
    if sidecar.get("schema") != RESULT_SCHEMA:
        problems.append(f"result schema {sidecar.get('schema')!r} is not {RESULT_SCHEMA!r} (stale deployed function?)")
    if sidecar.get("stage1_param_digest") != spec.digest():
        problems.append("the result was computed for a different specification (digest mismatch)")
    if sidecar.get("hf_converged") is not True:
        problems.append("the SCF did not converge")
    energy = sidecar.get("hf_energy")
    if not isinstance(energy, (int, float)) or not math.isfinite(float(energy)):
        problems.append("the HF energy is not finite")
    if problems:
        raise HFError("Stage 1 result rejected: " + "; ".join(problems))
    try:
        raw = base64.b64decode(result["mf_pkl_b64"], validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise HFError("Stage 1 result rejected: the mean-field pickle is missing or not valid base64") from exc
    recorded = (sidecar.get("artifact") or {}).get("mf_data_pkl_sha256")
    if not raw or hashlib.sha256(raw).hexdigest() != recorded:
        raise HFError("Stage 1 result rejected: the transferred pickle does not match its recorded SHA-256")
    return sidecar, raw


# ─────────────────────────────────────────────────────────────────────────────
#  The Challenge mixin
# ─────────────────────────────────────────────────────────────────────────────

class HartreeFockMixin:
    """Stage 0/1 operations of :class:`proteasome_challenge.Challenge`."""

    # ── paths ────────────────────────────────────────────────────────────────
    def _hf_location(self, spec: Stage1Spec) -> tuple[str, Path, Path, Optional[str]]:
        """``(namespace, sidecar, pickle, canonical_ligand)`` for ``spec``."""
        lig = _canonical_ligand(spec, self.bundle_dir)
        if lig is not None:
            meta_p, pkl_p = self._stage1_paths(lig)
            return "canonical", meta_p, pkl_p, lig
        base = self.ckpt_dir / "stage1"
        return "digest", base / f"{spec.tag}.json", base / f"{spec.tag}__mf_data.pkl", None

    def _hf_pending_path(self, spec: Stage1Spec) -> Path:
        return self.ckpt_dir / "stage1" / "pending" / f"{spec.digest()[:12]}.json"

    def _hf_pending(self, spec: Stage1Spec) -> Optional[dict]:
        p = self._hf_pending_path(spec)
        if not p.is_file():
            return None
        data = _read_sidecar(p)
        if data.get("stage1_param_digest") != spec.digest() or not isinstance(data.get("job_id"), str):
            raise HFError(f"pending-job record {p.name} is inconsistent; it is not used")
        return data

    @staticmethod
    def _as_spec(molecule: Any, config: Optional[HFConfig]) -> Stage1Spec:
        if isinstance(molecule, Stage1Spec):
            if config is not None:
                raise TypeError("pass either a Stage1Spec or (molecule, config)")
            return molecule
        if not isinstance(molecule, MoleculeSpec):
            raise TypeError("molecule must be a MoleculeSpec")
        cfg = HFConfig() if config is None else config
        if not isinstance(cfg, HFConfig):
            raise TypeError("config must be an HFConfig")
        return Stage1Spec(molecule, cfg)

    # ── read-only inspection ─────────────────────────────────────────────────
    def hf_status(self, molecule: Any, config: Optional[HFConfig] = None, *, run_ctx: Any = None,
                  stationarity: bool = True) -> HFStatus:
        """What exists for this exact specification, and can it be trusted?  Never submits, writes or publishes."""
        spec = self._as_spec(molecule, config)
        namespace, meta_p, pkl_p, lig = self._hf_location(spec)
        qualified, reasons = stage2_qualification(spec, self.bundle_dir)
        st = HFStatus(spec=spec, state=ABSENT, namespace=namespace, sidecar_path=meta_p, pkl_path=pkl_p,
                      qualified=qualified, qualification_reasons=reasons, run_checked=run_ctx is not None,
                      canonical_ligand=lig)
        st.pending = self._hf_pending(spec)

        if not meta_p.is_file():
            if st.pending:
                st.state = PENDING
            elif pkl_p.is_file():
                st.note = "an orphan mean-field pickle without a sidecar was found; it is ignored (never trusted)"
            return st
        try:
            sidecar = _read_sidecar(meta_p)
        except HFError as exc:
            st.state, st.problems = INVALID, [str(exc)]
            return st
        if sidecar.get("schema") == RESULT_SCHEMA and sidecar.get("stage1_param_digest") != spec.digest():
            st.state = STALE
            st.problems = [f"the checkpoint {meta_p.name} was produced for a different specification "
                           f"({str(sidecar.get('stage1_param_digest'))[:12]}... vs {spec.digest()[:12]}...)"]
            return st
        provenance, record, diag, problems = _verify_candidate(
            spec, sidecar, pkl_p, expect_system=lig, stationarity=stationarity)
        st.provenance = provenance
        if problems:
            wrong_spec = any("does not correspond to the requested specification" in p for p in problems)
            st.state, st.problems = (STALE if wrong_spec else INVALID), problems
            return st
        st.record, st.diagnostics = record, diag
        st.state = VERIFIED if provenance == "full" else VERIFIED_LEGACY
        if run_ctx is not None:
            self._attach_published_ref(st, run_ctx)
        return st

    @staticmethod
    def _attach_published_ref(st: HFStatus, run_ctx: Any) -> None:
        from proteasome_challenge.run_control_plane import StageSourceError, load_stage_ref
        try:
            ref = load_stage_ref(run_ctx, "meanfield")
        except StageSourceError as exc:
            st.problems.append(str(exc))
            return
        if ref is None:
            return
        recorded = (ref.meta or {}).get("stage1_param_digest")
        if ref.file_sha256 != st.record["mf_data_pkl_sha256"]:
            st.problems.append("the artifact published for this run differs from this checkpoint "
                               f"({ref.file_sha256[:12]}... vs {st.record['mf_data_pkl_sha256'][:12]}...)")
        elif recorded is not None and recorded != st.digest:
            st.problems.append("the artifact published for this run belongs to a different specification")
        else:
            st.artifact_ref = ref

    # ── the single operation ─────────────────────────────────────────────────
    def ensure_hf(self, molecule: Any, config: Optional[HFConfig] = None, *, run_ctx: Any = None,
                  allow_submit: bool = False, poll_seconds: int = 30) -> HFStatus:
        """Make a verified, persisted (and, when a run exists, published) Hartree-Fock result available.

        1. validate the specification;
        2. reuse a verified match — otherwise reconnect to this specification's pending job — otherwise, only with
           ``allow_submit=True``, submit ONE Serverless job;
        3. reject an invalid result (nothing is written);
        4. persist sidecar + pickle write-once; 5. publish to ``run_ctx`` if one is given (never created here).

        Idempotent.  An INVALID or STALE checkpoint is reported and left untouched.
        """
        spec = self._as_spec(molecule, config)
        st = self.hf_status(spec, run_ctx=run_ctx)
        if st.state in (INVALID, STALE):
            raise HFError("; ".join(st.problems) + " — an authorized new calculation needs the conflicting files "
                          "to be retired by the operator; nothing was changed")
        if not st.ok:
            if st.state == ABSENT and not allow_submit:
                raise HFNotAvailable(
                    f"no verified Hartree-Fock result exists for {spec.molecule.name} ({spec.digest()[:12]}...) and "
                    "paid work was not authorized (set RUN_STAGE1 = True to compute it). Nothing was submitted.")
            self._collect_hf(spec, st, allow_submit=allow_submit, poll_seconds=poll_seconds)
            st = self.hf_status(spec, run_ctx=run_ctx)
            if not st.ok:
                raise HFError("the persisted Stage 1 result failed re-verification: " + "; ".join(st.problems))
        if run_ctx is not None:
            if st.qualified:
                self._publish_hf(st, run_ctx)
                st = self.hf_status(spec, run_ctx=run_ctx)
            else:
                st.note = ("not published to the run: the run's single write-once 'meanfield' reference is reserved "
                           "for a specification qualified downstream")
        return st

    def abandon_hf_job(self, molecule: Any, config: Optional[HFConfig] = None, *, reason: str) -> None:
        """Forget this specification's pending job record (the only supported way to resubmit after losing a job)."""
        if not isinstance(reason, str) or not reason.strip():
            raise HFError("abandoning a pending Stage 1 job requires a reason")
        spec = self._as_spec(molecule, config)
        p = self._hf_pending_path(spec)
        if p.is_file():
            p.unlink()
            print(f"Pending Stage 1 job record removed ({reason.strip()[:200]}).")

    # ── submission / collection ──────────────────────────────────────────────
    def _collect_hf(self, spec: Stage1Spec, st: HFStatus, *, allow_submit: bool, poll_seconds: int) -> None:
        pending = st.pending
        if pending:
            job_id = pending["job_id"]
            try:
                job = self._connect().job(job_id)
            except Exception as exc:                            # noqa: BLE001
                raise HFError(
                    f"cannot reach pending Stage 1 job {job_id!r} ({type(exc).__name__}); its state is unknown, so it "
                    "is NOT resubmitted. Retry later, or call abandon_hf_job(..., reason=...) if it is known lost."
                ) from exc
            if job is None:
                raise HFError(f"pending Stage 1 job {job_id!r} is not known to the service; it is NOT resubmitted. "
                              "Call abandon_hf_job(..., reason=...) if it is known lost.")
            print(f"Stage 1 {spec.molecule.name}: reconnected to job_id={job_id}", flush=True)
        else:
            if not allow_submit:
                raise HFNotAvailable("submitting a Stage 1 job was not authorized")
            self._assert_bundle_deployed()
            job = self._load_function(1).run(function_size="s", spec=spec.to_dict())
            self._write_json_atomic(self._hf_pending_path(spec), {
                "schema": "stage1-pending/1", "stage1_param_digest": spec.digest(), "name": spec.molecule.name,
                "job_id": job.job_id, "submitted_utc": _utc_now()})
            print(f"Stage 1 {spec.molecule.name}: submitted job_id={job.job_id}", flush=True)

        try:
            result = self._wait_and_collect({spec.tag: job}, poll_seconds)[spec.tag]
        except ChallengeError:
            try:
                if str(job.status()).upper() in _TERMINAL_FAILURE:
                    self._hf_pending_path(spec).unlink(missing_ok=True)      # terminal failure: a fresh run is allowed
            except Exception:                                                # noqa: BLE001
                pass
            raise
        try:
            self._accept_remote_result(spec, st, result)
        finally:
            self._hf_pending_path(spec).unlink(missing_ok=True)              # the job is spent either way

    def _accept_remote_result(self, spec: Stage1Spec, st: HFStatus, result: Any) -> None:
        sidecar, raw = _decode_remote(spec, result)
        base = st.pkl_path.parent
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".hf_staging_", dir=base) as tmp:
            staged = Path(tmp) / "mf_data.pkl"
            staged.write_bytes(raw)
            provenance, _record_unused, _diag, problems = _verify_candidate(
                spec, sidecar, staged, expect_system=None, stationarity=True)
            if problems:
                raise HFError("Stage 1 result rejected after independent validation (nothing was persisted): "
                              + "; ".join(problems))
        _write_once_bytes(st.pkl_path, raw)                                 # pickle first, sidecar last = commit marker
        _write_once_bytes(st.sidecar_path, _json_bytes(sidecar))
        print(f"Stage 1 {spec.molecule.name}: E_HF = {sidecar['hf_energy']:.10f} Ha (checkpoint saved)", flush=True)

    # ── publication ──────────────────────────────────────────────────────────
    def _publish_hf(self, st: HFStatus, run_ctx: Any) -> None:
        from proteasome_challenge.run_control_plane import publish_stage1
        publish_stage1(run_ctx, st.pkl_path, st.sidecar_path,
                       ligand=str(st.canonical_ligand), basis=st.spec.config.basis, expected_hf=st.record)

    # ── downstream view ──────────────────────────────────────────────────────
    def hf_record(self, ligand: str) -> dict:
        """The Stage 1 record of a Challenge ``ligand`` for downstream consumers.

        Verified checkpoint (real charge/spin from the stored state) when one exists; the published reference
        ``hf_results.json`` entry only when there is no checkpoint at all.  A checkpoint that exists but fails
        verification raises — it is never silently replaced by the reference file.
        """
        meta_p, pkl_p = self._stage1_paths(ligand)
        if not meta_p.is_file() and not pkl_p.is_file():
            return {**self.load_hf_results()[ligand], "source": "published_reference", "provenance": None,
                    "stage2_qualified": True}
        st = self.hf_status(challenge_preset(self.bundle_dir, ligand), stationarity=False)
        if not st.ok:
            raise HFError(f"Stage 1 checkpoint for {ligand} is {st.state}: " + "; ".join(st.problems or [st.note]))
        return {**st.record, "stage2_qualified": st.qualified}


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
