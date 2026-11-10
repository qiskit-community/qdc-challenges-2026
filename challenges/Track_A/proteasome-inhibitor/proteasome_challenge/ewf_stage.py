"""
Stage 2 — EWF embedding, fragmentation, bath construction, and Hamiltonian export as ONE operation.

The ``EWFStageMixin`` provides:
* ``ewf_status(molecule, hf_config, ewf_config)``  read-only: what exists for this exact specification and is it trustworthy?
* ``ensure_ewf(molecule, hf_config, ewf_config)``  validate -> reuse a verified result, else (if authorized) submit or reconnect ->
                                                   reject invalid -> persist -> publish to the run (if run_ctx exists)
* ``load_ewf_record(ligand)``                      the verified Stage 2 record for a Challenge ligand.

Identity, integrity and agreement:
* identity   ``stage2_param_digest``  SHA-256 of the canonical scientific specification (``stage2_spec``)
* integrity  SHA-256 of ``ewf_dumpfile.h5``, recorded in the sidecar and published artifact reference
* agreement  numerical validation gates (coverage, dimensions, Hermiticity, ERI 8-fold symmetry, orthonormality)
"""
from __future__ import annotations

import base64
import binascii
import gzip
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from proteasome_challenge import BASIS, FCI_THRESHOLD, LIGANDS, SQD_THRESHOLD, ChallengeError
from proteasome_challenge.hf_stage import (
    ABSENT,
    INVALID,
    PENDING,
    STALE,
    VERIFIED,
    VERIFIED_LEGACY,
    challenge_preset,
)
from proteasome_challenge.solver_assignments import SolverAllocationPolicy, assign_solvers_by_size
from serverless_bundle.steps.stage1_spec import HFConfig, MoleculeSpec, Stage1Spec, sha256_file
from serverless_bundle.steps.stage2_spec import (
    RESULT_SCHEMA,
    BathConfig,
    EWFConfig,
    FragmentationConfig,
    Stage2Spec,
    Stage2SpecError,
    Stage2ValidationError,
    build_stage2_result_record,
    diff_stage2_specs,
    validate_ewf_dumpfile,
)

__all__ = [
    "FragmentationConfig", "BathConfig", "EWFConfig", "Stage2Spec",
    "Stage2SpecError", "Stage2ValidationError",
    "EWFError", "EWFNotAvailable", "EWFStatus", "EWFResult", "EWFStageMixin",
    "ewf_challenge_preset", "ewf_catalog_from_result",
    "VERIFIED", "VERIFIED_LEGACY", "PENDING", "ABSENT", "INVALID", "STALE",
]

_TERMINAL_FAILURE = frozenset({"ERROR", "FAILED", "CANCELED", "CANCELLED"})


class EWFError(ChallengeError):
    """A Stage 2 EWF result is missing, inconsistent or unsafe to reuse."""


class EWFNotAvailable(EWFError):
    """No verified Stage 2 result exists and submitting paid work was not authorized."""


# ─────────────────────────────────────────────────────────────────────────────
#  Challenge preset
# ─────────────────────────────────────────────────────────────────────────────

def ewf_challenge_preset(
    bundle_dir: str | Path,
    ligand: str = "ixazomib",
    *,
    stage1_spec: Optional[Stage1Spec] = None,
) -> Stage2Spec:
    """The canonical Proteasome Challenge Stage 2 specification.

    * Restricted closed-shell reference (RHF/sto-3g).
    * IAO atomic fragmentation.
    * MP2-augmented bath construction with truncation threshold 1e-4.
    """
    s1 = stage1_spec or challenge_preset(bundle_dir, ligand)
    ewf_cfg = EWFConfig(
        fragmentation=FragmentationConfig(scheme="atomic", basis="iao", iao_minao="auto"),
        bath=BathConfig(method="mp2", truncation=1e-4),
        solver="DUMP",
    )
    return Stage2Spec(stage1=s1, config=ewf_cfg)


# ─────────────────────────────────────────────────────────────────────────────
#  Status and Result objects
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EWFStatus:
    spec: Stage2Spec
    state: str
    namespace: str                                  # "canonical" | "digest"
    sidecar_path: Path
    dumpfile_path: Path
    provenance: Optional[str] = None                # "full" | "legacy_reconstructed"
    record: Optional[dict] = None                   # sidecar dictionary
    diagnostics: Optional[dict] = None
    problems: list[str] = field(default_factory=list)
    pending: Optional[dict] = None
    artifact_ref: Any = None
    run_checked: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.state in (VERIFIED, VERIFIED_LEGACY)

    @property
    def digest(self) -> str:
        return self.spec.digest()

    def summary(self) -> str:
        s2 = self.spec
        s1 = s2.stage1
        mol = s1.molecule
        cfg = s2.config
        lines = [
            f"Stage 2 · EWF [{cfg.fragmentation.scheme}/{cfg.fragmentation.basis} + {cfg.bath.method} bath] · {mol.name}",
            f"  spec digest {self.digest}",
            f"  bath: {cfg.bath.method.upper()} (truncation {cfg.bath.truncation:g}, kind {cfg.bath.truncation_kind})",
            f"  state: {self._state_text()}",
        ]
        if self.diagnostics:
            d = self.diagnostics
            n_frag = d.get("n_fragments", 0)
            norb_map = d.get("norb_by_fragment", {})
            fci_cnt = sum(1 for n in norb_map.values() if n <= FCI_THRESHOLD)
            ccsd_cnt = sum(1 for n in norb_map.values() if FCI_THRESHOLD < n < SQD_THRESHOLD)
            qpu_cnt = sum(1 for n in norb_map.values() if n >= SQD_THRESHOLD)
            lines += [
                f"  fragments: {n_frag} total  norb range [{d.get('norb_min')}, {d.get('norb_max')}]  mean {d.get('norb_mean', 0.0):.1f}",
                f"  size allocation: FCI (≤{FCI_THRESHOLD})={fci_cnt}, CCSD ({FCI_THRESHOLD}<n<{SQD_THRESHOLD})={ccsd_cnt}, QPU candidate (≥{SQD_THRESHOLD})={qpu_cnt}",
                f"  validation residuals: max_herm={d.get('max_hermiticity_residual', 0.0):.2e}  max_sym={d.get('max_symmetry_residual', 0.0):.2e}",
            ]
        if self.problems:
            lines.append("  problems:")
            for p in self.problems:
                lines.append(f"    * {p}")
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)

    def _state_text(self) -> str:
        if self.state == VERIFIED:
            return "VERIFIED (specification, SHA-256 and numerical validation all match)"
        if self.state == VERIFIED_LEGACY:
            return "VERIFIED LEGACY (reconstructed and validated against specification)"
        if self.state == PENDING:
            jid = (self.pending or {}).get("job_id", "unknown")
            return f"PENDING (Serverless job {jid} in flight)"
        if self.state == ABSENT:
            return "ABSENT (no checkpoint found for this specification)"
        if self.state == STALE:
            return "STALE (checkpoint belongs to a different specification)"
        if self.state == INVALID:
            return f"INVALID ({len(self.problems)} check(s) failed)"
        return self.state


@dataclass
class EWFResult:
    spec: Stage2Spec
    dumpfile_path: Path
    sidecar_path: Path
    sidecar: dict[str, Any]
    diagnostics: dict[str, Any]
    n_fragments: int
    fragment_ids: list[int]
    norb_by_fragment: dict[int, int]
    nocc_by_fragment: dict[int, int]
    artifact_ref: Any = None

    @property
    def digest(self) -> str:
        return self.spec.digest()

    @property
    def hamiltonian_hashes(self) -> dict[int, str]:
        """Hamiltonian identity of every embedded fragment, read from the Stage 2 dumpfile.

        Returns ``{fragment_id: sha256(heff.tobytes() + eris.tobytes())}`` using the
        same convention as :func:`~proteasome_challenge.classical_baseline.hamiltonian_hashes_from_dumpfile`.

        The dumpfile must exist (``ewf_status.ok`` is True) before calling this property.
        Raises :class:`EWFError` if the dumpfile is missing or contains no fragment groups.
        """
        from proteasome_challenge.classical_baseline import (
            hamiltonian_hashes_from_dumpfile, BaselineError,
        )
        if not self.dumpfile_path.is_file():
            raise EWFError(
                f"Stage 2 dumpfile not found at {self.dumpfile_path}. "
                "Run Stage 2 (ensure_ewf) before accessing hamiltonian_hashes.")
        try:
            return hamiltonian_hashes_from_dumpfile(self.dumpfile_path)
        except BaselineError as exc:
            raise EWFError(f"Could not read Hamiltonian hashes from {self.dumpfile_path}: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
#  EWFStageMixin
# ─────────────────────────────────────────────────────────────────────────────

class EWFStageMixin:
    """Mixin for ``Challenge`` providing Stage 2 EWF embedding operations."""

    def _resolve_stage2_spec(
        self,
        molecule_or_stage1: Any = None,
        hf_config: Optional[HFConfig | Mapping[str, Any]] = None,
        ewf_config: Optional[EWFConfig | Mapping[str, Any]] = None,
        *,
        ligand: str = "ixazomib",
    ) -> Stage2Spec:
        """Resolve a full Stage2Spec from various user input forms."""
        # 1. If already a Stage2Spec
        if isinstance(molecule_or_stage1, Stage2Spec):
            return molecule_or_stage1

        # 2. Stage 1 specification
        if isinstance(molecule_or_stage1, Stage1Spec):
            s1 = molecule_or_stage1
        elif hasattr(molecule_or_stage1, "spec") and isinstance(getattr(molecule_or_stage1, "spec"), Stage1Spec):
            s1 = molecule_or_stage1.spec
        elif molecule_or_stage1 is None and ligand in LIGANDS:
            s1 = challenge_preset(getattr(self, "bundle_dir", _ROOT / "serverless_bundle"), ligand)
        elif isinstance(molecule_or_stage1, (str, MoleculeSpec)):
            if isinstance(molecule_or_stage1, str):
                if molecule_or_stage1 in LIGANDS:
                    mol_spec = MoleculeSpec(
                        name=molecule_or_stage1,
                        path=str(Path(getattr(self, "bundle_dir", _ROOT / "serverless_bundle")) / "configs" / f"system_{molecule_or_stage1}.xyz"),
                        units="angstrom", charge=0, spin=0,
                    )
                else:
                    mol_spec = MoleculeSpec(name="custom", geometry=molecule_or_stage1, units="angstrom", charge=0, spin=0)
            else:
                mol_spec = molecule_or_stage1

            h_cfg = hf_config if isinstance(hf_config, HFConfig) else (HFConfig.from_dict(hf_config) if hf_config else HFConfig(basis=BASIS))
            s1 = Stage1Spec(molecule=mol_spec, config=h_cfg)
        else:
            raise Stage2SpecError(f"cannot resolve Stage 1 specification from {molecule_or_stage1!r}")

        # 3. EWF configuration
        if ewf_config is None:
            e_cfg = EWFConfig()
        elif isinstance(ewf_config, EWFConfig):
            e_cfg = ewf_config
        elif isinstance(ewf_config, Mapping):
            e_cfg = EWFConfig.from_dict(ewf_config)
        else:
            raise Stage2SpecError(f"ewf_config must be an EWFConfig or dict, got {ewf_config!r}")

        return Stage2Spec(stage1=s1, config=e_cfg)

    def _stage2_checkpoint_paths(self, spec: Stage2Spec) -> tuple[Path, Path, str]:
        """Return (sidecar_json_path, dumpfile_h5_path, namespace)."""
        ckpt_dir = Path(getattr(self, "ckpt_dir", _ROOT / "checkpoints")) / "stage2"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        bundle_dir = Path(getattr(self, "bundle_dir", _ROOT / "serverless_bundle"))
        mol_name = spec.stage1.molecule.name
        is_canonical = False
        if mol_name in LIGANDS:
            try:
                preset = ewf_challenge_preset(bundle_dir, mol_name)
                if preset.digest() == spec.digest():
                    is_canonical = True
            except Exception:
                is_canonical = False

        if is_canonical:
            rn = f"{mol_name}_ligand_{spec.stage1.config.basis}"
            return ckpt_dir / f"{rn}.json", ckpt_dir / f"{rn}__ewf_dumpfile.h5", "canonical"

        tag = spec.tag
        return ckpt_dir / f"{tag}.json", ckpt_dir / f"{tag}__ewf_dumpfile.h5", "digest"

    def ewf_status(
        self,
        molecule_or_stage1: Any = None,
        hf_config: Optional[HFConfig | Mapping[str, Any]] = None,
        ewf_config: Optional[EWFConfig | Mapping[str, Any]] = None,
        *,
        run_ctx: Any = None,
        ligand: str = "ixazomib",
    ) -> EWFStatus:
        """Inspect the Stage 2 state without submitting work or modifying files."""
        spec = self._resolve_stage2_spec(molecule_or_stage1, hf_config, ewf_config, ligand=ligand)
        sidecar_p, h5_p, ns = self._stage2_checkpoint_paths(spec)

        st = EWFStatus(
            spec=spec,
            state=ABSENT,
            namespace=ns,
            sidecar_path=sidecar_p,
            dumpfile_path=h5_p,
            run_checked=(run_ctx is not None),
        )

        if not h5_p.is_file():
            # Check pending job
            jobs_file = sidecar_p.parent / "jobs.json"
            if jobs_file.is_file():
                try:
                    jobs = json.loads(jobs_file.read_text(encoding="utf-8"))
                    tag = spec.digest()[:12]
                    if tag in jobs or spec.stage1.molecule.name in jobs:
                        st.state = PENDING
                        st.pending = jobs.get(tag, jobs.get(spec.stage1.molecule.name))
                except Exception:
                    pass
            return st

        # Dumpfile exists - validate
        sidecar = None
        if sidecar_p.is_file():
            try:
                sidecar = json.loads(sidecar_p.read_text(encoding="utf-8"))
            except Exception as e:
                st.problems.append(f"unreadable sidecar JSON: {e}")

        # Check digest match in sidecar if present
        if sidecar is not None:
            sidecar_digest = sidecar.get("stage2_param_digest")
            if sidecar_digest and sidecar_digest != spec.digest():
                st.state = STALE
                st.problems.append(f"sidecar digest {sidecar_digest[:12]} does not match requested {spec.digest()[:12]}")
                return st

        # Validate HDF5 dumpfile contents
        try:
            diag = validate_ewf_dumpfile(h5_p, spec=spec)
            st.diagnostics = diag
            st.record = sidecar or build_stage2_result_record(spec, h5_p, validation_diag=diag)
            st.state = VERIFIED if (sidecar and sidecar.get("schema") == RESULT_SCHEMA) else VERIFIED_LEGACY
            st.provenance = "full" if st.state == VERIFIED else "legacy_reconstructed"
        except Exception as exc:
            st.state = INVALID
            st.problems.append(str(exc))

        # Check published run reference if run_ctx is provided
        if run_ctx is not None and hasattr(run_ctx, "refs"):
            ref = run_ctx.refs.get("embedding_ewf")
            if ref is not None:
                st.artifact_ref = ref
                if ref.file_sha256 != sha256_file(h5_p):
                    st.problems.append("published run artifact hash differs from local checkpoint")

        return st

    def ensure_ewf(
        self,
        molecule_or_stage1: Any = None,
        hf_config: Optional[HFConfig | Mapping[str, Any]] = None,
        ewf_config: Optional[EWFConfig | Mapping[str, Any]] = None,
        *,
        run_ctx: Any = None,
        allow_submit: bool = False,
        poll_seconds: int = 30,
        ligand: str = "ixazomib",
    ) -> EWFResult:
        """Ensure a verified Stage 2 result exists (reuse verified -> submit if authorized -> validate -> publish)."""
        spec = self._resolve_stage2_spec(molecule_or_stage1, hf_config, ewf_config, ligand=ligand)
        status = self.ewf_status(spec, run_ctx=run_ctx)

        # 1. Reuse existing verified checkpoint
        if status.ok:
            print(f"[ensure_ewf] Reusing verified Stage 2 checkpoint for {spec.stage1.molecule.name} (digest {spec.digest()[:12]})")
            res = self._build_ewf_result_from_status(status)
            if run_ctx is not None:
                self._publish_stage2_to_run(run_ctx, res, spec)
            return res

        # 2. Check submission authorization
        if not allow_submit:
            raise EWFNotAvailable(
                f"No verified Stage 2 result exists for {spec.stage1.molecule.name} (digest {spec.digest()[:12]}), "
                f"and allow_submit=False. Status: {status.state}. Problems: {status.problems}"
            )

        # 3. Ensure Stage 1 mean-field is available
        hf_status = self.hf_status(spec.stage1.molecule, spec.stage1.config, run_ctx=run_ctx)
        if not hf_status.ok:
            print(f"[ensure_ewf] Stage 1 mean-field not ready ({hf_status.state}); running ensure_hf...")
            self.ensure_hf(spec.stage1.molecule, spec.stage1.config, run_ctx=run_ctx, allow_submit=True, poll_seconds=poll_seconds)
            hf_status = self.hf_status(spec.stage1.molecule, spec.stage1.config, run_ctx=run_ctx)
            if not hf_status.ok:
                raise EWFError(f"Failed ensuring Stage 1 reference for Stage 2: {hf_status.problems}")

        # 4. Execute Stage 2 (Serverless or local fallback)
        print(f"[ensure_ewf] Computing EWF fragmentation for {spec.stage1.molecule.name} (digest {spec.digest()[:12]})...")
        h5_p = status.dumpfile_path
        sidecar_p = status.sidecar_path

        self._execute_stage2_calculation(spec, hf_status, h5_p, sidecar_p, poll_seconds=poll_seconds)

        # 5. Validate output
        diag = validate_ewf_dumpfile(h5_p, spec=spec)
        sidecar = build_stage2_result_record(spec, h5_p, stage1_result=hf_status.record, validation_diag=diag)
        sidecar_p.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        res = EWFResult(
            spec=spec,
            dumpfile_path=h5_p,
            sidecar_path=sidecar_p,
            sidecar=sidecar,
            diagnostics=diag,
            n_fragments=diag["n_fragments"],
            fragment_ids=diag["fragment_ids"],
            norb_by_fragment=diag["norb_by_fragment"],
            nocc_by_fragment=diag["nocc_by_fragment"],
        )

        # 6. Publish to run if active
        if run_ctx is not None:
            self._publish_stage2_to_run(run_ctx, res, spec)

        return res

    def _execute_stage2_calculation(
        self,
        spec: Stage2Spec,
        hf_status: Any,
        dest_h5: Path,
        dest_sidecar: Path,
        poll_seconds: int = 30,
    ) -> None:
        """Execute Stage 2 via Serverless client or local fallback runner."""
        client = getattr(self, "client", None)
        pkl_path = hf_status.pkl_path

        # If serverless client is present and connected
        if client is not None and hasattr(client, "run"):
            with open(pkl_path, "rb") as fh:
                mf_bytes = fh.read()
            mf_b64 = base64.b64encode(mf_bytes).decode("ascii")

            job_args = {
                "run_name": f"{spec.stage1.molecule.name}_ligand_{spec.stage1.config.basis}",
                "mf_pkl_b64": mf_b64,
                "ewf_config": spec.config.to_dict(),
                "bath_type": spec.config.bath.method,
                "truncation": spec.config.bath.truncation,
                "fragmentation": spec.config.fragmentation.scheme,
                "verbose": spec.config.verbose,
            }

            pattern = getattr(self, "serverless_patterns", {}).get(2, {})
            title = pattern.get("title", "qfm-proteasome-stage2-fragments")
            print(f"[ensure_ewf] Submitting Serverless Stage 2 job: {title}")
            job = client.run(title, arguments=job_args)
            print(f"[ensure_ewf] Job submitted: {job.job_id}. Polling (interval {poll_seconds}s)...")
            res = job.result()
            if not isinstance(res, dict) or "dumpfile_b64" not in res:
                raise EWFError(f"Serverless Stage 2 job {job.job_id} returned invalid result payload: {res}")

            raw_gz = base64.b64decode(res["dumpfile_b64"])
            raw_h5 = gzip.decompress(raw_gz)
            dest_h5.write_bytes(raw_h5)
            return

        # Local fallback execution
        from serverless_bundle.steps.step2_fragments import run_fragments
        with tempfile.TemporaryDirectory(prefix="ewf_stage2_") as tmpdir:
            tmp_data = Path(tmpdir) / "data"
            tmp_data.mkdir(parents=True, exist_ok=True)
            shutil.copy(pkl_path, tmp_data / "mf_data.pkl")

            out = run_fragments(
                str(tmp_data),
                f"{spec.stage1.molecule.name}_stage2",
                ewf_config=spec.config,
                spec=spec,
            )
            shutil.copy(out["dumpfile"], dest_h5)

    def _build_ewf_result_from_status(self, status: EWFStatus) -> EWFResult:
        diag = status.diagnostics or validate_ewf_dumpfile(status.dumpfile_path, spec=status.spec)
        rec = status.record or build_stage2_result_record(status.spec, status.dumpfile_path, validation_diag=diag)
        return EWFResult(
            spec=status.spec,
            dumpfile_path=status.dumpfile_path,
            sidecar_path=status.sidecar_path,
            sidecar=rec,
            diagnostics=diag,
            n_fragments=diag["n_fragments"],
            fragment_ids=diag["fragment_ids"],
            norb_by_fragment=diag["norb_by_fragment"],
            nocc_by_fragment=diag["nocc_by_fragment"],
            artifact_ref=status.artifact_ref,
        )

    def _publish_stage2_to_run(self, run_ctx: Any, result: EWFResult, spec: Stage2Spec) -> None:
        from proteasome_challenge.run_control_plane import publish_stage2
        try:
            ref = publish_stage2(
                run_ctx,
                result.dumpfile_path,
                meta_json_path=result.sidecar_path if result.sidecar_path.is_file() else None,
                ligand=spec.stage1.molecule.name,
                basis=spec.stage1.config.basis,
                stage2_param_digest=spec.digest(),
            )
            result.artifact_ref = ref
            print(f"[ensure_ewf] Published Stage 2 artifact to run {run_ctx.run_id}: sha256 {ref.file_sha256[:12]}...")
        except Exception as exc:
            print(f"[ensure_ewf] Warning: failed publishing Stage 2 artifact to run: {exc}")

    def load_ewf_record(self, ligand: str = "ixazomib") -> EWFResult:
        """Load the verified Stage 2 result for a Challenge ligand."""
        preset = ewf_challenge_preset(getattr(self, "bundle_dir", _ROOT / "serverless_bundle"), ligand)
        st = self.ewf_status(preset, ligand=ligand)
        if not st.ok:
            raise EWFError(f"No verified Stage 2 record found for {ligand}. Status: {st.state}, Problems: {st.problems}")
        return self._build_ewf_result_from_status(st)


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helper: EWF result → fragment catalog DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def ewf_catalog_from_result(result: "EWFResult") -> "Any":
    """Build a minimal fragment-catalog DataFrame directly from an :class:`EWFResult`.

    This converts the data already present in Stage 2's ``norb_by_fragment``
    and ``nocc_by_fragment`` dictionaries into the same shape used by
    :meth:`~proteasome_challenge.Challenge.load_fragment_catalog`, making the
    notebook fragment-catalog cell work for an arbitrary molecular system
    without touching the protected challenge-data CSV.

    Columns returned: ``fragment_id``, ``n_orb``, ``n_occ``, ``n_vir``.
    The ``ligand`` column is set to the molecule name from the spec.

    Size-band indicator columns (``classical_fci_candidate``,
    ``classical_ccsd_candidate``, ``qpu_eligible``) are derived from
    ``FCI_THRESHOLD`` / ``SQD_THRESHOLD`` exactly as the main catalog loader
    does, so downstream code that inspects them behaves consistently.

    Parameters
    ----------
    result : EWFResult
        A verified Stage 2 result (returned by ``ensure_ewf`` or
        ``ewf_status`` when ``ok`` is True).

    Returns
    -------
    pandas.DataFrame
        One row per fragment, sorted by ``fragment_id``.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pandas is required for ewf_catalog_from_result") from exc

    rows = []
    for fid in sorted(result.fragment_ids):
        n_orb = int(result.norb_by_fragment.get(fid, 0))
        n_occ = int(result.nocc_by_fragment.get(fid, 0))
        rows.append({
            "fragment_id": fid,
            "n_orb": n_orb,
            "n_occ": n_occ,
            "n_vir": max(0, n_orb - n_occ),
        })

    mol_name = result.spec.stage1.molecule.name if result.spec is not None else "unknown"
    df = pd.DataFrame(rows)
    df.insert(0, "ligand", mol_name)
    df["classical_fci_candidate"] = df["n_orb"] <= FCI_THRESHOLD
    df["qpu_eligible"]            = df["n_orb"] >= SQD_THRESHOLD
    df["classical_ccsd_candidate"] = ~(df["classical_fci_candidate"] | df["qpu_eligible"])
    return df
