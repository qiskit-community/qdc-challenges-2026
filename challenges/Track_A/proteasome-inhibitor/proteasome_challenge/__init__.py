"""
proteasome_challenge — participant-facing helper facade for the QDC 2026
Proteasome Inhibitor challenge.

Usage (workflow order)
----------------------
::

    from proteasome_challenge import Challenge
    lab = Challenge.from_environment()
    lab.show_status()

    # Stage A — MP2 characterisation (loads only)
    cat   = lab.load_fragment_catalog()
    prelim = lab.save_preliminary_selection()          # provisional proposal, NOT authorising SQD

    # Stage B — all-classical (mixed FCI/CCSD) EWF baseline + CCSD diagnostics
    base  = lab.run_classical_baseline("ixazomib", execution="load")   # "run" submits ONE classical job
    # Stage C — CCSD-informed assessment (loads the saved baseline; launches nothing)
    diag  = lab.diagnostic_analysis_table("ixazomib")
    # Stage D — participant workflow: deterministic size rule (n_orb >= 20), no diagnostics
    final = lab.select_fragments_by_size()
    # Stage D — research workflow: explicit, user-approved freeze (no automatic scoring/thresholds)
    final = lab.freeze_final_selection(entries, approved_by=..., selection_basis=...,
                                       sqd_calibration_fragment_id=..., confirm=True)
    # Stage E — single-fragment SQD/TrimSQD (fragment comes from the frozen selection)
    ccsd  = lab.solve_fragment("ixazomib", method="ccsd", execution="prepared")
    grid  = lab.run_tuning_grid(execution="replay")
    sel   = lab.promote_configuration("<grid label>", grid)
    # Stage F — full EWF–SQD reusing the classical baseline
    full  = lab.solve_molecule("ixazomib", promoted=sel, execution="explicit")
    eff   = lab.substitution_effects("ixazomib")
    sub   = lab.validate_submission(answers)

Everything that touches Qiskit Serverless (authentication, deployment, bundle
fingerprinting, upload patches, job creation/polling, artifact serialization,
QPU budget enforcement) lives inside this module, not in the notebook.  The
notebook cells contain scientific configuration, analysis, and checkpoint
reporting only.

QPU policy
----------
A live QPU sampling job is only ever submitted when the caller passes
``execution="live_qpu"``.  Every other execution mode that needs measurement
data (``"replay"``, ``"explicit"``) raises :class:`ChallengeError` when the
frozen/live counts file is missing; it never falls back to sampling.  The QPU
budget counter is incremented in exactly one place,
:meth:`Challenge._record_qpu_sampling`, called from
:meth:`Challenge._submit_stage3`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

__all__ = [
    "Challenge",
    "ChallengeError",
    "LIGANDS",
    "BASIS",
    "FCI_THRESHOLD",
    "SQD_THRESHOLD",
    "SELECTED_LIGAND",
    "QPU_BACKEND_ENV_VAR",
    "resolve_qpu_backend",
    "FragmentationConfig",
    "BathConfig",
    "EWFConfig",
    "Stage2Spec",
    "Stage2SpecError",
    "Stage2ValidationError",
    "EWFStatus",
    "EWFResult",
    "EWFError",
    "EWFNotAvailable",
    "ewf_challenge_preset",
    "SolverAllocationPolicy",
    "assign_solvers_by_size",
    "generate_solver_assignments",
    # There is deliberately no fragment-id constant.  The quantum-treated
    # fragments come ONLY from the user-frozen final selection
    # (Challenge.load_final_selection()).
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIGANDS = ("ixazomib", "bortezomib")
BASIS = "sto-3g"
FCI_THRESHOLD = 10       # n_orb ≤ this → FCI
SQD_THRESHOLD = 20       # n_orb ≥ this → quantum-eligible (graded threshold)
EXPECTED_FRAGMENTS = {"ixazomib": 42, "bortezomib": 53}
SELECTED_LIGAND = "ixazomib"
# No fragment-id constant exists: the SQD calibration fragment and the
# promoted fragments are read from the frozen final selection manifest.
EXPECTED_BUNDLE_API = 3

# The QPU backend is configured in ``.env`` (``IBM_QUANTUM_BACKEND``).  The literal below is
# only the fallback used when that variable is unset, so existing setups keep working.
QPU_BACKEND_ENV_VAR = "IBM_QUANTUM_BACKEND"
FALLBACK_QPU_BACKEND = "ibm_kingston"
_QPU_BACKEND_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
DEFAULT_N_SHOTS = 100_000
DEFAULT_N_REPS = 1

QUANTUM_METHODS = ("sqd", "trimsqd")           # methods that may be promoted to Section 4
FRAGMENT_METHODS = ("sqd", "trimsqd", "both", "ccsd")
QUANTUM_SOLVER_LABELS = ("SQD", "TrimSQD")     # solver labels written by step3_solve

# Numerical tolerances for validate_full_ewf
HF_TOL = 1e-5                 # Ha, Stage 3 e_hf vs Stage 1 reference
CONSISTENCY_TOL = 1e-6        # Ha, e_total = e_hf + e_corr and Σ fragment e_corr = e_corr

STAGES = {
    1: dict(title="qfm-proteasome-stage1-hf",
            entrypoint="proteasome_entrypoint_stage1_hf.py",
            description="Proteasome Stage 1: RHF → mf_pkl_b64", size="s"),
    2: dict(title="qfm-proteasome-stage2-fragments",
            entrypoint="proteasome_entrypoint_stage2_fragments.py",
            description="Proteasome Stage 2: EWF fragmentation (IAO, MP2 bath)", size="m"),
    3: dict(title="qfm-proteasome-stage3-solve",
            entrypoint="proteasome_entrypoint_stage3_solve_reconstruct.py",
            description="Proteasome Stage 3: FCI/CCSD/SQD solve + EWF reconstruction", size="l"),
    # Task 5.9C.1: independent one-fragment FCI/CCSD worker.  Registered here so that the next
    # (organizer-initiated) deploy_bundle() uploads it; nothing is deployed by importing this module.
    4: dict(title="qfm-proteasome-classical-fragment",
            entrypoint="proteasome_entrypoint_classical_fragment.py",
            description="Proteasome 5.9C: one FCI/CCSD fragment per worker (File API transport)",
            size="l"),
    # Task 5.9C.2: independent SQD/TrimSQD postprocessing worker (File API transport).
    5: dict(title="qfm-proteasome-3qc-postprocess",
            entrypoint="proteasome_entrypoint_3qc_postprocess.py",
            description="Proteasome 5.9C: one SQD/TrimSQD quantum postprocessing worker (File API transport)",
            size="m"),
    # Task 5.9D: independent Stage 4 completion-barrier + EWF reconstruction worker (CPU, transport mode).
    # Registered so that the next organizer-initiated deploy_bundle() uploads it; nothing is deployed here.
    6: dict(title="qfm-proteasome-stage4-reconstruct",
            entrypoint="proteasome_entrypoint_stage4_reconstruct.py",
            description="Proteasome 5.9D: Stage 4 EWF reconstruction from verified fragment results (CPU)",
            size="l"),
}

class ChallengeError(RuntimeError):
    """Raised when a required artifact is missing or a safety gate fails."""


def resolve_qpu_backend(explicit: str | None = None, *, require: bool = False) -> str:
    """Return the IBM backend name to use.

    Precedence: an ``explicit`` argument, then the ``IBM_QUANTUM_BACKEND`` environment
    variable (normally set in ``.env``), then :data:`FALLBACK_QPU_BACKEND`.  Resolved at
    call time, so a ``.env`` loaded by :meth:`Challenge.from_environment` is honoured.
    The name is validated (allowlist) and a ``<placeholder>`` is rejected.

    Parameters
    ----------
    explicit : str or None
        An explicit backend name that takes precedence over the environment.
    require : bool
        When True and no backend is configured (neither ``explicit`` nor the env-var),
        raise :class:`ChallengeError` with actionable instructions instead of silently
        falling back to :data:`FALLBACK_QPU_BACKEND`.  Use ``require=True`` on all
        code paths that would immediately submit a live QPU job.
    """
    if explicit is not None:
        if not isinstance(explicit, str):
            raise ChallengeError(f"qpu_backend must be a string, got {type(explicit).__name__}")
        value, origin = explicit.strip(), "qpu_backend argument"
    else:
        value, origin = os.environ.get(QPU_BACKEND_ENV_VAR, "").strip(), QPU_BACKEND_ENV_VAR
        if not value:
            if require:
                raise ChallengeError(
                    f"No QPU backend configured. Set {QPU_BACKEND_ENV_VAR} in your .env file "
                    f"(e.g. {QPU_BACKEND_ENV_VAR}=ibm_pittsburgh) before requesting live QPU sampling."
                )
            return FALLBACK_QPU_BACKEND
    if not _QPU_BACKEND_RE.fullmatch(value):
        raise ChallengeError(
            f"Invalid backend name from {origin}: {value!r}. Expected an IBM backend name such as "
            f"'{FALLBACK_QPU_BACKEND}' (letters, digits, '_', '-', '.'; at most 64 characters). "
            f"Set {QPU_BACKEND_ENV_VAR} in .env."
        )
    return value


def _by_id(mapping: dict, fragment_id: int):
    """Look up a fragment id in a dict whose keys may be int or str (JSON round trip)."""
    if int(fragment_id) in mapping:
        return mapping[int(fragment_id)]
    return mapping.get(str(int(fragment_id)))


def _coerce_fragment_id(value: Any, what: str = "fragment id") -> int:
    """Return ``value`` as an ``int`` fragment id, or raise a clear :class:`ChallengeError`.

    Accepts Python/NumPy integers and integral floats (``3.0``).  Rejects ``None``,
    booleans (``True`` would otherwise silently become the integer one), non-integral
    floats, strings and containers; ``int()`` would truncate or mis-parse several of those.
    """
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ChallengeError(f"{what} must be an integer fragment id, got {value!r}.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    raise ChallengeError(f"{what} must be an integer fragment id, got {value!r}.")


def _normalize_single_fragment(value: Any, what: str) -> int | None:
    """Normalize a scalar id, or a one-element list/tuple, to ``int`` (``None`` stays ``None``).

    The Stage 3 counts staging handles exactly one fragment per job.  An empty
    sequence is invalid, and a sequence of several ids is refused explicitly instead of
    silently using one of them (which would replay one fragment's counts for another).
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ChallengeError(f"{what}: an empty fragment-id sequence is not a valid selection.")
        ids = [_coerce_fragment_id(v, what) for v in value]
        if len(ids) > 1:
            raise ChallengeError(
                f"{what}: counts for {len(ids)} fragments {ids} were requested in one job, but "
                "Stage 3 counts staging currently supports exactly one fragment per job "
                "(multi-fragment counts staging is not implemented yet). Nothing was submitted "
                "and no counts file was used."
            )
        return ids[0]
    return _coerce_fragment_id(value, what)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main facade
# ---------------------------------------------------------------------------

from proteasome_challenge.measurement_routing import MeasurementRoutingMixin  # noqa: E402
from proteasome_challenge.protocol_client import ProtocolTransferMixin  # noqa: E402
from proteasome_challenge.hf_stage import HartreeFockMixin, challenge_preset  # noqa: E402
from proteasome_challenge.ewf_stage import (  # noqa: E402
    EWFError,
    EWFNotAvailable,
    EWFResult,
    EWFStageMixin,
    EWFStatus,
    ewf_challenge_preset,
)
from proteasome_challenge.solver_assignments import (  # noqa: E402
    SolverAllocationPolicy,
    assign_solvers_by_size,
    generate_solver_assignments,
)
from serverless_bundle.steps.stage2_spec import (  # noqa: E402
    BathConfig,
    EWFConfig,
    FragmentationConfig,
    Stage2Spec,
    Stage2SpecError,
    Stage2ValidationError,
)


class Challenge(MeasurementRoutingMixin, ProtocolTransferMixin, HartreeFockMixin, EWFStageMixin):
    """Participant-facing facade for the QDC 2026 Proteasome challenge.

    Parameters
    ----------
    notebook_dir : Path
        Directory containing the participant notebook.  Defaults to ``cwd()``.
    allow_unverified_bundle : bool
        Skip the local-vs-deployed bundle fingerprint check (use only when an
        organizer pre-deployed the Serverless functions).
    """

    def __init__(self, notebook_dir: Path | None = None, *,
                 allow_unverified_bundle: bool = False):
        self.notebook_dir = (Path(notebook_dir) if notebook_dir else Path.cwd()).resolve()
        self.data_root = self.notebook_dir / "challenge_data"
        self.bundle_dir = self.notebook_dir / "serverless_bundle"
        self.ckpt_dir = self.notebook_dir / "checkpoints"
        self.allow_unverified_bundle = bool(allow_unverified_bundle)

        # Paths to canonical data files
        self.hf_path = self.data_root / "hf_results.json"
        self.cat_path = self.data_root / "fragment_catalog.csv"
        self.benchmark_path = self.data_root / "reference_results.json"
        self.archive_path = self.data_root / "archived_lab_results.json"
        self.frozen_counts_dir = self.data_root / "frozen_counts"
        self.chemistry_path = self.data_root / "fragment_chemistry.json"
        self.assignments_path = self.data_root / "solver_assignments.json"
        # Preliminary (pre-CCSD) MP2 proposal — informational, never authorises SQD.
        self.preliminary_selection_path = self.data_root / "preliminary_mp2_selection.json"
        # FINAL CCSD-informed selection — written only by freeze_final_selection().
        # User-produced, hence under checkpoints/ (git-ignored), not challenge_data/.
        self.final_selection_path = self.ckpt_dir / "final_experiment_selection.json"

        # QPU budget gate — kernel-restart-safe (persisted to disk).
        self._qpu_record_path = self.ckpt_dir / ".qpu_sampling_record.json"
        self._qpu_sampling_count: int = self._load_qpu_count()

        # Lazy Serverless connection and upload bookkeeping
        self._serverless = None
        self._fragment_jobs: dict[str, Any] = {}      # job_id -> live handle, this kernel only
        self._uploaded: set[tuple[str, str]] = set()
        self._upload_patched = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_environment(cls, notebook_dir: Path | None = None, *,
                         allow_unverified_bundle: bool = False) -> "Challenge":
        """Create a Challenge instance and load ``.env`` credentials.

        Credentials are loaded from ``<notebook_dir>/.env`` into ``os.environ``
        using python-dotenv.  Credentials are **never** returned to the caller or
        printed.
        """
        from dotenv import load_dotenv
        inst = cls(notebook_dir=notebook_dir, allow_unverified_bundle=allow_unverified_bundle)
        load_dotenv(inst.notebook_dir / ".env", override=False)
        os.environ.setdefault("REQUESTS_TIMEOUT_OVERRIDE", "300")
        return inst

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------

    def show_status(self) -> None:
        """Print a concise status table for all data artifacts and QPU usage.

        Reads files only; never launches a calculation.
        """
        final_state = "absent — quantum execution is blocked"
        try:
            fin = self.load_final_selection(check_baseline=False)
            final_state = f"frozen {fin['selected_fragment_ids']}"
        except ChallengeError as exc:
            if self.final_selection_path.is_file():
                final_state = f"INVALID: {str(exc).splitlines()[0]}"
        base_state = "absent"
        try:
            base = self.load_classical_baseline(SELECTED_LIGAND)
            base_state = f"loaded (E_total={base['energies']['e_total']:.8f} Ha)"
        except ChallengeError as exc:
            if self.baseline_path(SELECTED_LIGAND).is_file():
                base_state = f"INVALID: {str(exc).splitlines()[0]}"

        rows = []
        for name, path in [
            ("hf_results.json",                    self.hf_path),
            ("fragment_catalog.csv",               self.cat_path),
            ("preliminary_mp2_selection.json",     self.preliminary_selection_path),
            (f"{SELECTED_LIGAND}_classical_baseline.json", self.baseline_path(SELECTED_LIGAND)),
            ("final_experiment_selection.json",    self.final_selection_path),
            ("reference_results.json",             self.benchmark_path),
            ("archived_lab_results.json",          self.archive_path),
            ("promoted_configuration.json",        self.ckpt_dir / "promoted_configuration.json"),
            ("promotion_manifest.json",            self.promotion_manifest_path),
        ]:
            rows.append({"artifact": name, "present": path.is_file()})
        df = pd.DataFrame(rows)
        try:
            from IPython.display import display
            display(df)
        except ImportError:
            print(df.to_string(index=False))

        print(f"\nClassical baseline:        {base_state}")
        print(f"Final quantum selection:   {final_state}")
        print(f"QPU sampling jobs recorded (this kernel + previous sessions): "
              f"{self._qpu_sampling_count}")
        print(f"Data root:   {self.data_root}")
        print(f"Bundle dir:  {self.bundle_dir}")
        print(f"Checkpoints: {self.ckpt_dir}")

    def attempt(self, fn: Callable, *args, **kwargs):
        """Call ``fn``; on :class:`ChallengeError` print the actionable message and return None.

        Used by the notebook so a missing artifact leaves a checkpoint PENDING
        rather than aborting the whole run.  Nothing is fabricated.
        """
        try:
            return fn(*args, **kwargs)
        except ChallengeError as exc:
            print(f"PENDING — {exc}")
            return None

    # ------------------------------------------------------------------
    # Preliminary (pre-CCSD, MP2) selection — a PROVISIONAL PROPOSAL
    # ------------------------------------------------------------------

    def load_preliminary_selection(self) -> dict:
        """Load the preliminary MP2 proposal (``authorizes_sqd`` is always False).

        This is informational.  No quantum method accepts it as a selection.
        """
        from proteasome_challenge.experiment_selection import load_preliminary_selection as _load
        try:
            return _load(self.preliminary_selection_path)
        except ValueError as exc:
            raise ChallengeError(str(exc)) from exc

    def save_preliminary_selection(self, catalog_df=None, *, ligand: str = SELECTED_LIGAND) -> dict:
        """Run the deterministic MP2 screening procedure and record the proposal.

        Uses only pre-CCSD data (catalog + MP2 diagnostics).  The manifest is
        marked ``selection_stage='preliminary_mp2'`` and ``authorizes_sqd=False``;
        it does not authorise SQD execution and cannot be used as the final selection.
        """
        from proteasome_challenge.experiment_selection import (
            select_experimental_panel, save_preliminary_selection as _save)

        if catalog_df is None:
            catalog_df = self.load_fragment_catalog()
        try:
            result = select_experimental_panel(catalog_df, ligand=ligand)
        except ValueError as exc:
            raise ChallengeError(f"Preliminary MP2 selection failed: {exc}") from exc
        manifest = _save(result, self.preliminary_selection_path)
        print(
            f"Preliminary MP2 proposal saved → {self.preliminary_selection_path.name}\n"
            f"  Proposed (provisional) fragments: {result.proposed_fragment_ids}\n"
            f"  qubit group (sqd_n_qubits):       {result.selected_sqd_n_qubits}"
            f"  (group size: {result.group_size})\n"
            "  This proposal does NOT authorise SQD execution."
        )
        return manifest

    # ------------------------------------------------------------------
    # Final (CCSD-informed, user-frozen) selection
    # ------------------------------------------------------------------

    def load_final_selection(self, *, check_baseline: bool = True) -> dict:
        """Load the frozen final selection.  The ONLY source of quantum fragments.

        Raises :class:`ChallengeError` — with an actionable explanation — when the
        manifest is missing, unfrozen, edited, or (``check_baseline=True``) stale.
        A user-frozen CCSD-informed manifest (schema 1.0) is stale relative to the
        current classical baseline and CCSD diagnostics; a size-based manifest
        (schema 2.0, :meth:`select_fragments_by_size`) is stale relative to the
        current catalog and frozen solver map and needs neither the baseline nor any
        diagnostics.  It never falls back to the preliminary panel or to any fixed
        fragment.
        """
        from proteasome_challenge import final_selection as fs
        try:
            manifest = fs.load_final_manifest(self.final_selection_path)
        except fs.FinalSelectionError as exc:
            raise ChallengeError(str(exc)) from exc
        if fs.is_size_based(manifest):
            if check_baseline:
                try:
                    fs.check_size_manifest_current(
                        manifest, catalog_records=self._size_catalog_records(),
                        assignments_sha256=self._frozen_assignments_sha256(manifest["ligand"]))
                except fs.FinalSelectionError as exc:
                    raise ChallengeError(str(exc)) from exc
            return manifest
        if check_baseline:
            base = self.load_classical_baseline(manifest["ligand"])
            try:
                fs.check_not_stale(
                    manifest,
                    baseline_cache_key=base["identity"]["cache_key"],
                    diagnostics_sha256=base["diagnostics_sha256"])
            except fs.FinalSelectionError as exc:
                raise ChallengeError(str(exc)) from exc
        return manifest

    def final_selection_digest(self) -> str:
        """Digest of the (validated, non-stale) final selection; bound into downstream artifacts."""
        return self.load_final_selection()["manifest_digest"]

    def final_selection_fragment_ids(self) -> list[int]:
        return list(self.load_final_selection()["selected_fragment_ids"])

    def sqd_calibration_fragment_id(self) -> int:
        """The single-fragment SQD/TrimSQD calibration target — from the frozen selection only."""
        return int(self.load_final_selection()["sqd_calibration_fragment_id"])

    def _size_catalog_records(self) -> list[dict]:
        cat = self.load_fragment_catalog(with_diagnostics=False, with_chemistry=False)
        return cat[["ligand", "fragment_id", "n_orb", "n_elec"]].to_dict("records") \
            if "n_elec" in cat.columns else cat[["ligand", "fragment_id", "n_orb"]].to_dict("records")

    def _frozen_assignments_sha256(self, ligand: str) -> str:
        from proteasome_challenge import classical_baseline as cb
        return cb.assignments_digest(self.baseline_assignments(ligand))

    def select_fragments_by_size(
        self,
        ligand: str = SELECTED_LIGAND,
        *,
        min_n_orb: int = SQD_THRESHOLD,
    ) -> dict:
        """Create (or reload) the deterministic size-based quantum-fragment selection.

        Rule: every ``ligand`` fragment with ``n_orb >= min_n_orb`` (default
        ``SQD_THRESHOLD`` = 20).  No MP2/CCSD diagnostic, ranking, classical baseline
        or hand-picked fragment id is involved.  The manifest (schema 2.0, see
        :mod:`proteasome_challenge.final_selection`) is written to
        ``checkpoints/final_experiment_selection.json`` — the same location the
        existing loaders read — and is a pure function of the catalog and frozen
        solver map, so regenerating it is byte-identical.

        An existing current size-based manifest is reloaded untouched; a stale one is
        regenerated.  A user-frozen CCSD-informed manifest (schema 1.0) or an
        unreadable file is never overwritten silently: remove it deliberately first.
        """
        from proteasome_challenge import final_selection as fs
        records = self._size_catalog_records()
        a_sha = self._frozen_assignments_sha256(ligand)
        try:
            manifest = fs.build_size_manifest(
                ligand=ligand, catalog_records=records, min_n_orb=min_n_orb,
                assignments=self.baseline_assignments(ligand), assignments_sha256=a_sha)
        except fs.FinalSelectionError as exc:
            raise ChallengeError(str(exc)) from exc

        path = self.final_selection_path
        if path.is_file():
            try:
                existing = fs.load_final_manifest(path)
            except fs.FinalSelectionError as exc:
                raise ChallengeError(
                    f"{path.name} exists but is not a valid selection ({str(exc).splitlines()[0]}). "
                    "Delete it deliberately to regenerate; it is not overwritten silently.") from exc
            if not fs.is_size_based(existing):
                raise ChallengeError(
                    f"{path.name} is a user-frozen (schema {existing['schema_version']}) selection. "
                    "It is not overwritten by the size-based rule; delete it deliberately to switch.")
            if existing == manifest:
                return existing
            print(f"Stale size-based selection in {path.name} — regenerating.")
        fs.write_final_manifest(manifest, path)
        print(f"Size-based selection written → {path.name}\n"
              f"  rule: n_orb >= {int(min_n_orb)}   fragments: {manifest['selected_fragment_ids']}")
        return manifest

    def freeze_final_selection(
        self,
        entries: list[dict],
        *,
        approved_by: str,
        selection_basis: str,
        sqd_calibration_fragment_id: int,
        ligand: str = SELECTED_LIGAND,
        confirm: bool = False,
    ) -> dict:
        """Record and freeze the user's final quantum-fragment selection.

        There is **no automatic rule**: ``entries`` (``[{"fragment_id", "role",
        "rationale"}, ...]``), ``approved_by`` and ``selection_basis`` are written
        by the user after reviewing the CCSD diagnostics.  No T1/D1 threshold or
        weighted score is applied.  Requires a valid classical baseline and
        ``confirm=True``.  The manifest is bound to the exact baseline cache key
        and diagnostics digest; if either changes the selection becomes stale.
        """
        from proteasome_challenge import final_selection as fs
        if confirm is not True:
            raise ChallengeError(
                "freeze_final_selection requires confirm=True. Freezing is an explicit, "
                "user-approved scientific decision; nothing was written.")
        base = self.load_classical_baseline(ligand)       # raises if no valid baseline
        cat = self.load_fragment_catalog()
        sub = cat[cat["ligand"] == ligand].set_index("fragment_id")
        rows = {f["fragment_id"]: f for f in base["fragments"]}
        ctx: dict[int, dict] = {}
        for fid, f in rows.items():
            d = f["ccsd_diagnostics"]
            c = sub.loc[fid] if fid in sub.index else None
            def _num(col):
                if c is None or col not in c.index:
                    return None
                v = c[col]
                try:
                    return None if pd.isna(v) else v.item() if hasattr(v, "item") else v
                except (TypeError, ValueError):
                    return None
            ctx[fid] = {
                "resource_requirements": {
                    "n_orb": f["n_orb"], "n_occ": _num("n_occ"), "n_vir": _num("n_vir"),
                    "log10_fci_dimension": _num("log10_fci_dimension"),
                    "sqd_n_qubits": _num("sqd_n_qubits"),
                    "ccsd_t2_elements": _num("ccsd_t2_elements"),
                },
                "classical_solver_in_baseline": f["solver"],
                "mp2_max_abs_t2": _num("mp2_max_abs_t2"),
                "ccsd_diagnostics_summary": {
                    "diagnostics_status": d["diagnostics_status"],
                    "ccsd_t1_diagnostic": d["ccsd_t1_diagnostic"],
                    "ccsd_d1_diagnostic": d["ccsd_d1_diagnostic"],
                    "ccsd_max_abs_t1": d["ccsd_max_abs_t1"],
                    "ccsd_max_abs_t2": d["ccsd_max_abs_t2"],
                    "ccsd_converged": d["ccsd_converged"],
                },
            }
        prelim_digest = None
        try:
            prelim_digest = self.load_preliminary_selection()["manifest_digest"]
        except ChallengeError:
            pass
        try:
            manifest = fs.build_final_manifest(
                ligand=ligand, entries=entries, approved_by=approved_by,
                selection_basis=selection_basis,
                sqd_calibration_fragment_id=sqd_calibration_fragment_id,
                baseline_identity=base["identity"],
                baseline_e_total_ha=base["energies"]["e_total"],
                diagnostics_sha256=base["diagnostics_sha256"],
                fragment_context=ctx, preliminary_manifest_digest=prelim_digest)
            for e in manifest["fragments"]:
                if e["classical_solver_in_baseline"] not in ("FCI", "CCSD"):
                    raise fs.FinalSelectionError("baseline fragment has a non-classical solver")
            fs.write_final_manifest(manifest, self.final_selection_path)
        except fs.FinalSelectionError as exc:
            raise ChallengeError(str(exc)) from exc
        print(f"Final selection FROZEN → {self.final_selection_path.name}\n"
              f"  fragments: {manifest['selected_fragment_ids']}  "
              f"(SQD calibration fragment: {manifest['sqd_calibration_fragment_id']})\n"
              f"  bound to baseline {manifest['source_provenance']['baseline_cache_key'][:12]}…")
        return manifest

    # ------------------------------------------------------------------
    # Fragment chemistry
    # ------------------------------------------------------------------

    def load_fragment_chemistry(self) -> "pd.DataFrame":
        """Load fragment chemistry as a DataFrame (cached on self)."""
        if not hasattr(self, "_chemistry_df"):
            import json as _json
            with self.chemistry_path.open("r") as fh:
                self._chemistry_df = pd.DataFrame(_json.load(fh))
        return self._chemistry_df

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def load_hf_results(self) -> dict:
        """Load and validate the Stage 1 HF results JSON.

        Returns
        -------
        dict mapping ligand name → {hf_energy, hf_converged, basis, n_ao,
        n_electrons, charge, spin, source}

        Raises
        ------
        ChallengeError
            If the file is missing, a ligand entry is absent, HF did not
            converge, or an energy value is non-finite.
        """
        if not self.hf_path.is_file():
            raise ChallengeError(
                f"hf_results.json not found at {self.hf_path}.\n"
                "This published reference file is not produced by Stage 1; ask the organizers for it "
                "(Stage 1 results live in checkpoints/stage1 and are read through hf_record())."
            )
        with self.hf_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for lig in LIGANDS:
            if lig not in data:
                raise ChallengeError(f"hf_results.json is missing an entry for '{lig}'")
            row = data[lig]
            if not row.get("hf_converged", False):
                raise ChallengeError(
                    f"HF is marked non-converged for {lig} in hf_results.json. "
                    "Contact the organizers."
                )
            e = row.get("hf_energy")
            if e is None or not np.isfinite(float(e)):
                raise ChallengeError(f"Non-finite HF energy for {lig}: {e!r}")
            if row.get("basis", BASIS).lower() != BASIS:
                raise ChallengeError(
                    f"Unexpected basis for {lig}: {row.get('basis')!r} (expected {BASIS!r})"
                )
        return data

    def load_fragment_catalog(
        self,
        *,
        with_diagnostics: bool = True,
        with_chemistry: bool = True,
    ) -> pd.DataFrame:
        """Load and validate the Stage 2 fragment catalog CSV.

        The enriched catalog includes nine pre-execution computational resource
        descriptors derived from the embedded-cluster spin occupations:
        ``n_alpha``, ``n_beta``, ``n_occ``, ``n_vir``, ``fci_dimension``,
        ``log10_fci_dimension``, ``fci_vector_bytes``, ``ccsd_t2_elements``,
        ``ccsd_t2_bytes``, ``sqd_n_qubits``.  These are read directly from the
        CSV — no solver execution is required.

        When ``with_diagnostics=True`` (default) and
        ``challenge_data/fragment_diagnostics.csv`` exists, the MP2
        correlation-hardness columns (``mp2_max_abs_t2``, ``mp2_t2_norm``,
        ``mp2_t2_rms``, ``mp2_corr_energy_ha``, ``hf_gap_ha``,
        ``diagnostic_status``) are left-joined onto the catalog by
        ``(ligand, fragment_id)``.  Columns already present in the catalog
        (e.g. ``n_occ``, ``n_vir``) are not re-imported from diagnostics.

        When ``with_chemistry=True`` (default) and
        ``challenge_data/fragment_chemistry.json`` exists, chemical-identity
        columns (``center_element``, ``chemical_environment``,
        ``bonded_neighbor_elements``, ``contains_transition_metal``) are
        left-joined in the same way.  The catalog file is never modified;
        all joins are done in memory only.

        Returns
        -------
        pandas.DataFrame with columns: ligand, fragment_id, n_orb, n_elec,
        n_alpha, n_beta, n_occ, n_vir, fci_dimension, log10_fci_dimension,
        fci_vector_bytes, ccsd_t2_elements, ccsd_t2_bytes, sqd_n_qubits,
        classical_fci_candidate, qpu_eligible, classical_ccsd_candidate,
        and (optionally) MP2 diagnostic columns and chemistry columns.

        Raises
        ------
        ChallengeError
            If the file is missing, required columns are absent, or fragment
            counts do not match the expected 42/53 values.
        """
        if not self.cat_path.is_file():
            raise ChallengeError(
                f"fragment_catalog.csv not found at {self.cat_path}.\n"
                "Run Stage 2 or ask the organizers for the published file."
            )
        cat = pd.read_csv(self.cat_path)
        required = {"ligand", "fragment_id", "n_orb"}
        missing = required - set(cat.columns)
        if missing:
            raise ChallengeError(
                f"fragment_catalog.csv is missing required columns: {sorted(missing)}"
            )
        cat["ligand"] = cat["ligand"].astype(str).str.lower().str.strip()
        for col in ("fragment_id", "n_orb"):
            vals = pd.to_numeric(cat[col], errors="raise")
            if vals.isna().any() or not np.all(vals == vals.astype(int)):
                raise ChallengeError(f"Non-integer values in fragment catalog column '{col}'")
            cat[col] = vals.astype(int)
        if not cat["ligand"].isin(LIGANDS).all():
            raise ChallengeError("fragment_catalog.csv contains unknown ligand names")
        if cat.duplicated(["ligand", "fragment_id"]).any():
            raise ChallengeError("Duplicate (ligand, fragment_id) pairs in fragment_catalog.csv")
        if not (cat["n_orb"] > 0).all():
            raise ChallengeError("All n_orb values must be positive in fragment_catalog.csv")
        actual = cat.groupby("ligand").size().to_dict()
        for lig, count in actual.items():
            expected = EXPECTED_FRAGMENTS[lig]
            if count != expected:
                raise ChallengeError(
                    f"Expected {expected} {lig} fragments, found {count}. "
                    "Check geometry, bath parameters, and fragment indexing."
                )
        # Computational-size descriptors ONLY (orbital-count bands).  They do NOT
        # decide which solver runs: the solver for each fragment comes from the
        # frozen map in challenge_data/solver_assignments.json (see
        # load_solver_assignments / routing_plan).
        cat["classical_fci_candidate"] = cat["n_orb"] <= FCI_THRESHOLD
        cat["qpu_eligible"] = cat["n_orb"] >= SQD_THRESHOLD
        cat["classical_ccsd_candidate"] = ~(cat["classical_fci_candidate"] | cat["qpu_eligible"])

        # ── Merge MP2 diagnostics if available ────────────────────────────────
        if with_diagnostics:
            diag_path = self.data_root / "fragment_diagnostics.csv"
            if diag_path.is_file():
                try:
                    diag = pd.read_csv(diag_path)
                    # Skip a catalog column only if it is present AND has at
                    # least one non-null value.  A column that exists but is
                    # entirely NaN (e.g. after an external edit wiped the
                    # enriched values) is treated as absent so diagnostics can
                    # repopulate it.
                    _join_keys = {"ligand", "fragment_id"}
                    _already_in_cat = {
                        c for c in cat.columns
                        if c not in _join_keys and cat[c].notna().any()
                    }
                    _want_from_diag = [
                        "n_occ", "n_vir",
                        "mp2_max_abs_t2", "mp2_t2_norm", "mp2_t2_rms",
                        "mp2_corr_energy_ha", "hf_gap_ha",
                        "diagnostic_status",
                    ]
                    _new_cols = [c for c in _want_from_diag
                                 if c in diag.columns and c not in _already_in_cat]
                    if _new_cols:
                        _diag_cols = list(_join_keys) + _new_cols
                        diag = diag[[c for c in _diag_cols if c in diag.columns]].copy()
                        diag["ligand"]       = diag["ligand"].astype(str).str.lower().str.strip()
                        diag["fragment_id"]  = diag["fragment_id"].astype(int)
                        # Float columns: empty string → NaN
                        for fc in ("mp2_max_abs_t2", "mp2_t2_norm", "mp2_t2_rms",
                                    "mp2_corr_energy_ha", "hf_gap_ha"):
                            if fc in diag.columns:
                                diag[fc] = pd.to_numeric(diag[fc], errors="coerce")
                        # Drop all-NaN versions of new_cols from cat before merge
                        # to avoid pandas creating _x/_y suffix columns.
                        _to_drop = [c for c in _new_cols if c in cat.columns]
                        if _to_drop:
                            cat = cat.drop(columns=_to_drop)
                        cat = cat.merge(diag, on=["ligand", "fragment_id"], how="left")
                except Exception:
                    pass   # diagnostics are optional; never break catalog loading

        # ── Merge fragment chemistry if available ─────────────────────────────
        if with_chemistry and self.chemistry_path.is_file():
            try:
                chem_rows = json.loads(self.chemistry_path.read_text(encoding="utf-8"))
                chem = pd.DataFrame(chem_rows)
                _chem_cols = [
                    "ligand", "fragment_id",
                    "center_element", "chemical_environment",
                    "bonded_neighbor_elements", "contains_transition_metal",
                ]
                chem = chem[[c for c in _chem_cols if c in chem.columns]].copy()
                chem["ligand"]      = chem["ligand"].astype(str).str.lower().str.strip()
                chem["fragment_id"] = chem["fragment_id"].astype(int)
                cat = cat.merge(chem, on=["ligand", "fragment_id"], how="left")
            except Exception:
                pass   # chemistry is optional; never break catalog loading

        # ── Derive n_occ / n_vir from n_elec when the catalog is in protected form ──
        # The on-disk catalog stores only n_orb and n_elec (the two quantities that
        # come directly from the Vayesta Stage 2 output).  For closed-shell RHF
        # clusters n_occ = n_elec / 2 and n_vir = n_orb - n_occ.  We derive them
        # in memory; we never write back to the protected CSV file.
        try:
            _nocc_blank = "n_occ" not in cat.columns or cat["n_occ"].isna().all()
            if _nocc_blank and "n_elec" in cat.columns and cat["n_elec"].notna().all():
                _n_elec = pd.to_numeric(cat["n_elec"], errors="coerce")
                if (_n_elec % 2 == 0).all():
                    cat["n_occ"] = (_n_elec // 2).astype(int)
                    cat["n_vir"] = cat["n_orb"] - cat["n_occ"]
        except Exception:
            pass   # derivation is best-effort; never break loading

        # ── Ensure resource descriptor columns are populated ──────────────────
        # Compute all nine resource fields in memory from n_orb / n_occ / n_vir.
        # The protected CSV is never modified; these columns are always in-memory.
        try:
            from proteasome_challenge.resource_descriptors import (
                RESOURCE_COLUMNS as _RC,
                enrich_catalog as _enrich,
            )
            _needs_enrich = (
                any(c not in cat.columns for c in _RC)
                or any(cat[c].isna().any() for c in _RC if c in cat.columns)
            )
            if _needs_enrich and "n_occ" in cat.columns and "n_vir" in cat.columns:
                _enrich(cat)
        except Exception:
            pass   # resource descriptors are informational; never break loading

        return cat

    def load_fragment_diagnostics(self, *, recompute: bool = False) -> pd.DataFrame:
        """Load (or compute) MP2 correlation-hardness diagnostics for all fragments.

        This method reads ``challenge_data/fragment_diagnostics.csv`` if it
        exists.  When *recompute* is True, or the CSV is absent, it reads the
        Stage 2 HDF5 dumpfiles from ``checkpoints/stage2/`` and computes the
        diagnostics in memory, then saves the result for future use.

        Returns
        -------
        pandas.DataFrame with one row per fragment and columns matching those
        described in :mod:`proteasome_challenge.mp2_diagnostics`.

        Raises
        ------
        ChallengeError
            If neither the CSV nor the HDF5 dumpfiles are available.
        """
        from proteasome_challenge.mp2_diagnostics import (
            compute_all_diagnostics,
            load_diagnostics_csv,
            save_diagnostics_csv,
        )

        diag_path = self.data_root / "fragment_diagnostics.csv"

        if not recompute and diag_path.is_file():
            rows = load_diagnostics_csv(diag_path)
            return pd.DataFrame(rows)

        # Compute from HDF5 dumpfiles
        all_rows = []
        for lig in LIGANDS:
            h5p = self.ckpt_dir / "stage2" / f"{lig}_ligand_{BASIS}__ewf_dumpfile.h5"
            if not h5p.is_file():
                raise ChallengeError(
                    f"Stage 2 HDF5 dumpfile not found: {h5p}\n"
                    "Run Stage 2 first, or ask the organizers for "
                    "challenge_data/fragment_diagnostics.csv."
                )
            print(f"[diagnostics] Computing MP2 diagnostics for {lig} ({h5p.name})…",
                  flush=True)
            rows = compute_all_diagnostics(h5p, lig)
            ok  = sum(1 for r in rows if r["diagnostic_status"] == "ok")
            bad = sum(1 for r in rows if r["diagnostic_status"] != "ok")
            print(f"  {len(rows)} fragments: {ok} ok, {bad} failed", flush=True)
            all_rows.extend(rows)

        # Save to challenge_data/ so future calls are instant
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        save_diagnostics_csv(all_rows, diag_path)
        print(f"[diagnostics] Saved {len(all_rows)} rows to {diag_path}", flush=True)

        return pd.DataFrame(all_rows)

    def load_fragment_chemistry(self, *, recompute: bool = False) -> pd.DataFrame:
        """Load (or compute) fragment chemical-identity metadata.

        Reads ``challenge_data/fragment_chemistry.json`` when present.
        When *recompute* is True or the file is absent, recomputes from the
        Stage 2 HDF5 dumpfiles and the geometry XYZ files.

        Each row describes the chemically defined fragment center (one atom per
        fragment in IAO atomic fragmentation), NOT the full DMET embedded
        cluster.

        Returns
        -------
        pandas.DataFrame with columns: ligand, fragment_id, center_atom_idx,
        center_element, bonded_neighbor_indices, bonded_neighbor_elements,
        chemical_environment, contains_transition_metal, norb_dmet_cluster,
        assignment_method.

        Raises
        ------
        ChallengeError
            If neither the JSON file nor both HDF5+XYZ files are available.
        """
        from proteasome_challenge.fragment_chemistry import (
            build_fragment_chemistry,
            load_fragment_chemistry_json,
            save_fragment_chemistry_json,
        )

        if not recompute and self.chemistry_path.is_file():
            rows = load_fragment_chemistry_json(self.chemistry_path)
            return pd.DataFrame(rows)

        # Compute from HDF5 dumpfiles + XYZ geometry files
        all_rows = []
        for lig in LIGANDS:
            h5p  = self.ckpt_dir / "stage2" / f"{lig}_ligand_{BASIS}__ewf_dumpfile.h5"
            xyz_p = self.bundle_dir / "configs" / f"system_{lig}.xyz"
            if not h5p.is_file():
                raise ChallengeError(
                    f"Stage 2 HDF5 dumpfile not found: {h5p}\n"
                    "Run Stage 2 first, or ask the organizers for "
                    "challenge_data/fragment_chemistry.json."
                )
            if not xyz_p.is_file():
                raise ChallengeError(
                    f"Geometry file not found: {xyz_p}\n"
                    "The serverless_bundle/configs/ directory may be incomplete."
                )
            print(f"[chemistry] Building fragment chemistry for {lig}…", flush=True)
            rows = build_fragment_chemistry(h5p, xyz_p, lig)
            print(f"  {len(rows)} fragments processed.", flush=True)
            all_rows.extend(rows)

        self.chemistry_path.parent.mkdir(parents=True, exist_ok=True)
        save_fragment_chemistry_json(all_rows, self.chemistry_path)
        print(f"[chemistry] Saved {len(all_rows)} rows to {self.chemistry_path}", flush=True)

        return pd.DataFrame(all_rows)

    def describe_fragment(self, ligand: str, fragment_id: int) -> pd.Series:
        """Look up a single fragment from the catalog.

        Parameters
        ----------
        ligand : str
            ``"ixazomib"`` or ``"bortezomib"``.
        fragment_id : int
            Fragment index (1-based as stored in the HDF5 dump).

        Returns
        -------
        pandas.Series with fragment metadata.

        Raises
        ------
        ChallengeError
            If the fragment does not exist.
        """
        cat = self.load_fragment_catalog()
        matches = cat.loc[(cat["ligand"] == ligand) & (cat["fragment_id"] == int(fragment_id))]
        if len(matches) != 1:
            raise ChallengeError(
                f"Fragment ({ligand}, {fragment_id}) not found in the catalog. "
                "Available IDs: " + str(sorted(cat.loc[cat["ligand"] == ligand, "fragment_id"].tolist()))
            )
        return matches.iloc[0]

    # ------------------------------------------------------------------
    # Reference data loaders
    # ------------------------------------------------------------------

    def load_reference_results(self) -> dict:
        """Load the organizer's reference_results.json (if present)."""
        if not self.benchmark_path.is_file():
            return {}
        with self.benchmark_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def load_archived_results(self) -> dict:
        """Load the organizer's archived_lab_results.json (if present)."""
        if not self.archive_path.is_file():
            return {}
        with self.archive_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def load_tuning_grid(self) -> list[dict]:
        """Return the organizer-fixed tuning grid (``reference_results.json["tuning_grid"]``)."""
        grid = self.load_reference_results().get("tuning_grid", [])
        if not grid:
            raise ChallengeError(
                "reference_results.json has no 'tuning_grid'. Ask the organizers for the "
                "published tuning grid before running Section 3c."
            )
        labels = [p.get("label") for p in grid]
        if len(set(labels)) != len(labels) or any(not l for l in labels):
            raise ChallengeError(f"Tuning grid labels must be unique and non-empty: {labels}")
        return grid

    # ------------------------------------------------------------------
    # Counts helpers
    # ------------------------------------------------------------------

    def _run_name(self, ligand: str) -> str:
        return f"{ligand}_ligand_{BASIS}"

    def _frozen_counts_path(self, ligand: str, fragment_id: int) -> Path:
        return self.frozen_counts_dir / f"{ligand}_fragment_{int(fragment_id)}_counts.npy"

    def _live_counts_path(self, ligand: str, fragment_id: int) -> Path:
        return (self.ckpt_dir / "counts"
                / f"{self._run_name(ligand)}__counts_frag{int(fragment_id)}.npy")

    def resolve_counts(self, ligand: str, fragment_id: int) -> Path | None:
        """Return path to the best available counts file (live run > frozen > None)."""
        for p in (self._live_counts_path(ligand, fragment_id),
                  self._frozen_counts_path(ligand, fragment_id)):
            if p.is_file():
                return p
        return None

    def counts_provenance(self, ligand: str, fragment_id: int) -> str:
        """Human-readable description of which counts are available."""
        live = self._live_counts_path(ligand, fragment_id)
        frozen = self._frozen_counts_path(ligand, fragment_id)
        if live.is_file():
            return f"live_run ({live.name})"
        if frozen.is_file():
            return f"organizer_frozen ({frozen.name})"
        return "none_available"

    def _counts_source(self, ligand: str, fragment_id: int) -> str | None:
        """``"live"``, ``"frozen"`` or None, matching :meth:`resolve_counts` priority."""
        if self._live_counts_path(ligand, fragment_id).is_file():
            return "live"
        if self._frozen_counts_path(ligand, fragment_id).is_file():
            return "frozen"
        return None

    def _require_counts(self, ligand: str, fragment_id: int, context: str) -> Path:
        """Return the counts file or raise.  Never falls back to QPU sampling."""
        p = self.resolve_counts(ligand, fragment_id)
        if p is None:
            raise ChallengeError(
                f"{context}: no saved or frozen counts for ({ligand}, fragment {fragment_id}).\n"
                "This call replays measurement data and never samples the QPU implicitly. "
                "Obtain the organizer's frozen counts (challenge_data/frozen_counts/) or, "
                "organizers only, take the single authorized sampling job with "
                "solve_fragment(..., execution='live_qpu')."
            )
        return p

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def energy_of(result: dict | None, method: str) -> float | None:
        """Extract the ``method`` energy (Ha) from a single-fragment Stage 3 result.

        ``method`` is ``"ccsd"``, ``"sqd"`` or ``"trimsqd"``.  Returns None when
        the result does not contain that energy (e.g. ``sqd`` from a
        ``trimsqd``-only run).  A Stage 3 ``trimsqd`` result carries its energy in
        ``energy_ha`` (``trimsqd_energy_ha`` is only written by ``both`` mode), so
        both fields are consulted.
        """
        if not isinstance(result, dict):
            return None
        field = {"ccsd": "ccsd_energy_ha", "sqd": "sqd_energy_ha",
                 "trimsqd": "trimsqd_energy_ha"}.get(method)
        if field is None:
            raise ChallengeError(f"energy_of: unknown method {method!r}")
        candidates = [result.get(field)]
        if str(result.get("method_used", "")).lower() == method:
            candidates.append(result.get("energy_ha"))
        for v in candidates:
            if v is None:
                continue
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(x):
                return x
        return None

    # ------------------------------------------------------------------
    # Serverless: connection, deployment, uploads
    # ------------------------------------------------------------------

    def _connect(self, *, _force_refresh: bool = False):
        """Return an authenticated QiskitServerless client (lazy).

        Pass ``_force_refresh=True`` (or set ``self._serverless = None``) to
        discard a stale cached client and create a fresh one.  This is called
        automatically after a transient ``QiskitServerlessException`` so that
        the next retry gets a new TCP connection rather than reusing the dead
        socket.
        """
        if _force_refresh:
            self._serverless = None
        if self._serverless is None:
            from qiskit_ibm_catalog import QiskitServerless
            qrmi_prefix = resolve_qpu_backend().lower()      # QRMI keys are <backend>_QRMI_*
            token = (os.environ.get("IBM_QUANTUM_TOKEN")
                     or os.environ.get(f"{qrmi_prefix}_QRMI_IBM_QCS_IAM_APIKEY"))
            instance = (os.environ.get("IBM_QUANTUM_INSTANCE")
                        or os.environ.get(f"{qrmi_prefix}_QRMI_IBM_QCS_SERVICE_CRN"))
            if not token or not instance or token.startswith("<") or instance.startswith("<"):
                raise ChallengeError(
                    "IBM credentials missing or still placeholders.\n"
                    "Copy .env.example to .env and fill IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE.\n"
                    "Nothing was submitted."
                )
            self._serverless = QiskitServerless(token=token, instance=instance)
            print("Authenticated to Qiskit Serverless.")
        return self._serverless

    def _load_function(self, stage: int):
        return self._connect().load(STAGES[stage]["title"])

    def _stage3_fn(self):
        return self._load_function(3)

    def bundle_fingerprint(self) -> str:
        """SHA-256 over the relative names and bytes of every file in ``serverless_bundle/``."""
        h = hashlib.sha256()
        for p in sorted(self.bundle_dir.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                h.update(str(p.relative_to(self.bundle_dir)).encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    def _assert_bundle_deployed(self) -> None:
        """Fail if the local bundle differs from the last deployed fingerprint."""
        if self.allow_unverified_bundle:
            return
        fp_path = self.ckpt_dir / ".deployed_bundle_sha256"
        if not fp_path.exists() or fp_path.read_text().strip() != self.bundle_fingerprint():
            raise ChallengeError(
                "serverless_bundle/ differs from the last deployed version (or was never deployed).\n"
                "Set RUN_DEPLOY=True and re-run the setup cell, or set ALLOW_UNVERIFIED_BUNDLE=True "
                "if an organizer deployed the functions."
            )

    def _install_upload_patch(self) -> None:
        """Wrap uploads as .tar.gz because the file API rejects .pkl/.h5/.npy directly."""
        if self._upload_patched:
            return
        import io
        import tarfile
        import requests
        from qiskit_serverless.core import files as _f
        from qiskit_serverless.core.files import QiskitServerlessException as _Err
        from qiskit_serverless.utils.urls import url_path_join as _join
        from qiskit_serverless.utils.http import get_headers as _headers
        import qiskit_serverless.core.clients.serverless_client as _sc
        import qiskit_serverless.core.constants as _c
        _c.REQUESTS_TIMEOUT = _sc.REQUESTS_TIMEOUT = 300

        def _targz_upload(self_inner, file, function, sub_path=None):
            fname = os.path.basename(file)
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(file, arcname=fname)
            buf.seek(0)
            resp = requests.post(
                _join(self_inner._files_url, "upload/"),
                files={"file": (fname + ".tar.gz", buf, "application/gzip")},
                params={"provider": function.provider, "function": function.title},
                headers=_headers(token=self_inner._token, instance=self_inner._instance,
                                 channel=self_inner._channel),
                stream=True, timeout=600,
            )
            if resp.ok:
                return resp.text
            raise _Err(f"Upload failed: {resp.status_code}")

        _f.GatewayFilesClient.upload = _targz_upload
        self._upload_patched = True

    def _upload_file_once(self, src: Path, remote_name: str, fn) -> None:
        """Upload ``src`` to the Serverless file API as ``remote_name`` (idempotent per session)."""
        key = (fn.title, remote_name)
        if key in self._uploaded:
            return
        import shutil
        import tempfile
        self._install_upload_patch()
        src = Path(src)
        if src.name != remote_name:
            tmp = Path(tempfile.mkdtemp()) / remote_name
            shutil.copyfile(src, tmp)
            src = tmp
        self._connect().file_upload(str(src), fn)
        self._uploaded.add(key)
        print(f"  uploaded {remote_name}", flush=True)

    def _stage1_paths(self, ligand: str) -> tuple[Path, Path]:
        base = self.ckpt_dir / "stage1"
        rn = self._run_name(ligand)
        return base / f"{rn}.json", base / f"{rn}__mf_data.pkl"

    def _stage2_paths(self, ligand: str) -> tuple[Path, Path, Path]:
        base = self.ckpt_dir / "stage2"
        rn = self._run_name(ligand)
        return (base / f"{rn}.json", base / f"{rn}__embedding_data.pkl",
                base / f"{rn}__ewf_dumpfile.h5")

    @property
    def _stage2_jobs_path(self) -> Path:
        """Maps ligand name → Serverless job_id; written right after Stage 2 submission."""
        return self.ckpt_dir / "stage2" / "jobs.json"

    def _ensure_stage3_files(self, ligand: str, fn) -> None:
        """Ensure mf_data.pkl / ewf_dumpfile.h5 are on the file API."""
        _, pkl_p = self._stage1_paths(ligand)
        _, emb_p, h5_p = self._stage2_paths(ligand)
        for p in (pkl_p, h5_p):
            if not p.is_file():
                raise ChallengeError(
                    f"Missing Stage 1/2 artifact: {p}\n"
                    "Run Stage 1 and Stage 2 first, or use an organizer-published "
                    "prepared-data package."
                )
        rn = self._run_name(ligand)
        uploads = [(pkl_p, "mf_data.pkl"), (h5_p, "ewf_dumpfile.h5")]
        if emb_p.is_file():
            uploads.append((emb_p, "embedding_data.pkl"))
        for src, name in uploads:
            self._upload_file_once(src, f"{rn}__{name}", fn)

    def deploy_bundle(self) -> None:
        """Upload the three Serverless functions and record the bundle fingerprint."""
        from qiskit_ibm_catalog import QiskitFunction
        self._install_upload_patch()
        svc = self._connect()
        for spec in STAGES.values():
            svc.upload(QiskitFunction(
                title=spec["title"], entrypoint=spec["entrypoint"],
                working_dir=str(self.bundle_dir), description=spec["description"],
                runner="fleets",
                sizes_map={"s": "8x32", "m": "16x128", "l": "48x768"},
                default_size=spec["size"]))
            print(f"Uploaded: {spec['title']}")
        fp = self.ckpt_dir / ".deployed_bundle_sha256"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(self.bundle_fingerprint())
        print("Bundle fingerprint recorded.")

    # ------------------------------------------------------------------
    # Stage 2 (classical, organizer / live-recompute path).  Stage 0/1: see hf_stage.HartreeFockMixin
    # ------------------------------------------------------------------

    def run_stage2(self, poll_seconds: int = 30) -> None:
        """Run IAO/MP2-bath EWF fragmentation on Serverless and refresh ``fragment_catalog.csv``.

        **Kernel-restart safe:** job IDs are written to
        ``checkpoints/stage2/jobs.json`` immediately after submission.
        On re-entry, already-submitted jobs are reconnected rather than
        resubmitted, and already-collected ligands are skipped entirely.
        """
        import base64
        self._assert_bundle_deployed()

        # Load any job IDs persisted by a previous (interrupted) invocation.
        saved_job_ids: dict[str, str] = {}
        if self._stage2_jobs_path.is_file():
            try:
                saved_job_ids = json.loads(self._stage2_jobs_path.read_text())
            except Exception:
                saved_job_ids = {}

        # Determine which ligands still need work (no completed checkpoint yet).
        pending_ligands = [
            lig for lig in LIGANDS
            if not self._stage2_paths(lig)[0].is_file()   # meta JSON not yet written
        ]
        if not pending_ligands:
            print("[run_stage2] All Stage 2 checkpoints already present — skipping.")
            self._export_fragment_catalog()
            return

        fn2 = self._load_function(2)
        jobs: dict[str, Any] = {}

        for lig in pending_ligands:
            if lig in saved_job_ids:
                # Reconnect to a job submitted in a previous kernel session.
                job_id = saved_job_ids[lig]
                try:
                    job = self._connect().job(job_id)
                    if job is None:
                        raise ValueError("returned None")
                    jobs[lig] = job
                    print(f"Stage 2 {lig}: reconnected to job_id={job_id}", flush=True)
                except Exception as exc:
                    raise ChallengeError(
                        f"Could not reconnect to Stage 2 job {job_id!r} for {lig!r}: {exc}\n"
                        "Delete checkpoints/stage2/jobs.json to resubmit."
                    ) from exc
            else:
                # Stage 2 consumes only a verified Stage 1 result of the qualified (Challenge) specification.
                s1 = self.hf_status(challenge_preset(self.bundle_dir, lig), stationarity=False)
                if not (s1.ok and s1.qualified):
                    raise ChallengeError(
                        f"Stage 1 for {lig} is {s1.state}"
                        + (f" ({'; '.join(s1.problems)})" if s1.problems else "")
                        + "; run Stage 1 (RUN_STAGE1) first.")
                pkl_p = s1.pkl_path
                job = fn2.run(
                    function_size="m", run_name=self._run_name(lig),
                    mf_pkl_b64=base64.b64encode(pkl_p.read_bytes()).decode("ascii"),
                    bath_type="mp2", truncation=1e-4, fragmentation="iao", verbose=3)
                jobs[lig] = job
                saved_job_ids[lig] = job.job_id
                print(f"Stage 2 {lig}: submitted job_id={job.job_id}", flush=True)

        # Persist job IDs so a kernel restart can reconnect without resubmitting.
        self._stage2_jobs_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(self._stage2_jobs_path, saved_job_ids)

        for lig, res in self._wait_and_collect(jobs, poll_seconds).items():
            meta_p, emb_p, h5_p = self._stage2_paths(lig)
            self._write_json(meta_p, {k: v for k, v in res.items()
                                      if k not in ("embedding_pkl_b64", "dumpfile_b64")})
            self._b64_to_file(res["embedding_pkl_b64"], emb_p, gunzip=True)
            self._b64_to_file(res["dumpfile_b64"], h5_p, gunzip=True)
            print(f"Stage 2 {lig}: {res['n_fragments']} fragments (checkpoint saved)")

        # Clean up the in-progress jobs file now that everything completed.
        if self._stage2_jobs_path.is_file():
            try:
                self._stage2_jobs_path.unlink()
            except Exception:
                pass
        self._export_fragment_catalog()

    def _export_fragment_catalog(self) -> bool:
        """Build fragment_catalog.csv from ewf_dumpfile.h5 norb/nocc attrs (no vayesta import)."""
        import h5py
        rows = []
        for lig in LIGANDS:
            h5_p = self._stage2_paths(lig)[2]
            if not h5_p.is_file():
                continue
            with h5py.File(h5_p, "r") as hf:
                for key in hf:
                    if key.startswith("fragment_"):
                        rows.append({
                            "ligand": lig,
                            "fragment_id": int(key.split("_")[1]),
                            "n_orb": int(hf[key].attrs["norb"]),
                            "n_elec": 2 * int(hf[key].attrs["nocc"]),
                        })
        if not rows:
            return False
        old = (pd.read_csv(self.cat_path) if self.cat_path.is_file()
               else pd.DataFrame(columns=["ligand", "fragment_id", "n_orb"]))
        built = pd.DataFrame(rows)
        keep = old[~old["ligand"].isin(built["ligand"].unique())]
        self.cat_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([keep, built]).sort_values(["ligand", "fragment_id"]).to_csv(
            self.cat_path, index=False)
        print("fragment_catalog.csv refreshed from Stage 2 HDF5 dump files.")
        return True

    # ------------------------------------------------------------------
    # QPU budget gate (single point of truth)
    # ------------------------------------------------------------------

    def _load_qpu_count(self) -> int:
        if self._qpu_record_path.exists():
            try:
                return int(json.loads(self._qpu_record_path.read_text()).get("count", 0))
            except Exception:
                pass
        return 0

    def _record_qpu_sampling(self) -> None:
        """Increment and persist the QPU sampling count.

        This is the ONLY place the counter changes.  It is called exactly once
        per accepted sampling job, from :meth:`_submit_stage3`.
        """
        self._qpu_sampling_count += 1
        self._qpu_record_path.parent.mkdir(parents=True, exist_ok=True)
        self._qpu_record_path.write_text(
            json.dumps({"count": self._qpu_sampling_count}, indent=2)
        )

    def _check_qpu_budget(self, allow_second: bool = False) -> None:
        if self._qpu_sampling_count >= 1 and not allow_second:
            raise ChallengeError(
                f"QPU budget gate: {self._qpu_sampling_count} live QPU sampling job(s) already "
                "recorded for this account/kernel session.\n"
                "Reuse the saved counts (challenge_data/frozen_counts/ or checkpoints/counts/), "
                "or set allow_second_qpu=True to explicitly authorize another QPU job."
            )

    # ------------------------------------------------------------------
    # Stage 3 submission (single submission point)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_solver_settings(sqd_config: dict | None, trim_config: dict | None) -> None:
        for key in ("iterations", "n_batches", "max_dim", "samples_per_batch"):
            if sqd_config and key in sqd_config:
                v = sqd_config[key]
                if not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1:
                    raise ChallengeError(f"sqd_config['{key}'] must be an integer ≥ 1, got {v!r}")
        if trim_config and "trim_ratio" in trim_config:
            r = trim_config["trim_ratio"]
            if not isinstance(r, (int, float)) or not (0 < r <= 1):
                raise ChallengeError(f"trim_config['trim_ratio'] must satisfy 0 < x ≤ 1, got {r!r}")

    def _submit_stage3(
        self,
        ligand: str,
        *,
        label: str,
        solver_mode: str,
        qpu_backend: str | None,
        sqd_config: dict | None,
        trim_config: dict | None,
        n_shots: int,
        n_reps: int,
        reconstruct: bool,
        fragment_ids: list[int] | None = None,
        quantum_fragment_ids: list[int] | None = None,
        solver_assignments: dict[int, str] | None = None,
        partial_assignments: bool = False,
        replay_fragment: "int | list[int] | tuple[int, ...] | None" = None,
        allow_sampling: bool = False,
        allow_second_qpu: bool = False,
    ):
        """Submit one Stage 3 job and return the Serverless job handle.

        A job *samples the QPU* iff ``qpu_backend`` is set and no counts file is
        replayed.  That is only permitted when ``allow_sampling`` is True, the
        budget gate passes, and the counter is then incremented exactly once —
        here, after the gateway accepted the job.

        ``replay_fragment`` names the fragment whose counts are replayed.  It may be an
        integer or a one-element list/tuple (``solve_molecule`` passes a list).  An empty
        or invalid value raises, and so does a sequence of several ids: Stage 3 stages the
        counts of exactly one fragment per job, and the first id is never chosen silently.
        This normalization runs before any budget check, bundle check or network call.

        ``solver_assignments`` is by default a *complete* map: Stage 3 is then told not to
        fall back to the orbital-count policy for any fragment, and rejects a map that does
        not cover every fragment in the embedding.  ``partial_assignments=True`` is for a
        job restricted by ``fragment_ids`` whose map names only the fragments it solves:
        the listed fragments are routed explicitly, and the threshold policy stays enabled
        only for fragments the job never solves.

        This is the single-fragment-counts path.  Fragment-specific measurement plans (several
        fragments, each with its own live/replay mode) use :meth:`_submit_planned_stage3`.
        """
        replay_fragment = _normalize_single_fragment(replay_fragment, "_submit_stage3 replay_fragment")
        if replay_fragment is not None:
            # Counts must belong to a fragment this job actually solves ...
            if fragment_ids is not None and replay_fragment not in {
                    _coerce_fragment_id(i, "_submit_stage3 fragment_ids") for i in fragment_ids}:
                raise ChallengeError(
                    f"_submit_stage3: counts for fragment {replay_fragment} were supplied, but the "
                    f"job's fragment_ids are {sorted(int(i) for i in fragment_ids)}. Counts of one "
                    "fragment are never replayed for another.")
            # ... and no other fragment may need quantum counts that are not staged: with a
            # backend set, the container would otherwise sample the QPU for it.
            needs_counts = {_coerce_fragment_id(i, "_submit_stage3 quantum_fragment_ids")
                            for i in (quantum_fragment_ids or [])}
            needs_counts |= {_coerce_fragment_id(k, "_submit_stage3 solver_assignments key")
                             for k, v in (solver_assignments or {}).items()
                             if str(v).lower() in ("sqd", "trimsqd")}
            unstaged = sorted(needs_counts - {replay_fragment})
            if unstaged:
                raise ChallengeError(
                    f"_submit_stage3: only the counts of fragment {replay_fragment} can be staged, "
                    f"but the job also routes fragment(s) {unstaged} to SQD/TrimSQD. Multi-fragment "
                    "counts staging is not implemented yet; nothing was submitted.")
        will_sample = qpu_backend is not None and replay_fragment is None
        if will_sample:
            if not allow_sampling:
                raise ChallengeError(
                    f"[{label}] this call would sample the QPU, but live sampling was not "
                    "requested (execution='live_qpu'). Nothing was submitted."
                )
            self._check_qpu_budget(allow_second=allow_second_qpu)

        self._assert_bundle_deployed()

        kw: dict[str, Any] = dict(
            run_name=self._run_name(ligand),
            basis=BASIS,
            fci_threshold=FCI_THRESHOLD,
            sqd_threshold=SQD_THRESHOLD,
            qpu_backend=qpu_backend,
            n_shots=int(n_shots),
            n_reps=int(n_reps),
            solver_mode=solver_mode,
            sqd_config=dict(sqd_config) if sqd_config else None,
            trim_config=dict(trim_config) if trim_config else None,
            reconstruct=bool(reconstruct),
        )
        if fragment_ids is not None:
            kw["fragment_ids"] = [int(i) for i in fragment_ids]
        if quantum_fragment_ids is not None:
            kw["quantum_fragment_ids"] = [int(i) for i in quantum_fragment_ids]
        if solver_assignments is not None:
            # JSON-safe: string keys.  A complete map switches the threshold policy off in
            # Stage 3; a partial map (single-fragment jobs) leaves it on for unsolved ids.
            kw["solver_assignments"] = {str(int(k)): str(v) for k, v in solver_assignments.items()}
            kw["use_threshold_fallback"] = bool(partial_assignments)
        uploads = []
        if replay_fragment is not None:
            counts_p = self._require_counts(ligand, replay_fragment, label)
            kw["counts_fragment_id"] = int(replay_fragment)
            remote = self._live_counts_path(ligand, replay_fragment).name
            uploads.append((counts_p, remote))
            kw["counts_file"] = remote

        print(f"Submitting [{label}] {ligand}: solver_mode={solver_mode}, backend={qpu_backend}, "
              f"replay={'yes' if replay_fragment is not None else 'no'}, "
              f"live QPU sampling={'YES' if will_sample else 'no'}")

        job = self._dispatch_stage3(ligand, kw, uploads)
        print(f"  job_id: {job.job_id}")
        if will_sample:
            self._record_qpu_sampling()
        return job

    def _dispatch_stage3(self, ligand: str, kw: dict, uploads=()):
        """Upload ``uploads`` ([(src, remote_name)]) and submit one Stage 3 job; return the handle.

        Retries once on transient network failures (connection reset / AUTH1001).  On failure the
        cached client is cleared so the retry gets a fresh socket.
        """
        _MAX_SUBMIT_ATTEMPTS = 2
        for _attempt in range(_MAX_SUBMIT_ATTEMPTS):
            try:
                fn = self._stage3_fn()
                # Upload THIS checkout's Stage 1/2 artifacts so the solver reads exactly the
                # Hamiltonians whose identity the notebook verifies afterwards.
                self._ensure_stage3_files(ligand, fn)
                for src, remote in uploads:
                    self._upload_file_once(src, remote, fn)
                return fn.run(function_size="l", **kw)
            except Exception as _net_exc:
                _exc_str = str(_net_exc)
                _is_transient = (
                    "Connection" in _exc_str
                    or "ConnectionReset" in _exc_str
                    or "AUTH1001" in _exc_str
                    or "ProtocolError" in _exc_str
                )
                if _attempt < _MAX_SUBMIT_ATTEMPTS - 1 and _is_transient:
                    print(
                        f"  [retry] transient network error on attempt {_attempt + 1} "
                        f"— refreshing connection and retrying...\n  ({_exc_str[:120]})",
                        flush=True,
                    )
                    self._serverless = None   # force fresh client on next _connect()
                    self._uploaded.clear()    # re-upload files with the new function handle
                    continue
                raise  # non-transient or exhausted retries

    # ------------------------------------------------------------------
    # Fragment solve
    # ------------------------------------------------------------------

    def solve_fragment(
        self,
        ligand: str,
        fragment_id: int,
        method: str = "sqd",
        *,
        execution: str = "replay",
        sqd_config: dict | None = None,
        trim_config: dict | None = None,
        n_shots: int = DEFAULT_N_SHOTS,
        n_reps: int = DEFAULT_N_REPS,
        qpu_backend: str | None = None,
        allow_second_qpu: bool = False,
        poll_seconds: int = 30,
    ) -> dict:
        """Run a single-fragment CCSD / SQD / TrimSQD solve via Qiskit Serverless Stage 3.

        Parameters
        ----------
        ligand : str
            ``"ixazomib"`` or ``"bortezomib"``.
        fragment_id : int
            Fragment index.
        method : str
            ``"sqd"``, ``"trimsqd"``, ``"both"`` (SQD, then TrimSQD on the same
            counts), or ``"ccsd"``.
        execution : str
            ``"prepared"``  — load from a saved checkpoint or archive; no submission.
            ``"replay"``    — replay frozen/live counts (classical post-processing
                              only).  Raises if no counts exist; never samples.
            ``"live_qpu"``  — submit a new QPU sampling job.  Only mode that can
                              touch the QPU; subject to the budget gate.
        sqd_config, trim_config : dict, optional
            Classical post-processing settings.
        n_shots, n_reps, qpu_backend
            QPU circuit settings (recorded; used only by ``live_qpu``).  ``qpu_backend=None``
            resolves to ``IBM_QUANTUM_BACKEND`` from ``.env``.
        allow_second_qpu : bool
            Override the QPU budget gate (organizer authorization).

        Returns
        -------
        dict with method_used, energy_ha, sqd_energy_ha, trimsqd_energy_ha,
        ccsd_energy_ha, hamiltonian_sha256, qpu_job_id, counts_b64 (when sampled).
        """
        self._check_fragment_request(method, execution)
        # Quantum experiments are restricted to the frozen final selection (Stage D).
        fragment_id = self._resolve_quantum_fragment(fragment_id, f"solve_fragment({method!r})")
        if execution == "prepared":
            return self._load_fragment_checkpoint(ligand, fragment_id, method)
        job, record = self._submit_fragment_job(
            ligand, fragment_id, method, execution=execution, sqd_config=sqd_config,
            trim_config=trim_config, n_shots=n_shots, n_reps=n_reps, qpu_backend=qpu_backend,
            allow_second_qpu=allow_second_qpu)
        return self._collect_fragment(job, record, poll_seconds)

    # ------------------------------------------------------------------
    # Fragment solve in two phases: submit_fragment() and retrieve_fragment()
    # ------------------------------------------------------------------

    def submit_fragment(
        self,
        ligand: str,
        fragment_id: int,
        method: str = "sqd",
        *,
        execution: str = "replay",
        sqd_config: dict | None = None,
        trim_config: dict | None = None,
        n_shots: int = DEFAULT_N_SHOTS,
        n_reps: int = DEFAULT_N_REPS,
        qpu_backend: str | None = None,
        allow_second_qpu: bool = False,
    ) -> dict:
        """Submit one single-fragment Stage 3 job and return without waiting.

        Same arguments, validation and QPU budget gate as :meth:`solve_fragment` (with
        ``execution`` ``"replay"`` or ``"live_qpu"``).  The Serverless job id is written to
        ``checkpoints/stage3/pending/`` as soon as the gateway accepts the job, so
        :meth:`retrieve_fragment` can collect the result later, also from a new kernel.

        **A second job is never submitted for the same (ligand, fragment, method).**  While a
        job record exists this method only reconnects to that job, to prove it is reachable,
        and returns the existing record.  No budget check applies then, because nothing new is
        created.  If the arguments differ from the saved job's, it refuses instead of silently
        reusing a job that was started with other settings.
        """
        self._check_fragment_request(method, execution)
        fragment_id = self._resolve_quantum_fragment(fragment_id, f"submit_fragment({method!r})")
        if execution == "prepared":
            raise ChallengeError(
                "submit_fragment: execution='prepared' loads a saved result and submits nothing. "
                "Use retrieve_fragment(..., execution='prepared').")
        _, record = self._submit_fragment_job(
            ligand, fragment_id, method, execution=execution, sqd_config=sqd_config,
            trim_config=trim_config, n_shots=n_shots, n_reps=n_reps, qpu_backend=qpu_backend,
            allow_second_qpu=allow_second_qpu)
        return record

    def retrieve_fragment(
        self,
        ligand: str,
        fragment_id: int,
        method: str = "sqd",
        *,
        execution: str | None = None,
        job_id: str | None = None,
        poll_seconds: int = 30,
    ) -> dict:
        """Collect the result of the job started by :meth:`submit_fragment`.

        Reconnects to the saved Serverless job and waits until it ends.  Interrupting the wait
        is safe: the job keeps running and its record is untouched, so the call can be repeated.
        The result then gets exactly the checks :meth:`solve_fragment` applies (bundle API,
        requested fragment, requested method) and is saved as
        ``checkpoints/stage3/{ligand}_frag{id}_{method}.json``; the job record is removed.

        ``execution=None`` uses the mode the job was submitted with.  ``"prepared"`` loads the
        saved result and contacts nothing.  A job that ended in ERROR or was cancelled has its
        record renamed ``*.failed-<job id>.json``, which allows a new submission.  A job whose
        result is *refused* keeps its record (its output is the evidence; the QPU time is spent).

        ``job_id`` adopts a job that was submitted *without* a record (for example by the
        single-call cell this API replaced).  ``execution`` must then be given explicitly, as
        ``"live_qpu"`` if that job sampled the QPU (so its counts are saved) or ``"replay"``.
        The result is still checked for the requested fragment and method, so a wrong id
        cannot be saved as this fragment's result.
        """
        if execution is not None and execution not in ("prepared", "replay", "live_qpu"):
            raise ChallengeError(
                f"execution must be None, 'prepared', 'replay' or 'live_qpu', got {execution!r}")
        self._check_fragment_request(method, execution or "replay")
        fragment_id = self._resolve_quantum_fragment(fragment_id, f"retrieve_fragment({method!r})")
        if execution == "prepared":
            return self._load_fragment_checkpoint(ligand, fragment_id, method)
        path = self._pending_job_path(ligand, fragment_id, method)
        record = self._read_job_record(path, ligand, fragment_id, method)
        if job_id is not None:
            if execution not in ("replay", "live_qpu"):
                raise ChallengeError(
                    "retrieve_fragment: adopting a job_id needs execution='replay' or 'live_qpu' "
                    "(the mode that job was submitted with).")
            if record is not None and record["job_id"] != str(job_id):
                raise ChallengeError(
                    f"retrieve_fragment: a different job is already recorded for this fragment "
                    f"(job_id={record['job_id']}). Nothing was collected.")
            if record is None:
                record = {"ligand": ligand, "fragment_id": int(fragment_id), "method": method,
                          "execution": execution, "job_id": str(job_id),
                          "live_qpu_sampling": execution == "live_qpu", "adopted": True}
                self._write_json(path, record)
        if record is None:
            saved = self.ckpt_dir / "stage3" / f"{ligand}_frag{fragment_id}_{method}.json"
            if job_id is None and execution is None and saved.is_file():
                print(f"No job is pending for {ligand} fragment {fragment_id} {method!r}; "
                      f"loading the saved result {saved.name}.")
                return self._load_fragment_checkpoint(ligand, fragment_id, method)
            raise ChallengeError(
                f"retrieve_fragment: no submitted {method!r} job for {ligand} fragment {fragment_id} "
                f"({path.name} not found). Run the submission step first.")
        if execution is not None and record["execution"] != execution:
            raise ChallengeError(
                f"retrieve_fragment: the saved job was submitted with execution={record['execution']!r}, "
                f"but execution={execution!r} was requested. Nothing was collected.")
        job = self._reconnect_fragment_job(record, path)
        print(f"Retrieving [{ligand} fragment {fragment_id} {method}] job_id={record['job_id']}")
        return self._collect_fragment(job, record, poll_seconds)

    @staticmethod
    def _check_fragment_request(method: str, execution: str) -> None:
        """Validate the method and execution mode of a single-fragment request."""
        if method not in FRAGMENT_METHODS:
            raise ChallengeError(f"method must be one of {FRAGMENT_METHODS}, got {method!r}")
        if execution not in ("prepared", "replay", "live_qpu"):
            raise ChallengeError(
                f"execution must be 'prepared', 'replay' or 'live_qpu', got {execution!r}")

    def _pending_job_path(self, ligand: str, fragment_id: int, method: str) -> Path:
        return self.ckpt_dir / "stage3" / "pending" / f"{ligand}_frag{int(fragment_id)}_{method}.job.json"

    def _read_job_record(self, path: Path, ligand: str, fragment_id: int, method: str) -> dict | None:
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ChallengeError(f"Unreadable job record {path}: {exc}. Nothing was submitted.") from exc
        if (not isinstance(record, dict) or not record.get("job_id")
                or record.get("ligand") != ligand or record.get("method") != method
                or record.get("fragment_id") != int(fragment_id)):
            raise ChallengeError(
                f"Job record {path} does not describe {ligand} fragment {fragment_id} {method!r}. "
                "Nothing was submitted; inspect or delete the file.")
        return record

    def _reconnect_fragment_job(self, record: dict, path: Path):
        live = self._fragment_jobs.get(record["job_id"])
        if live is not None:
            return live                       # same kernel that submitted it: no network round-trip
        try:
            job = self._connect().job(record["job_id"])
            if job is None:
                raise ValueError("the gateway returned no job")
            self._fragment_jobs[record["job_id"]] = job
        except Exception as exc:
            raise ChallengeError(
                f"Could not reconnect to Serverless job {record['job_id']!r}: {exc}\n"
                f"Nothing was submitted.  If the job no longer exists, delete {path} to submit a new one."
            ) from exc
        return job

    def _submit_fragment_job(self, ligand, fragment_id, method, *, execution, sqd_config,
                             trim_config, n_shots, n_reps, qpu_backend, allow_second_qpu):
        """Submit (or, if a job record exists, re-attach to) a single-fragment job.

        Returns ``(job, record)``.  The record is persisted before this method returns.
        """
        if method == "ccsd":
            if execution == "live_qpu":
                raise ChallengeError("method='ccsd' is classical; it never uses the QPU.")
            backend = None
        else:
            self._validate_solver_settings(sqd_config, trim_config)
            backend = resolve_qpu_backend(qpu_backend)
        classical = method == "ccsd"
        request = {
            "ligand": ligand, "fragment_id": int(fragment_id), "method": method,
            "execution": execution, "qpu_backend": backend,
            "n_shots": None if classical else int(n_shots),
            "n_reps": None if classical else int(n_reps),
            "sqd_config": None if classical else sqd_config,
            "trim_config": None if classical else trim_config,
        }

        path = self._pending_job_path(ligand, fragment_id, method)
        saved = self._read_job_record(path, ligand, fragment_id, method)
        if saved is not None:
            canon = lambda v: json.dumps(v, sort_keys=True, default=float)  # noqa: E731
            differs = sorted(k for k, v in request.items() if canon(v) != canon(saved.get(k)))
            if differs:
                raise ChallengeError(
                    f"A {method!r} job for {ligand} fragment {fragment_id} was already submitted "
                    f"(job_id={saved['job_id']}) with different {differs}. Nothing was submitted. "
                    "Retrieve that job, or delete "
                    f"{path} to discard it and submit again.")
            job = self._reconnect_fragment_job(saved, path)
            print(f"Already submitted: job_id={saved['job_id']} — reconnected, nothing resubmitted.")
            return job, saved

        if method == "ccsd":
            # solver_mode is only a generic Stage 3 argument here (no backend is given, so no
            # quantum solver is ever built).  CCSD is requested explicitly for this fragment
            # rather than reached through the absence of a quantum solver.
            solver_mode, replay_fragment, sampling = "sqd", None, False
            assignments, partial = {int(fragment_id): "CCSD"}, True
        else:
            assignments, partial = None, False
            solver_mode = method
            sampling = execution == "live_qpu"
            if sampling:
                existing = self.resolve_counts(ligand, fragment_id)
                if existing is not None and not allow_second_qpu:
                    raise ChallengeError(
                        f"Counts already exist ({existing.name}); refusing to sample the QPU "
                        "again. Use execution='replay'."
                    )
                replay_fragment = None
            else:
                self._require_counts(ligand, fragment_id, f"solve_fragment({method!r}, replay)")
                replay_fragment = int(fragment_id)

        job = self._submit_stage3(
            ligand, label=f"fragment {fragment_id} {method}",
            solver_mode=solver_mode, qpu_backend=backend,
            sqd_config=sqd_config, trim_config=trim_config,
            n_shots=n_shots, n_reps=n_reps, reconstruct=False,
            fragment_ids=[int(fragment_id)], replay_fragment=replay_fragment,
            solver_assignments=assignments, partial_assignments=partial,
            allow_sampling=sampling, allow_second_qpu=allow_second_qpu,
        )
        record = {**request, "job_id": str(job.job_id), "solver_mode": solver_mode,
                  "live_qpu_sampling": bool(sampling),
                  "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        # Written immediately: this id is the only handle to a (possibly QPU-consuming) job.
        self._write_json(path, record)
        self._fragment_jobs[record["job_id"]] = job
        return job, record

    def _collect_fragment(self, job, record: dict, poll_seconds: int) -> dict:
        """Wait for ``job``, validate its result, save it and retire the job record."""
        ligand, fragment_id, method = record["ligand"], int(record["fragment_id"]), record["method"]
        path = self._pending_job_path(ligand, fragment_id, method)
        try:
            result = self._wait_and_collect({"fragment": job}, poll_seconds)["fragment"]
        except ChallengeError:
            try:
                ended_badly = str(job.status()).upper() in ("ERROR", "FAILED", "CANCELED", "CANCELLED")
            except Exception:
                ended_badly = False
            if ended_badly and path.is_file():
                failed = path.with_name(path.name[: -len(".json")] + f".failed-{record['job_id']}.json")
                path.rename(failed)
                self._fragment_jobs.pop(record["job_id"], None)
                print(f"The job ended unsuccessfully; its record was kept as {failed.name}. "
                      "A new submission is now possible.")
            raise
        self._validate_bundle_api(result, "fragment solve")
        self._check_result_fragment(result, fragment_id, f"solve_fragment({method!r})")
        self._check_method_used(result, method)

        # Persist counts ONLY when this job produced them (a replay returns the same bytes).
        if record.get("live_qpu_sampling") and result.get("counts_b64"):
            self._b64_to_file(result["counts_b64"],
                              self._live_counts_path(ligand, fragment_id), gunzip=True)

        slim = {k: v for k, v in result.items() if k != "counts_b64"}
        self._write_json(
            self.ckpt_dir / "stage3" / f"{ligand}_frag{fragment_id}_{method}.json", slim)
        path.unlink(missing_ok=True)
        self._fragment_jobs.pop(record["job_id"], None)
        return result

    @staticmethod
    def _check_method_used(result: dict, method: str) -> None:
        used = str(result.get("method_used", "")).lower()
        ok = {
            "ccsd": {"ccsd"},
            "sqd": {"sqd", "both"},
            "trimsqd": {"trimsqd", "both"},
            "both": {"sqd", "trimsqd", "both"},
        }[method]
        if used not in ok:
            raise ChallengeError(
                f"Requested method {method!r} but Stage 3 reports method_used={used!r}. "
                "The reported solver differs from the requested one; check the job logs. "
                "This result is not used."
            )

    def _load_fragment_checkpoint(self, ligand: str, fragment_id: int, method: str) -> dict:
        ckpt_p = self.ckpt_dir / "stage3" / f"{ligand}_frag{fragment_id}_{method}.json"
        if ckpt_p.is_file():
            with ckpt_p.open("r") as fh:
                return json.load(fh)
        key = f"{ligand}:{fragment_id}"
        if method == "ccsd":
            ref = self.load_reference_results().get(key, {})
            if ref.get("ccsd_energy_ha") is not None:
                print(f"[REFERENCE] CCSD baseline for {key} from reference_results.json")
                return {"method_used": "ccsd", "fragment_id": int(fragment_id),
                        "ccsd_energy_ha": float(ref["ccsd_energy_ha"]),
                        "provenance": "reference_results"}
        else:
            archived = self.load_archived_results()
            if isinstance(archived.get(key), dict):
                print(f"[ARCHIVED] Loaded {key} from archived_lab_results.json")
                return {**archived[key], "provenance": "archived"}
        raise ChallengeError(
            f"No saved fragment result for ({ligand}, {fragment_id}, {method}).\n"
            "Run solve_fragment with execution='replay' (needs counts) or ask the "
            "organizers for archived_lab_results.json."
        )

    # ------------------------------------------------------------------
    # Tuning grid (Section 3c)
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_method(point: dict) -> str:
        method = point.get("solver_method") or point.get("solver_mode")
        if method not in QUANTUM_METHODS:
            raise ChallengeError(
                f"Tuning-grid point {point.get('label')!r} has solver {method!r}; "
                f"grid points must be one of {QUANTUM_METHODS} (not 'both')."
            )
        return method

    def _merge_grid_definitions(self, rows: list[dict]) -> list[dict]:
        """Attach the organizer grid's settings to rows that only carry label/energy."""
        try:
            defs = {p["label"]: p for p in self.load_tuning_grid()}
        except ChallengeError:
            defs = {}
        merged = []
        for r in rows:
            r = dict(r)
            d = defs.get(r.get("label"), {})
            r.setdefault("solver_method", r.get("solver_mode") or d.get("solver_mode")
                         or d.get("solver_method"))
            r.setdefault("solver_mode", r["solver_method"])
            if r.get("sqd_config") is None and d.get("sqd_config") is not None:
                r["sqd_config"] = dict(d["sqd_config"])
            if r.get("trim_config") is None and d.get("trim_config") is not None:
                r["trim_config"] = dict(d["trim_config"])
            merged.append(r)
        return merged

    # Paths for tuning-grid results and resumability checkpoints.  All of them are keyed by
    # (ligand, fragment) so that a grid evaluated for one fragment can never be loaded,
    # resumed or promoted as another fragment's grid.  The jobs/partial files are written
    # and read during run_tuning_grid(execution="replay") only.
    def _grid_results_path(self, ligand: str, fragment_id: int) -> "Path":
        """Final grid rows for one fragment (what ``execution='prepared'`` reloads)."""
        return self.ckpt_dir / "stage3" / f"{ligand}_frag{int(fragment_id)}_tuning_results.json"

    def _grid_jobs_path(self, ligand: str, fragment_id: int) -> "Path":
        """Maps grid label → Serverless job_id (written immediately after submission)."""
        return self.ckpt_dir / "stage3" / f"{ligand}_frag{int(fragment_id)}_tuning_jobs.json"

    def _grid_partial_path(self, ligand: str, fragment_id: int) -> "Path":
        """Maps grid label → completed result row (written as each job finishes)."""
        return self.ckpt_dir / "stage3" / f"{ligand}_frag{int(fragment_id)}_tuning_results_partial.json"

    @staticmethod
    def _check_result_fragment(result: dict, fragment_id: int, context: str) -> None:
        """A single-fragment Stage 3 result that names its fragment must name the requested one."""
        reported = result.get("fragment_id") if isinstance(result, dict) else None
        if reported is not None and _coerce_fragment_id(reported, f"{context} result fragment_id") != fragment_id:
            raise ChallengeError(
                f"{context}: requested fragment {fragment_id} but the Stage 3 result is for "
                f"fragment {reported}. The result is not used.")

    def run_tuning_grid(
        self,
        *,
        execution: str = "prepared",
        ligand: str = SELECTED_LIGAND,
        fragment_id: int | None = None,
        n_shots: int = DEFAULT_N_SHOTS,
        n_reps: int = DEFAULT_N_REPS,
        qpu_backend: str | None = None,
        poll_seconds: int = 30,
    ) -> list[dict]:
        """Evaluate every organizer grid point on the SAME frozen counts.

        ``execution="prepared"`` loads saved or archived results.
        ``execution="replay"`` submits one classical post-processing job per grid
        point; it requires counts and never samples the QPU.

        **Kernel-restart safe:** on re-entry the method reconnects to any
        in-flight Serverless jobs using job IDs persisted to
        ``checkpoints/stage3/tuning_jobs.json`` and skips grid points whose
        results were already written to
        ``checkpoints/stage3/tuning_results_partial.json``.  Only genuinely
        un-submitted points are re-submitted.

        Returns a list of rows ``{label, solver_method, energy_ha, sqd_config,
        trim_config, hamiltonian_sha256, provenance, fragment_id}``.

        Every result, job-id and partial-progress file is specific to ``(ligand,
        fragment_id)``; a grid evaluated for one fragment is never loaded for another.
        """
        if execution not in ("prepared", "replay"):
            raise ChallengeError(f"execution must be 'prepared' or 'replay', got {execution!r}")
        # The fragment comes from the frozen final selection only (``None`` → its default).
        fragment_id = self._resolve_quantum_fragment(
            fragment_id, f"run_tuning_grid({execution})")
        qpu_backend = resolve_qpu_backend(qpu_backend)
        saved_p = self._grid_results_path(ligand, fragment_id)
        jobs_p = self._grid_jobs_path(ligand, fragment_id)
        partial_p = self._grid_partial_path(ligand, fragment_id)

        if execution == "prepared":
            if saved_p.is_file():
                saved = json.loads(saved_p.read_text())
                if saved.get("fragment_id") is not None and int(saved["fragment_id"]) != fragment_id:
                    raise ChallengeError(
                        f"{saved_p.name} belongs to fragment {saved['fragment_id']}, not "
                        f"{fragment_id}; refusing to reuse it.")
                rows = saved.get("tuning_results", [])
                prov = "saved"
            else:
                archived = self.load_archived_results()
                rows = list(archived.get(f"{ligand}:{fragment_id}:tuning_results") or [])
                if not rows and archived.get("tuning_results_fragment_id") is not None \
                        and int(archived["tuning_results_fragment_id"]) == fragment_id:
                    rows = list(archived.get("tuning_results", []))
                prov = "archived"
            if not rows:
                raise ChallengeError(
                    f"No saved or archived tuning-grid results for ({ligand}, fragment "
                    f"{fragment_id}). Run with execution='replay' (requires that fragment's "
                    "counts) or ask the organizers for archived results for this fragment "
                    "(archived_lab_results.json)."
                )
            print(f"Loaded {len(rows)} {prov} tuning-grid results for fragment {fragment_id}.")
            return [{**r, "fragment_id": fragment_id, "provenance": r.get("provenance", prov)}
                    for r in self._merge_grid_definitions(rows)]

        # ── execution == "replay" ─────────────────────────────────────────────
        grid = self.load_tuning_grid()
        for p in grid:
            self._grid_method(p)
        self._require_counts(ligand, fragment_id, "run_tuning_grid(replay)")
        self._validate_all_grid_settings(grid)

        # ── Load partial progress from a previous (interrupted) run ──────────
        # partial_rows: label → already-complete result row
        partial_rows: dict[str, dict] = {}
        if partial_p.is_file():
            try:
                partial_rows = json.loads(partial_p.read_text())
                print(
                    f"[run_tuning_grid] Resuming: {len(partial_rows)} / {len(grid)} "
                    "grid points already complete from a previous run.",
                    flush=True,
                )
            except Exception:
                partial_rows = {}

        # Saved job IDs from a previous (incomplete) submission round
        saved_job_ids: dict[str, str] = {}
        if jobs_p.is_file():
            try:
                saved_job_ids = json.loads(jobs_p.read_text())
            except Exception:
                saved_job_ids = {}

        # ── Submit / reconnect only the outstanding grid points ───────────────
        pending_grid = [p for p in grid if p["label"] not in partial_rows]

        if not pending_grid:
            print("[run_tuning_grid] All grid points already complete — skipping submission.")
        else:
            # Determine which labels need a fresh submit vs reconnect
            needs_submit = [p for p in pending_grid if p["label"] not in saved_job_ids]
            needs_reconnect = [p for p in pending_grid if p["label"] in saved_job_ids]

            if needs_reconnect:
                print(
                    f"[run_tuning_grid] Reconnecting to {len(needs_reconnect)} in-flight job(s).",
                    flush=True,
                )
            if needs_submit:
                print(
                    f"[run_tuning_grid] Submitting {len(needs_submit)} new grid job(s).",
                    flush=True,
                )

            new_jobs: dict[str, Any] = {}
            for p in needs_submit:
                label = p["label"]
                job = self._submit_stage3(
                    ligand, label=f"grid {label}",
                    solver_mode=self._grid_method(p), qpu_backend=qpu_backend,
                    sqd_config=p.get("sqd_config"), trim_config=p.get("trim_config"),
                    n_shots=n_shots, n_reps=n_reps, reconstruct=False,
                    fragment_ids=[int(fragment_id)], replay_fragment=int(fragment_id),
                    allow_sampling=False,
                )
                saved_job_ids[label] = job.job_id
                new_jobs[label] = job

            # Persist job IDs immediately so a restart can reconnect
            self._write_json(jobs_p, saved_job_ids)

            # Reconnect to jobs submitted in a previous kernel session via the
            # QiskitServerless client's .job(job_id) lookup.
            reconnected_jobs: dict[str, Any] = {}
            if needs_reconnect:
                client = self._connect()
                for p in needs_reconnect:
                    label = p["label"]
                    job_id = saved_job_ids[label]
                    try:
                        job = client.job(job_id)
                        if job is None:
                            raise ValueError("returned None")
                        reconnected_jobs[label] = job
                        print(f"  Reconnected {label}: job_id={job_id}", flush=True)
                    except Exception as exc:
                        raise ChallengeError(
                            f"Could not reconnect to Serverless job {job_id!r} for grid point "
                            f"{label!r}: {exc}\n"
                            f"If the job expired, delete checkpoints/stage3/{jobs_p.name} "
                            "and rerun to resubmit."
                        ) from exc

            # Collect pending jobs, checkpointing each result as it arrives
            all_pending_jobs = {**new_jobs, **reconnected_jobs}
            grid_by_label = {p["label"]: p for p in grid}
            self._collect_grid_with_checkpoint(
                all_pending_jobs, grid_by_label, partial_rows, poll_seconds,
                fragment_id=fragment_id, partial_path=partial_p,
            )

        # ── Assemble final ordered rows from partial_rows ─────────────────────
        rows = [{**partial_rows[p["label"]], "fragment_id": fragment_id} for p in grid]
        self._write_json(saved_p, {"ligand": ligand, "fragment_id": fragment_id,
                                   "tuning_results": rows})
        # Clean up the in-progress files now that everything is complete
        for p in (jobs_p, partial_p):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass
        return rows

    def _collect_grid_with_checkpoint(
        self,
        jobs: dict,
        grid_by_label: dict,
        partial_rows: dict,
        poll_seconds: int,
        *,
        fragment_id: int,
        partial_path: "Path",
    ) -> None:
        """Poll *jobs* until all are terminal, checkpointing each result as it
        completes.  Results are written into *partial_rows* in-place and also
        persisted to ``partial_path`` (specific to the fragment) after every completion.

        Raises :class:`ChallengeError` if any job ends in a non-DONE state, or if a result
        names a fragment other than ``fragment_id``.
        """
        terminal = {"DONE", "ERROR", "FAILED", "CANCELED", "CANCELLED"}
        pending = dict(jobs)

        while pending:
            still_pending: dict[str, Any] = {}
            for label, job in pending.items():
                status = str(job.status()).upper()
                print(f"  {label}: {status}", flush=True)
                if status not in terminal:
                    still_pending[label] = job
                    continue
                # Job reached a terminal state — collect immediately
                if status != "DONE":
                    try:
                        print(str(job.logs())[-4000:])
                    except Exception:
                        pass
                    raise ChallengeError(
                        f"Serverless job '{label}' (id={job.job_id}) ended with status {status}"
                    )
                res = job.result()
                p = grid_by_label[label]
                method = self._grid_method(p)
                self._validate_bundle_api(res, f"grid point {label}")
                self._check_result_fragment(res, fragment_id, f"grid point {label}")
                self._check_method_used(res, method)
                energy = self.energy_of(res, method)
                if energy is None:
                    raise ChallengeError(
                        f"Grid point {label!r}: no finite {method} energy returned."
                    )
                row = {
                    "label": label, "solver_method": method, "solver_mode": method,
                    "fragment_id": fragment_id,
                    "energy_ha": energy,
                    "sqd_config": dict(p.get("sqd_config") or {}),
                    "trim_config": dict(p.get("trim_config") or {}),
                    "hamiltonian_sha256": res.get("hamiltonian_sha256"),
                    "provenance": "replay",
                }
                partial_rows[label] = row
                # Write partial progress to disk so a kernel restart can resume
                self._write_json(partial_path, partial_rows)
                print(
                    f"  [checkpoint] {label} done — "
                    f"E={energy:.8f} Ha  "
                    f"({len(partial_rows)} / {len(grid_by_label)} complete)",
                    flush=True,
                )

            pending = still_pending
            if pending:
                time.sleep(poll_seconds)

    def _validate_all_grid_settings(self, grid: list[dict]) -> None:
        for p in grid:
            self._validate_solver_settings(p.get("sqd_config"), p.get("trim_config"))

    @staticmethod
    def grid_table(results: list[dict]) -> pd.DataFrame:
        """Tabulate grid results: energy, distance from best, classical-cost proxy, settings.

        ``diag_calls`` is an analytic proxy for classical cost: ``n_batches × iterations``
        batch diagonalizations, plus one merged (reporting) diagonalization per
        iteration for TrimSQD.  It is not a measured wall time.
        """
        cols = ["label", "solver_method", "energy_ha", "delta_vs_best_mHa", "diag_calls",
                "max_dim", "n_batches", "iterations", "samples_per_batch",
                "trim_ratio", "max_strings_per_trim", "max_carryover"]
        rows = []
        for r in results:
            s = r.get("sqd_config") or {}
            t = r.get("trim_config") or {}
            method = r.get("solver_method") or r.get("solver_mode")
            nb, it = s.get("n_batches"), s.get("iterations")
            diag = (nb * it + (it if method == "trimsqd" else 0)) if nb and it else None
            rows.append({
                "label": r.get("label"), "solver_method": method,
                "energy_ha": r.get("energy_ha"), "diag_calls": diag,
                "max_dim": s.get("max_dim"), "n_batches": nb, "iterations": it,
                "samples_per_batch": s.get("samples_per_batch"),
                "trim_ratio": t.get("trim_ratio") if method == "trimsqd" else None,
                "max_strings_per_trim": t.get("max_strings_per_trim") if method == "trimsqd" else None,
                "max_carryover": t.get("max_carryover") if method == "trimsqd" else None,
            })
        df = pd.DataFrame(rows, columns=[c for c in cols if c != "delta_vs_best_mHa"])
        if df.empty:
            return df.assign(delta_vs_best_mHa=pd.Series(dtype=float))[cols]
        df["energy_ha"] = pd.to_numeric(df["energy_ha"], errors="coerce")
        df["delta_vs_best_mHa"] = (df["energy_ha"] - df["energy_ha"].min()) * 1e3
        return df[cols].sort_values("energy_ha").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Configuration promotion
    # ------------------------------------------------------------------

    def _resolve_quantum_fragment(self, fragment_id: int | None, caller: str) -> int:
        """Fragment for a quantum experiment, validated against the frozen final selection.

        ``None`` → the selection's ``sqd_calibration_fragment_id`` (a convenience default
        only; it carries no special treatment).  An explicit id must be an integer and one of
        the frozen selected fragments.  A missing / unfrozen / stale final selection
        raises :class:`ChallengeError`; there is no fallback to the preliminary panel or to
        any fixed fragment.
        """
        sel = self.load_final_selection()
        if fragment_id is None:
            return int(sel["sqd_calibration_fragment_id"])
        fid = _coerce_fragment_id(fragment_id, f"{caller}: fragment_id")
        if fid not in sel["selected_fragment_ids"]:
            raise ChallengeError(
                f"{caller}: fragment {fid} is not in the frozen final selection "
                f"{sel['selected_fragment_ids']}.  Quantum execution is restricted to the "
                "fragments approved in Stage D.")
        return fid

    def validate_target_fragment(self, fragment_id: Any, *, ligand: str = SELECTED_LIGAND) -> int:
        """Validate the participant's Stage E fragment against the frozen final selection.

        Returns the fragment id as an ``int``.  Raises :class:`ChallengeError` for a
        non-integer value (including ``None`` and booleans), a fragment outside the frozen
        selection, or a selection that belongs to a different ligand.  This is the single
        gate Stage E uses; it adds no second selection policy: the allowed ids are exactly
        :meth:`final_selection_fragment_ids`.
        """
        sel = self.load_final_selection()
        if sel["ligand"] != ligand:
            raise ChallengeError(
                f"The frozen selection is for {sel['ligand']!r}, not {ligand!r}.")
        if fragment_id is None:
            raise ChallengeError(
                "No target fragment chosen. Set TARGET_FRAGMENT_ID to one of the selected "
                f"fragments {sel['selected_fragment_ids']}.")
        return self._resolve_quantum_fragment(fragment_id, "Stage E target fragment")

    def _known_hamiltonian_sha(self, ligand: str, fragment_id: int) -> str | None:
        for method in ("both", "sqd", "trimsqd", "ccsd"):
            p = self.ckpt_dir / "stage3" / f"{ligand}_frag{fragment_id}_{method}.json"
            if p.is_file():
                sha = json.loads(p.read_text()).get("hamiltonian_sha256")
                if sha:
                    return sha
        archived = self.load_archived_results().get(f"{ligand}:{fragment_id}")
        if isinstance(archived, dict):
            return archived.get("hamiltonian_sha256")
        return None

    def promote_configuration(
        self,
        grid_label: str,
        results: list[dict],
        *,
        ligand: str | None = None,
        fragment_id: int | None = None,
        sqd_config: dict | None = None,
        trim_config: dict | None = None,
        solver_mode: str | None = None,
        n_shots: int = DEFAULT_N_SHOTS,
        n_reps: int = DEFAULT_N_REPS,
        qpu_backend: str | None = None,
        measurement_artifact: str | None = None,
        hamiltonian_sha256: str | None = None,
        rationale: str | None = None,
        persist: bool = True,
    ) -> dict:
        """Freeze one evaluated grid point as the configuration Section 4 will use.

        Call this only AFTER :meth:`run_tuning_grid` and after inspecting
        :meth:`grid_table`.  The settings are taken from the selected grid row
        itself, so the label and the parameters cannot disagree.

        Parameters
        ----------
        grid_label : str
            Label of the grid point being promoted.
        results : list[dict]
            The full grid results.  Must be non-empty.
        sqd_config, trim_config, solver_mode : optional
            Only needed for rows that carry no settings of their own; if the row
            does carry settings and these disagree, promotion fails.
        measurement_artifact, hamiltonian_sha256 : optional
            Provenance overrides; otherwise derived from the counts file and the
            saved fragment result.
        rationale : str, optional
            Participant's free-text reason for the selection (stored verbatim).
        persist : bool
            ``True`` (default, historical behaviour) writes the single-fragment
            ``promoted_configuration.json``.  ``False`` only builds and returns the record
            (used by :meth:`promote_fragment_to_manifest`); nothing is written.

        Returns
        -------
        dict — frozen record with ``grid_label``, ``solver_method`` (sqd|trimsqd),
        all classical parameters, ``fragment_id``, ``hamiltonian_sha256``,
        ``counts_source`` (frozen|live) and ``counts_hash``.
        """
        if not results:
            raise ChallengeError(
                "No tuning-grid results to promote from (results is empty). Evaluate the grid "
                "with run_tuning_grid() and inspect grid_table() before promoting."
            )
        if not isinstance(grid_label, str) or not grid_label:
            raise ChallengeError(f"grid_label must be a non-empty string, got {grid_label!r}")
        row = next((r for r in results if r.get("label") == grid_label), None)
        if row is None:
            labels = [r.get("label") for r in results]
            raise ChallengeError(
                f"Grid label {grid_label!r} not found in results. Available labels: {labels}"
            )
        energy = row.get("energy_ha")
        try:
            energy = float(energy)
        except (TypeError, ValueError):
            energy = float("nan")
        if not np.isfinite(energy):
            raise ChallengeError(f"Grid point {grid_label!r} has no finite energy; cannot promote.")

        method = row.get("solver_method") or row.get("solver_mode") or solver_mode
        if method not in QUANTUM_METHODS:
            raise ChallengeError(
                f"Cannot promote solver {method!r}: Section 4 requires one of {QUANTUM_METHODS} "
                "(never 'both')."
            )
        if solver_mode is not None and solver_mode != method:
            raise ChallengeError(
                f"solver_mode={solver_mode!r} disagrees with grid point {grid_label!r} ({method!r})."
            )

        def _resolve(name: str, from_row: dict | None, given: dict | None) -> dict:
            if from_row is not None and given is not None and dict(given) != dict(from_row):
                raise ChallengeError(
                    f"{name} passed to promote_configuration differs from grid point "
                    f"{grid_label!r}; the promoted settings must be the grid point's."
                )
            chosen = from_row if from_row is not None else given
            if chosen is None:
                raise ChallengeError(
                    f"Grid point {grid_label!r} carries no {name}; pass {name}= explicitly.")
            return dict(chosen)

        sqd_cfg = _resolve("sqd_config", row.get("sqd_config"), sqd_config)
        trim_cfg = (_resolve("trim_config", row.get("trim_config"), trim_config)
                    if method == "trimsqd" or row.get("trim_config") is not None
                    or trim_config is not None else {})
        self._validate_solver_settings(sqd_cfg, trim_cfg)

        ligand = ligand or SELECTED_LIGAND
        fid = self._resolve_quantum_fragment(fragment_id, "promote_configuration")
        # A grid row that names its fragment must be a row for the fragment being promoted;
        # otherwise a grid evaluated for one fragment could be promoted as another.
        row_fid = row.get("fragment_id")
        if row_fid is not None and _coerce_fragment_id(row_fid, "grid row fragment_id") != fid:
            raise ChallengeError(
                f"Grid point {grid_label!r} was evaluated for fragment {row_fid}, but fragment "
                f"{fid} is being promoted. Pass fragment_id={row_fid} or evaluate the grid "
                f"for fragment {fid}.")
        final_sel = self.load_final_selection()
        if final_sel["ligand"] != ligand:
            raise ChallengeError(
                f"promote_configuration: ligand {ligand!r} differs from the frozen selection's "
                f"ligand {final_sel['ligand']!r}.")
        base_ham = None
        for f in self.load_classical_baseline(ligand)["fragments"]:
            if int(f["fragment_id"]) == fid:
                base_ham = f["hamiltonian_hash"]
        counts_p = self.resolve_counts(ligand, fid)
        record = {
            "selected_grid_label": grid_label,
            "grid_label": grid_label,
            "ligand": ligand,
            "fragment_id": fid,
            "solver_method": method,
            "solver_mode": method,           # same value; kept for older readers
            "sqd_config": sqd_cfg,
            "trim_config": trim_cfg,
            "n_shots": int(n_shots),
            "n_reps": int(n_reps),
            "qpu_backend": resolve_qpu_backend(qpu_backend),
            "grid_energy_ha": energy,
            "measurement_artifact": measurement_artifact or (str(counts_p) if counts_p else None),
            "hamiltonian_sha256": (hamiltonian_sha256 or row.get("hamiltonian_sha256")
                                   or self._known_hamiltonian_sha(ligand, fid)),
            "counts_source": self._counts_source(ligand, fid),
            "counts_hash": _sha256_file(counts_p) if counts_p else None,
            "rationale": rationale,
            # Binding to the frozen final selection and the baseline Hamiltonian.
            "final_selection_digest": final_sel["manifest_digest"],
            "baseline_hamiltonian_hash": base_ham,
        }
        if not persist:
            return record
        cfg_path = self.ckpt_dir / "promoted_configuration.json"
        self._write_json(cfg_path, record)
        print(f"Promoted configuration saved → {cfg_path}")
        return record

    def load_promoted_configuration(self) -> dict:
        """Reload a previously promoted configuration (survives kernel restart).

        The record must be bound to the *current* frozen final selection: a record without
        ``final_selection_digest`` (e.g. one created for the historical fragment-1
        experiment), with a different digest, or for a fragment outside the selection is
        rejected as stale.
        """
        cfg_path = self.ckpt_dir / "promoted_configuration.json"
        sel = self.load_final_selection()          # a missing selection is the first blocker
        if not cfg_path.is_file():
            raise ChallengeError(
                "No promoted configuration found. Complete Stage E (evaluate the grid, set "
                "SELECTED_GRID_LABEL) and call promote_configuration() before Stage F."
            )
        with cfg_path.open("r") as fh:
            record = json.load(fh)
        digest = record.get("final_selection_digest")
        if digest is None:
            raise ChallengeError(
                f"{cfg_path.name} carries no final_selection_digest: it predates the "
                "diagnostic-driven workflow (archived data) and cannot be used. "
                "Re-run Stage E and promote again.")
        if digest != sel["manifest_digest"]:
            raise ChallengeError(
                f"{cfg_path.name} was promoted under a different final selection "
                "(digest mismatch); it is stale. Re-run Stage E and promote again.")
        if int(record.get("fragment_id", -1)) not in sel["selected_fragment_ids"]:
            raise ChallengeError(
                f"{cfg_path.name}: fragment {record.get('fragment_id')} is not in the frozen "
                f"selection {sel['selected_fragment_ids']}.")
        return record

    # ------------------------------------------------------------------
    # Multi-fragment TrimSQD promotion manifest (separate from promoted_configuration.json)
    # ------------------------------------------------------------------

    @property
    def promotion_manifest_path(self) -> Path:
        return self.ckpt_dir / "promotion_manifest.json"

    def _manifest_context(self, ligand: str) -> tuple[dict, dict]:
        return self.load_final_selection(), self.reference_baseline(ligand)

    def start_promotion_manifest(self, ligand: str | None = None, *, overwrite: bool = False) -> dict:
        """Create an empty DRAFT manifest bound to the frozen selection and classical baseline.

        Never touches ``promoted_configuration.json``.  An existing manifest is not replaced
        unless ``overwrite=True``.
        """
        from proteasome_challenge import promotion_manifest as pm
        ligand = ligand or SELECTED_LIGAND
        if self.promotion_manifest_path.is_file() and not overwrite:
            raise ChallengeError(
                f"{self.promotion_manifest_path.name} already exists; load it, or pass overwrite=True.")
        sel, base = self._manifest_context(ligand)
        try:
            manifest = pm.new_manifest(ligand=ligand, selection=sel, baseline=base)
        except pm.PromotionManifestError as exc:
            raise ChallengeError(str(exc)) from exc
        pm.write_manifest(manifest, self.promotion_manifest_path)
        return manifest

    def _read_manifest(self) -> dict:
        from proteasome_challenge import promotion_manifest as pm
        if not self.promotion_manifest_path.is_file():
            raise ChallengeError(
                f"No {self.promotion_manifest_path.name}; call start_promotion_manifest() first.")
        try:
            return pm.read_manifest(self.promotion_manifest_path)
        except pm.PromotionManifestError as exc:
            raise ChallengeError(str(exc)) from exc

    def load_promotion_manifest(self, *, require_complete: bool = True,
                                require_measurements: bool = False) -> dict:
        """Load and strictly validate the multi-fragment manifest.

        ``require_complete=True`` (experiment mode) rejects a draft.  ``False`` accepts an
        explicitly labelled draft for inspection only.
        """
        from proteasome_challenge import promotion_manifest as pm
        manifest = self._read_manifest()
        sel, base = self._manifest_context(manifest.get("ligand") or SELECTED_LIGAND)
        try:
            return pm.validate_manifest(
                manifest, selection=sel, baseline=base, require_complete=require_complete,
                config_digest_fn=self._config_digest, require_measurements=require_measurements)
        except pm.PromotionManifestError as exc:
            raise ChallengeError(str(exc)) from exc

    def add_to_promotion_manifest(self, record: dict, *, replace: bool = False) -> dict:
        """Add one fragment's TrimSQD promotion record to the DRAFT manifest.

        ``record`` is the dict returned by :meth:`promote_configuration` (or
        :meth:`promote_fragment_to_manifest`).  Validated-result and measurement references
        are recorded only when the evidence exists on disk; otherwise they stay ``None``.
        """
        from proteasome_challenge import promotion_manifest as pm
        manifest = self.load_promotion_manifest(require_complete=False)
        ligand = manifest["ligand"]
        fid = _coerce_fragment_id(record.get("fragment_id"), "record fragment_id")
        label = record.get("selected_grid_label") or record.get("grid_label")
        validated = None
        grid_p = self._grid_results_path(ligand, fid)
        if grid_p.is_file():
            with grid_p.open("r", encoding="utf-8") as fh:
                rows = json.load(fh).get("tuning_results", [])
            row = next((r for r in rows if r.get("label") == label), None)
            if row is not None:
                validated = {"source": "tuning_results", "file": grid_p.name,
                             "sha256": _sha256_file(grid_p), "grid_label": label,
                             "hamiltonian_sha256": row.get("hamiltonian_sha256")}
        measurement = None
        if record.get("counts_hash"):
            art = record.get("measurement_artifact")
            measurement = {"counts_hash": record["counts_hash"],
                           "counts_source": record.get("counts_source"),
                           "artifact": Path(art).name if art else None}
        try:
            entry = pm.entry_from_promoted_record(
                record, config_digest=self._config_digest(record),
                validated_result=validated, measurement=measurement)
            updated = pm.with_entry(manifest, entry, replace=replace)
            sel, base = self._manifest_context(ligand)
            pm.validate_manifest(updated, selection=sel, baseline=base, require_complete=False,
                                 config_digest_fn=self._config_digest)
        except pm.PromotionManifestError as exc:
            raise ChallengeError(str(exc)) from exc
        pm.write_manifest(updated, self.promotion_manifest_path)
        return updated

    def promote_fragment_to_manifest(self, grid_label: str, results: list[dict], *,
                                     fragment_id: int, replace: bool = False, **kwargs) -> dict:
        """Promote one fragment's grid point into the manifest WITHOUT writing
        ``promoted_configuration.json`` (independent per-fragment promotion)."""
        record = self.promote_configuration(
            grid_label, results, fragment_id=fragment_id, persist=False, **kwargs)
        return self.add_to_promotion_manifest(record, replace=replace)

    def finalize_promotion_manifest(self, *, require_measurements: bool = False) -> dict:
        """Mark the manifest COMPLETE iff all selected fragments are present and valid."""
        from proteasome_challenge import promotion_manifest as pm
        manifest = self._read_manifest()
        sel, base = self._manifest_context(manifest.get("ligand") or SELECTED_LIGAND)
        try:
            pm.validate_manifest(manifest, selection=sel, baseline=base, require_complete=False,
                                 config_digest_fn=self._config_digest)
            done = pm.finalize(manifest, selection=sel, baseline=base,
                               config_digest_fn=self._config_digest,
                               require_measurements=require_measurements)
        except pm.PromotionManifestError as exc:
            raise ChallengeError(str(exc)) from exc
        pm.write_manifest(done, self.promotion_manifest_path)
        return done

    def promotion_manifest_report(self) -> list[dict]:
        """Per selected fragment: configured / validated result / measurement / genuine."""
        from proteasome_challenge import promotion_manifest as pm
        return pm.provenance_report(self.load_promotion_manifest(require_complete=False))

    @staticmethod
    def _promoted_method(promoted: dict) -> str:
        method = promoted.get("solver_method") or promoted.get("solver_mode")
        if method not in QUANTUM_METHODS:
            raise ChallengeError(
                f"Promoted configuration has solver {method!r}; Section 4 dispatches exactly the "
                f"promoted solver, one of {QUANTUM_METHODS}. 'both' is only for Sections 3a–3c."
            )
        return method

    @classmethod
    def _config_digest(cls, promoted: dict) -> str:
        payload = {
            "label": promoted.get("selected_grid_label") or promoted.get("grid_label"),
            "method": cls._promoted_method(promoted),
            "fragment_id": int(promoted["fragment_id"]),
            "sqd_config": promoted.get("sqd_config") or {},
            "trim_config": promoted.get("trim_config") or {},
            "counts_hash": promoted.get("counts_hash"),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    # ------------------------------------------------------------------
    # Full-molecule solve (Section 4)
    # ------------------------------------------------------------------

    def load_solver_assignments(self, ligand: str = SELECTED_LIGAND) -> list[dict] | None:  # noqa: E501
        """Load and validate the frozen per-fragment solver map for ``ligand``.

        Returns ``None`` when no map exists for the ligand (the file is
        ixazomib-only, or has not been published); callers then use the legacy
        orbital-threshold policy.  Raises :class:`ChallengeError` if the file
        exists but does not cover exactly the catalog's fragments.
        """
        from proteasome_challenge.solver_assignments import (
            load_solver_assignments as _load, validate_assignments)

        if not self.assignments_path.is_file():
            return None
        with self.assignments_path.open("r") as fh:
            meta = json.load(fh)
        if isinstance(meta, dict) and meta.get("ligand", ligand) != ligand:
            return None
        assignments = _load(self.assignments_path)
        cat = self.load_fragment_catalog(with_diagnostics=False, with_chemistry=False)
        expected = set(cat.loc[cat["ligand"] == ligand, "fragment_id"].astype(int))
        try:
            validate_assignments(assignments, expected)
        except ValueError as exc:
            raise ChallengeError(f"solver_assignments.json is inconsistent with the catalog: {exc}")
        return assignments

    def effective_solver_assignments(self, ligand: str, promoted=None) -> list[dict] | None:
        """Frozen classical baseline map with the participant's promotion applied.

        The frozen map contains only FCI and CCSD assignments.  If ``promoted``
        is supplied, the fragment identified by the promotion record is upgraded
        to SQD or TrimSQD; every other fragment keeps its classical assignment.

        ``promoted`` may be a single promote_configuration() dict or a list of
        such dicts.  Returns ``None`` if no frozen map exists for the ligand.
        """
        base = self.load_solver_assignments(ligand)
        if base is None:
            return None
        if promoted is None:
            promoted = []
        if isinstance(promoted, dict):
            promoted = [promoted]
        pmap = {int(p["fragment_id"]): p for p in promoted}
        out: list[dict] = []
        for a in base:
            a = dict(a)
            fid = int(a["fragment_id"])
            if fid in pmap:
                p = pmap[fid]
                a["solver"] = {"sqd": "SQD", "trimsqd": "TrimSQD"}[self._promoted_method(p)]
                a["assignment_source"] = "participant_promotion"
                a["solver_config"] = {
                    "sqd_config": p.get("sqd_config") or {},
                    "trim_config": p.get("trim_config") or {},
                    "grid_label": p.get("selected_grid_label") or p.get("grid_label"),
                }
            out.append(a)
        unknown = set(pmap) - {int(a["fragment_id"]) for a in out}
        if unknown:
            raise ChallengeError(f"Promoted fragments {sorted(unknown)} are not in the catalog.")
        return out

    def routing_plan(self, ligand: str, promoted=None) -> pd.DataFrame:
        """Planned solver for every fragment.

        Uses the explicit frozen assignment map (with ``promoted`` overriding the
        baseline) when ``challenge_data/solver_assignments.json`` covers the
        ligand; otherwise falls back to the legacy orbital-count thresholds.

        ``promoted`` may be a single promoted-configuration dict (backwards
        compatible) or a list of dicts, one per quantum-treated fragment.  Each
        dict must carry ``fragment_id`` and ``solver_method`` (``sqd`` or
        ``trimsqd``).
        """
        cat = self.load_fragment_catalog()
        sub = cat[cat["ligand"] == ligand]
        if promoted is None:
            promoted = []
        if isinstance(promoted, dict):
            promoted = [promoted]
        out = sub[["ligand", "fragment_id", "n_orb"]].copy()

        eff = self.effective_solver_assignments(ligand, promoted)
        if eff is not None:
            smap = {int(a["fragment_id"]): a["solver"] for a in eff}
            src = {int(a["fragment_id"]): a["assignment_source"] for a in eff}
            out["planned_solver"] = sub["fragment_id"].map(lambda f: smap[int(f)])
            out["assignment_source"] = sub["fragment_id"].map(lambda f: src[int(f)])
            return out.reset_index(drop=True)

        qmap: dict[int, str] = {}
        for p in promoted:
            method = self._promoted_method(p)
            qmap[int(p["fragment_id"])] = {"sqd": "SQD", "trimsqd": "TrimSQD"}[method]

        def plan(r) -> str:
            if r["n_orb"] <= FCI_THRESHOLD:
                return "FCI"
            if r["n_orb"] < SQD_THRESHOLD:
                return "CCSD"
            return qmap.get(int(r["fragment_id"]), "CCSD")

        out["planned_solver"] = sub.apply(plan, axis=1)
        return out.reset_index(drop=True)

    @staticmethod
    def solver_assignments(result: dict) -> pd.DataFrame:
        """Per-fragment solver actually used, joined with the cumulant reconstruction record."""
        summaries = result.get("fragment_summaries") or {}
        frags = result.get("fragments") or {}
        rows = []
        for k, s in summaries.items():
            rec = _by_id(frags, int(k)) or {}
            rows.append({
                "fragment_id": int(k), "solver": s.get("solver"), "n_orb": s.get("n_orb"),
                "e_corr_solver_ha": s.get("e_corr_ha"),
                "e_corr_cumulant_ha": rec.get("e_corr"),
                "rdm_integrated": rec.get("rdm_integrated"),
            })
        return pd.DataFrame(rows).sort_values("fragment_id").reset_index(drop=True)

    def solve_molecule(
        self,
        ligand: str,
        *,
        promoted=None,
        protocol: "dict | None" = None,
        measurement_plan: "dict | None" = None,
        execution: str = "prepared_or_explicit",
        allow_second_qpu: bool = False,
        poll_seconds: int = 30,
    ) -> dict:
        """Run the full EWF solve via Stage 3.

        Two calling conventions are supported.

        **Legacy (single or multi-fragment via promotion records)**::

            result = lab.solve_molecule("ixazomib", promoted=sel, execution="explicit")

        ``promoted`` is a list of promoted-configuration dicts (a single dict is
        accepted for backwards compatibility), one per quantum-treated fragment.
        Each dict must carry ``fragment_id`` and ``solver_method`` (``sqd`` or
        ``trimsqd``) plus the classical post-processing settings.  All quantum
        fragments are solved in a single Stage 3 job; every other quantum-sized
        fragment falls back to CCSD.  This path is preserved unchanged.

        **Full-experiment path (Task 4C.1)**::

            protocol = lab.load_shared_protocol()
            plan     = lab.build_measurement_plan(...)   # or load a sealed plan
            result   = lab.solve_molecule(
                "ixazomib",
                protocol=protocol,
                measurement_plan=plan,
                execution="explicit",
            )

        When ``protocol`` and ``measurement_plan`` are supplied (``promoted`` is
        not required), the full heterogeneous execution path is used.  The shared
        TrimSQD protocol determines which fragments are quantum-treated and what
        settings every solver uses; those fragment IDs come exclusively from
        ``protocol["target_fragment_ids"]`` (a runtime artifact derived from the
        frozen selection manifest — never a hard-coded list).  The measurement
        plan specifies live/replay mode for each of those fragments.  The
        checkpoint is written to a per-experiment path that encodes the source
        fragment ID, the protocol digest, and the plan digest; the legacy
        single-fragment artifact is never touched.

        Parameters
        ----------
        ligand : str
            ``"ixazomib"`` (graded) or ``"bortezomib"``.
        promoted : dict or list[dict], optional
            Legacy path: promoted-configuration record(s).
        protocol : dict, optional
            Full-experiment path: validated shared TrimSQD protocol (from
            :meth:`load_shared_protocol`).  Requires ``measurement_plan``; mutually
            exclusive with the legacy multi-fragment counts path.
        measurement_plan : dict, optional
            Full-experiment path: validated measurement plan covering all selected
            quantum fragments (from :meth:`build_measurement_plan`).
        execution : str
            ``"prepared_or_explicit"`` — use a saved result if one exists and matches,
                                         else submit.
            ``"prepared"``             — saved only; fail if absent or mismatched.
            ``"explicit"``             — submit a replay job now (needs counts).
            ``"live_qpu"``             — submit and sample the QPU (budget-gated).
        allow_second_qpu : bool
            Override the QPU budget gate (organizer authorization).

        Raises
        ------
        ChallengeError
            Configuration or identity mismatches, missing counts, validation failures.
        """
        if execution not in ("prepared_or_explicit", "prepared", "explicit", "live_qpu"):
            raise ChallengeError(f"Unknown execution mode {execution!r}")

        # Route: full-experiment path when protocol and/or measurement_plan are given.
        if protocol is not None or measurement_plan is not None:
            return self._solve_molecule_full_ewf(
                ligand,
                protocol=protocol,
                measurement_plan=measurement_plan,
                execution=execution,
                allow_second_qpu=allow_second_qpu,
                poll_seconds=poll_seconds,
            )

        # Legacy path (promoted records).
        return self._solve_molecule_legacy(
            ligand,
            promoted=promoted,
            execution=execution,
            allow_second_qpu=allow_second_qpu,
            poll_seconds=poll_seconds,
        )

    def _solve_molecule_legacy(
        self,
        ligand: str,
        *,
        promoted,
        execution: str,
        allow_second_qpu: bool,
        poll_seconds: int,
    ) -> dict:
        """Legacy single/multi-fragment promoted-configuration path (unchanged semantics)."""
        if isinstance(promoted, dict):
            promoted = [promoted]
        if not promoted:
            raise ChallengeError("solve_molecule requires at least one promoted configuration.")
        for p in promoted:
            self._promoted_method(p)  # validates solver is sqd/trimsqd
        fids = [int(p["fragment_id"]) for p in promoted]
        if len(set(fids)) != len(fids):
            raise ChallengeError("Duplicate fragment_id in promoted configurations.")
        for p in promoted:
            if p.get("ligand", ligand) != ligand:
                raise ChallengeError(
                    f"Promoted configuration is for {p.get('ligand')!r}, not {ligand!r}.")

        # Stage F consumes (i) the frozen final selection and (ii) a compatible classical
        # baseline; both are validated before anything is read or submitted.
        final_sel = self.load_final_selection()           # blocks if missing / stale
        base = self.load_classical_baseline(ligand)       # blocks if missing / stale
        if final_sel["ligand"] != ligand:
            raise ChallengeError(
                f"The frozen selection is for {final_sel['ligand']!r}, not {ligand!r}.")
        outside = [f for f in fids if f not in final_sel["selected_fragment_ids"]]
        if outside:
            raise ChallengeError(
                f"Promoted fragment(s) {outside} are not in the frozen final selection "
                f"{final_sel['selected_fragment_ids']}.")
        for p in promoted:
            if p.get("final_selection_digest") != final_sel["manifest_digest"]:
                raise ChallengeError(
                    f"Promotion for fragment {p.get('fragment_id')} is not bound to the current "
                    "frozen final selection (stale or archived). Promote again in Stage E.")
        provenance_ids = {
            "baseline_cache_key": base["identity"]["cache_key"],
            "final_selection_digest": final_sel["manifest_digest"],
        }
        digests = [self._config_digest(p) for p in promoted]
        expected_solvers = [p["solver_method"] for p in promoted]

        # --- saved / archived first ---
        if execution in ("prepared_or_explicit", "prepared"):
            saved = self._load_full_ewf_checkpoint(ligand)
            if saved is not None:
                result, provenance = saved
                for k, v in provenance_ids.items():
                    if result.get(k) != v:
                        raise ChallengeError(
                            f"The saved full-EWF result is stale: its {k} differs from the "
                            "current baseline / frozen selection. Re-run with execution='explicit'.")
                saved_digests = result.get("promoted_digests")
                if saved_digests is None:
                    saved_digests = result.get("promoted_digest")
                    saved_digests = [saved_digests] if saved_digests else []
                if not any(d in digests for d in saved_digests):
                    raise ChallengeError(
                        "The saved full-EWF result was produced with a different promoted "
                        "configuration than the current one. Re-run with execution='explicit'."
                    )
                if provenance == "archived":
                    print("[ARCHIVED] organizer result: solvers verified against the promotions, "
                          "classical settings cannot be.")
                self.validate_full_ewf(result, ligand, fids, expected_solver=expected_solvers)
                return result
            if execution == "prepared":
                raise ChallengeError(
                    "execution='prepared' requested but no saved full-EWF result found. "
                    "Use execution='explicit' to run it."
                )

        live = execution == "live_qpu"
        if live:
            for fid in fids:
                if self.resolve_counts(ligand, fid) is not None and not allow_second_qpu:
                    raise ChallengeError(
                        f"Counts already exist for fragment {fid}; refusing to sample again. "
                        "Use execution='explicit'."
                    )
            replay_fragment = None
        else:
            for fid in fids:
                counts_p = self._require_counts(ligand, fid, "solve_molecule")
                expected_hash = next(
                    p for p in promoted if int(p["fragment_id"]) == fid
                ).get("counts_hash")
                if expected_hash and _sha256_file(counts_p) != expected_hash:
                    raise ChallengeError(
                        f"The counts file for fragment {fid} changed since promotion "
                        f"({counts_p.name}). Re-evaluate the grid and promote again."
                    )
            replay_fragment = fids

        # Use a single shared classical post-processing config for all quantum fragments.
        # Stage 3 applies one sqd_config/trim_config to every quantum-sized fragment, so
        # every quantum fragment in the pilot is post-processed with the same validated
        # settings.  Each fragment's individual config is recorded for provenance.
        primary = promoted[0]
        sqd_cfg = primary.get("sqd_config")
        trim_cfg = primary.get("trim_config")
        solver_mode = primary["solver_method"]
        for p in promoted[1:]:
            if p["solver_method"] != solver_mode:
                raise ChallengeError(
                    "All quantum fragments in one Stage 3 job must share a solver_mode "
                    f"({solver_mode!r} vs {p['solver_method']!r}).")

        # Explicit routing: frozen baseline + participant promotion.  When no
        # frozen map is published for this ligand, Stage 3 keeps the legacy
        # threshold policy (backward compatible).
        eff = self.effective_solver_assignments(ligand, promoted)
        step3_assign = None
        if eff is not None:
            from proteasome_challenge.solver_assignments import assignments_to_step3_params
            params = assignments_to_step3_params(eff)
            if sorted(params["quantum_fragment_ids"]) != sorted(fids):
                raise ChallengeError(
                    "Explicit assignment map and promoted fragments disagree on the "
                    f"quantum fragments ({params['quantum_fragment_ids']} vs {sorted(fids)}).")
            step3_assign = params["solver_assignments"]

        job = self._submit_stage3(
            ligand, label=f"full EWF {solver_mode}",
            solver_mode=solver_mode,
            qpu_backend=resolve_qpu_backend(primary.get("qpu_backend")),
            sqd_config=sqd_cfg, trim_config=trim_cfg,
            n_shots=primary.get("n_shots", DEFAULT_N_SHOTS),
            n_reps=primary.get("n_reps", DEFAULT_N_REPS),
            reconstruct=True, quantum_fragment_ids=fids,
            solver_assignments=step3_assign,
            replay_fragment=replay_fragment,
            allow_sampling=live, allow_second_qpu=allow_second_qpu,
        )
        result = self._wait_and_collect({"full": job}, poll_seconds)["full"]
        self._validate_bundle_api(result, "full EWF solve")

        # Validate (numerically) before anything is saved
        self.validate_full_ewf(result, ligand, fids, expected_solver=expected_solvers)

        slim = {k: v for k, v in result.items() if k != "counts_b64"}
        slim.update(provenance_ids)
        slim["promoted_digests"] = digests
        slim["promoted_grid_labels"] = [p.get("selected_grid_label") for p in promoted]
        slim["quantum_fragment_ids"] = fids
        if eff is not None:
            slim["solver_assignments"] = {str(a["fragment_id"]): a["solver"] for a in eff}
        # Consistency of the re-solved classical fragments with the saved baseline.
        dev = self._baseline_fragment_deviation(result, base, exclude=fids)
        slim["baseline_max_classical_fragment_deviation_ha"] = dev
        if dev is not None and dev > CONSISTENCY_TOL:
            print(f"WARNING: re-solved classical fragments deviate from the saved baseline by "
                  f"{dev:.2e} Ha (> {CONSISTENCY_TOL:.0e}); inspect before comparing energies.")
        self._write_json(self.ckpt_dir / "stage3" / f"{ligand}_full_ewf_sqd.json", slim)
        # One artifact per promotion set, so substitution effects can be compared.
        tag = "-".join(str(f) for f in sorted(fids))
        self._write_json(self.ckpt_dir / "stage3" / "substitutions" / f"{ligand}_ewf_{tag}.json", slim)
        return result

    def _full_ewf_ckpt_path(self, ligand: str, *, source_fid: int, protocol_digest: str,
                             plan_digest: str) -> Path:
        """Per-experiment checkpoint path (never overlaps the legacy single-fragment file).

        Encodes the source fragment, protocol digest and plan digest so that a new
        experiment with different settings creates a new file rather than overwriting.
        """
        tag = f"src{source_fid}_{protocol_digest[:12]}_{plan_digest[:12]}"
        return self.ckpt_dir / "stage3" / f"{ligand}_full_ewf_11frag_{tag}.json"

    def _load_full_ewf_11frag_checkpoint(
        self, ligand: str, *,
        protocol_digest: str,
        plan_digest: str,
        protocol: "dict | None" = None,
        quantum_fids: "list[int] | None" = None,
        baseline: "dict | None" = None,
    ) -> "dict | None":
        """Return a previously saved full-experiment result matching this protocol+plan, or None.

        Scans every ``{ligand}_full_ewf_11frag_*.json`` file in the stage3 directory and
        returns the first one whose ``protocol_digest`` and ``measurement_plan_digest``
        fields match.  A checkpoint from a different protocol or plan is never reused.

        Every candidate record is re-validated through the full ``validate_full_ewf``
        contract (when ``protocol``, ``quantum_fids``, and ``baseline`` are supplied) before
        being returned.  A checkpoint that fails any validation gate is rejected with a
        clear warning — it is never silently presented as a valid result.

        When ``protocol`` / ``quantum_fids`` / ``baseline`` are not supplied, a lighter
        structural pass (finite energies, non-empty quantum_fragment_ids) is applied.
        The caller (``_solve_molecule_full_ewf``) always supplies all three.
        """
        stage3_dir = self.ckpt_dir / "stage3"
        if not stage3_dir.is_dir():
            return None
        for p in sorted(stage3_dir.glob(f"{ligand}_full_ewf_11frag_*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not (rec.get("protocol_digest") == protocol_digest
                    and rec.get("measurement_plan_digest") == plan_digest):
                continue

            # ── Primary gate: full validate_full_ewf contract ─────────────────
            if protocol is not None and quantum_fids is not None:
                n_q = len(quantum_fids)
                try:
                    self.validate_full_ewf(
                        rec, ligand, quantum_fids,
                        expected_solver=["trimsqd"] * n_q,
                        protocol=protocol,
                        baseline=baseline,
                    )
                except ChallengeError as exc:
                    print(f"  WARNING: checkpoint {p.name} failed revalidation "
                          f"({exc}); skipped.")
                    continue
            else:
                # Lighter structural pass when full context is unavailable.
                qfids_rec = rec.get("quantum_fragment_ids")
                if not qfids_rec:
                    print(f"  WARNING: checkpoint {p.name} has no quantum_fragment_ids; skipped.")
                    continue
                try:
                    e_tot = float(rec.get("e_total", "nan"))
                except (TypeError, ValueError):
                    e_tot = float("nan")
                if not np.isfinite(e_tot):
                    print(f"  WARNING: checkpoint {p.name} has non-finite e_total; skipped.")
                    continue
                try:
                    e_cor = float(rec.get("e_corr", "nan"))
                except (TypeError, ValueError):
                    e_cor = float("nan")
                if not np.isfinite(e_cor):
                    print(f"  WARNING: checkpoint {p.name} has non-finite e_corr; skipped.")
                    continue

            print(f"Loaded saved full-experiment EWF result from {p.name}")
            return rec
        return None

    def _solve_molecule_full_ewf(
        self,
        ligand: str,
        *,
        protocol: "dict | None",
        measurement_plan: "dict | None",
        execution: str,
        allow_second_qpu: bool,
        poll_seconds: int,
    ) -> dict:
        """Full heterogeneous EWF path: one shared TrimSQD protocol, per-fragment plan.

        Steps:
        1. Load/validate frozen classical baseline.
        2. Validate the shared TrimSQD protocol against the frozen selection and baseline.
        3. Validate the measurement plan against the selected fragments and Hamiltonians.
        4. Check live/replay constraints.
        5. Construct the complete solver assignment from the baseline map + protocol targets.
        6. Check for a saved result matching this protocol+plan.
        7. Build the job payload and submit via _submit_planned_stage3.
        8. Validate the complete result before saving.
        9. Write the per-experiment artifact; the legacy checkpoint is never touched.

        The quantum fragment IDs come solely from ``protocol["target_fragment_ids"]`` —
        a runtime value derived from the frozen selection manifest — never from a
        hard-coded list.
        """
        from collections import Counter as _Counter
        from proteasome_challenge import protocol_transfer as pt
        from proteasome_challenge import measurement_plan as mp
        from proteasome_challenge.solver_assignments import assignments_to_step3_params

        # 1. Frozen classical baseline.
        final_sel = self.load_final_selection()
        base = self.load_classical_baseline(ligand)
        if final_sel["ligand"] != ligand:
            raise ChallengeError(
                f"The frozen selection is for {final_sel['ligand']!r}, not {ligand!r}.")
        provenance_ids = {
            "baseline_cache_key": base["identity"]["cache_key"],
            "final_selection_digest": final_sel["manifest_digest"],
        }

        # 2. Validate the shared protocol.
        if protocol is None:
            raise ChallengeError(
                "solve_molecule full-experiment path: protocol is required "
                "(call load_shared_protocol() and pass protocol=...).")
        try:
            protocol = pt.validate_protocol(protocol, selection=final_sel, baseline=base)
        except pt.ProtocolTransferError as exc:
            raise ChallengeError(f"shared protocol is invalid: {exc}") from exc
        proto_digest = protocol["protocol_digest"]
        source_fid = int(protocol["source"]["fragment_id"])
        # Fragment IDs come from the protocol, which was itself derived from the frozen
        # selection manifest.  No hard-coded list exists anywhere in this code.
        quantum_fids = sorted(int(i) for i in protocol["target_fragment_ids"])
        n_quantum = len(quantum_fids)

        # 3. Validate the measurement plan.
        if measurement_plan is None:
            raise ChallengeError(
                "solve_molecule full-experiment path: measurement_plan is required "
                "(call build_measurement_plan() and pass measurement_plan=...).")
        try:
            measurement_plan = mp.validate_plan(
                measurement_plan, selection=final_sel, baseline=base,
                require_full_selection=True)
        except mp.MeasurementPlanError as exc:
            raise ChallengeError(f"measurement plan is invalid: {exc}") from exc
        plan_digest = measurement_plan["plan_digest"]
        # Every spec must be a target of the shared protocol (same fragment set).
        spec_fids = {s["fragment_id"] for s in measurement_plan["specs"]}
        not_targets = sorted(spec_fids - set(quantum_fids))
        if not_targets:
            raise ChallengeError(
                f"measurement plan includes fragment(s) {not_targets} that are not targets "
                f"of the shared protocol (targets: {quantum_fids})")
        missing_specs = sorted(set(quantum_fids) - spec_fids)
        if missing_specs:
            raise ChallengeError(
                f"measurement plan has no spec for quantum fragment(s) {missing_specs}; "
                "every protocol target needs a measurement spec")

        # 4. Live/replay constraints.
        live_specs = [s for s in measurement_plan["specs"] if s["mode"] == "live"]
        will_sample = bool(live_specs)
        if will_sample:
            if execution == "explicit":
                raise ChallengeError(
                    "execution='explicit' means replay-only; the plan includes live fragment(s) "
                    f"{[s['fragment_id'] for s in live_specs]}. Use execution='live_qpu'.")
            if execution not in ("live_qpu", "prepared_or_explicit"):
                raise ChallengeError(
                    f"live fragments in the plan require execution='live_qpu', got {execution!r}.")
            self._check_qpu_budget(allow_second=allow_second_qpu)
        elif execution == "live_qpu":
            raise ChallengeError(
                "execution='live_qpu' requested but the measurement plan has no live specs; "
                "pass execution='explicit' for a replay-only job.")

        # 5. Complete solver assignment: baseline FCI/CCSD + protocol targets → TrimSQD.
        base_map = self.load_solver_assignments(ligand)
        if base_map is None:
            raise ChallengeError(
                f"No frozen solver-assignment map for {ligand}; the full-experiment path "
                "requires challenge_data/solver_assignments.json.")
        eff_list: list[dict] = []
        for a in base_map:
            a = dict(a)
            fid = int(a["fragment_id"])
            if fid in quantum_fids:
                a["solver"] = "TrimSQD"
                a["assignment_source"] = "shared_protocol"
            eff_list.append(a)
        params = assignments_to_step3_params(eff_list)
        if sorted(params["quantum_fragment_ids"]) != quantum_fids:
            raise ChallengeError(
                f"Assignment map quantum fragments {params['quantum_fragment_ids']} "
                f"differ from protocol targets {quantum_fids}.")
        step3_assign = params["solver_assignments"]

        # Validate solver composition (fail before any network call).
        composition = _Counter(a["solver"] for a in eff_list)
        if composition.get("TrimSQD", 0) != n_quantum:
            raise ChallengeError(
                f"Expected {n_quantum} TrimSQD fragment(s) in the assignment, "
                f"got {composition.get('TrimSQD', 0)}")

        # Solver settings come from the protocol, not from any promotion record.
        tr = protocol["transferable"]
        sqd_cfg = dict(tr["sqd_config"])
        trim_cfg = dict(tr["trim_config"])

        # 6. Saved result matching this protocol+plan.
        #    _load_full_ewf_11frag_checkpoint applies the full validate_full_ewf contract
        #    (including delta reconciliation) before returning a candidate; a corrupt or
        #    stale checkpoint is rejected with a warning and skipped.
        if execution in ("prepared_or_explicit", "prepared"):
            saved = self._load_full_ewf_11frag_checkpoint(
                ligand,
                protocol_digest=proto_digest,
                plan_digest=plan_digest,
                protocol=protocol,
                quantum_fids=quantum_fids,
                baseline=base,
            )
            if saved is not None:
                # Provenance identity check: the checkpoint must match the current
                # baseline and frozen selection (catches stale artifacts from a previous run).
                for k, v in provenance_ids.items():
                    if saved.get(k) != v:
                        raise ChallengeError(
                            f"The saved full-experiment result is stale: its {k} differs "
                            "from the current baseline / frozen selection. "
                            "Re-run with execution='explicit'.")
                # Re-verify the shared protocol was actually applied — a prepared checkpoint
                # must satisfy both validate_full_ewf (already done above) and the
                # per-fragment protocol-application check (fail-closed, not skippable).
                self.verify_protocol_result(protocol, saved)
                return saved
            if execution == "prepared":
                raise ChallengeError(
                    "execution='prepared' requested but no saved full-experiment result found "
                    "matching this protocol and plan. Use execution='explicit' to run it.")

        # 7. Build payload and submit.
        job = self._submit_planned_stage3(
            ligand,
            label=f"full EWF {n_quantum}×TrimSQD src={source_fid}",
            plan=measurement_plan,
            specs=measurement_plan["specs"],
            reconstruct=True,
            quantum_fragment_ids=quantum_fids,
            solver_assignments=step3_assign,
            sqd_config=sqd_cfg,
            trim_config=trim_cfg,
            protocol=protocol,
            allow_sampling=will_sample,
            allow_second_qpu=allow_second_qpu,
        )
        result = self._wait_and_collect({"full": job}, poll_seconds)["full"]
        self._validate_bundle_api(result, f"full EWF {n_quantum}×TrimSQD solve")

        # 8. Full validation before saving anything (including delta reconciliation
        #    against the frozen classical baseline — not bypassable by the caller).
        self.validate_full_ewf(
            result, ligand, quantum_fids,
            expected_solver=["trimsqd"] * n_quantum,
            protocol=protocol, baseline=base)
        self.verify_protocol_result(protocol, result)

        # 9. Per-experiment artifact (legacy checkpoint untouched).
        slim = {k: v for k, v in result.items() if k not in ("counts_b64", "counts_b64_by_fragment")}
        slim.update(provenance_ids)
        slim["quantum_fragment_ids"] = quantum_fids
        slim["protocol_digest"] = proto_digest
        slim["protocol_document_digest"] = protocol["document_digest"]
        slim["applied_config_digest"] = protocol["applied_config_digest"]
        slim["source_fragment_id"] = source_fid
        slim["measurement_plan_digest"] = plan_digest
        slim["measurement_plan_execution"] = measurement_plan["execution"]
        # Per-fragment measurement identity.
        meas_identity: dict[str, "str | None"] = {}
        for s in measurement_plan["specs"]:
            rep = s.get("replay")
            meas_identity[str(s["fragment_id"])] = rep["counts_sha256"] if rep else None
        slim["measurement_identity_by_fragment"] = meas_identity
        # Per-quantum-fragment Hamiltonian identity.
        summ_all = result.get("fragment_summaries") or {}
        ham_identity: dict[str, "str | None"] = {}
        for fid in quantum_fids:
            s = _by_id(summ_all, fid)
            ham_identity[str(fid)] = s.get("hamiltonian_hash") if isinstance(s, dict) else None
        slim["hamiltonian_identity_by_fragment"] = ham_identity
        slim["solver_assignments"] = {str(a["fragment_id"]): a["solver"] for a in eff_list}
        dev = self._baseline_fragment_deviation(result, base, exclude=quantum_fids)
        slim["baseline_max_classical_fragment_deviation_ha"] = dev
        if dev is not None and dev > CONSISTENCY_TOL:
            print(f"WARNING: re-solved classical fragments deviate from the saved baseline by "
                  f"{dev:.2e} Ha (> {CONSISTENCY_TOL:.0e}); inspect before comparing energies.")
        ckpt = self._full_ewf_ckpt_path(
            ligand, source_fid=source_fid,
            protocol_digest=proto_digest, plan_digest=plan_digest)
        self._write_json_atomic(ckpt, slim)
        tag = "-".join(str(f) for f in quantum_fids)
        self._write_json_atomic(
            self.ckpt_dir / "stage3" / "substitutions" / f"{ligand}_ewf_{tag}.json", slim)
        return result

    @staticmethod
    def _baseline_fragment_deviation(result: dict, base: dict, exclude) -> float | None:
        """Max |E_fragment(result) − E_fragment(baseline)| over non-promoted fragments."""
        summ = result.get("fragment_summaries") or {}
        devs = []
        for f in base["fragments"]:
            fid = int(f["fragment_id"])
            if fid in set(int(x) for x in exclude):
                continue
            s = _by_id(summ, fid)
            if isinstance(s, dict) and s.get("energy_ha") is not None:
                devs.append(abs(float(s["energy_ha"]) - float(f["energy_ha"])))
        return max(devs) if devs else None

    def substitution_effects(self, ligand: str = SELECTED_LIGAND) -> pd.DataFrame:
        """Individual and cumulative effect of quantum-solver substitutions on E_total.

        Reads only the saved full-EWF artifacts in ``checkpoints/stage3/substitutions/`` that
        are bound to the *current* classical baseline and frozen final selection; incompatible
        artifacts are ignored (and counted in ``attrs['n_ignored_stale']``).  Nothing is run.
        ``delta_vs_baseline_ha`` = E_total(substituted) − E_total(baseline).
        ``delta_incremental_ha`` is relative to the largest strict-subset artifact (or the
        baseline), i.e. the marginal effect of the additional substitution(s).
        Returns an empty frame when no compatible artifact exists.
        """
        cols = ["fragments", "n_substituted", "kind", "e_total_ha",
                "delta_vs_baseline_ha", "delta_incremental_ha", "incremental_over"]
        base = self.load_classical_baseline(ligand)
        sel = self.load_final_selection()
        want = {"baseline_cache_key": base["identity"]["cache_key"],
                "final_selection_digest": sel["manifest_digest"]}
        e_base = float(base["energies"]["e_total"])
        arts: dict[tuple, float] = {}
        n_stale = 0
        d = self.ckpt_dir / "stage3" / "substitutions"
        for p in sorted(d.glob(f"{ligand}_ewf_*.json")) if d.is_dir() else []:
            rec = json.loads(p.read_text())
            if any(rec.get(k) != v for k, v in want.items()):
                n_stale += 1
                continue
            try:
                arts[tuple(sorted(int(x) for x in rec["quantum_fragment_ids"]))] = float(rec["e_total"])
            except (KeyError, TypeError, ValueError):
                n_stale += 1
        rows = []
        for key in sorted(arts, key=lambda k: (len(k), k)):
            subs = [k for k in arts if set(k) < set(key)]
            prev = max(subs, key=len) if subs else None
            e_prev = arts[prev] if prev else e_base
            rows.append({
                "fragments": list(key), "n_substituted": len(key),
                "kind": "individual" if len(key) == 1 else "cumulative",
                "e_total_ha": arts[key], "delta_vs_baseline_ha": arts[key] - e_base,
                "delta_incremental_ha": arts[key] - e_prev,
                "incremental_over": list(prev) if prev else "baseline",
            })
        df = pd.DataFrame(rows, columns=cols)
        df.attrs["n_ignored_stale"] = n_stale
        df.attrs["baseline_e_total_ha"] = e_base
        return df

    def _load_full_ewf_checkpoint(self, ligand: str) -> tuple[dict, str] | None:
        p = self.ckpt_dir / "stage3" / f"{ligand}_full_ewf_sqd.json"
        if p.is_file():
            print(f"Loaded saved full-EWF result from {p.name}")
            with p.open("r") as fh:
                return json.load(fh), "saved"
        archived = self.load_archived_results()
        key = f"{ligand}:full_ewf"
        if isinstance(archived.get(key), dict):
            print(f"[ARCHIVED] Loaded {key} from archived_lab_results.json")
            return archived[key], "archived"
        return None

    # ------------------------------------------------------------------
    # Validation gate
    # ------------------------------------------------------------------

    def validate_full_ewf(
        self, result: dict, ligand: str, quantum_frag, expected_solver=None,
        *, protocol: "dict | None" = None, baseline: "dict | None" = None,
    ) -> dict:
        """Strict validation gate for full EWF–SQD results.

        ``quantum_frag`` may be a single fragment id (backwards compatible) or a
        list of ids, one per quantum-treated fragment.  ``expected_solver`` may
        likewise be a single solver name or a list aligned with ``quantum_frag``.

        When ``protocol`` is supplied (full-experiment path), additional checks are
        applied:

        - Exactly as many TrimSQD fragments as the protocol has targets.
        - Every quantum fragment is a target of the protocol.
        - No quantum fragment lies outside the protocol's target set.
        - The result's ``protocol_digest`` / ``applied_config_digest`` match the
          protocol (when present in the result — the live-result dict from the
          entrypoint carries them via ``quantum_protocol``; the saved artifact
          carries them in top-level fields written by ``_solve_molecule_full_ewf``).
        - The result contains no CCSD or FCI fragments where TrimSQD is expected.
        - Exactly the expected solver composition (FCI + CCSD + TrimSQD) when the
          full fragment_summaries are present — verified against the *actual*
          ``fragment_summaries`` entries, not the result metadata counters.

        When ``baseline`` is supplied (delta-energy reconciliation path):

        - R_delta = ΔE_total − (ΔE_HF + Σ ΔC_F) ≈ 0  (additive reconciliation).
        - ΔE_HF ≈ 0 (HF energy is geometry-fixed, not solver-dependent).
        - ΔC_F ≈ 0 for all classical (non-quantum) fragments (cumulant unchanged).
        - ΔE_total ≈ Σ_{F∈quantum} ΔC_F  (quantum-only decomposition).

        Structural checks
        -----------------
        - e_hf matches the Stage 1 reference within 1e-5 Ha
        - correct number of fragments solved
        - ``n_quantum_fragments`` equals the number of quantum-treated fragments
        - ``quantum_rdm_integrated`` flag is True
        - every quantum fragment id is present and matches the selection
        - ``fragments`` has no duplicate keys; every id is in [0, expected_n)

        Numerical checks (the flag alone is not trusted)
        ------------------------------------------------
        - each quantum fragment's cumulant ``e_corr`` is present and finite; a value of exactly
          zero is not rejected by itself, because the contribution is accepted on the strength of
          the RDM integration below (a placeholder is caught by ``rdm_integrated``)
        - ``fragments[<id>]["rdm_integrated"]`` is True for each quantum fragment
        - ``fragment_summaries`` shows each quantum fragment solved by SQD/TrimSQD,
          and no others; with ``expected_solver`` each must match its promotion
        - ``e_total`` and ``e_corr`` are finite and ``e_total ≈ e_hf + e_corr``
          (1e-6 Ha, R_total check via math.fsum)
        - Σ fragment ``e_corr`` ≈ ``e_corr`` via math.fsum (R_corr) when all fragments present

        Call this before checkpoints 10–12 are assigned.

        Returns
        -------
        dict — validated summary (e_hf_ha, e_total_ha, e_corr_ewf_ha, n_fragments,
        n_quantum_fragments, quantum_fragments, quantum_solvers).

        Raises
        ------
        ChallengeError
            On any failed check.
        """
        if not isinstance(result, dict):
            raise ChallengeError("Expected a result dict from Stage 3")

        # Normalise to lists.
        if isinstance(quantum_frag, (int, np.integer)):
            quantum_frags = [int(quantum_frag)]
        else:
            quantum_frags = [int(q) for q in quantum_frag]
        if expected_solver is None:
            expected_solvers = [None] * len(quantum_frags)
        elif isinstance(expected_solver, str):
            expected_solvers = [expected_solver] * len(quantum_frags)
        else:
            expected_solvers = list(expected_solver)
        if len(expected_solvers) != len(quantum_frags):
            raise ChallengeError(
                "expected_solver(s) must align one-to-one with quantum_frag(s)."
            )

        # HF reference check
        e_hf = float(result.get("e_hf", float("nan")))
        e_hf_ref = float(self.hf_record(ligand)["hf_energy"])
        if not np.isfinite(e_hf):
            raise ChallengeError(f"Non-finite e_hf in full EWF result: {e_hf!r}")
        if abs(e_hf - e_hf_ref) > HF_TOL:
            raise ChallengeError(
                f"Stage 3 e_hf = {e_hf:.8f} Ha differs from Stage 1 reference "
                f"{e_hf_ref:.8f} Ha by {abs(e_hf - e_hf_ref):.2e} Ha. "
                "Geometry or basis mismatch — do not use this result."
            )

        # Fragment count check
        expected_n = EXPECTED_FRAGMENTS[ligand]
        n_frags = int(result.get("n_fragments", -1))
        if n_frags != expected_n:
            raise ChallengeError(
                f"Expected {expected_n} fragments for {ligand}, got {n_frags}. "
                "The full EWF solve may have skipped fragments."
            )

        # Quantum fragment count check
        if "n_quantum_fragments" not in result:
            raise ChallengeError(
                "Stage 3 result is missing 'n_quantum_fragments'. "
                "The deployed Stage 3 function may be stale — redeploy the bundle."
            )
        n_quantum = int(result["n_quantum_fragments"])
        if n_quantum != len(quantum_frags):
            raise ChallengeError(
                f"Expected {len(quantum_frags)} quantum (SQD/TrimSQD) fragment(s), "
                f"got {n_quantum}. Check for a classical fallback or an extra QPU fragment."
            )

        # RDM integration flag
        if not result.get("quantum_rdm_integrated", False):
            raise ChallengeError(
                "quantum_rdm_integrated is False: the quantum fragments' 1- and 2-RDMs "
                "did not enter the EWF cumulant reconstruction.\n"
                "This is the 'e_corr=0.000000' blocker. Do not label E_total as "
                "validated EWF–SQD until this gate passes."
            )

        # Fragment id(s) check.
        # The entrypoint writes quantum_fragment_id (singular, int or None) when there is
        # exactly one quantum fragment, and quantum_fragment_ids (plural, list) always.
        # The saved artifact written by _solve_molecule_full_ewf carries the list form.
        # Accept either form so the same gate works for single- and multi-fragment results.
        actual_qfrags = result.get("quantum_fragment_ids")
        if actual_qfrags is None:
            actual_qfrags = result.get("quantum_fragment_id")
        if isinstance(actual_qfrags, (int, np.integer)):
            actual_qfrags = [int(actual_qfrags)]
        elif actual_qfrags is None:
            actual_qfrags = []
        else:
            actual_qfrags = [int(q) for q in actual_qfrags]
        if sorted(actual_qfrags) != sorted(quantum_frags):
            raise ChallengeError(
                f"quantum_fragment_id(s) = {actual_qfrags!r} do not match the selected "
                f"fragment(s) {sorted(quantum_frags)!r}."
            )

        # --- Numerical self-consistency (math.fsum for faithful summation) ---
        e_total = float(result["e_total"])
        if not np.isfinite(e_total):
            raise ChallengeError(f"Non-finite e_total: {e_total!r}")
        e_corr_raw = result.get("e_corr")
        try:
            e_corr_top = float(e_corr_raw)
        except (TypeError, ValueError):
            e_corr_top = float("nan")
        if not np.isfinite(e_corr_top):
            raise ChallengeError(f"e_corr missing or non-finite in full EWF result: {e_corr_raw!r}")
        # R_total: e_hf + e_corr − e_total ≈ 0
        r_total = math.fsum([e_hf, e_corr_top, -e_total])
        if abs(r_total) > CONSISTENCY_TOL:
            raise ChallengeError(
                f"Inconsistent energies: e_total={e_total:.8f} but e_hf+e_corr="
                f"{math.fsum([e_hf, e_corr_top]):.8f} Ha (|R_total|={abs(r_total):.2e})."
            )

        frags = result.get("fragments")
        summaries = result.get("fragment_summaries")
        if not isinstance(frags, dict) or not frags or not isinstance(summaries, dict) or not summaries:
            raise ChallengeError(
                "Stage 3 result lacks per-fragment 'fragments' / 'fragment_summaries'; the "
                "quantum fragments' e_corr cannot be verified numerically. Redeploy the bundle."
            )

        # Fragment-key completeness checks.
        # Duplicate-ID check applies everywhere; range-membership check only for the
        # full-experiment path (protocol is not None) so legacy single-fragment tests
        # that use 1-based synthetic fixtures are not affected.
        frag_ids_present = [int(k) for k in frags]
        if len(frag_ids_present) != len(set(frag_ids_present)):
            raise ChallengeError(
                f"fragments dict contains duplicate fragment IDs: "
                f"{sorted(k for k in frag_ids_present if frag_ids_present.count(k) > 1)}")
        if protocol is not None and len(frags) == n_frags:
            all_fids = set(range(n_frags))
            extra = set(frag_ids_present) - all_fids
            missing = all_fids - set(frag_ids_present)
            if extra:
                raise ChallengeError(
                    f"fragments dict contains unexpected fragment IDs {sorted(extra)}.")
            if missing:
                raise ChallengeError(
                    f"fragments dict is missing fragment IDs {sorted(missing)}.")

        quantum_details = []
        for qf, exp_solver in zip(quantum_frags, expected_solvers):
            qrec = _by_id(frags, qf)
            qsum = _by_id(summaries, qf)
            if not isinstance(qrec, dict) or not isinstance(qsum, dict):
                raise ChallengeError(f"Quantum fragment {qf} is missing from the per-fragment records.")
            if not qrec.get("rdm_integrated", False):
                raise ChallengeError(
                    f"Fragment {qf}: rdm_integrated is False in the per-fragment reconstruction "
                    "record — its energy came from the e_corr placeholder, not its RDMs."
                )
            try:
                e_corr_q = float(qrec.get("e_corr"))
            except (TypeError, ValueError):
                e_corr_q = float("nan")
            if not np.isfinite(e_corr_q):
                raise ChallengeError(
                    f"Quantum fragment {qf} cumulant e_corr = {e_corr_q!r} Ha; the RDM-integrated "
                    "contribution must be present and finite."
                )
            solver = str(qsum.get("solver", ""))
            if solver not in QUANTUM_SOLVER_LABELS:
                raise ChallengeError(
                    f"Fragment {qf} was solved by {solver!r}, not SQD/TrimSQD (silent CCSD fallback?)."
                )
            if exp_solver is not None and solver.lower() != exp_solver.lower():
                raise ChallengeError(
                    f"Fragment {qf} was solved by {solver}, but the promoted solver is "
                    f"{exp_solver!r}. The result does not correspond to the promotion."
                )
            quantum_details.append({
                "fragment_id": qf,
                "e_corr_ha": e_corr_q,
                "solver": solver,
            })

        n_quantum_summ = sum(1 for s in summaries.values()
                             if isinstance(s, dict) and s.get("solver") in QUANTUM_SOLVER_LABELS)
        if n_quantum_summ != len(quantum_frags):
            raise ChallengeError(
                f"fragment_summaries lists {n_quantum_summ} SQD/TrimSQD fragments; "
                f"expected {len(quantum_frags)}."
            )
        if len(frags) == n_frags:
            # R_corr: Σ fragment C_F − e_corr ≈ 0 (math.fsum for faithful summation).
            # C_F is fragments[fid]["e_corr"] — the cumulant-path EWF contribution from Step 4.
            frag_ecorr_vals = [float(f.get("e_corr", 0.0)) for f in frags.values()]
            r_corr = math.fsum(frag_ecorr_vals + [-e_corr_top])
            if abs(r_corr) > CONSISTENCY_TOL:
                frag_sum = math.fsum(frag_ecorr_vals)
                raise ChallengeError(
                    f"Σ fragment e_corr = {frag_sum:.10f} Ha does not match e_corr = "
                    f"{e_corr_top:.10f} Ha (|R_corr|={abs(r_corr):.2e})."
                )

        # --- Full-experiment protocol checks (Task 4C.1) ---
        if protocol is not None:
            from proteasome_challenge import protocol_transfer as _pt
            proto_targets = sorted(int(i) for i in protocol["target_fragment_ids"])

            # Every quantum fragment must be a target; no target may be missing.
            outside_proto = sorted(set(quantum_frags) - set(proto_targets))
            if outside_proto:
                raise ChallengeError(
                    f"Quantum fragment(s) {outside_proto} are not targets of the shared protocol "
                    f"(targets: {proto_targets}).")
            missing_proto = sorted(set(proto_targets) - set(quantum_frags))
            if missing_proto:
                raise ChallengeError(
                    f"Protocol target(s) {missing_proto} are absent from the quantum fragment list; "
                    "a result with fewer TrimSQD fragments than the protocol targets does not "
                    "satisfy the full-experiment requirement.")

            # Protocol digest consistency: check both the live result echo (quantum_protocol)
            # and the saved artifact's top-level fields (written by _solve_molecule_full_ewf).
            # ``document_digest`` covers the complete document including source identity —
            # it is the most specific identifier and must agree when present.
            # ``protocol_digest`` covers the transferable settings only (the same for any
            # two protocols built with identical settings from different source fragments).
            qp = result.get("quantum_protocol") or {}
            for field in ("document_digest", "protocol_digest", "applied_config_digest"):
                want = protocol[field]
                # Try both the quantum_protocol sub-object (live result) and top-level (artifact).
                got_live = qp.get(field)
                got_top = result.get(field)
                if got_live is not None and got_live != want:
                    raise ChallengeError(
                        f"quantum_protocol.{field} in the result ({got_live[:12]}...) "
                        f"does not match the expected protocol ({want[:12]}...).")
                if got_top is not None and got_top != want:
                    raise ChallengeError(
                        f"result.{field} ({got_top[:12]}...) does not match the expected "
                        f"protocol ({want[:12]}...).")

            # Exact solver composition when full summaries are present.
            if summaries and len(summaries) == n_frags:
                from collections import Counter as _Counter
                comp = _Counter(s.get("solver") for s in summaries.values()
                                if isinstance(s, dict))
                n_trimsqd = comp.get("TrimSQD", 0)
                n_fci_r = comp.get("FCI", 0)
                n_ccsd_r = comp.get("CCSD", 0)
                if n_trimsqd != len(proto_targets):
                    raise ChallengeError(
                        f"Expected {len(proto_targets)} TrimSQD fragments but the result "
                        f"has {n_trimsqd}; a result with {n_trimsqd} does not satisfy the "
                        "full-experiment requirement.")
                n_expected_fci = result.get("n_fci")
                n_expected_ccsd = result.get("n_ccsd")
                if n_expected_fci is not None and n_fci_r != int(n_expected_fci):
                    raise ChallengeError(
                        f"FCI count mismatch: result.n_fci={n_expected_fci} but "
                        f"fragment_summaries has {n_fci_r} FCI fragments.")
                if n_expected_ccsd is not None and n_ccsd_r != int(n_expected_ccsd):
                    raise ChallengeError(
                        f"CCSD count mismatch: result.n_ccsd={n_expected_ccsd} but "
                        f"fragment_summaries has {n_ccsd_r} CCSD fragments.")

        # --- Baseline delta-energy reconciliation (Task 4C.2) ---
        # Applied only when a classical baseline is supplied and all per-fragment
        # data are present.  Uses math.fsum throughout.
        if baseline is not None and len(frags) == n_frags:
            base_frags_map: dict[int, dict] = {
                int(bf["fragment_id"]): bf for bf in baseline["fragments"]
            }
            base_energies = baseline.get("energies", {})
            e_hf_base = float(base_energies.get("e_hf", e_hf))
            e_total_base = float(base_energies.get("e_total", float("nan")))
            e_corr_base = float(base_energies.get("e_corr", float("nan")))

            if np.isfinite(e_total_base) and np.isfinite(e_corr_base):
                # R_delta: ΔE_total − (ΔE_HF + Σ ΔC_F) ≈ 0
                d_e_total = e_total - e_total_base
                d_e_hf = math.fsum([e_hf, -e_hf_base])

                # ΔE_HF ≈ 0 — HF energy is geometry/basis fixed, not solver-dependent.
                if abs(d_e_hf) > HF_TOL:
                    raise ChallengeError(
                        f"ΔE_HF = {d_e_hf:.2e} Ha exceeds tolerance {HF_TOL:.0e}; "
                        "the result and baseline use different geometries or basis sets.")

                # ΔC_F for every fragment.
                # The Step 4 cumulant contribution is stored as:
                #   result fragments[fid]["e_corr"]          — from Stage 3/4 solve
                #   baseline fragments[fid]["e_corr_cumulant_ha"] — from Stage B classical baseline
                # These two keys are intentionally different; we must use the correct one for each.
                quantum_fids_set = set(quantum_frags)
                delta_c: dict[int, float] = {}
                for fid_str, frec in frags.items():
                    fid_i = int(fid_str)
                    c_f_new = float(frec.get("e_corr", 0.0))
                    base_frec = base_frags_map.get(fid_i)
                    if base_frec is None:
                        continue  # fragment absent from baseline — skip silently
                    # Prefer e_corr_cumulant_ha (Stage B classical baseline schema);
                    # fall back to e_corr for synthetic test fixtures that use it directly.
                    c_f_base = float(
                        base_frec.get("e_corr_cumulant_ha",
                                      base_frec.get("e_corr", 0.0)))
                    delta_c[fid_i] = math.fsum([c_f_new, -c_f_base])

                # Unchanged-fragment check: ΔC_F ≈ 0 for all classical fragments.
                for fid_i, dc in delta_c.items():
                    if fid_i not in quantum_fids_set and abs(dc) > CONSISTENCY_TOL:
                        raise ChallengeError(
                            f"Classical fragment {fid_i} cumulant changed by {dc:.2e} Ha "
                            f"(|ΔC_F| > {CONSISTENCY_TOL:.0e}); the classical re-solve "
                            "differs from the saved baseline.")

                # Quantum-only decomposition: ΔE_total ≈ Σ_{F∈quantum} ΔC_F.
                sum_quantum_dc = math.fsum(
                    dc for fid_i, dc in delta_c.items() if fid_i in quantum_fids_set)
                r_qonly = math.fsum([d_e_total, -sum_quantum_dc])
                if abs(r_qonly) > CONSISTENCY_TOL:
                    raise ChallengeError(
                        f"Quantum-only decomposition fails: ΔE_total={d_e_total:.10f} Ha but "
                        f"Σ_quantum ΔC_F={sum_quantum_dc:.10f} Ha "
                        f"(|R_delta|={abs(r_qonly):.2e}).")

                # R_delta: full additive check ΔE_total − ΔE_HF − Σ ΔC_F ≈ 0.
                sum_all_dc = math.fsum(delta_c.values())
                r_delta = math.fsum([d_e_total, -d_e_hf, -sum_all_dc])
                if abs(r_delta) > CONSISTENCY_TOL:
                    raise ChallengeError(
                        f"Energy delta reconciliation failed: "
                        f"ΔE_total={d_e_total:.10f}, ΔE_HF={d_e_hf:.2e}, "
                        f"Σ ΔC_F={sum_all_dc:.10f} Ha (|R_delta|={abs(r_delta):.2e}).")

        return {
            "ligand": ligand,
            "e_hf_ha": e_hf,
            "e_total_ha": e_total,
            "e_corr_ewf_ha": e_total - e_hf,
            "n_fragments": expected_n,
            "n_quantum_fragments": n_quantum,
            "quantum_fragments": quantum_details,
        }

    # ------------------------------------------------------------------
    # Stage B — all-classical (mixed FCI/CCSD) EWF baseline + CCSD diagnostics
    # ------------------------------------------------------------------

    def baseline_path(self, ligand: str) -> Path:
        return self.ckpt_dir / "stage3" / f"{ligand}_classical_baseline.json"

    def _legacy_classical_ewf_path(self, ligand: str) -> Path:
        """Pre-diagnostics checkpoint (historical; carries no CCSD amplitudes)."""
        return self.ckpt_dir / "stage3" / f"{ligand}_classical_ewf.json"

    def _current_hamiltonian_hashes(self, ligand: str) -> dict[int, str]:
        """Hamiltonian identity of every embedded fragment, read from the Stage 2 dump.

        Pure file read — no chemistry is run.  Cached per (path, mtime, size).
        """
        from proteasome_challenge import classical_baseline as cb
        _, _, h5_p = self._stage2_paths(ligand)
        if not h5_p.is_file():
            raise ChallengeError(
                f"Cannot verify Hamiltonian identity: Stage 2 dump missing ({h5_p}). "
                "Run Stage 2 (or obtain the prepared-data package) first.")
        st = h5_p.stat()
        key = (str(h5_p), st.st_mtime_ns, st.st_size)
        cache = self.__dict__.setdefault("_ham_cache", {})
        if key not in cache:
            try:
                cache[key] = cb.hamiltonian_hashes_from_dumpfile(h5_p)
            except cb.BaselineError as exc:
                raise ChallengeError(str(exc)) from exc
        return cache[key]

    def baseline_assignments(self, ligand: str) -> dict[int, str]:
        """Frozen FCI/CCSD assignment map used by the baseline (never any SQD)."""
        from proteasome_challenge import classical_baseline as cb
        from proteasome_challenge.solver_assignments import assignments_to_step3_params
        eff = self.effective_solver_assignments(ligand, None)
        if eff is None:
            raise ChallengeError(
                f"No frozen solver-assignment map for {ligand}; the baseline is defined by "
                "challenge_data/solver_assignments.json.")
        amap = assignments_to_step3_params(eff)["solver_assignments"]
        try:
            cb.assert_classical_assignments(amap)
        except cb.BaselineError as exc:
            raise ChallengeError(str(exc)) from exc
        return {int(k): v for k, v in amap.items()}

    def expected_baseline_identity(self, ligand: str) -> dict:
        """Cache identity a valid baseline must have *now* (Hamiltonians + map + config)."""
        from proteasome_challenge import classical_baseline as cb
        try:
            return cb.compute_identity(
                ligand, self._current_hamiltonian_hashes(ligand),
                self.baseline_assignments(ligand),
                cb.default_solver_config(FCI_THRESHOLD, SQD_THRESHOLD),
                bundle_api=EXPECTED_BUNDLE_API)
        except cb.BaselineError as exc:
            raise ChallengeError(str(exc)) from exc

    def structural_reference(self, ligand: str = SELECTED_LIGAND) -> dict:
        """Stage 2 structural view (identity, Hamiltonian hashes, n_orb/n_occ).  **No calculation, no job.**

        Carries the same cache identity a computed baseline would have for these Stage 2 Hamiltonians, but no
        energies and no CCSD amplitudes (``baseline_computed`` is False).  It is NOT a classical baseline and is
        rejected by every energy consumer.
        """
        from proteasome_challenge import classical_baseline as cb
        _, _, h5_p = self._stage2_paths(ligand)
        try:
            return cb.build_structural_reference(
                ligand, identity=self.expected_baseline_identity(ligand),
                assignments=self.baseline_assignments(ligand), dims=cb.fragment_dims_from_dumpfile(h5_p))
        except cb.BaselineError as exc:
            raise ChallengeError(str(exc)) from exc

    def reference_baseline(self, ligand: str = SELECTED_LIGAND) -> dict:
        """What protocol / manifest / plan binding reads: the structural view in a full live trial, else the baseline.

        ``use_structural_reference`` is False by default, so every legacy workflow still requires (and validates)
        the saved classical baseline exactly as before.
        """
        if getattr(self, "use_structural_reference", False):
            return self.structural_reference(ligand)
        return self.load_classical_baseline(ligand)

    def load_classical_baseline(self, ligand: str = SELECTED_LIGAND) -> dict:
        """Reload and validate the saved baseline.  **Never launches a calculation.**

        Validates schema, internal consistency, the diagnostics digest and — against
        the *current* Stage 2 Hamiltonians, frozen solver map and solver config — the
        cache identity.  A stale or pre-diagnostics baseline is rejected, with a clear
        statement of whether a fresh user-executed run is required.
        """
        from proteasome_challenge import classical_baseline as cb
        path = self.baseline_path(ligand)
        if not path.is_file():
            legacy = cb.legacy_baseline_status(self._legacy_classical_ewf_path(ligand))
            msg = [f"No classical baseline for {ligand} ({path.name})."]
            if legacy["present"] and not legacy["diagnostics_available"]:
                msg.append(
                    f"A historical classical EWF checkpoint exists (E_total = "
                    f"{legacy.get('e_total_ha')}), but it retained no CCSD amplitudes for its "
                    f"{legacy['n_ccsd_fragments']} CCSD fragments. {cb.NOT_RETAINED_REASON}")
            else:
                msg.append("Run Stage B with run_classical_baseline(ligand, execution='run') "
                           "(one classical Stage 3 job, no QPU).")
            raise ChallengeError(" ".join(msg))
        with path.open("r", encoding="utf-8") as fh:
            record = json.load(fh)
        try:
            return cb.validate_baseline_record(
                record, expected_identity=self.expected_baseline_identity(ligand))
        except cb.BaselineError as exc:
            raise ChallengeError(f"{path.name}: {exc}") from exc

    def run_classical_baseline(
        self, ligand: str = SELECTED_LIGAND, *, execution: str = "load",
        force: bool = False, poll_seconds: int = 30,
    ) -> dict:
        """Load (default) or run the all-classical mixed FCI/CCSD EWF baseline.

        ``execution="load"``  validate and return the saved baseline; submits nothing.
        ``execution="run"``   if a compatible baseline is already saved it is returned
                              unchanged (no duplicate calculation) unless ``force=True``;
                              otherwise submits ONE classical Stage 3 job.

        The job uses the frozen FCI/CCSD map (zero SQD) with ``qpu_backend=None``; CCSD
        diagnostics are extracted from the amplitudes of those same solves.
        """
        from proteasome_challenge import classical_baseline as cb
        if execution not in ("load", "run"):
            raise ChallengeError(f"execution must be 'load' or 'run', got {execution!r}")
        if execution == "load":
            return self.load_classical_baseline(ligand)
        if not force:
            try:
                rec = self.load_classical_baseline(ligand)
                print(f"A compatible classical baseline already exists for {ligand}; "
                      "reusing it (pass force=True to recompute).")
                return rec
            except ChallengeError as exc:
                print(f"No reusable baseline ({str(exc).splitlines()[0][:120]}) — submitting.")
        ident = self.expected_baseline_identity(ligand)
        amap = self.baseline_assignments(ligand)
        job = self._submit_stage3(
            ligand, label="classical baseline", solver_mode="sqd", qpu_backend=None,
            sqd_config=None, trim_config=None, n_shots=DEFAULT_N_SHOTS,
            n_reps=DEFAULT_N_REPS, reconstruct=True, solver_assignments=amap)
        result = self._wait_and_collect({"baseline": job}, poll_seconds)["baseline"]
        self._validate_bundle_api(result, "classical baseline")
        try:
            record = cb.build_baseline_record(
                result, ligand=ligand, expected_identity=ident, assignments=amap,
                expected_bundle_api=EXPECTED_BUNDLE_API)
        except cb.BaselineError as exc:
            raise ChallengeError(f"Baseline result rejected, nothing saved: {exc}") from exc
        self._write_json(self.baseline_path(ligand), record)
        n_av = sum(1 for f in record["fragments"]
                   if f["ccsd_diagnostics"]["diagnostics_status"] == "available")
        print(f"Baseline saved → {self.baseline_path(ligand).name}: "
              f"E_total = {record['energies']['e_total']:.8f} Ha; "
              f"CCSD diagnostics available for {n_av}/{record['solver_counts']['n_ccsd']} CCSD fragments.")
        return record

    def solve_classical_ewf(self, *a, **k):  # pragma: no cover - removed API
        raise ChallengeError(
            "solve_classical_ewf() was replaced by run_classical_baseline(ligand, execution='load'|'run'), "
            "which runs from the frozen FCI/CCSD map and extracts CCSD diagnostics.")

    def diagnostic_state(self, ligand: str = SELECTED_LIGAND) -> dict:
        """Load the baseline diagnostics table or return an explicit missing-results state.

        Returns ``{"available": bool, "message": str, "table": DataFrame | None,
        "baseline": dict | None}``.  Nothing is computed or simulated.
        """
        try:
            base = self.load_classical_baseline(ligand)
        except ChallengeError as exc:
            return {"available": False, "message": str(exc), "table": None, "baseline": None}
        from proteasome_challenge import diagnostic_analysis as da
        table = da.merged_table(base, self.load_fragment_catalog(), ligand)
        n_av = int((table["diagnostics_status"] == "available").sum())
        if n_av == 0:
            return {"available": False, "baseline": base, "table": table,
                    "message": "The baseline is loaded but no fragment has CCSD diagnostics available."}
        return {"available": True, "baseline": base, "table": table,
                "message": f"CCSD diagnostics available for {n_av} fragment(s)."}

    def diagnostic_analysis_table(self, ligand: str = SELECTED_LIGAND) -> pd.DataFrame:
        """Fragment-resolved MP2 + CCSD table; raises if the baseline is not available."""
        st = self.diagnostic_state(ligand)
        if not st["available"]:
            raise ChallengeError(st["message"])
        return st["table"]

    # ------------------------------------------------------------------
    # Submission validation
    # ------------------------------------------------------------------

    def validate_submission(self, answers: dict) -> pd.DataFrame:
        """Validate the 12 checkpoint answers and return a summary table.

        Parameters
        ----------
        answers : dict
            Mapping from checkpoint key (e.g. ``"01_ixazomib_hf_ha"``) to
            float or None.

        Returns
        -------
        pandas.DataFrame with columns: checkpoint, section, points, value,
        status.
        """
        RUBRIC = [
            ("01_ixazomib_hf_ha",         "1",   10),
            ("02_bortezomib_hf_ha",        "1",   10),
            ("03_ixazomib_max_norb",       "2",    5),
            ("04_bortezomib_max_norb",     "2",    5),
            ("05_ixazomib_n_orb_ge_20",    "2",    5),
            ("06_bortezomib_n_orb_ge_20",  "2",    5),
            # Checkpoints 07-12 depend on the SQD calibration fragment / quantum panel, which
            # is only defined by the user-frozen final selection (Stage D).  Key
            # 07_ccsd_ixazomib_initial_ha is the key the backend and grader both use.
            ("07_ccsd_ixazomib_initial_ha",  "3a",   5),
            ("08_sqd_frozen_counts_ha",    "3b",  10),
            ("09_best_tuned_energy_ha",    "3c",  10),
            ("10_full_ewf_sqd_ecorr_ha",   "4a",  15),
            ("11_full_ewf_sqd_etotal_ha",  "4b",  15),
            ("12_delta_vs_classical_ha",   "4c",   5),
        ]
        assert sum(p for _, _, p in RUBRIC) == 100

        # Experiment-dependent checkpoints are only gradable against a valid frozen final
        # selection.  Without one, any value supplied is flagged INCOMPATIBLE (it can only
        # stem from the historical fragment-1 experiment or an unfrozen selection) and is
        # never graded against the old fragment-1 values.
        selection_reason = None
        try:
            self.load_final_selection()
        except ChallengeError as exc:
            selection_reason = str(exc).splitlines()[0][:160]
        experiment_keys = {k for k, s, _ in RUBRIC if s in ("3a", "3b", "3c", "4a", "4b", "4c")}

        rows = []
        for key, section, pts in RUBRIC:
            v = answers.get(key)
            note = ""
            if v is None:
                status = "PENDING"
                display = "PENDING"
                if key in experiment_keys and selection_reason:
                    note = "awaiting final selection: " + selection_reason
            elif key in experiment_keys and selection_reason:
                status = "INCOMPATIBLE"
                display = str(v)
                note = ("not gradable: no valid frozen final selection (" + selection_reason + "). "
                        "Values from the historical fragment-1 experiment are not accepted.")
            else:
                try:
                    fv = float(v)
                    if not np.isfinite(fv):
                        raise ValueError
                    display = f"{fv:.10f}"
                    status = "ready"
                except (TypeError, ValueError):
                    display = str(v)
                    status = "ERROR (non-finite)"
            rows.append({"checkpoint": key, "section": section, "points": pts,
                         "value": display, "status": status, "note": note})

        df = pd.DataFrame(rows)
        n_ready = sum(r["points"] for r in rows if r["status"] == "ready")
        pending = sum(1 for r in rows if r["status"] == "PENDING")
        incompatible = sum(1 for r in rows if r["status"] == "INCOMPATIBLE")
        print(f"Checkpoints with a value: {n_ready}/100 points  "
              f"({pending} pending, {incompatible} incompatible)")
        return df

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _wait_and_collect(self, jobs: dict, poll_seconds: int = 30) -> dict:
        """Poll all Serverless jobs until terminal; raise on non-DONE status.

        Transient network errors (``AUTH1001``, ``ConnectionReset``) on
        ``job.status()`` are retried up to ``_MAX_POLL_RETRIES`` times with a
        short back-off, refreshing the Serverless client on each retry so the
        dead TCP socket is replaced.
        """
        _MAX_POLL_RETRIES = 3
        _RETRY_BACKOFF    = 5   # seconds between poll retries

        def _is_transient(exc: Exception) -> bool:
            s = str(exc)
            return any(kw in s for kw in (
                "Connection", "ConnectionReset", "AUTH1001", "ProtocolError",
            ))

        def _status_with_retry(job) -> str:
            for attempt in range(_MAX_POLL_RETRIES):
                try:
                    return str(job.status()).upper()
                except Exception as exc:
                    if attempt < _MAX_POLL_RETRIES - 1 and _is_transient(exc):
                        print(
                            f"    [poll retry {attempt + 1}/{_MAX_POLL_RETRIES - 1}] "
                            f"transient error on status check — refreshing connection… "
                            f"({str(exc)[:80]})",
                            flush=True,
                        )
                        self._connect(_force_refresh=True)
                        time.sleep(_RETRY_BACKOFF)
                    else:
                        raise

        terminal = {"DONE", "ERROR", "FAILED", "CANCELED", "CANCELLED"}
        pending = dict(jobs)
        while pending:
            for name, job in list(pending.items()):
                status = _status_with_retry(job)
                print(f"  {name}: {status}", flush=True)
                if status in terminal:
                    pending.pop(name)
            if pending:
                time.sleep(poll_seconds)
        results = {}
        for name, job in jobs.items():
            status = _status_with_retry(job)
            if status != "DONE":
                try:
                    print(str(job.logs())[-4000:])
                except Exception:
                    pass
                raise ChallengeError(
                    f"Serverless job '{name}' (id={job.job_id}) ended with status {status}"
                )
            results[name] = job.result()
        return results

    def _validate_bundle_api(self, result: dict, context: str) -> None:
        if not isinstance(result, dict) or result.get("bundle_api") != EXPECTED_BUNDLE_API:
            raise ChallengeError(
                f"{context}: result is missing bundle_api={EXPECTED_BUNDLE_API}. "
                "The deployed Stage 3 function is stale. Set RUN_DEPLOY=True and redeploy."
            )

    @staticmethod
    def _b64_to_file(b64: str, path: Path, *, gunzip: bool) -> None:
        import base64
        import gzip
        raw = base64.b64decode(b64)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.decompress(raw) if gunzip else raw)

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=float)

    @staticmethod
    def _write_json_atomic(path: Path, obj: Any) -> None:
        """Write *obj* to *path* atomically via a sibling .tmp file.

        The destination file is only replaced after the content has been fully
        written and flushed, so a crash or signal during the write cannot leave
        a partially-written checkpoint at the target path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(obj, fh, indent=2, default=float)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise
