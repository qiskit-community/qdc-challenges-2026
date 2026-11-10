"""
Step 4 — Partitioned cumulant energy reconstruction.

Assembles per-fragment RDMs into the total EWF correlation energy and writes
a JSON summary.  Identical in structure to
proteasome_ligand_challenge/steps/step4_reconstruct.py.

All fragments fail closed.  Every fragment — quantum (SQD/TrimSQD) or classical
(FCI/CCSD) — must supply complete, finite 1-RDM and 2-RDM data.  The partitioned-
cumulant contraction is the only permitted energy source: ``metadata.e_corr`` and
zero are never substituted.  Missing or invalid RDMs raise
``QuantumReconstructionError`` (quantum) or ``ClassicalReconstructionError`` (classical)
with the fragment id and reason.  An exactly-zero cumulant contribution is legitimate.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

# Import canonical constant from the authoritative source.
# Do NOT hardcode the quantum-solver set here — the registry is the single source of truth.
try:
    _pkg_root = Path(__file__).resolve().parents[2]
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from proteasome_challenge.solver_assignments import QUANTUM_SOLVERS as _QS_FROZENSET
    QUANTUM_SOLVERS: tuple[str, ...] = tuple(sorted(_QS_FROZENSET))
except ImportError:
    # Fallback when bundle runs without the proteasome_challenge package
    QUANTUM_SOLVERS = ("SQD", "TrimSQD")

# Metadata written only by the quantum solvers / only by the CCSD solver (see sqd.py,
# trim_sqd.py, ccsd.py).  Used to cross-check the label step 3 stamped on a result.
_QUANTUM_RESULT_MARKERS = ("measurement_sha256", "policy", "trim_ratio")
_CCSD_RESULT_MARKERS = ("ccsd_diagnostics", "e_hf", "conv_tol")


class QuantumReconstructionError(RuntimeError):
    """A quantum fragment's RDMs could not be integrated into the EWF energy."""

    def __init__(self, fragment_id, solver, reason):
        self.fragment_id = int(fragment_id)
        self.solver = str(solver)
        self.reason = str(reason)
        super().__init__(f"fragment {self.fragment_id} ({self.solver}): {self.reason}")


class ClassicalReconstructionError(RuntimeError):
    """A classical fragment's RDMs could not be integrated into the EWF energy."""

    def __init__(self, fragment_id, solver, reason):
        self.fragment_id = int(fragment_id)
        self.solver = str(solver)
        self.reason = str(reason)
        super().__init__(f"fragment {self.fragment_id} ({self.solver}): {self.reason}")


def _is_quantum_result(frag_id, metadata) -> bool:
    """True when the solver that actually executed was SQD or TrimSQD.

    Step 3 stamps ``solver_label`` after the requested quantum solver has run and its
    identity has been verified, so it is the executed solver, not the requested map.  The
    label is cross-checked against the result's own metadata: quantum-only keys under a
    non-quantum label (or CCSD-only keys under a quantum label) mean the label is wrong,
    and the fragment is refused rather than reconstructed under the wrong rules.
    """
    label = metadata.get("solver_label", "unknown")
    quantum_label = label in QUANTUM_SOLVERS
    quantum_keys = [k for k in _QUANTUM_RESULT_MARKERS if k in metadata]
    if quantum_keys and not quantum_label:
        raise QuantumReconstructionError(
            frag_id, label,
            f"result carries quantum-solver metadata {quantum_keys} but is labelled {label!r}; "
            "refusing to reconstruct a quantum result under a classical label")
    if quantum_label:
        ccsd_keys = [k for k in _CCSD_RESULT_MARKERS if k in metadata]
        if ccsd_keys:
            raise QuantumReconstructionError(
                frag_id, label,
                f"result is labelled {label} but carries CCSD-only metadata {ccsd_keys}")
    return quantum_label


def _quantum_fragment_energy(ccsd_helper, frag_id, label, result, fock_emb, h2e, proj, nocc):
    """Cumulant contribution of one SQD/TrimSQD fragment; raises on any failure."""
    if result.rdm1 is None:
        raise QuantumReconstructionError(frag_id, label, "one-body RDM (rdm1) is missing")
    if result.rdm2 is None:
        raise QuantumReconstructionError(frag_id, label, "two-body RDM (rdm2) is missing")
    try:
        e1_pc, e2_pc = ccsd_helper.compute_fragment_energy(result, fock_emb, h2e, proj, nocc)
    except Exception as exc:
        raise QuantumReconstructionError(
            frag_id, label,
            f"cumulant contraction failed ({type(exc).__name__}: {exc})") from exc
    try:
        finite = math.isfinite(float(e1_pc)) and math.isfinite(float(e2_pc)) \
            and math.isfinite(float(e1_pc + e2_pc))
    except Exception as exc:
        raise QuantumReconstructionError(
            frag_id, label,
            f"cumulant contribution is not a number ({type(exc).__name__}: {exc})") from exc
    if not finite:
        raise QuantumReconstructionError(
            frag_id, label,
            f"reconstructed contribution is not finite (e1_pc={e1_pc!r}, e22_pc={e2_pc!r})")
    return e1_pc + e2_pc


def _load_ewf_inputs(data_dir: str):
    """Load mf_data.pkl and ewf_dumpfile.h5 from *data_dir*.

    Returns ``(mf, e_hf, dumpfile_path, norb_by_frag)`` where ``norb_by_frag``
    maps fragment_id → norb as read from the HDF5 attrs.
    """
    import h5py
    data_path = Path(data_dir)

    with open(data_path / "mf_data.pkl", "rb") as fh:
        mf_data = pickle.load(fh)
    mf = mf_data["mf"]
    e_hf = float(mf.e_tot)

    dumpfile = str(data_path / "ewf_dumpfile.h5")
    norb_by_frag: dict = {}
    with h5py.File(dumpfile, "r") as hf:
        for k in hf:
            if k.startswith("fragment_"):
                fid = int(k.split("_")[1])
                norb_by_frag[fid] = int(hf[k].attrs["norb"])

    return mf, e_hf, dumpfile, norb_by_frag


def _accumulate_cumulant_energies(
    mf,
    dumpfile: str,
    norb_by_frag: dict,
    fragment_results: dict,
    run_name: str,
) -> tuple[dict, float]:
    """Core partitioned-cumulant accumulation loop shared by both entry points.

    Parameters
    ----------
    mf             : PySCF mean-field object (provides overlap and Fock matrices).
    dumpfile       : path to the Vayesta EWF HDF5 dump file.
    norb_by_frag   : dict mapping fragment_id → norb (from HDF5 attrs).
    fragment_results : dict mapping fragment_id → SolverResult.
    run_name       : short label for log messages.

    Returns
    -------
    (fragment_energies, e_corr_total)
        ``fragment_energies`` maps int fragment_id → dict with keys
        ``e_corr``, ``solver``, ``n_orb``, ``converged``, ``rdm_integrated``.
        ``e_corr_total`` is the sum of all per-fragment correlation energies.

    Raises
    ------
    QuantumReconstructionError
        When a quantum fragment's RDMs are missing, non-finite, or the
        cumulant contraction fails.
    ClassicalReconstructionError
        When a classical fragment's RDMs are missing, non-finite, or the
        cumulant contraction fails.  ``metadata.e_corr`` is never used as a
        fallback — missing or invalid RDMs always raise.
    """
    import h5py
    import numpy as np
    from quantum_fragment_methods.application.solvers.classical_zoo.ccsd import CCSD

    ccsd_helper = CCSD()
    ovlp = mf.get_ovlp()
    fock_global = mf.get_fock()

    e_corr_total = 0.0
    fragment_energies: dict = {}

    for frag_id, result in fragment_results.items():
        _norb = norb_by_frag.get(int(frag_id), result.metadata.get("nocc", 1) * 2)
        nocc = int(result.metadata.get("nocc", _norb // 2))

        with h5py.File(dumpfile, "r") as hf:
            g = hf[f"fragment_{int(frag_id)}"]
            h2e = g["eris"][:]
            c_frag = g["c_frag"][:]
            c_cluster = g["c_cluster"][:]
            fock_emb = g["fock"][:] if "fock" in g else (c_cluster.T @ fock_global @ c_cluster)

        # Fragment projector in cluster basis: P = (c_frag^T S c_cls)^T (c_frag^T S c_cls)
        proj_frag = c_frag.T @ ovlp @ c_cluster
        proj = proj_frag.T @ proj_frag

        rdm_integrated = False
        _quantum = _is_quantum_result(frag_id, result.metadata)
        if _quantum:
            frag_e_corr = _quantum_fragment_energy(
                ccsd_helper, frag_id, result.metadata["solver_label"],
                result, fock_emb, h2e, proj, nocc)
            rdm_integrated = True
        else:
            # Classical fragment — require complete RDMs; no fallback to metadata.e_corr.
            label = result.metadata.get("solver_label", "classical")
            if result.rdm1 is None:
                raise ClassicalReconstructionError(
                    frag_id, label, "one-body RDM (rdm1) is missing")
            if result.rdm2 is None:
                raise ClassicalReconstructionError(
                    frag_id, label, "two-body RDM (rdm2) is missing")
            try:
                e1_pc, e2_pc = ccsd_helper.compute_fragment_energy(
                    result, fock_emb, h2e, proj, nocc)
            except (QuantumReconstructionError, ClassicalReconstructionError):
                raise
            except Exception as exc:
                raise ClassicalReconstructionError(
                    frag_id, label,
                    f"cumulant contraction failed ({type(exc).__name__}: {exc})") from exc
            try:
                finite = math.isfinite(float(e1_pc)) and math.isfinite(float(e2_pc)) \
                    and math.isfinite(float(e1_pc + e2_pc))
            except Exception as exc:
                raise ClassicalReconstructionError(
                    frag_id, label,
                    f"cumulant contribution is not a number ({type(exc).__name__}: {exc})") from exc
            if not finite:
                raise ClassicalReconstructionError(
                    frag_id, label,
                    f"reconstructed contribution is not finite (e1_pc={e1_pc!r}, e2_pc={e2_pc!r})")
            frag_e_corr = e1_pc + e2_pc
            rdm_integrated = True

        fragment_energies[int(frag_id)] = {
            "e_corr":         frag_e_corr,
            "solver":         result.metadata.get("solver_label", "unknown"),
            "n_orb":          _norb,
            "converged":      result.metadata.get("converged", True),
            "rdm_integrated": rdm_integrated,
        }
        e_corr_total += frag_e_corr
        print(
            f"  Fragment {frag_id}: e_corr={frag_e_corr:.6f} Ha"
            f"  solver={result.metadata.get('solver_label', '?')}"
        )

    return fragment_energies, e_corr_total


def run_reconstruct(
    data_dir: str,
    run_name: str,
    results_dir: str | None = None,
    solver_results_pkl: str | None = None,
) -> dict:
    """
    Reconstruct total EWF energy from per-fragment solver results (pkl entry point).

    Parameters
    ----------
    data_dir           : directory containing mf_data.pkl, ewf_dumpfile.h5,
                         and the solver results pickle
    run_name           : short label for logging / output file naming
    results_dir        : where to write summary.json
                         (default: data_dir/../../results/<run_name>)
    solver_results_pkl : explicit path to the solver results pickle.
                         Defaults to <data_dir>/solver_results.pkl when None.

    Returns
    -------
    dict with keys: e_hf, e_corr, e_total, run_name, fragments
    """
    data_path = Path(data_dir)
    mf, e_hf, dumpfile, norb_by_frag = _load_ewf_inputs(data_dir)

    _solver_pkl = solver_results_pkl or str(data_path / "solver_results.pkl")
    with open(_solver_pkl, "rb") as fh:
        fragment_results = pickle.load(fh)

    print(f"[step4] {run_name}: HF energy = {e_hf:.8f} Ha")
    fragment_energies, e_corr_total = _accumulate_cumulant_energies(
        mf, dumpfile, norb_by_frag, fragment_results, run_name)

    e_total = e_hf + e_corr_total
    print(
        f"[step4] {run_name}: "
        f"E_HF={e_hf:.8f}  E_corr={e_corr_total:.8f}  E_total={e_total:.8f} Ha"
    )

    # ── Write summary JSON ────────────────────────────────────────────────────
    if results_dir is None:
        results_dir = str(data_path.parent.parent / "results" / run_name)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name":    run_name,
        "e_hf":        e_hf,
        "e_corr":      e_corr_total,
        "e_total":     e_total,
        "n_fragments": len(fragment_results),
        "fragments":   fragment_energies,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[step4] {run_name}: wrote {summary_path}")

    return {
        "run_name": run_name,
        "e_hf":    e_hf,
        "e_corr":  e_corr_total,
        "e_total": e_total,
        "fragments": fragment_energies,
    }


def run_reconstruct_from_refs(
    data_dir: str,
    run_name: str,
    fragment_result_refs: dict,
    *,
    results_dir: str | None = None,
    expected_fragment_ids: list | None = None,
) -> dict:
    """
    Reconstruct total EWF energy from :class:`FragmentResultReference` objects.

    This is an **additive** entry point that accepts explicit fragment-result
    references instead of a single ``solver_results.pkl``.  It performs identity
    and integrity validation on every loaded result and then delegates to the
    same partitioned cumulant loop as :func:`run_reconstruct`.

    Parameters
    ----------
    data_dir               : directory containing ``mf_data.pkl`` and
                             ``ewf_dumpfile.h5`` (outputs of steps 1 & 2).
    run_name               : short label for logging / output file naming.
    fragment_result_refs   : mapping of ``fragment_id`` → one of:
                             * :class:`~fragment_result_store.FragmentResultReference`
                             * ``str`` / :class:`~pathlib.Path` — loaded with no
                               identity validation beyond the file checksum embedded
                               in the artifact; the ``run_id`` on the artifact must
                               equal *run_name* unless the caller provides a
                               ``FragmentResultReference`` with a different ``run_id``.
    results_dir            : where to write ``summary.json``
                             (default: ``data_dir/../../results/<run_name>``).
    expected_fragment_ids  : when provided, the ref dict must cover exactly these
                             fragment IDs; missing or extra IDs raise.

    Returns
    -------
    dict with keys: ``e_hf``, ``e_corr``, ``e_total``, ``run_name``, ``fragments``
        Identical format to :func:`run_reconstruct`.

    Raises
    ------
    ValueError
        When fragment coverage is incomplete or an identity check fails.
    QuantumReconstructionError
        When a quantum fragment's RDMs cannot be integrated (fail-closed).
    """
    try:
        # Relative import when loaded as part of the `steps` package
        from .fragment_result_store import (
            FragmentResultReference,
            FragmentStoreError,
            load_fragment_result,
            load_fragment_result_from_ref,
        )
    except ImportError:
        # Fallback for script-level / sys.path-based loading
        from fragment_result_store import (  # type: ignore[no-redef]
            FragmentResultReference,
            FragmentStoreError,
            load_fragment_result,
            load_fragment_result_from_ref,
        )

    # ── Normalise the reference dict keys to int ──────────────────────────────
    refs: dict = {int(k): v for k, v in fragment_result_refs.items()}

    # ── Coverage check ─────────────────────────────────────────────────────────
    if expected_fragment_ids is not None:
        expected = {int(i) for i in expected_fragment_ids}
        missing = expected - set(refs)
        extra   = set(refs) - expected
        if missing:
            raise ValueError(
                f"run_reconstruct_from_refs: missing fragment result(s) {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"run_reconstruct_from_refs: unexpected fragment result(s) {sorted(extra)}"
            )

    # ── Load all SolverResults from the references ────────────────────────────
    # Duck-type the FragmentResultReference check by attribute rather than by
    # isinstance().  This avoids module-identity mismatches when the store module
    # is loaded under a different name (e.g. via spec_from_file_location in tests).
    #
    # A valid reference duck-type must satisfy ALL of:
    #   .path       — a str or os.PathLike that resolves to a .h5 file
    #   .file_sha256 — a 64-character lowercase hex string
    #   .fragment_id — an int
    #   .run_id      — a non-empty str
    #   .solver      — a non-empty str
    #   .hamiltonian_sha256 — a 64-character lowercase hex string
    # Any object missing these fields, or with malformed values, is rejected
    # immediately rather than silently falling through to the plain-path branch.

    import re as _re_mod
    _HEX64 = _re_mod.compile(r"^[0-9a-f]{64}$")

    def _validate_ref_duck(ref, fid: int):
        """Raise ValueError if ref looks like a reference but fails validation."""
        required = ("file_sha256", "path", "fragment_id", "run_id",
                    "solver", "hamiltonian_sha256")
        missing = [a for a in required if not hasattr(ref, a)]
        if missing:
            raise ValueError(
                f"fragment {fid}: reference object is missing required fields "
                f"{missing}; got {type(ref).__name__!r}"
            )
        # Validate file_sha256 format
        fsha = getattr(ref, "file_sha256", "")
        if not isinstance(fsha, str) or not _HEX64.match(fsha):
            raise ValueError(
                f"fragment {fid}: reference.file_sha256 must be a 64-char lowercase "
                f"hex string, got {fsha!r}"
            )
        # Validate hamiltonian_sha256 format
        hsha = getattr(ref, "hamiltonian_sha256", "")
        if not isinstance(hsha, str) or not _HEX64.match(hsha):
            raise ValueError(
                f"fragment {fid}: reference.hamiltonian_sha256 must be a 64-char "
                f"lowercase hex string, got {hsha!r}"
            )
        # Validate path is path-like and ends in .h5
        ref_path = getattr(ref, "path", None)
        try:
            ref_path_obj = Path(ref_path)
        except (TypeError, ValueError):
            raise ValueError(
                f"fragment {fid}: reference.path cannot be converted to a Path: "
                f"{ref_path!r}"
            )
        if ref_path_obj.suffix.lower() != ".h5":
            raise ValueError(
                f"fragment {fid}: reference.path must end in .h5, got {ref_path_obj}"
            )
        # fragment_id, run_id, solver: basic type checks
        if not isinstance(getattr(ref, "fragment_id", None), (int,)):
            raise ValueError(
                f"fragment {fid}: reference.fragment_id must be int, "
                f"got {type(ref.fragment_id).__name__!r}"
            )
        for _field in ("run_id", "solver"):
            _val = getattr(ref, _field, "")
            if not isinstance(_val, str) or not _val.strip():
                raise ValueError(
                    f"fragment {fid}: reference.{_field} must be a non-empty str, "
                    f"got {_val!r}"
                )

    fragment_results: dict = {}
    for fid, ref in sorted(refs.items()):
        try:
            if hasattr(ref, "file_sha256") or hasattr(ref, "hamiltonian_sha256"):
                # Object looks like a reference — apply full duck-type validation
                # before attempting to load.
                _validate_ref_duck(ref, fid)
                result = load_fragment_result_from_ref(ref)
            else:
                # Plain path (str / Path): load with fragment_id identity check
                result = load_fragment_result(
                    ref,
                    expected_fragment_id=fid,
                )
        except FragmentStoreError as exc:
            raise ValueError(
                f"run_reconstruct_from_refs: failed to load fragment {fid}: {exc}"
            ) from exc
        fragment_results[fid] = result

    # ── Reconstruct directly — no temp pickle ────────────────────────────────
    mf, e_hf, dumpfile, norb_by_frag = _load_ewf_inputs(data_dir)
    print(f"[step4] {run_name}: HF energy = {e_hf:.8f} Ha")
    fragment_energies, e_corr_total = _accumulate_cumulant_energies(
        mf, dumpfile, norb_by_frag, fragment_results, run_name)

    e_total = e_hf + e_corr_total
    print(
        f"[step4] {run_name}: "
        f"E_HF={e_hf:.8f}  E_corr={e_corr_total:.8f}  E_total={e_total:.8f} Ha"
    )

    # ── Write summary JSON ────────────────────────────────────────────────────
    data_path = Path(data_dir)
    if results_dir is None:
        results_dir = str(data_path.parent.parent / "results" / run_name)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name":    run_name,
        "e_hf":        e_hf,
        "e_corr":      e_corr_total,
        "e_total":     e_total,
        "n_fragments": len(fragment_results),
        "fragments":   fragment_energies,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[step4] {run_name}: wrote {summary_path}")

    return {
        "run_name": run_name,
        "e_hf":    e_hf,
        "e_corr":  e_corr_total,
        "e_total": e_total,
        "fragments": fragment_energies,
    }
