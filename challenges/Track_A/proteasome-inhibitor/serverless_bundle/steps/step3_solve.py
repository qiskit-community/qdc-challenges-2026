"""
Step 3 — Per-fragment solver (FCI / CCSD / SQD / TrimSQD).

Three-tier adaptive strategy matching the glycine notebook demo:

    n_orb <= fci_threshold              → FCI   (exact; H fragments at 3–4 orb)
    fci_threshold < n_orb < sqd_threshold → CCSD  (N, Cα, O fragments at 13–17 orb)
    n_orb >= sqd_threshold              → SQD / TrimSQD / both
                                           (CCSD only in a deliberately classical
                                            run: qpu_backend=None, solver_mode="sqd")

Quantum requests fail closed.  A fragment whose SQD / TrimSQD treatment was requested
-- by an explicit ``solver_assignments`` entry, or through the size-threshold route in a
run that was given a ``qpu_backend`` or a non-default ``solver_mode`` -- ends in exactly
one of two ways: the requested solver ran and its result carries that solver's identity,
or ``QuantumSolverError`` is raised.  CCSD is never substituted.

Default thresholds (from config_glycine_sto-3g.yaml):
    fci_threshold  = 10
    sqd_threshold  = 19

solver_mode controls which quantum solver(s) run on fragments >= sqd_threshold:
    "sqd"     — standard SQD only (default)
    "trimsqd" — TrimSQD only
    "both"    — standard SQD first, then TrimSQD post-processing on the same
                QPU counts (no second QPU job); both energies saved in result
"""

from __future__ import annotations

import hashlib
import pickle
import sys
import time
from pathlib import Path

# Import canonical constants from the authoritative source (solver_assignments).
# Do NOT duplicate these lists here — the registry is the single source of truth.
try:
    _pkg_root = Path(__file__).resolve().parents[2]
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from proteasome_challenge.solver_assignments import (
        QUANTUM_SOLVERS as _QS_FROZENSET,
        is_valid_solver as _is_valid_solver,
    )
    QUANTUM_SOLVERS: tuple[str, ...] = tuple(sorted(_QS_FROZENSET))
    _VALID_SOLVER_FN = _is_valid_solver
except ImportError:
    # Fallback when bundle runs without the proteasome_challenge package
    QUANTUM_SOLVERS = ("SQD", "TrimSQD")
    _VALID_SOLVER_FN = lambda s: s in ("FCI", "CCSD", "SQD", "TrimSQD")  # noqa: E731

SOLVER_MODES = ("sqd", "trimsqd", "both")
# Metadata keys only the CCSD solver writes; their presence means "this is a CCSD result".
_CCSD_RESULT_MARKERS = ("ccsd_diagnostics", "e_hf", "conv_tol")


class QuantumSolverError(RuntimeError):
    """An explicitly requested SQD / TrimSQD fragment could not be solved by that solver."""


def _check_counts_file(path: Path, norb: int) -> None:
    """Raise QuantumSolverError unless *path* holds usable frozen measurement counts."""
    import numpy as np

    try:
        raw = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        raise QuantumSolverError(
            f"measurement counts {path} cannot be read ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(raw, dict) or not raw:
        raise QuantumSolverError(f"measurement counts {path} are empty or not a counts dict")
    try:
        shots = sum(int(v) for v in raw.values())
        non_positive = any(int(v) <= 0 for v in raw.values())
    except Exception as exc:
        raise QuantumSolverError(f"measurement counts {path} have non-numeric values") from exc
    if shots <= 0 or non_positive:
        raise QuantumSolverError(f"measurement counts {path} have non-positive shot counts")
    keys = list(raw)
    if all(isinstance(k, str) for k in keys):
        widths = {len(k) for k in keys}
        if widths != {2 * norb} or any(set(k) - {"0", "1"} for k in keys):
            raise QuantumSolverError(
                f"measurement counts {path} are not {2 * norb}-bit strings for a {norb}-orbital "
                f"fragment (bit widths found: {sorted(widths)}); counts of another fragment?"
            )
    elif not all(isinstance(k, (int, np.integer)) for k in keys):
        raise QuantumSolverError(f"measurement counts {path} have unsupported key types")
    elif max(int(k) for k in keys).bit_length() > 2 * norb:
        raise QuantumSolverError(
            f"measurement counts {path} contain outcomes wider than {2 * norb} bits"
        )


_MEASUREMENT_MODES = ("live", "replay")
_MEASUREMENT_KEYS = {"mode", "solver", "n_qubits", "backend", "hamiltonian_sha256", "counts_file",
                     "counts_sha256"}


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _normalize_measurement_plan(raw) -> dict:
    """``{fragment_id: entry}`` from the JSON ``measurement_plan`` argument; QuantumSolverError if malformed.

    Every entry must name its mode explicitly: there is no default and no inference from "counts
    happen to be staged".
    """
    if not isinstance(raw, dict) or not raw:
        raise QuantumSolverError("measurement_plan must be a non-empty {fragment_id: entry} object")
    out = {}
    for key, entry in raw.items():
        try:
            fid = int(key)
        except (TypeError, ValueError):
            raise QuantumSolverError(f"measurement_plan key {key!r} is not a fragment id") from None
        if fid in out:
            raise QuantumSolverError(f"measurement_plan names fragment {fid} more than once")
        if not isinstance(entry, dict) or set(entry) != _MEASUREMENT_KEYS:
            raise QuantumSolverError(
                f"measurement_plan[{fid}] must carry exactly the keys {sorted(_MEASUREMENT_KEYS)}")
        if entry["mode"] not in _MEASUREMENT_MODES:
            raise QuantumSolverError(
                f"measurement_plan[{fid}]: mode must be one of {_MEASUREMENT_MODES}, got {entry['mode']!r}")
        if entry["solver"] not in QUANTUM_SOLVERS:
            raise QuantumSolverError(
                f"measurement_plan[{fid}]: solver must be one of {QUANTUM_SOLVERS}, got {entry['solver']!r}")
        if not isinstance(entry["backend"], str) or not entry["backend"]:
            raise QuantumSolverError(f"measurement_plan[{fid}]: a backend identity is required")
        for field in ("hamiltonian_sha256", "counts_sha256"):
            v = entry[field]
            if v is not None and not (isinstance(v, str) and len(v) == 64 and set(v) <= set("0123456789abcdef")):
                raise QuantumSolverError(f"measurement_plan[{fid}].{field} is not a SHA-256: {v!r}")
        if entry["hamiltonian_sha256"] is None:
            raise QuantumSolverError(f"measurement_plan[{fid}]: hamiltonian_sha256 is required")
        if entry["mode"] == "replay" and (entry["counts_sha256"] is None or not entry["counts_file"]):
            raise QuantumSolverError(f"measurement_plan[{fid}]: a replay entry pins counts_file and counts_sha256")
        if entry["mode"] == "live" and (entry["counts_sha256"] is not None or entry["counts_file"]):
            raise QuantumSolverError(f"measurement_plan[{fid}]: a live entry carries no counts to replay")
        out[fid] = dict(entry)
    return out


def _check_planned_measurement(fid: int, label: str, entry: dict, norb: int, ham_sha: str,
                               counts_p: Path, qpu_backend) -> None:
    """Pre-flight identity checks of one quantum fragment against its plan entry (before any solve)."""
    if entry["solver"] != label:
        raise QuantumSolverError(
            f"fragment {fid}: the measurement plan says {entry['solver']} but the fragment is routed to "
            f"{label}. Nothing was solved.")
    if entry["n_qubits"] != 2 * norb:
        raise QuantumSolverError(
            f"fragment {fid}: the plan expects {entry['n_qubits']} qubits but the embedded Hamiltonian "
            f"has {norb} orbitals = {2 * norb} qubits. Nothing was solved.")
    if entry["hamiltonian_sha256"] != ham_sha:
        raise QuantumSolverError(
            f"fragment {fid}: the plan was made for a different embedded Hamiltonian "
            f"({entry['hamiltonian_sha256'][:12]}... != {ham_sha[:12]}...). Nothing was solved.")
    if entry["mode"] == "live":
        if qpu_backend != entry["backend"]:
            raise QuantumSolverError(
                f"fragment {fid}: live sampling is planned on {entry['backend']!r} but the job's "
                f"qpu_backend is {qpu_backend!r}. Nothing was solved.")
        if counts_p.exists():
            raise QuantumSolverError(
                f"fragment {fid}: planned as live, but counts are already staged at {counts_p}; they would "
                "be replayed silently instead of sampling. Nothing was solved.")
        return
    if not counts_p.exists():
        raise QuantumSolverError(f"fragment {fid}: replay planned but its counts {counts_p} are not staged")
    _check_counts_file(counts_p, norb)
    if _sha256_of_file(counts_p) != entry["counts_sha256"]:
        raise QuantumSolverError(
            f"fragment {fid}: the staged counts do not match the SHA-256 pinned in the plan; they are "
            "not the planned measurement. Nothing was solved.")


_PROTOCOL_SQD_KEYS = ("classical_backend", "symmetrize_spin", "n_batches", "iterations",
                      "samples_per_batch", "max_dim", "energy_tol", "occupancies_tol", "seed")
_PROTOCOL_TRIM_KEYS = ("trim_ratio", "max_strings_per_trim", "max_carryover", "sector_trim")
_PROTOCOL_KEYS = {"schema_version", "kind", "protocol_digest", "document_digest", "applied_config",
                  "applied_config_digest", "adaptations", "source_fragment_id", "sqd_config",
                  "trim_config", "targets"}
_PROTOCOL_TARGET_KEYS = {"role", "hamiltonian_sha256", "n_orb", "n_occ", "n_elec",
                         "effective_config_digest"}


def _config_digest_of(cfg: dict) -> str:
    """The SQD solvers' ``_config_hash`` convention (``sqd_config_hash`` in their metadata)."""
    import json
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def _canonical_digest_of(obj) -> str:
    import json
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _normalize_shared_protocol(raw) -> dict:
    """``{fragment_id: target}`` plus the protocol, from the JSON ``shared_protocol`` argument.

    Everything is re-derived here from the transferable settings and compared with the digests the
    client sent, so an edited or inconsistent protocol is refused before any solve.
    """
    if not isinstance(raw, dict) or set(raw) != _PROTOCOL_KEYS:
        raise QuantumSolverError(
            f"shared_protocol must be an object with exactly the keys {sorted(_PROTOCOL_KEYS)}")
    if raw["kind"] != "shared_trimsqd_protocol" or raw["schema_version"] != "1.0":
        raise QuantumSolverError("shared_protocol is not a shared_trimsqd_protocol schema 1.0")
    if raw["adaptations"] != []:
        raise QuantumSolverError(
            f"shared_protocol names adaptation rule(s) {raw['adaptations']!r}; none is approved in this "
            "bundle. The protocol is applied unchanged or not at all.")
    sqd, trim = raw["sqd_config"], raw["trim_config"]
    if not isinstance(sqd, dict) or not isinstance(trim, dict):
        raise QuantumSolverError("shared_protocol carries no sqd_config / trim_config")
    miss = [k for k in _PROTOCOL_SQD_KEYS if k not in sqd] + [k for k in _PROTOCOL_TRIM_KEYS if k not in trim]
    if miss:
        raise QuantumSolverError(f"shared_protocol lacks required transferable setting(s) {miss}")
    unknown = sorted(set(sqd) - set(_PROTOCOL_SQD_KEYS) - {"carryover_threshold"}) \
        + sorted(set(trim) - set(_PROTOCOL_TRIM_KEYS))
    if unknown:
        raise QuantumSolverError(f"shared_protocol carries non-transferable setting(s) {unknown}")
    view = {k: sqd[k] for k in _PROTOCOL_SQD_KEYS}
    view["trim"] = {k: trim[k] for k in _PROTOCOL_TRIM_KEYS}
    if raw["applied_config"] != view or raw["applied_config_digest"] != _config_digest_of(view):
        raise QuantumSolverError("shared_protocol: applied_config does not match the transferable settings")
    if raw["protocol_digest"] != _canonical_digest_of(
            {"solver": "TrimSQD", "sqd_config": sqd, "trim_config": trim, "adaptations": []}):
        raise QuantumSolverError("shared_protocol: protocol_digest does not match the transferable settings")
    eff = _canonical_digest_of({"solver": "TrimSQD", "applied": view, "adaptations": []})
    targets = {}
    if not isinstance(raw["targets"], dict) or not raw["targets"]:
        raise QuantumSolverError("shared_protocol.targets must be a non-empty {fragment_id: target} object")
    for key, t in raw["targets"].items():
        try:
            fid = int(key)
        except (TypeError, ValueError):
            raise QuantumSolverError(f"shared_protocol target key {key!r} is not a fragment id") from None
        if not isinstance(t, dict) or set(t) != _PROTOCOL_TARGET_KEYS:
            raise QuantumSolverError(
                f"shared_protocol.targets[{fid}] must carry exactly {sorted(_PROTOCOL_TARGET_KEYS)}")
        if t["effective_config_digest"] != eff:
            raise QuantumSolverError(
                f"fragment {fid}: the target's effective configuration differs from the protocol's")
        targets[fid] = dict(t)
    if raw["source_fragment_id"] not in targets or targets[raw["source_fragment_id"]]["role"] != "source" \
            or sum(t["role"] == "source" for t in targets.values()) != 1:
        raise QuantumSolverError("shared_protocol must name exactly one source among its targets")
    return {"protocol": raw, "targets": targets, "effective_config_digest": eff, "applied": view}


def _check_protocol_applied(result, fid: int, spec: dict) -> dict:
    """Prove the TrimSQD solver that produced *result* ran with the protocol's settings.

    The solver records the configuration it actually used (``metadata['sqd_config']`` and its hash);
    a result whose record differs from the protocol is refused.  Returns the per-fragment record.
    """
    md = getattr(result, "metadata", None) or {}
    applied = spec["applied"]
    if md.get("sqd_config") != applied:
        raise QuantumSolverError(
            f"fragment {fid}: the solver's recorded configuration differs from the shared protocol "
            f"(recorded {md.get('sqd_config')!r}, protocol {applied!r}); nothing is silently retuned")
    if md.get("sqd_config_hash") != spec["protocol"]["applied_config_digest"]:
        raise QuantumSolverError(
            f"fragment {fid}: the solver's sqd_config_hash does not match the shared protocol")
    return {"verified": True, "protocol_digest": spec["protocol"]["protocol_digest"],
            "applied_config_digest": spec["protocol"]["applied_config_digest"],
            "effective_config_digest": spec["effective_config_digest"],
            "source_fragment_id": spec["protocol"]["source_fragment_id"],
            "role": spec["targets"][fid]["role"], "adaptations": []}


def _check_quantum_identity(result, label: str, frag_id: int) -> None:
    """Raise QuantumSolverError unless *result* was produced by the *label* solver.

    SQD and TrimSQD results both carry ``measurement_sha256`` (the counts they consumed);
    only TrimSQD adds ``policy`` / ``trim_ratio``; only CCSD writes the markers in
    ``_CCSD_RESULT_MARKERS``.
    """
    md = getattr(result, "metadata", None) or {}
    ccsd_keys = [k for k in _CCSD_RESULT_MARKERS if k in md]
    if ccsd_keys:
        raise QuantumSolverError(
            f"fragment {frag_id}: {label} was requested but the returned result is a CCSD result "
            f"(metadata keys {ccsd_keys}); a CCSD result never satisfies a quantum request"
        )
    if not md.get("measurement_sha256"):
        raise QuantumSolverError(
            f"fragment {frag_id}: {label} result records no measurement provenance "
            "(metadata['measurement_sha256']); it cannot have consumed quantum measurements"
        )
    is_trim = "policy" in md and "trim_ratio" in md
    if label == "TrimSQD" and not is_trim:
        raise QuantumSolverError(
            f"fragment {frag_id}: TrimSQD was requested but the result has no TrimSQD policy "
            "metadata; standard SQD must not stand in for TrimSQD"
        )
    if label == "SQD" and ("policy" in md or "trim_ratio" in md):
        raise QuantumSolverError(
            f"fragment {frag_id}: SQD was requested but the result carries TrimSQD metadata"
        )


def _software_versions() -> dict:
    """Versions of the packages that determine the classical numerics."""
    out = {}
    for name in ("pyscf", "numpy", "scipy"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:  # pragma: no cover - environment dependent
            out[name] = "unavailable"
    return out


def run_solve(
    data_dir: str,
    run_name: str,
    fci_threshold: int = 10,
    sqd_threshold: int = 19,
    qpu_backend: str | None = None,
    n_shots: int = 100_000,
    n_reps: int = 2,
    sqd_config: dict | None = None,
    solver_mode: str = "sqd",
    trim_config: dict | None = None,
    resume_job_ids: dict | None = None,
    compute_rdms: bool = True,
    basis: str | None = None,
    fragment_ids: list[int] | None = None,
    quantum_fragment_ids: list[int] | None = None,
    solver_assignments: dict | None = None,
    use_threshold_fallback: bool = True,
    replay_fragment_ids: list[int] | None = None,
    measurement_plan: dict | None = None,
    shared_protocol: dict | None = None,
    stage3_config=None,
) -> dict:
    """
    Solve all EWF fragments loaded from *data_dir* using the three-tier strategy.

    Parameters
    ----------
    data_dir        : directory containing mf_data.pkl + embedding_data.pkl +
                      ewf_dumpfile.h5 (outputs of steps 1 & 2)
    run_name        : short label for logging (e.g. "ixazomib_ligand_sto-3g")
    fci_threshold   : n_orb <= this → FCI
    sqd_threshold   : n_orb >= this → quantum solver; else CCSD
    qpu_backend     : IBM backend name for SQD / TrimSQD.  None → a deliberately
                      classical run (threshold route only); an explicit SQD / TrimSQD
                      assignment without a backend raises QuantumSolverError.
    n_shots         : shots per SQD QPU circuit
    n_reps          : LUCJ circuit repetitions (n_reps=1 matches challenge config)
    sqd_config      : dict of extra SQD classical post-processing options.
                      When ``stage3_config`` is given, this is ignored.
    solver_mode     : "sqd" | "trimsqd" | "both"
                      "both" runs SQD then TrimSQD on the same QPU counts.
    trim_config     : TrimSQD-specific config (trim_ratio, max_strings_per_trim, …).
                      When ``stage3_config`` is given, this is ignored.
    compute_rdms    : whether to compute 1- and 2-RDMs (needed for step 4).
                      Always required for reconstruction; default True.
    basis           : basis set label (e.g. "sto-3g").  Used as an IBM Quantum job
                      tag alongside the system name and fragment ID.  When None,
                      inferred by stripping the last "_<token>" from run_name.
    fragment_ids    : when given, ONLY these fragment IDs are solved (single-fragment
                      runs).  None (default) solves every fragment.
    quantum_fragment_ids : when given, ONLY these fragment IDs may use SQD/TrimSQD;
                      every other fragment >= sqd_threshold falls back to CCSD.
                      None (default) lets every fragment >= sqd_threshold use the QPU.
                      Legacy threshold path only; ignored for fragments in
                      *solver_assignments*.
    solver_assignments : explicit {fragment_id: solver_name} routing map.
                      Solver names must be registered in the solver registry.
                      A fragment present in the map is routed exactly as
                      assigned, regardless of n_orb.  "SQD"/"TrimSQD" are explicit
                      quantum requests: if that solver could not be built (no QPU
                      backend, mismatching solver_mode, initialization failure) or
                      its solve fails, QuantumSolverError is raised -- never CCSD.
    shared_protocol : single-source TrimSQD protocol (Task 4B.2).  When given, every TrimSQD
                      fragment runs with these frozen settings: ``sqd_config`` / ``trim_config``
                      must equal the protocol's, every TrimSQD fragment must be a protocol target
                      whose Hamiltonian / n_orb / n_occ match, and after each solve the solver's
                      recorded configuration must equal the protocol's.  Nothing is retuned.
    use_threshold_fallback : when True (default), fragments absent from
                      *solver_assignments* (or all fragments, if it is None) use
                      the legacy n_orb threshold policy.  When False,
                      *solver_assignments* is mandatory and must cover every
                      solved fragment; otherwise ValueError is raised.
    stage3_config   : optional ``Stage3Config`` (or dict).  When given, its per-solver
                      settings override the individual ``sqd_config`` / ``trim_config`` /
                      ``n_reps`` parameters.  Allocation thresholds (``fci_threshold``,
                      ``sqd_threshold``) are not part of Stage3Config and must be
                      supplied separately.

    Returns
    -------
    dict with keys: n_fragments, n_fci, n_ccsd, n_sqd, n_trimsqd, e_corr_total,
                    solver_results_pkl
                    (plus e_corr_total_trimsqd when solver_mode=="both")
    """
    import h5py

    from quantum_fragment_methods.application.solvers.classical_zoo.ccsd import CCSD
    from quantum_fragment_methods.application.solvers.classical_zoo.fci import FCI

    # ── Apply Stage3Config overrides ──────────────────────────────────────────
    # stage3_config is the preferred interface.  When given, its per-solver
    # settings take precedence over the individual sqd_config / trim_config /
    # n_reps / conv_tol / max_cycle kwargs.
    #
    # Propagation:
    #   stage3_config.fci    → FCI(conv_tol=, max_cycle=, spin=)
    #   stage3_config.ccsd   → CCSD(conv_tol=, max_cycle=, diis_space=, level_shift=)
    #   stage3_config.sqd    → sqd_config dict fed to SQDSolver
    #   stage3_config.trimsqd→ sqd_config + trim_config dicts fed to TrimSQDSolver
    _fci_kwargs:  dict = {}
    _ccsd_kwargs: dict = {}
    if stage3_config is not None:
        try:
            from proteasome_challenge.solver_config import Stage3Config as _S3C
            if isinstance(stage3_config, dict):
                stage3_config = _S3C.from_dict(stage3_config)
            if not isinstance(stage3_config, _S3C):
                raise ValueError(
                    f"stage3_config must be a Stage3Config or dict, got {type(stage3_config).__name__!r}"
                )
        except ImportError:
            pass   # running outside proteasome_challenge package — treat as opaque dict

        if hasattr(stage3_config, "fci"):
            _fci_kwargs = stage3_config.fci.to_dict()

        if hasattr(stage3_config, "ccsd"):
            _ccsd_kwargs = stage3_config.ccsd.to_dict()

        if hasattr(stage3_config, "sqd"):
            # SQD base config: use SQDConfig (not overwritten by TrimSQD below)
            sqd_config = stage3_config.sqd.to_sqd_dict()
            n_reps = stage3_config.sqd.n_reps

        if hasattr(stage3_config, "trimsqd"):
            # TrimSQD trim sub-dict extracted separately from the SQD base
            trim_config = stage3_config.trimsqd.to_sqd_dict().get("trim", {})
            # When running in trimsqd or both mode, TrimSQD also needs its own
            # shared SQD-level parameters (iterations, n_batches, etc.).  These
            # live in trimsqd.to_sqd_dict() minus the "trim" key.
            if solver_mode in ("trimsqd", "both"):
                _tsqd_dict = stage3_config.trimsqd.to_sqd_dict()
                sqd_config = {k: v for k, v in _tsqd_dict.items() if k != "trim"}
                n_reps = stage3_config.trimsqd.n_reps
            # If solver_mode == "sqd", keep sqd_config from stage3_config.sqd above.

    data_path = Path(data_dir)

    dumpfile = str(data_path / "ewf_dumpfile.h5")

    # ── Build fragment stubs from HDF5 — avoids unpickling vayesta objects ───
    # The embedding pickle contains vayesta EWF objects whose __reduce__ chain
    # imports vayesta.libs.libcore (a C extension that requires libopenblas.so.0).
    # That shared library is absent in the Stage 3 container, causing an
    # AttributeError on pickle.load().  Everything step3 actually needs from the
    # embedding result is already stored as HDF5 attrs: norb and nocc per fragment.
    class _FragStub:
        """Lightweight stand-in for a vayesta Fragment object."""
        __slots__ = ("n_orbitals",)
        def __init__(self, n_orbitals: int):
            self.n_orbitals = n_orbitals

    class _EmbStub:
        """Lightweight stand-in for a vayesta EmbeddingResult object."""
        __slots__ = ("fragments",)
        def __init__(self, fragments: dict):
            self.fragments = fragments

    with h5py.File(dumpfile, "r") as _hf:
        _frag_stubs = {
            int(k.split("_")[1]): _FragStub(int(_hf[k].attrs["norb"]))
            for k in _hf
            if k.startswith("fragment_")
        }
    embedding_result = _EmbStub(_frag_stubs)

    fragments = embedding_result.fragments
    n_frag = len(fragments)
    _assign = None
    if solver_assignments:
        _assign = {int(k): str(v) for k, v in solver_assignments.items()}
        # Use the registry as the authoritative source of valid solver names.
        _bad = {k: v for k, v in _assign.items() if not _VALID_SOLVER_FN(v)}
        if _bad:
            raise ValueError(
                f"solver_assignments contains unregistered solvers: {_bad}. "
                f"Register the solver contract before dispatch."
            )
    if not use_threshold_fallback:
        if _assign is None:
            raise ValueError(
                "use_threshold_fallback=False requires an explicit solver_assignments map."
            )
        _uncovered = sorted(int(k) for k in fragments if int(k) not in _assign)
        if _uncovered:
            raise ValueError(
                f"solver_assignments does not cover fragments {_uncovered} "
                "and use_threshold_fallback=False."
            )
    print(
        f"[step3] {run_name}: {n_frag} fragments (from HDF5 attrs)  "
        f"FCI<=fci_threshold={fci_threshold}  "
        f"CCSD<sqd_threshold={sqd_threshold}  "
        f"quantum>={sqd_threshold}  solver_mode={solver_mode}"
    )

    # ── Instantiate solvers ───────────────────────────────────────────────────
    if solver_mode not in SOLVER_MODES:
        raise ValueError(f"solver_mode must be one of {SOLVER_MODES}, got {solver_mode!r}")

    fci_solver  = FCI(**_fci_kwargs)
    ccsd_solver = CCSD(**_ccsd_kwargs)

    sqd_solver      = None
    trimsqd_solver  = None
    # Kept, not swallowed: it is raised (chained) as soon as a fragment needs the solver.
    _quantum_init_error: Exception | None = None

    if qpu_backend is not None:
        try:
            from qiskit_serverless import get_runtime_service
            from quantum_fragment_methods.application.solvers.quantum_zoo.sqd import SQDSolver
            from quantum_fragment_methods.qpu.qiskit_ibm_runtime import IBMQuantumBackend

            # get_runtime_service() is the only egress-allowed path inside Fleets.
            # QiskitRuntimeService(channel=..., token=..., instance=...) triggers a
            # TCP connection to the IBM Cloud resource-controller that hangs for 1080s.
            _svc     = get_runtime_service()
            _backend = _svc.backend(qpu_backend)
            print(
                f"[step3] {run_name}: QPU via get_runtime_service() → "
                f"{_backend.name}  {_backend.num_qubits} qubits",
                flush=True,
            )

            # Wrap in IBMQuantumBackend with the already-fetched service + backend
            # injected directly — initialize() / get_backend() are never called,
            # so no second network round-trip occurs.
            backend_obj = IBMQuantumBackend({
                "provider":     "ibm_quantum",
                "backend_name": qpu_backend,
                "credentials": {
                    "channel":  "ibm_cloud",
                    "token":    "",     # not used — service already live
                    "instance": "",
                },
                "sampler_options": {"default_shots": n_shots},
            })
            backend_obj.service         = _svc
            backend_obj.backend         = _backend
            backend_obj._qiskit_backend = _backend

            _base_config = {
                "lucj": {"n_reps": n_reps},
                "classical_backend": "python",
                **(sqd_config or {}),
            }

            if solver_mode in ("sqd", "both"):
                sqd_solver = SQDSolver(
                    qpu_backend=backend_obj,
                    config=_base_config,
                )
                print(f"[step3] {run_name}: SQDSolver ready  shots={n_shots}", flush=True)

            if solver_mode in ("trimsqd", "both"):
                from quantum_fragment_methods.application.solvers.quantum_zoo.trim_sqd import (
                    TrimSQDSolver,
                )
                _trim_config = {**_base_config, "trim": trim_config or {}}
                trimsqd_solver = TrimSQDSolver(
                    qpu_backend=backend_obj,
                    config=_trim_config,
                )
                print(
                    f"[step3] {run_name}: TrimSQDSolver ready  "
                    f"trim_ratio={(_trim_config.get('trim') or {}).get('trim_ratio', 0.5)}",
                    flush=True,
                )

        except Exception as exc:
            sqd_solver = trimsqd_solver = None      # never leave a half-built pair behind
            _quantum_init_error = exc
            print(
                f"[step3] ERROR: could not initialise the quantum solver(s): "
                f"{type(exc).__name__}: {exc}.  A fragment that needs SQD/TrimSQD will fail; "
                "CCSD is never substituted.",
                flush=True,
            )

    # ── Routing plan (decided for every solved fragment BEFORE any solve) ────
    # Intent is read from the arguments the client sends:
    #   * a "SQD"/"TrimSQD" entry in solver_assignments is an explicit quantum request;
    #   * on the size-threshold route, a quantum-sized fragment is a quantum request when
    #     the run was given a qpu_backend or asked for solver_mode "trimsqd"/"both"
    #     (the frozen classical baseline and explicit CCSD jobs send qpu_backend=None and
    #     solver_mode="sqd" with a CCSD/FCI map, so they are never quantum requests);
    #   * quantum_fragment_ids names fragments whose quantum treatment is requested.
    _threshold_quantum = qpu_backend is not None or solver_mode in ("trimsqd", "both")
    _only_ids    = None if fragment_ids is None else {int(i) for i in fragment_ids}
    _quantum_ids = None if quantum_fragment_ids is None else {int(i) for i in quantum_fragment_ids}
    _replay_ids  = None if replay_fragment_ids is None else {int(i) for i in replay_fragment_ids}
    _mplan = None
    if measurement_plan is not None:
        if replay_fragment_ids is not None:
            raise QuantumSolverError(
                "measurement_plan and replay_fragment_ids are two ways to say where measurements come "
                "from; give one. Nothing was solved.")
        _mplan = _normalize_measurement_plan(measurement_plan)
    if _only_ids is not None:
        _unknown = _only_ids - {int(k) for k in fragments}
        if _unknown:
            raise KeyError(f"fragment_ids not present in the embedding: {sorted(_unknown)}")

    def _route(fid: int, norb: int) -> tuple[str, bool]:
        """Return (solver label, is_quantum_request) for one fragment."""
        if _assign is not None and fid in _assign:
            return _assign[fid], _assign[fid] in QUANTUM_SOLVERS
        # Quantum treatment of THIS fragment was asked for: it is in quantum_fragment_ids, or
        # the job is a single-fragment threshold job given a backend / non-default solver_mode.
        requested = (
            (_quantum_ids is not None and fid in _quantum_ids)
            or (_quantum_ids is None and _only_ids is not None and len(_only_ids) == 1
                and _threshold_quantum)
        )
        if requested and norb < sqd_threshold:
            raise QuantumSolverError(
                f"fragment {fid} (n_orb={norb}) was requested for quantum treatment but is below "
                f"sqd_threshold={sqd_threshold}; the threshold route would hand it to FCI/CCSD. "
                "Nothing was solved."
            )
        if norb <= fci_threshold:
            return "FCI", False
        if norb < sqd_threshold:
            return "CCSD", False
        if _quantum_ids is not None and fid not in _quantum_ids:
            return "CCSD", False        # quantum-sized but deliberately not selected
        if requested or _threshold_quantum:
            return ("TrimSQD" if solver_mode == "trimsqd" else "SQD"), True
        return "CCSD", False            # deliberately classical reference run

    plan: dict[int, tuple[str, bool]] = {}
    for frag_id, frag in fragments.items():
        if _only_ids is None or int(frag_id) in _only_ids:
            plan[int(frag_id)] = _route(int(frag_id), frag.n_orbitals)

    _pspec = None
    if shared_protocol is not None:
        _pspec = _normalize_shared_protocol(shared_protocol)
        _pp = _pspec["protocol"]
        if solver_mode != "trimsqd":
            raise QuantumSolverError(
                f"a shared TrimSQD protocol needs solver_mode='trimsqd', got {solver_mode!r}. "
                "Nothing was solved.")
        if dict(sqd_config or {}) != _pp["sqd_config"] or dict(trim_config or {}) != _pp["trim_config"]:
            raise QuantumSolverError(
                "the job's sqd_config / trim_config differ from the shared protocol's transferable "
                "settings; the solvers would not run the protocol. Nothing was solved.")
        if not any(q for _, q in plan.values()):
            raise QuantumSolverError(
                "a shared protocol was given but this job routes no fragment to a quantum solver. "
                "Nothing was solved.")
        with h5py.File(dumpfile, "r") as _hfp:
            for _pf, (_pl, _pq) in plan.items():
                _t = _pspec["targets"].get(_pf)
                if _t is None:
                    if _pq:
                        raise QuantumSolverError(
                            f"fragment {_pf}: routed to {_pl} but it is not a target of the shared "
                            f"protocol (targets: {sorted(_pspec['targets'])}). Nothing was solved.")
                    continue
                if not (_pq and _pl == "TrimSQD"):
                    raise QuantumSolverError(
                        f"fragment {_pf} is a target of the shared TrimSQD protocol but is routed to "
                        f"{_pl}; it would silently run another solver. Nothing was solved.")
                _gp = _hfp[f"fragment_{_pf}"]
                _hp = hashlib.sha256(_gp["heff"][:].tobytes() + _gp["eris"][:].tobytes()).hexdigest()
                _nop, _nocp = int(_gp.attrs["norb"]), int(_gp.attrs["nocc"])
                if (_hp, _nop, _nocp, 2 * _nocp) != (_t["hamiltonian_sha256"], _t["n_orb"],
                                                      _t["n_occ"], _t["n_elec"]):
                    raise QuantumSolverError(
                        f"fragment {_pf}: its Hamiltonian / orbital / electron counts "
                        f"(n_orb={_nop}, n_occ={_nocp}) are not the protocol target's "
                        f"(n_orb={_t['n_orb']}, n_occ={_t['n_occ']}, hash {_t['hamiltonian_sha256'][:12]}...). "
                        "Nothing was solved.")

    _need_trim_secondary = False
    for fid, (label, is_quantum) in plan.items():
        if not is_quantum:
            continue
        solver = sqd_solver if label == "SQD" else trimsqd_solver
        if solver is None:
            if _quantum_init_error is not None:
                why = (f"quantum solver initialisation failed "
                       f"({type(_quantum_init_error).__name__}: {_quantum_init_error})")
            elif qpu_backend is None:
                why = "no qpu_backend was given"
            else:
                why = f"solver_mode={solver_mode!r} does not construct {label}"
            raise QuantumSolverError(
                f"fragment {fid}: {label} was requested but cannot run: {why}. "
                "CCSD is not substituted; nothing was solved."
            ) from _quantum_init_error
        if solver_mode == "both" and label == "SQD":
            _need_trim_secondary = True
        # Measurements: a replayed fragment must have valid staged counts; a fragment that
        # would sample the QPU is refused when the job is a replay job.
        counts_p = data_path / f"fragment_{fid}" / "counts.npy"
        if _mplan is not None:
            # Fragment-specific plan: the mode is explicit per fragment; nothing is inferred.
            if fid not in _mplan:
                raise QuantumSolverError(
                    f"fragment {fid}: {label} was requested but the measurement plan has no entry for it "
                    f"(planned: {sorted(_mplan)}); running it could sample the QPU or replay the wrong "
                    "counts. Nothing was solved.")
            with h5py.File(dumpfile, "r") as hf:
                _g0 = hf[f"fragment_{fid}"]
                _ham0 = hashlib.sha256(_g0["heff"][:].tobytes() + _g0["eris"][:].tobytes()).hexdigest()
            _check_planned_measurement(fid, label, _mplan[fid], fragments[fid].n_orbitals, _ham0,
                                       counts_p, qpu_backend)
            continue
        if _replay_ids is not None and fid not in _replay_ids:
            raise QuantumSolverError(
                f"fragment {fid}: {label} was requested but no frozen counts are staged for it "
                f"(staged: {sorted(_replay_ids)}); running it would sample the QPU. Nothing was solved."
            )
        if (_replay_ids is not None and fid in _replay_ids) or counts_p.exists():
            if not counts_p.exists():
                raise QuantumSolverError(
                    f"fragment {fid}: {label} replay requested but {counts_p} is missing"
                )
            _check_counts_file(counts_p, fragments[fid].n_orbitals)
    if _mplan is not None:
        _not_quantum = sorted(f for f in _mplan if not (f in plan and plan[f][1]))
        if _not_quantum:
            raise QuantumSolverError(
                f"the measurement plan names fragment(s) {_not_quantum}, but this job does not route "
                "them to SQD/TrimSQD; a measurement is never attached to a classical fragment. "
                "Nothing was solved.")
    if _need_trim_secondary and trimsqd_solver is None:
        raise QuantumSolverError(
            "solver_mode='both' requires TrimSQD as well as SQD, but TrimSQD is unavailable"
            + (f" ({type(_quantum_init_error).__name__}: {_quantum_init_error})"
               if _quantum_init_error is not None else "")
            + ". Nothing was solved."
        ) from _quantum_init_error

    if not any(q for _, q in plan.values()) and qpu_backend is None and solver_mode == "sqd":
        print("[step3] NOTE: no quantum fragment requested (classical reference run).", flush=True)

    counters = {"fci": 0, "ccsd": 0, "sqd": 0, "trimsqd": 0}
    fragment_results     = {}   # primary result (SQD or TrimSQD or classical)
    fragment_results_trim = {}  # TrimSQD secondary result (mode=="both" only)
    fragment_summaries   = {}   # per-fragment audit record returned to the caller

    # ── Solve each fragment ───────────────────────────────────────────────────
    for frag_id, frag in fragments.items():
        if _only_ids is not None and int(frag_id) not in _only_ids:
            continue
        norb = frag.n_orbitals

        solver_label, _quantum_request = plan[int(frag_id)]
        # Built-in solvers are looked up from the pre-constructed objects above.
        # An extended solver registered via register_solver_factory() is
        # instantiated on demand here.  The factory receives no arguments by
        # default; pass solver-specific kwargs through stage3_config.
        _builtin_map = {
            "FCI": fci_solver, "CCSD": ccsd_solver,
            "SQD": sqd_solver, "TrimSQD": trimsqd_solver,
        }
        if solver_label in _builtin_map:
            primary_solver = _builtin_map[solver_label]
        else:
            try:
                from proteasome_challenge.solver_assignments import get_solver_factory
                _factory = get_solver_factory(solver_label)
                primary_solver = _factory()
            except Exception as _fe:
                raise QuantumSolverError(
                    f"fragment {frag_id}: solver {solver_label!r} is not a built-in solver and "
                    f"could not be instantiated from its registered factory "
                    f"({type(_fe).__name__}: {_fe}). Nothing was solved."
                ) from _fe

        _tag_suffix = f"  tags=[{run_name.rsplit('_',1)[0][:20]!r}, 'fragment_{int(frag_id)}', {solver_label.lower()!r}, ...]" if solver_label in ("SQD", "TrimSQD") else ""
        print(f"  Fragment {frag_id}: n_orb={norb}  solver={solver_label}{_tag_suffix}", end="", flush=True)

        with h5py.File(dumpfile, "r") as hf:
            frag_key = f"fragment_{int(frag_id)}"
            g = hf[frag_key]
            h1e  = g["heff"][:]
            h2e  = g["eris"][:]
            norb_ = int(g.attrs["norb"])
            nocc  = int(g.attrs["nocc"])

        # Per-fragment writable checkpoint directory.
        # os.getcwd() is '/' in Fleets (no write permission) so we must pass an
        # explicit path under /tmp/ — the same subtree the entrypoint already uses.
        frag_work_dir = data_path / f"fragment_{int(frag_id)}"
        frag_work_dir.mkdir(parents=True, exist_ok=True)

        # Look up any pre-existing QPU job ID for this fragment (resume path).
        _resume_id = (resume_job_ids or {}).get(str(frag_id)) or (resume_job_ids or {}).get(int(frag_id))

        # Build per-fragment IBM Quantum job tags.
        # Format: [system_name, fragment_id, solver, basis]
        # e.g.  ["ixazomib_ligand", "fragment_20", "trimsqd", "sto-3g"]
        #
        # IBM Quantum limits: ≤5 tags, ≤20 chars each.
        # system_name is derived from run_name by stripping the trailing "_<basis>"
        # suffix (e.g. "ixazomib_ligand_sto-3g" → "ixazomib_ligand").
        _basis_tag = (basis or "").lower() or run_name.rsplit("_", 1)[-1]
        _system_name = run_name[: -(len(_basis_tag) + 1)] if run_name.endswith("_" + _basis_tag) else run_name
        # Truncate to 20 chars to stay within IBM Quantum tag limit
        _system_name = _system_name[:20]
        _basis_tag   = _basis_tag[:20]
        _frag_tags = [_system_name, f"fragment_{int(frag_id)}", solver_label.lower(), _basis_tag]

        t0 = time.time()
        try:
            result = primary_solver.solve_from_integrals(
                h1e, h2e, norb_, nocc,
                workflow_path=str(frag_work_dir),
                compute_rdms=compute_rdms,
                job_tags=_frag_tags,
                **({"resume_job_id": _resume_id} if _resume_id and solver_label in ("SQD", "TrimSQD") else {}),
            )
        except Exception as exc:
            if _quantum_request:
                # Fail closed: no CCSD substitute.  The original exception is chained.
                print(f" [{solver_label} FAILED: {type(exc).__name__}: {exc}]", flush=True)
                raise QuantumSolverError(
                    f"fragment {frag_id}: {solver_label} was requested and failed "
                    f"({type(exc).__name__}: {exc}). CCSD is not substituted."
                ) from exc
            raise
        if _quantum_request:
            _check_quantum_identity(result, solver_label, int(frag_id))
        _protocol_rec = None
        if _pspec is not None and _quantum_request:
            _protocol_rec = _check_protocol_applied(result, int(frag_id), _pspec)

        # Planned measurement: prove the solve consumed exactly the planned measurement.
        _measurement = None
        if _mplan is not None and int(frag_id) in _mplan:
            _entry = _mplan[int(frag_id)]
            _cp = frag_work_dir / "counts.npy"
            if not _cp.exists():
                raise QuantumSolverError(
                    f"fragment {frag_id}: planned as {_entry['mode']} but no counts.npy exists after the "
                    "solve; the measurements it consumed cannot be identified.")
            _check_counts_file(_cp, norb)
            _csha = _sha256_of_file(_cp)
            if _entry["mode"] == "replay" and _csha != _entry["counts_sha256"]:
                raise QuantumSolverError(
                    f"fragment {frag_id}: the replayed counts changed during the solve "
                    "(SHA-256 differs from the plan).")
            _measurement = {"mode": _entry["mode"], "backend": _entry["backend"],
                            "n_qubits": _entry["n_qubits"], "counts_sha256": _csha,
                            "planned_counts_sha256": _entry["counts_sha256"]}

        elapsed   = time.time() - t0
        converged = result.metadata.get("converged", True)
        e_corr    = result.metadata.get("e_corr", 0.0)
        print(f"  converged={converged}  e_corr={e_corr:.6f} Ha  ({elapsed:.1f}s)")

        # Attach fragment projector if available
        with h5py.File(dumpfile, "r") as hf:
            g = hf[f"fragment_{int(frag_id)}"]
            if "c_frag" in g:
                result.metadata["c_frag"] = g["c_frag"][:]

        result.metadata["solver_label"] = solver_label
        result.metadata["nocc"] = nocc
        fragment_results[frag_id] = result
        counters[solver_label.lower()] += 1

        _job_id_file = frag_work_dir / "job_id.txt"
        fragment_summaries[int(frag_id)] = {
            "solver":             solver_label,
            "n_orb":              int(norb),
            "energy_ha":          float(result.energy),
            "e_corr_ha":          float(e_corr),
            "converged":          bool(converged),
            "qpu_job_id":         _job_id_file.read_text().strip() if _job_id_file.exists() else None,
            "counts_file":        str(frag_work_dir / "counts.npy")
                                  if (frag_work_dir / "counts.npy").exists() else None,
            # --- CCSD-diagnostic provenance (bundle API 3) ---
            # Identity of the exact embedded Hamiltonian: sha256(heff.tobytes()+eris.tobytes()).
            "hamiltonian_hash":   hashlib.sha256(h1e.tobytes() + h2e.tobytes()).hexdigest(),
            "n_occ":              int(nocc),
            # Diagnostics come from the amplitudes of the SAME CCSD solve that supplies
            # this fragment's energy/RDMs.  None for non-CCSD solvers (not applicable)
            # or when the solver did not retain amplitudes (reported as unavailable by
            # the notebook-side loader, never as zero).
            "ccsd_diagnostics":   (result.metadata.get("ccsd_diagnostics")
                                   if solver_label == "CCSD" else None),
            # --- Reproducibility provenance (Phase 1) ---
            "seed":               result.metadata.get("seed"),
            "sqd_config_hash":    result.metadata.get("sqd_config_hash"),
            "sqd_version":        result.metadata.get("sqd_version"),
            "hamiltonian_sha256": result.metadata.get("hamiltonian_sha256"),
            "measurement_sha256": result.metadata.get("measurement_sha256"),
        }
        if _measurement is not None:
            fragment_summaries[int(frag_id)]["measurement"] = _measurement
        # Quantum solvers report no convergence flag; "converged" above is a placeholder default.
        fragment_summaries[int(frag_id)]["convergence_reported"] = "converged" in result.metadata
        if _protocol_rec is not None:
            fragment_summaries[int(frag_id)]["protocol"] = _protocol_rec

        # ── TrimSQD secondary pass (mode=="both", fragment is quantum-sized) ──
        if (
            solver_mode == "both"
            and _quantum_request
            and solver_label == "SQD"         # primary SQD succeeded (else we raised above)
        ):
            print(
                f"  Fragment {frag_id}: n_orb={norb}  solver=TrimSQD (reusing QPU counts)",
                end="", flush=True,
            )
            t0 = time.time()
            try:
                # TrimSQDSolver reads counts.npy from the same checkpoint dir as SQD —
                # the file was written there by the primary SQD run above.
                result_trim = trimsqd_solver.solve_from_integrals(
                    h1e, h2e, norb_, nocc,
                    workflow_path=str(frag_work_dir),
                    compute_rdms=compute_rdms,
                )
                _check_quantum_identity(result_trim, "TrimSQD", int(frag_id))
                elapsed_trim = time.time() - t0
                e_corr_trim  = result_trim.metadata.get("e_corr", 0.0)
                print(
                    f"  converged=True  e_corr={e_corr_trim:.6f} Ha  ({elapsed_trim:.1f}s)"
                )
                result_trim.metadata["solver_label"] = "TrimSQD"
                result_trim.metadata["nocc"] = nocc
                fragment_results_trim[frag_id] = result_trim
                counters["trimsqd"] += 1
                fragment_summaries[int(frag_id)]["trimsqd_energy_ha"] = float(result_trim.energy)
                fragment_summaries[int(frag_id)]["trimsqd_e_corr_ha"] = float(e_corr_trim)
            except Exception as exc_trim:
                # solver_mode="both" promises both energies: a missing one is a failure.
                print(f" [TrimSQD secondary FAILED: {type(exc_trim).__name__}: {exc_trim}]", flush=True)
                if isinstance(exc_trim, QuantumSolverError):
                    raise
                raise QuantumSolverError(
                    f"fragment {frag_id}: solver_mode='both' requires TrimSQD after SQD, and "
                    f"TrimSQD failed ({type(exc_trim).__name__}: {exc_trim})"
                ) from exc_trim

    # ── Persist primary results ───────────────────────────────────────────────
    out_pkl = data_path / "solver_results.pkl"
    with open(out_pkl, "wb") as fh:
        pickle.dump(fragment_results, fh)
    print(f"[step3] {run_name}: wrote {out_pkl}")

    e_corr_total = sum(
        r.metadata.get("e_corr", 0.0) for r in fragment_results.values()
    )

    out = {
        "n_fragments":  n_frag,
        "n_fci":        counters["fci"],
        "n_ccsd":       counters["ccsd"],
        "n_sqd":        counters["sqd"],
        "n_trimsqd":    counters["trimsqd"],
        "e_corr_total": float(e_corr_total),
        "solver_results_pkl": str(out_pkl),
        "fragment_summaries": fragment_summaries,
        "software": _software_versions(),
        # Solver configuration actually used (recorded for cache identity / provenance).
        # Values are read from the instantiated solver objects — they reflect exactly
        # what ran, regardless of whether Stage3Config or raw kwargs were used.
        "solver_config": {
            "fci_threshold":          int(fci_threshold),
            "sqd_threshold":          int(sqd_threshold),
            "use_threshold_fallback": bool(use_threshold_fallback),
            "fci": {
                "conv_tol":  getattr(fci_solver, "conv_tol", None),
                "max_cycle": getattr(fci_solver, "max_cycle", None),
                "spin":      getattr(fci_solver, "spin", None),
            },
            "ccsd": {
                "conv_tol":    getattr(ccsd_solver, "conv_tol", None),
                "max_cycle":   getattr(ccsd_solver, "max_cycle", None),
                "diis_space":  getattr(ccsd_solver, "diis_space", None),
                "level_shift": getattr(ccsd_solver, "level_shift", None),
            },
            # Stage3Config digest — None when caller used raw kwargs.
            "stage3_config_digest": (
                stage3_config.digest()
                if hasattr(stage3_config, "digest")
                else None
            ),
        },
    }

    if _pspec is not None:
        out["quantum_protocol"] = {
            "protocol_api": 1,
            "protocol_digest": _pspec["protocol"]["protocol_digest"],
            "document_digest": _pspec["protocol"]["document_digest"],
            "applied_config_digest": _pspec["protocol"]["applied_config_digest"],
            "source_fragment_id": _pspec["protocol"]["source_fragment_id"],
            "adaptations": [],
            "applied_fragment_ids": sorted(f for f, s in fragment_summaries.items()
                                           if s.get("protocol") is not None),
        }

    # ── Persist TrimSQD secondary results (mode=="both") ──────────────────────
    if fragment_results_trim:
        trim_pkl = data_path / "solver_results_trim.pkl"
        with open(trim_pkl, "wb") as fh:
            pickle.dump(fragment_results_trim, fh)
        print(f"[step3] {run_name}: wrote {trim_pkl}")

        e_corr_trim_total = sum(
            r.metadata.get("e_corr", 0.0) for r in fragment_results_trim.values()
        )
        out["e_corr_total_trimsqd"]    = float(e_corr_trim_total)
        out["solver_results_trim_pkl"] = str(trim_pkl)

    return out
