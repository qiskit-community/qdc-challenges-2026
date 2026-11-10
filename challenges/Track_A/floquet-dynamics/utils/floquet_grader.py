"""Self-contained local grader for the QDC 2026 Floquet-dynamics challenge.

Unlike QDC 2025 (whose ``qc_grader.challenges.qdc_2025`` runs server-side), this
grader runs entirely in the participant's notebook. The default 21-qubit target
is the corrected paper-coloured exact trace in
``reference/traces_paper_colored_21q_v1.csv``; the 51-qubit target is digitized
from the paper's Fig. 3. Corrected exact 28/35-qubit traces are loaded from the
configured release paths when supplied. Missing targets remain pending in
hardware diagnostics. Historical automatic-colouring traces in ``reference/traces.csv``
remain accessible by passing ``reference_path`` explicitly.

Exercises (mirroring README §4.1) and the leaderboard metrics (§4.2):

    grade_ex1(floquet_cycle)                -- U_F circuit correctness (gate-level)
    grade_ex3(m_unmit, qpu_seconds=...)     -- unmitigated run: observe the noise wall
    grade_ex4(m_mit,   qpu_seconds=...)     -- mitigated accuracy (primary score)
    grade_submission(Submission(...))       -- full §4.2 scorecard + leaderboard number

Traces are scored against a model-matched ``M_ref(n)`` when one is available
(default 21 qubits), over ``N_c = 30`` Floquet cycles at the paper's selected
angles ``(pi/6, pi/27, pi/3)``. Closer to the reference scores higher.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ContextManager, Mapping

import numpy as np

from reference import (
    FloquetParams,
    FloquetResult,
    dominant_period,
    fit_amplitude_scaling,
    load_paper_51q,
    load_results,
    oscillation_amplitude,
)
from reference.analysis import fit_late_cycle, fit_late_cycle_anchored

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from reference import HeavyHexLattice

# --------------------------------------------------------------------------- #
# Fixed problem definition (README §4)
# --------------------------------------------------------------------------- #
FIXED_SIZE = 21                 # default graded size (smallest paper lattice)
SCALING_SIZES = (21, 28, 35)    # the paper's finite-size-scaling family (Fig. 3b/c)
N_CYCLES = 30                   # long enough for the subharmonic analysis to resolve
PARAMS = FloquetParams()        # (pi/6, pi/27, pi/3)

# tolerances (README §4.2)
TOL_PASS = 0.05                 # accuracy pass threshold (MAE)
TOL_FULL = 0.02                 # full-marks threshold (the paper's unbiased target)
TOL_PERIOD = 1.5                # cycles, subharmonic period match
TOL_AMP = 0.02                  # absolute, subharmonic amplitude match
QPU_CAP_S = 30 * 60             # 30-minute total QPU budget

_HERE = Path(__file__).parent
_TRACES_CSV = _HERE.parent / "reference" / "traces.csv"  # legacy, explicit opt-in only
_COLORED_21_CSV = _HERE.parent / "reference" / "traces_paper_colored_21q_v1.csv"
# Keep all supplied targets in one mapping. Adding the corrected 28/35q files
# at these paths enables their comparisons without using hardware archives.
REFERENCE_PATHS = {
    n: _HERE.parent / "reference" / f"traces_paper_colored_{n}q_v1.csv"
    for n in (21, 28, 35)
}
_REF: dict[int, FloquetResult] = {}

# Walkthrough points are a provisional teaching rubric, separate from the
# challenge's historical leaderboard metrics implemented below.
INSTANCE_ID = "floquet-paper-colored-2026"
MAX_POINTS = {"Part 1": 10, "Part 2": 15, "Part 3": 25, "Part 4": 25, "Part 5": 25}
# Part 5 comprises extend_sizes (15) and the separately maintained QESEM exercise (10).
# Historical hardware comparison weights are diagnostics, outside the 100-point rubric.
HARDWARE_POINTS = {"Part 4": 5, "Part 5": 25}
HARDWARE_SIZE_POINTS = {21: 5, 28: 5, 35: 5, 51: 15}
SPEC = {
    "instance_id": INSTANCE_ID,
    "backend_name": "ibm_boston",
    "sizes": tuple(HARDWARE_SIZE_POINTS),
    "n_cycles": N_CYCLES,
    "params": PARAMS,
    "factors": (1, 3, 5),
    "shots": 8192,
    "qpu_cap_seconds": QPU_CAP_S,
}


# --------------------------------------------------------------------------- #
# Reference access (provided CSVs -- no classical simulation in the grader)
# --------------------------------------------------------------------------- #

def _reference_result(n_qubits: int, reference_path: str | Path | None = None) -> FloquetResult:
    """Load a supplied corrected exact trace or the digitized 51q target.

    The legacy 21/28/35 automatic-colouring CSV is never silently used for the
    paper-coloured instance. Its historical metrics remain reproducible through
    ``reference_path``.
    """
    if reference_path is not None:
        selected = [r for r in load_results(Path(reference_path)) if r.n_qubits == n_qubits]
        if len(selected) != 1:
            raise ValueError(f"Expected one N={n_qubits} trace in {reference_path}, found {len(selected)}")
        return selected[0]
    if n_qubits in REFERENCE_PATHS:
        path = REFERENCE_PATHS[n_qubits]
        if not path.is_file():
            raise FileNotFoundError(
                f"Corrected paper-coloured exact reference for N={n_qubits} "
                f"is not yet supplied: {path}"
            )
        return _reference_result(n_qubits, path)
    if n_qubits == 51:
        if n_qubits not in _REF:
            _REF[n_qubits], _ = load_paper_51q()
        return _REF[n_qubits]
    raise ValueError(f"No paper-coloured reference for N={n_qubits}; supported sizes are 21, 28, 35, and 51")


def load_reference(n_qubits: int, reference_path: str | Path | None = None) -> FloquetResult:
    """Load a challenge target for plotting and grading, never a measurement archive."""
    return _reference_result(n_qubits, reference_path)


def _cycle_labels(cycles, n_cycles: int, *, n_points: int | None = None) -> np.ndarray:
    """Validate a complete trace's physical labels, including known cycle zero."""
    labels = np.arange(n_cycles + 1) if cycles is None else np.asarray(cycles)
    minimum_points = 1 if n_cycles == 0 else 2
    if (labels.ndim != 1 or labels.size < minimum_points or labels.dtype.kind not in "iuf"
            or not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all()):
        raise ValueError("cycles must be a one-dimensional sequence of finite integers")
    if labels[0] != 0 or np.any(np.diff(labels.astype(float)) <= 0) or labels[-1] > n_cycles:
        raise ValueError(f"cycles must start at 0, increase strictly, and end at or before {n_cycles}")
    if n_points is not None and len(labels) != n_points:
        raise ValueError(f"expected {len(labels)} points for the labelled cycles, got {n_points}")
    return labels.astype(int)


def _anchor_fit(n_qubits: int, n_cycles: int,
                reference_path: str | Path | None = None):
    """Dense Eq. 3 fit of the exact (or digitized 51q) ``M`` trace, used to seed
    sparse fits; the nearest supplied size stands in for a missing target."""
    candidates = [n_qubits] + sorted({*REFERENCE_PATHS, 51} - {n_qubits},
                                     key=lambda n: abs(n - n_qubits))
    for size in candidates:
        try:
            result = _reference_result(size, reference_path if size == n_qubits else None)
        except (FileNotFoundError, ValueError):
            continue
        fit = fit_late_cycle(result.cycles, result.magnetization, window=(6, n_cycles))
        if fit.valid:
            return fit
    return None


def _sampled_fit(labels, values, n_cycles: int, anchor):
    """Eq. 3 fit of a sparse trace: anchored + bootstrapped when a target fit exists."""
    if anchor is None:
        return fit_late_cycle(labels, values, window=(6, n_cycles))
    return fit_late_cycle_anchored(labels, values, anchor, window=(6, n_cycles))


def reference_magnetization(n_qubits: int = FIXED_SIZE,
                            n_cycles: int = N_CYCLES,
                            reference_path: str | Path | None = None,
                            *, cycles=None) -> np.ndarray:
    """Reference full-lattice ``M(n)`` at the requested physical cycle labels.

    The default 21q trace uses the corrected paper-coloured circuit. Supply
    ``reference_path`` to reproduce metrics against a historical CSV.
    """
    result = _reference_result(n_qubits, reference_path)
    labels = np.asarray(result.cycles)
    required = _cycle_labels(cycles, n_cycles)
    if (labels.ndim != 1 or labels.dtype.kind not in "iuf"
            or not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all()
            or len(np.unique(labels)) != len(labels) or not np.all(np.isin(required, labels))):
        raise ValueError(
            f"reference for N={n_qubits} does not contain every requested labelled cycle."
        )
    by_cycle = dict(zip(labels, np.asarray(result.magnetization, dtype=float)))
    return np.asarray([by_cycle[n] for n in required], dtype=float)


# --------------------------------------------------------------------------- #
# The Exercise-1 answer (kept here, not in the participant-facing helpers)
# --------------------------------------------------------------------------- #

def _ref_cycle(qc, groups, p: FloquetParams) -> None:
    """One correct Floquet cycle (paper Eq. 1, ``prod_{r=1}^{3} C_r.RZ.RX``).

    The ``RX``/``RZ`` layer is re-applied **before each** of the three ``ZZ``
    colour groups -- it does not commute with ``ZZ``, so it cannot be pulled out
    (the effective per-cycle drive is ``3 theta_x = pi/2``, which is what produces
    the period-~4 subharmonic). Collapsing it to a single ``RX``/``RZ`` layer
    followed by all bonds simulates a *different* unitary and is the classic wrong
    answer here.
    """
    for grp in groups:
        for q in range(qc.num_qubits):
            qc.rx(p.theta_x, q)
        for q in range(qc.num_qubits):
            qc.rz(p.theta_z, q)
        for (i, j) in grp:
            qc.rzz(p.theta_zz, i, j)


def _pass(msg: str) -> bool:
    print("✅ " + msg)
    return True


def _fail(msg: str) -> bool:
    print("❌ " + msg)
    return False


# --------------------------------------------------------------------------- #
# Exercise 1 -- build U_F (gate-level correctness)
# --------------------------------------------------------------------------- #

def grade_ex1(floquet_cycle) -> bool:
    """Grade the Floquet-cycle builder ``floquet_cycle(qc, groups, params)``.

    Checks gate-level equivalence (up to global phase) against the reference cycle
    on a small lattice -- so a right-behaviour-wrong-gates circuit still passes,
    but a wrong layer order / missing bond fails.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    from utils.floquet_helpers import _test_lattice
    lat = _test_lattice()
    groups = lat.color_groups()

    qc_p = QuantumCircuit(lat.n_qubits)
    try:
        floquet_cycle(qc_p, groups, PARAMS)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"grade_ex1: your floquet_cycle raised {type(exc).__name__}: {exc}")

    qc_r = QuantumCircuit(lat.n_qubits)
    _ref_cycle(qc_r, groups, PARAMS)

    if qc_p.num_qubits != qc_r.num_qubits:
        return _fail("grade_ex1: circuit acts on the wrong number of qubits.")
    if Operator(qc_p).equiv(Operator(qc_r)):
        return _pass("grade_ex1: U_F is correct (matches the reference cycle).")
    return _fail(
        "grade_ex1: U_F does not match. Check that you re-apply RX(theta_x) and "
        "RZ(theta_z) on all qubits BEFORE each of the three RZZ(theta_zz) colour "
        "groups (three sub-layers per cycle, not one) -- the angles and one RZZ "
        "per bond in every colour group."
    )


# --------------------------------------------------------------------------- #
# metrics shared by ex3/ex4/submission
# --------------------------------------------------------------------------- #

def _trace_metrics(m, n_qubits: int, *, n_cycles: int = N_CYCLES,
                   reference_path: str | Path | None = None, cycles=None) -> dict:
    """MAE, trusted depth, and subharmonic (period, amplitude) vs the reference."""
    m = np.asarray(m, dtype=float).ravel()
    labels = _cycle_labels(cycles, n_cycles, n_points=m.size)
    ref = reference_magnetization(n_qubits, n_cycles, reference_path, cycles=labels)
    sampled = not np.array_equal(labels, np.arange(n_cycles + 1))

    err = np.abs(m[1:] - ref[1:])                       # exclude known cycle zero
    finite = np.isfinite(err)
    if sampled:
        mae = float(err.mean()) if finite.all() else float("inf")
    else:
        # Keep historical dense grading reproducible; sparse acquisition must
        # never earn an accuracy pass by dropping a missing measured point.
        mae = float(err[finite].mean()) if finite.any() else float("nan")

    trusted = 0                                         # largest n* with all k<=n* in tol
    for n, e in zip(labels[1:], err):
        if np.isfinite(e) and e <= TOL_PASS:
            trusted = int(n)
        else:
            break

    metrics = {
        "mae": mae,
        "trusted_depth": trusted,
        "sampled_depth": sampled,
    }
    if sampled:
        # Sparse grids leave the free 5-parameter fit ill-conditioned; seed both
        # fits from the dense exact trace and bootstrap the uncertainties.
        anchor = _anchor_fit(n_qubits, n_cycles, reference_path)
        measured_fit = _sampled_fit(labels, m, n_cycles, anchor)
        reference_fit = _sampled_fit(labels, ref, n_cycles, anchor)
        for prefix, fit in (("", measured_fit), ("ref_", reference_fit)):
            metrics.update({
                prefix + "period": float(fit.T) if fit.valid else float("nan"),
                prefix + "amplitude": float(fit.amplitude) if fit.valid else float("nan"),
                prefix + "amplitude_err": (float(getattr(fit, "amplitude_err", np.nan))
                                           if fit.valid else float("nan")),
                prefix + "fit_valid": bool(fit.valid),
                prefix + "fit_failure_reason": fit.failure_reason,
            })
        metrics["diagnostic_method"] = ("anchored nonlinear fit" if anchor is not None
                                        else "nonlinear fit")
    else:
        metrics.update(period=dominant_period(m), amplitude=oscillation_amplitude(m),
                       ref_period=dominant_period(ref), ref_amplitude=oscillation_amplitude(ref),
                       diagnostic_method="Fourier")
    return metrics


def _diagnostic_value(metrics, key: str, decimals: int) -> str:
    """Display unavailable fitted estimates without printing numeric NaNs."""
    prefix = "ref_" if key.startswith("ref_") else ""
    if metrics["sampled_depth"] and not metrics[prefix + "fit_valid"]:
        return "unavailable"
    return f"{metrics[key]:.{decimals}f}"


# --------------------------------------------------------------------------- #
# Exercise 3 -- unmitigated (observe the noise wall)
# --------------------------------------------------------------------------- #

def grade_ex3(m_unmit, qpu_seconds: float, n_qubits: int = FIXED_SIZE,
              *, n_cycles: int = N_CYCLES,
              reference_path: str | Path | None = None, cycles=None) -> dict:
    """Record the unmitigated run and report where it breaks from the reference."""
    mm = _trace_metrics(m_unmit, n_qubits, n_cycles=n_cycles,
                        reference_path=reference_path, cycles=cycles)
    depth_label = "sampled trusted depth" if mm["sampled_depth"] else "trusted depth"
    print(f"Unmitigated: MAE = {mm['mae']:.3f}, {depth_label} n* = {mm['trusted_depth']} "
          f"/ {n_cycles}  (QPU {qpu_seconds:.0f}s)")
    if mm["trusted_depth"] >= n_cycles:
        print("   (No visible noise wall -- unusual for an unmitigated deep circuit; "
              "double-check this is really the raw hardware trace.)")
    else:
        _pass(f"grade_ex3: noise wall observed after cycle {mm['trusted_depth']}. "
              "This is what error mitigation must push back.")
    mm["qpu_seconds"] = float(qpu_seconds)
    return mm


# --------------------------------------------------------------------------- #
# Exercise 4 -- mitigated (primary accuracy score)
# --------------------------------------------------------------------------- #

def grade_ex4(m_mit, qpu_seconds: float, n_qubits: int = FIXED_SIZE,
              *, n_cycles: int = N_CYCLES,
              reference_path: str | Path | None = None, cycles=None) -> dict:
    """Primary score: mitigated-trajectory accuracy against the reference."""
    mm = _trace_metrics(m_mit, n_qubits, n_cycles=n_cycles,
                        reference_path=reference_path, cycles=cycles)
    mm["qpu_seconds"] = float(qpu_seconds)

    scored_label = "measured nonzero cycles" if mm["sampled_depth"] else f"n=1..{n_cycles}"
    depth_label = "Sampled trusted depth" if mm["sampled_depth"] else "Trusted depth n*"
    print(f"Mitigated accuracy (MAE over {scored_label}): {mm['mae']:.4f}")
    print(f"{depth_label}: {mm['trusted_depth']} / {n_cycles}")
    print(f"Subharmonic period ({mm['diagnostic_method']}): {_diagnostic_value(mm, 'period', 2)} "
          f"(reference {_diagnostic_value(mm, 'ref_period', 2)})")
    print(f"Subharmonic amp.  : {_diagnostic_value(mm, 'amplitude', 3)} "
          f"(reference {_diagnostic_value(mm, 'ref_amplitude', 3)})")
    if mm["sampled_depth"]:
        for prefix, description in (("", "measured"), ("ref_", "reference")):
            if not mm[prefix + "fit_valid"]:
                print(f"   {description} fitted estimates unavailable: {mm[prefix + 'fit_failure_reason']}")

    if mm["mae"] <= TOL_FULL:
        mm["grade"] = "full"
        _pass(f"grade_ex4: FULL MARKS -- MAE {mm['mae']:.4f} ≤ {TOL_FULL} "
              "(paper-level precision).")
    elif mm["mae"] <= TOL_PASS:
        mm["grade"] = "pass"
        _pass(f"grade_ex4: PASS -- MAE {mm['mae']:.4f} ≤ {TOL_PASS}. Tighten below "
              f"{TOL_FULL} for full marks.")
    else:
        mm["grade"] = "fail"
        _fail(f"grade_ex4: MAE {mm['mae']:.4f} > {TOL_PASS}. Improve mitigation "
              "(readout + ZNE), shot count, or transpilation.")
    return mm


# --------------------------------------------------------------------------- #
# Full submission -- leaderboard scorecard (README §4.2)
# --------------------------------------------------------------------------- #

@dataclass
class Submission:
    """What the grader consumes for the final leaderboard score.

    ``m_mit`` is the mitigated trace at ``n_qubits``. ``cycles`` optionally gives
    its physical labels, including the known initial value; omitted labels mean
    cycles 0..N_c. ``qpu_seconds``
    is the *total* Heron r3 time across the whole challenge. ``scaling`` optionally
    maps ``{n_qubits: mitigated_trace}`` for the finite-size-scaling points, with
    corresponding labels in ``scaling_cycles`` (defaulting to ``cycles``).
    """

    team: str
    m_mit: np.ndarray
    qpu_seconds: float
    n_qubits: int = FIXED_SIZE
    scaling: dict[int, np.ndarray] = field(default_factory=dict)
    # Historical automatic-colouring traces require an explicit reference path;
    # the default 21q reference is the corrected paper-coloured trace.
    n_cycles: int = N_CYCLES
    reference_path: str | Path | None = None
    cycles: np.ndarray | None = None
    scaling_cycles: dict[int, np.ndarray] = field(default_factory=dict)


def grade_submission(sub: Submission) -> dict:
    """Compute and print the §4.2 scorecard; return the metric dict."""
    m = _trace_metrics(sub.m_mit, sub.n_qubits, n_cycles=sub.n_cycles,
                       reference_path=sub.reference_path, cycles=sub.cycles)
    within_budget = sub.qpu_seconds <= QPU_CAP_S

    period_ok = abs(m["period"] - m["ref_period"]) <= TOL_PERIOD
    amp_ok = abs(m["amplitude"] - m["ref_amplitude"]) <= TOL_AMP
    subharmonic_ok = bool(period_ok and amp_ok)

    # accuracy per QPU-second (leaderboard tiebreak): reward low error & low time
    acc_per_s = (1.0 - min(m["mae"], 1.0)) / max(sub.qpu_seconds, 1.0)

    print("=" * 60)
    print(f"SUBMISSION — team {sub.team!r}  (N = {sub.n_qubits})")
    print("=" * 60)
    print(f"Accuracy (MAE)        : {m['mae']:.4f}   "
          f"[{'FULL' if m['mae'] <= TOL_FULL else 'PASS' if m['mae'] <= TOL_PASS else 'FAIL'}]")
    depth_label = "Sampled trusted depth" if m["sampled_depth"] else "Trusted depth n*"
    print(f"{depth_label}: {m['trusted_depth']} / {sub.n_cycles}")
    print(f"Subharmonic recovery ({m['diagnostic_method']}): {'✅' if subharmonic_ok else '❌'}  "
          f"(period {_diagnostic_value(m, 'period', 2)} vs {_diagnostic_value(m, 'ref_period', 2)}, "
          f"amp {_diagnostic_value(m, 'amplitude', 3)} vs {_diagnostic_value(m, 'ref_amplitude', 3)})")
    if m["sampled_depth"]:
        for prefix, description in (("", "measured"), ("ref_", "reference")):
            if not m[prefix + "fit_valid"]:
                print(f"   {description} fitted estimates unavailable: {m[prefix + 'fit_failure_reason']}")
    print(f"QPU time              : {sub.qpu_seconds/60:.2f} / {QPU_CAP_S/60:.0f} min  "
          f"{'✅ within budget' if within_budget else '❌ OVER BUDGET (penalized)'}")
    print(f"Accuracy per QPU-sec  : {acc_per_s:.2e}   (leaderboard tiebreak)")

    scaling_result = None
    scaling_failures = {}
    if sub.scaling:
        sizes, amps, errs = [], [], []
        for size in sorted(sub.scaling):
            trace = np.asarray(sub.scaling[size], float).reshape(-1)
            labels = _cycle_labels(sub.scaling_cycles.get(size, sub.cycles), sub.n_cycles,
                                   n_points=trace.size)
            error = float("nan")
            if np.array_equal(labels, np.arange(sub.n_cycles + 1)):
                amplitude = oscillation_amplitude(trace)
            else:
                fit = _sampled_fit(labels, trace, sub.n_cycles,
                                   _anchor_fit(size, sub.n_cycles))
                if not fit.valid:
                    scaling_failures[size] = fit.failure_reason
                    continue
                amplitude = fit.amplitude
                error = float(getattr(fit, "amplitude_err", np.nan))
            if not np.isfinite(amplitude):
                scaling_failures[size] = "amplitude is non-finite"
                continue
            sizes.append(size)
            amps.append(amplitude)
            errs.append(error)
        if len(sizes) >= 3:
            errs = np.array(errs, float)
            # Weight by the bootstrap amplitude errors when every size has one.
            sigma = (np.maximum(errs, 1e-4) if np.all(np.isfinite(errs)) else None)
            fit = fit_amplitude_scaling(np.array(sizes, float), np.array(amps), sigma=sigma)
            scaling_result = fit
            print(f"Finite-size scaling   : A(∞)=c={fit.c:.3f}  "
                  f"{'→ survives thermodynamic limit' if fit.survives_thermodynamic_limit else '→ vanishes'}")
        else:
            print(f"Finite-size scaling unavailable: requires at least three valid amplitudes; got {len(sizes)}")
        for size, reason in scaling_failures.items():
            print(f"   {size}q amplitude unavailable: {reason}")
    print("=" * 60)

    m.update(
        team=sub.team,
        within_budget=within_budget,
        subharmonic_ok=subharmonic_ok,
        qpu_seconds=float(sub.qpu_seconds),
        accuracy_per_qpu_second=acc_per_s,
        scaling_fit=scaling_result,
        scaling_failures=scaling_failures,
    )
    return m


# --------------------------------------------------------------------------- #
# Walkthrough reference implementations (public, for failed local checks)
# --------------------------------------------------------------------------- #

def reference_lattice_groups(n_qubits: int = FIXED_SIZE):
    """Return the published ordered colour groups for a paper lattice."""
    from reference import paper_colored_lattice

    lattice = paper_colored_lattice(n_qubits)
    return lattice, lattice.color_groups()


def reference_floquet_cycle(qc, groups, params: FloquetParams) -> None:
    """Apply one paper Floquet cycle to ``qc`` in place."""
    _ref_cycle(qc, groups, params)


def reference_logical_circuit(depth, lattice, groups, measured=True, *,
                              floquet_cycle=reference_floquet_cycle, params=PARAMS):
    """Stack the accepted cycle and optionally measure in logical qubit order."""
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(lattice.n_qubits)
    for _ in range(depth):
        floquet_cycle(circuit, groups, params)
    if measured:
        circuit.measure_all()
    return circuit


def reference_baseline_magnetizations(counts_list, lattice):
    """Convert raw counts to both traces, including the polarized cycle zero."""
    from utils import floquet_helpers as fh

    counts_list = list(counts_list)
    m = [1.0]
    mz2 = [1.0]
    for counts in counts_list:
        m.append(fh.magnetization_from_counts(counts, lattice.n_qubits))
        mz2.append(fh.magnetization_z2_from_counts(counts, lattice))
    return np.asarray(m, dtype=float), np.asarray(mz2, dtype=float)


def reference_corrected_mags(counts_list, lattice, correction=None):
    """Convert each measured circuit to ``(M, Mz2)`` after optional correction.

    ``correction(counts, index)`` returns a counts or quasi-probability mapping.
    The arrays omit cycle zero because only measured circuits are supplied.
    """
    from utils import floquet_helpers as fh

    m = []
    mz2 = []
    n = lattice.n_qubits
    for index, counts in enumerate(counts_list):
        quasi = counts if correction is None else correction(counts, index)
        clean = {
            (key.replace(" ", "").zfill(n) if isinstance(key, str) else format(int(key), f"0{n}b")): float(value)
            for key, value in quasi.items()
        }
        m.append(fh.magnetization_from_counts(clean, n))
        mz2.append(fh.magnetization_z2_from_counts(clean, lattice))
    return np.asarray(m, dtype=float), np.asarray(mz2, dtype=float)


def reference_make_readout_mitigator(
    isa_list: list[QuantumCircuit], *, use_hardware: bool,
    backend_standard: Any,
    calibration_tracker: Callable[[Any, list[int]], ContextManager[dict[str, Any]]],
) -> Any | None:
    """Calibrate mthree on the physical-qubit union for first-pass readout.

    This organizer fallback matches the participant's one-argument interface
    once the notebook binds the keyword dependencies. ``isa_list`` contains
    measured ISA circuits for ``backend_standard``. ``calibration_tracker``
    supplies settings and records the calibration job's QPU charge.

    Return the calibrated mitigator, or ``None`` for a local run or a missing
    optional mthree installation. Calibration failures propagate so an invalid
    hardware correction cannot silently receive a grade.
    """
    if not use_hardware:
        return None
    try:
        import mthree
    except ImportError as exc:
        print("Readout mitigation unavailable, using raw counts:", exc)
        return None
    mitigator = mthree.M3Mitigation(backend_standard)
    qubits = sorted({int(q) for circuit in isa_list
                     for q in circuit.layout.final_index_layout()})
    with calibration_tracker(mitigator, qubits) as settings:
        mitigator.cals_from_system(qubits=qubits, **settings)
    return mitigator


def reference_corrected_isa_mags(
    counts_list: list[Mapping[str | int, float]], isa_list: list[QuantumCircuit],
    mitigator: Any | None = None, *, lattice: HeavyHexLattice,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first-pass ``(M, Mz2)`` using each circuit's measurement layout.

    ``counts_list`` and ``isa_list`` share circuit order. When provided, mthree
    corrects each mapping in its physical-qubit order; circuits without layout
    metadata use logical indices ``0, ..., lattice.n_qubits - 1``. The canonical
    converter accepts string or integer quasi-probability keys and retains
    signed weights. Returned arrays omit the known cycle-zero value; the
    notebook prepends it only after extrapolating the measured cycles.
    """
    def correction(counts: Mapping[str | int, float], index: int) -> Mapping[str | int, float]:
        """Correct one circuit in its original physical measurement order."""
        circuit = isa_list[index]
        qubits = (circuit.layout.final_index_layout() if circuit.layout is not None
                  else list(range(lattice.n_qubits)))
        return mitigator.apply_correction(counts, qubits)

    return reference_corrected_mags(
        counts_list, lattice, None if mitigator is None else correction,
    )


def reference_folded_isa(
    depth: int, scale: int, *,
    logical_circuit: Callable[..., QuantumCircuit], lattice: HeavyHexLattice,
    groups: list[list[tuple[int, int]]],
) -> QuantumCircuit:
    """Globally fold an unmeasured depth circuit, then measure logical qubits.

    This organizer fallback uses the notebook's active ``logical_circuit``
    callback and fixed lattice/colour groups, so preceding exercise choices are
    retained. An odd positive ``scale`` repeats circuit/inverse pairs without
    changing the ideal unitary. Measurements are appended after folding in
    logical order. Despite the historical function name, the returned circuit
    is logical; the hardware run cell performs ISA transpilation afterward.
    Invalid folding factors raise through ``fold_circuit_global``.
    """
    from utils import floquet_helpers as fh

    logical = logical_circuit(depth, lattice, groups, measured=False)
    folded = fh.fold_circuit_global(logical, scale)
    folded.measure_all()
    return folded


def reference_zne_extrapolate(per_scale, scales=(1, 3, 5)):
    """Return the per-cycle intercept of a linear fit versus noise factor."""
    scales = tuple(scales)
    if len(scales) < 2 or len(set(scales)) != len(scales):
        raise ValueError("Provide at least two distinct noise factors")
    if any(scale not in per_scale for scale in scales):
        raise ValueError("A chosen noise factor is missing")
    rows = [np.asarray(per_scale[scale], dtype=float).reshape(-1) for scale in scales]
    if len({row.size for row in rows}) != 1 or not np.isfinite(rows).all():
        raise ValueError("Noise-factor measurements must be finite and equally sized")
    return np.asarray(np.polyfit(np.asarray(scales, dtype=float), np.stack(rows), 1)[1], dtype=float)


def reference_normalize_inverse_rzz(qc):
    """Replace negative-angle RZZ by an equivalent positive-angle RZZ."""
    normalized = qc.copy_empty_like()
    for inst in qc.data:
        if inst.operation.name == "rzz" and float(inst.operation.params[0]) < 0:
            a, b = (qc.find_bit(qubit).index for qubit in inst.qubits)
            normalized.x(a)
            normalized.barrier(a, b)
            normalized.rzz(-float(inst.operation.params[0]), a, b)
            normalized.barrier(a, b)
            normalized.x(a)
            normalized.barrier(a, b)
        else:
            normalized.append(inst)
    return normalized


def reference_choose_layout(backend, lattice, groups, limit=10, missing_error=0.01):
    """Choose the cheapest of the first ``limit`` SWAP-free RZZ embeddings."""
    import rustworkx as rx

    if backend is None or "rzz" not in backend.target.operation_names:
        raise RuntimeError("Select a fractional-RZZ backend before choosing a layout")
    if limit < 1:
        raise ValueError("limit must be positive")
    target = backend.target
    edges = {tuple(sorted(pair)) for pair in target["rzz"] if pair is not None and len(pair) == 2}
    if not edges:
        raise ValueError("RZZ target has no calibrated edges")
    bonds = [tuple(edge) for group in groups for edge in group]
    device = rx.PyGraph()
    device.add_nodes_from(range(backend.num_qubits))
    device.add_edges_from_no_data(sorted(edges))
    logical = rx.PyGraph()
    logical.add_nodes_from(range(lattice.n_qubits))
    logical.add_edges_from_no_data(bonds)

    def error(gate, qargs):
        props = target[gate].get(tuple(qargs))
        if props is None and len(qargs) == 2:
            props = target[gate].get(tuple(reversed(qargs)))
        value = None if props is None else props.error
        return missing_error if value is None else float(value)

    best, best_cost = None, float("inf")
    for seen, mapping in enumerate(rx.vf2_mapping(device, logical, subgraph=True, induced=False, id_order=False), 1):
        inverse = {logical_q: physical_q for physical_q, logical_q in mapping.items()}
        layout = [inverse[q] for q in range(lattice.n_qubits)]
        cost = sum(error("rzz", (layout[i], layout[j])) for i, j in bonds)
        cost += sum(error("measure", (physical,)) for physical in layout) / N_CYCLES
        if cost < best_cost:
            best, best_cost = layout, cost
        if seen >= limit:
            break
    if best is None:
        raise ValueError(f"No SWAP-free {lattice.n_qubits}q embedding found")
    return best


def reference_select_mapomatic_layout(qc, backend):
    """Return ``(layout, estimated_error)`` for Mapomatic's best placement."""
    import mapomatic as mm

    layouts = mm.matching_layouts(qc, backend, strict_direction=False)
    scores = mm.evaluate_layouts(qc, layouts, backend)
    if not scores:
        raise ValueError("Mapomatic found no compatible layout")
    layout, estimated_error = min(scores, key=lambda item: float(item[1]))
    return list(layout), float(estimated_error)


def reference_build_estimator(backend, allowance=600, *, shots=8192,
                              randomizations=128, cal_randomizations=256):
    """Create the paper workflow's TREX Estimator; construction submits no job."""
    from qiskit_ibm_runtime.executor_estimator import Estimator

    if backend is None or "rzz" not in backend.target.operation_names:
        raise RuntimeError("Executor Estimator requires a fractional-RZZ backend")
    estimator = Estimator(mode=backend)
    options = estimator.options
    options.max_execution_time = allowance
    options.resilience_level = 0
    options.resilience.zne_mitigation = False
    options.default_shots = shots
    options.resilience.measure_mitigation = True
    options.resilience.measure_noise_learning.num_randomizations = cal_randomizations
    options.twirling.enable_measure = True
    options.twirling.enable_gates = False
    options.twirling.strategy = "active-accum"
    options.twirling.num_randomizations = randomizations
    options.twirling.shots_per_randomization = "auto"
    options.dynamical_decoupling.enable = False
    return estimator


def reference_make_observables(lattice):
    """Return ``[M, Mz2]`` as logical SparsePauliOp objects."""
    from qiskit.quantum_info import SparsePauliOp

    n = lattice.n_qubits
    z2 = [q for q, degree in enumerate(lattice.degrees()) if degree == 2]
    if not z2:
        raise ValueError("No degree-two sites")
    return [
        SparsePauliOp.from_sparse_list([("Z", [q], 1 / n) for q in range(n)], num_qubits=n),
        SparsePauliOp.from_sparse_list([("Z", [q], 1 / len(z2)) for q in z2], num_qubits=n),
    ]


def reference_fold_cycles(depth, lattice, groups, factor):
    """Fold one logical Floquet cycle before composing ``depth`` copies."""
    from qiskit import QuantumCircuit
    from utils import floquet_helpers as fh

    one = QuantumCircuit(lattice.n_qubits)
    reference_floquet_cycle(one, groups, PARAMS)
    folded_one = fh.fold_circuit_global(one, factor)
    circuit = QuantumCircuit(lattice.n_qubits)
    for _ in range(depth):
        circuit.compose(folded_one, inplace=True)
    return circuit


def _reference_zne_data(
    run: Mapping[str, Any], chosen_factors: Sequence[int], observable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retain the grader's private data-selection API using the shared fitter."""
    from utils.floquet_zne import _zne_data

    return _zne_data(run, chosen_factors, observable)


def _reference_qmit_zne(
    run: Mapping[str, Any], chosen_factors: Sequence[int], observable: str,
    model: str, custom_fit_factory: Callable[..., Any] | None = None,
) -> np.ndarray:
    """Organizer fallback for the participant's common Qiskit Mitigation fitter."""
    from utils.floquet_zne import _zne_data
    from qiskit_mitigation.zne.extrapolation import _fit_extrapolation_models

    factors, values, stds = _zne_data(run, tuple(chosen_factors), observable)
    result = np.empty(len(values), dtype=float)
    for index, observed in enumerate(values):
        custom_fit = (None if custom_fit_factory is None
                      else custom_fit_factory(factors, observed))
        fitted, _ = _fit_extrapolation_models(
            observed, stds[index], factors, models=[model],
            extrapolated_noise_factor=0.0, custom_fit=custom_fit,
        )
        result[index] = fitted[0, 0]
    failed = np.flatnonzero(~np.isfinite(result))
    if failed.size:
        cycles = np.asarray(run["cycles"])[failed].tolist()
        raise RuntimeError(f"{model} fit failed for cycles {cycles}")
    return result


def reference_zne_linear_extrapolator(run, chosen_factors=(1, 3, 5), observable="M"):
    return _reference_qmit_zne(run, chosen_factors, observable, model="linear")


def reference_zne_richardson_extrapolator(run, chosen_factors=(1, 3, 5), observable="M"):
    degree = len(tuple(chosen_factors)) - 1
    if degree > 7:
        raise ValueError("Qiskit Mitigation supports polynomial degrees up to 7")
    return _reference_qmit_zne(run, chosen_factors, observable, model=f"polynomial_degree_{degree}")


def reference_zne_exponential_extrapolator(
    run, chosen_factors=(1, 3, 5), observable="M", rate_bounds=(0.0, 2.0),
):
    lower, upper = map(float, rate_bounds)
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 <= lower < upper):
        raise ValueError("rate_bounds must satisfy 0 <= lower < upper")

    def fit_spec(factors, observed):
        def exponential(factor, amplitude, rate):
            return amplitude * np.exp(-rate * factor)

        rate0 = lower + 0.1 * (upper - lower)
        j = int(np.argmax(np.abs(observed)))
        amplitude0 = float(observed[j] * np.exp(rate0 * factors[j]))
        return exponential, [amplitude0, rate0], ([-np.inf, lower], [np.inf, upper])

    return _reference_qmit_zne(run, chosen_factors, observable, model="custom", custom_fit_factory=fit_spec)


def align_zne_reference(run, reference_trace) -> np.ndarray:
    """Align a full labelled or dense reference with measured nonzero run cycles.

    Legacy arrays already restricted to run cycles, optionally prepended with
    cycle zero, remain supported. Full arrays are indexed by physical labels.
    """
    raw_cycles = np.asarray(run["cycles"])
    if (raw_cycles.ndim != 1 or raw_cycles.size == 0 or raw_cycles.dtype.kind not in "iuf"
            or not np.isfinite(raw_cycles).all()
            or not np.equal(raw_cycles, np.floor(raw_cycles)).all()
            or raw_cycles[0] <= 0 or np.any(np.diff(raw_cycles.astype(float)) <= 0)):
        raise ValueError("run cycles must be strictly increasing positive integers")
    cycles = raw_cycles.astype(int)
    if hasattr(reference_trace, "cycles"):
        labels = np.asarray(reference_trace.cycles)
        values = _magnetization_trace(reference_trace)
    elif isinstance(reference_trace, Mapping) and "cycles" in reference_trace:
        labels = np.asarray(reference_trace["cycles"])
        values = _magnetization_trace(reference_trace)
    else:
        reference = np.asarray(reference_trace, dtype=float).reshape(-1)
        if reference.size > cycles[-1]:
            return reference[cycles]
        if reference.size == cycles.size + 1:
            return reference[1:]
        if reference.size == cycles.size:
            return reference
        raise ValueError("Reference and run have different cycle counts")
    if (labels.ndim != 1 or len(labels) != len(values) or labels.dtype.kind not in "iuf"
            or not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all()
            or len(np.unique(labels)) != len(labels) or not np.isin(cycles, labels).all()):
        raise ValueError("Reference does not contain every measured cycle")
    by_cycle = dict(zip(labels, values))
    return np.asarray([by_cycle[cycle] for cycle in cycles], dtype=float)


def reference_select_zne_model(run21, reference_trace):
    """Pick the source notebook's lowest 21q M-trace RMSE method and factors."""
    reference = align_zne_reference(run21, reference_trace)
    methods = {
        "linear": reference_zne_linear_extrapolator,
        "richardson": reference_zne_richardson_extrapolator,
        "exp": reference_zne_exponential_extrapolator,
    }
    best, best_cost = None, float("inf")
    for factors in ((1, 3), (1, 3, 5)):
        for name, extrapolator in methods.items():
            predicted = extrapolator(run21, chosen_factors=factors)
            cost = float(np.sqrt(np.sum((predicted - reference) ** 2)))
            if cost < best_cost:
                best, best_cost = (factors, name), cost
    return best


def reference_extend_sizes(
    lattice, run21, *, cycles, selected_factors, use_hardware, backend,
    backend_name, ledger_path, saved_run_settings, mapomatic_layout,
    measure_cycle_folded, qpu_time_spent,
):
    """Carry the accepted measurement pipeline to the three larger paper patches.

    All execution callbacks are supplied by the notebook, retaining its active
    ledger, backend and hardware guards. Importing this reference submits no jobs.
    """
    from utils import floquet_helpers as fh

    lattices = {21: lattice, **{size: fh.paper_colored_lattice(size) for size in (28, 35, 51)}}
    groups = {size: fh.color_groups(lat) for size, lat in lattices.items()}
    runs = {21: run21} if run21 is not None else {}
    layouts = {21: list(run21["layout"])} if run21 is not None else {}
    if use_hardware:
        for size in (28, 35, 51):
            previous = saved_run_settings(
                f"trex-{size}q", backend_name=backend_name, ledger_path=ledger_path,
            )
            layout = (list(previous["layout"]) if previous is not None
                      else mapomatic_layout(lattices[size], groups[size], backend))
            runs[size] = measure_cycle_folded(
                lattices[size], groups[size], backend, layout,
                cycles=cycles, factors=selected_factors,
            )
            layouts[size] = layout
    return {
        "lattices_by_size": lattices, "groups_by_size": groups,
        "runs_by_size": runs, "layout_by_size": layouts,
        "qpu_seconds": qpu_time_spent(ledger_path) if use_hardware else 0.0,
    }


# --------------------------------------------------------------------------- #
# Deterministic local walkthrough checks (draft points, no QPU calls)
# --------------------------------------------------------------------------- #

def _finish_check(name: str, earned: float, maximum: float, detail: str = "") -> float:
    suffix = f" — {detail}" if detail else ""
    print(f"{name}: {earned:g}/{maximum}{suffix}")
    return float(earned)


def _check(name: str, maximum: float, verify) -> float:
    try:
        detail = verify() or ""
    except Exception as exc:  # noqa: BLE001 - a bad answer should not stop the walkthrough
        return _finish_check(name, 0, maximum, f"{type(exc).__name__}: {exc}")
    return _finish_check(name, maximum, maximum, detail)


def _assert_array(actual, expected, message, *, atol=1e-8):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{message}: expected shape {expected.shape}, got {actual.shape}"
    assert np.allclose(actual, expected, rtol=0, atol=atol), message


def check_lattice_groups(lattice, groups) -> float:
    """Check the 21q ordered published lattice and all three colour layers."""
    def verify():
        expected, expected_groups = reference_lattice_groups(21)
        assert lattice.n_qubits == expected.n_qubits, "wrong number of qubits"
        lattice.validate()
        assert {tuple(sorted(edge)) for edge in lattice.edges} == {
            tuple(sorted(edge)) for edge in expected.edges
        }, "bonds differ from the paper graph"
        assert len(groups) == len(expected_groups) == 3, "three ordered colour groups are required"
        for got, wanted in zip(groups, expected_groups):
            assert {tuple(sorted(edge)) for edge in got} == {
                tuple(sorted(edge)) for edge in wanted
            }, "colour assignment or group order differs"
        assert lattice.color_groups() == expected_groups, "lattice edge colours differ"
        return f"{lattice.n_qubits} qubits, {lattice.n_edges} bonds"

    return _check("lattice_groups", 5, verify)


def check_floquet_cycle(fn) -> float:
    """Check one-cycle gate equivalence on a three-colour tiny lattice."""
    def verify():
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        from utils.floquet_helpers import _test_lattice

        lattice = _test_lattice()
        groups = lattice.color_groups()
        actual = QuantumCircuit(lattice.n_qubits)
        expected = QuantumCircuit(lattice.n_qubits)
        fn(actual, groups, PARAMS)
        reference_floquet_cycle(expected, groups, PARAMS)
        assert Operator(actual).equiv(Operator(expected)), "Floquet unitary differs"
        return "paper layer order and angles match"

    return _check("floquet_cycle", 5, verify)


def check_logical_circuit(fn) -> float:
    """Check cycle stacking and optional logical-order measurements locally."""
    def verify():
        from qiskit.quantum_info import Operator
        from utils.floquet_helpers import _test_lattice

        lattice = _test_lattice()
        groups = lattice.color_groups()
        for depth in (0, 1, 2):
            actual = fn(depth, lattice, groups, measured=False)
            expected = reference_logical_circuit(depth, lattice, groups, measured=False)
            assert actual.num_qubits == lattice.n_qubits, "wrong circuit size"
            assert actual.num_clbits == 0, "unmeasured circuit has classical bits"
            assert Operator(actual).equiv(Operator(expected)), "cycle stacking changed the unitary"
            measured = fn(depth, lattice, groups)
            pairs = [(measured.find_bit(inst.qubits[0]).index,
                      measured.find_bit(inst.clbits[0]).index)
                     for inst in measured.data if inst.operation.name == "measure"]
            assert pairs == [(q, q) for q in range(lattice.n_qubits)], "measurement mapping"
            assert Operator(measured.remove_final_measurements(inplace=False)).equiv(
                Operator(expected)
            ), "measured circuit changed the drive"
        return "stacked cycles and optional measurements match"

    return _check("logical_circuit", 2.5, verify)


def check_baseline_magnetization(fn) -> float:
    """Check both count-derived observables and the initial cycle-zero value."""
    def verify():
        from utils.floquet_helpers import _test_lattice

        lattice = _test_lattice()
        counts = [
            {"0000000": 6, "0000010": 2},
            {"0000000": 3, "0001000": 2, "1000000": 3},
        ]
        expected = reference_baseline_magnetizations(counts, lattice)
        actual = fn(counts, lattice)
        assert len(actual) == 2, "return (M, Mz2)"
        for got, wanted, label in zip(actual, expected, ("M", "Mz2")):
            _assert_array(got, wanted, label)
        return "both observables include cycle zero"

    return _check("baseline_magnetization", 2.5, verify)


def check_first_mitigation(corrected_mags, zne_extrapolate) -> float:
    """Award five points each for count correction and linear ZNE."""
    from utils.floquet_helpers import _test_lattice

    earned = 0
    failed = []
    try:
        lattice = _test_lattice()
        counts = [{"0000000": 7, "0000010": 1}, {"0000000": 4, "1000000": 4}]

        def correction(_counts, index):
            return {0: 0.75 - 0.1 * index, (1 << lattice.n_qubits) - 1: 0.25 + 0.1 * index}

        for remedy in (None, correction):
            actual = corrected_mags(counts, lattice, remedy)
            expected = reference_corrected_mags(counts, lattice, remedy)
            assert len(actual) == 2, "return (M, Mz2)"
            for got, wanted in zip(actual, expected):
                _assert_array(got, wanted, "readout-corrected magnetization")
        earned += 5
    except Exception as exc:  # noqa: BLE001
        failed.append(f"correction: {type(exc).__name__}: {exc}")
    try:
        intercept = np.array([0.3, -0.2, 0.55])
        slope = np.array([-0.02, 0.03, -0.04])
        per_scale = {factor: intercept + factor * slope for factor in (1, 3, 5)}
        for scales in ((1, 3), (1, 3, 5)):
            _assert_array(zne_extrapolate(per_scale, scales), intercept, "linear ZNE intercept")
        earned += 5
    except Exception as exc:  # noqa: BLE001
        failed.append(f"ZNE: {type(exc).__name__}: {exc}")
    return _finish_check("first_mitigation", earned, 10, "; ".join(failed))


@contextmanager
def _first_mitigation_candidate(
    fn: Callable[..., Any], replacements: Mapping[str, Any],
) -> Iterator[Callable[..., Any]]:
    """Apply a local fixture to an exercise function or bound fallback.

    Exercise definitions resolve their notebook globals, which are patched and
    restored for the check. Organizer fallbacks have explicit keyword
    dependencies bound with ``partial``; construct a temporary partial with
    fixture keywords instead of altering the grader's module globals. The
    calibration callback's notebook name maps to the reference adapter's
    ``calibration_tracker`` argument. This also makes rerunning the check cell
    after fallback installation safe.
    """
    from functools import partial
    from inspect import signature
    from unittest.mock import patch

    if isinstance(fn, partial):
        parameters = signature(fn.func).parameters
        keywords = dict(fn.keywords)
        for name, value in replacements.items():
            if name == "tracked_readout_calibration" and "calibration_tracker" in parameters:
                name = "calibration_tracker"
            if name in parameters:
                keywords[name] = value
        yield partial(fn.func, *fn.args, **keywords)
    else:
        with patch.dict(fn.__globals__, replacements):
            yield fn


def check_readout_setup(fn: Callable[..., Any]) -> str:
    """Verify the participant's mthree setup with mocked calibration only.

    ``fn`` is the notebook-defined ``make_readout_mitigator`` or its bound
    organizer fallback. Exercise globals or fallback keywords receive a backend,
    hardware flag and tracking callback; mthree is mocked in ``sys.modules``.
    Check the sorted physical-qubit union, calibration settings, returned
    mitigator, local/missing-library fallbacks, and visible calibration failures.
    Return a success description;
    assertions or candidate exceptions propagate to the notebook's scoring
    loop. No calibration or QPU submission occurs.
    """
    import sys
    from contextlib import nullcontext
    from unittest.mock import Mock, patch

    layouts = [[2, 5, 8], [5, 1, 9]]
    circuits = [SimpleNamespace(layout=SimpleNamespace(final_index_layout=lambda q=q: q))
                for q in layouts]
    expected = Mock()
    fake_backend = object()
    fake_mthree = SimpleNamespace(M3Mitigation=Mock(return_value=expected))
    settings = {"shots": 128, "method": "balanced", "async_cal": False}
    tracker = Mock(return_value=nullcontext(settings))
    replacements = {"use_hardware": True, "backend_standard": fake_backend,
                    "tracked_readout_calibration": tracker}
    with _first_mitigation_candidate(fn, replacements) as candidate, \
            patch.dict(sys.modules, {"mthree": fake_mthree}):
        assert candidate(circuits) is expected, "return the calibrated mitigator"
        fake_mthree.M3Mitigation.assert_called_once_with(fake_backend)
        tracker.assert_called_once_with(expected, [1, 2, 5, 8, 9])
        expected.cals_from_system.assert_called_once_with(qubits=[1, 2, 5, 8, 9], **settings)
        expected.cals_from_system.side_effect = RuntimeError("calibration failed")
        try:
            candidate(circuits)
        except RuntimeError:
            pass
        else:
            raise AssertionError("hardware calibration failures must remain visible")
    for hardware in (False, True):
        with _first_mitigation_candidate(fn, {"use_hardware": hardware}) as candidate, \
                patch.dict(sys.modules, {"mthree": None}), patch("builtins.print"):
            assert candidate(circuits) is None, "local/missing-mthree fallback"
    return "backend, physical-qubit union, calibration, and fallbacks passed"


def check_corrected_observables(fn: Callable[..., Any]) -> str:
    """Verify both count-derived observables and physical readout ordering.

    ``fn`` is the notebook-defined ``corrected_mags`` or its bound organizer
    fallback. Its lattice global or bound keyword receives a seven-qubit fixture.
    Compare raw counts and signed integer/string quasi-probabilities with the
    organizer converter, and check layout-free circuits use logical qubit order. Return
    the success description or raise to the notebook's scoring loop. mthree's
    correction is mocked, so this check uses no hardware.
    """
    from unittest.mock import Mock, call
    from utils import floquet_helpers as fh

    fixture = fh._test_lattice()
    n = fixture.n_qubits
    physical = list(range(n + 2, 2, -1))
    circuits = [SimpleNamespace(layout=SimpleNamespace(final_index_layout=lambda: physical)),
                SimpleNamespace(layout=None)]
    counts = [{"0" * n: 7, "0" * (n - 2) + "10": 1},
              {"0" * n: 4, "1" + "0" * (n - 1): 4}]
    quasi = [{0: 1.1, (1 << n) - 1: -0.1}, {"0" * n: 0.6, "1" * n: 0.4}]
    mitigator = Mock(apply_correction=Mock(side_effect=quasi))
    with _first_mitigation_candidate(fn, {"lattice": fixture}) as candidate:
        for remedy in (None, mitigator):
            expected = reference_corrected_mags(
                counts, fixture, None if remedy is None else lambda cnts, index: quasi[index],
            )
            actual = candidate(counts, circuits, remedy)
            assert len(actual) == 2, "return (M, Mz2)"
            for got, wanted in zip(actual, expected):
                np.testing.assert_allclose(got, wanted, rtol=0, atol=1e-8)
    assert mitigator.apply_correction.call_args_list == [
        call(counts[0], physical), call(counts[1], list(range(n))),
    ], "apply correction in each circuit's measurement order"
    return "raw/signed quasi counts, both observables, and layouts passed"


def check_global_folds(
    fn: Callable[..., QuantumCircuit], *,
    logical_circuit: Callable[..., QuantumCircuit], factors: Sequence[int] = (1, 3, 5),
) -> str:
    """Verify whole-circuit folds preserve the unitary and measurement order.

    ``fn`` is the participant's ``folded_isa`` or its bound organizer fallback.
    Temporarily replace its lattice and groups with a small fixture, and use the
    supplied ``logical_circuit`` callback for the ideal depth-one/depth-two circuits.
    For every supplied factor, check the scaled RZZ count and final logical
    measurements as well as the ideal unitary. Return the success description;
    failures propagate to the notebook's scoring loop. Only local operators are
    evaluated, with no transpilation or hardware jobs.
    """
    from qiskit.quantum_info import Operator
    from utils import floquet_helpers as fh

    fixture = fh._test_lattice()
    fixture_groups = fh.color_groups(fixture)
    replacements = {"lattice": fixture, "groups": fixture_groups}
    with _first_mitigation_candidate(fn, replacements) as candidate:
        for depth in (1, 2):
            ideal = Operator(logical_circuit(depth, fixture, fixture_groups, measured=False))
            for scale in factors:
                folded = candidate(depth, scale)
                assert Operator(folded.remove_final_measurements(inplace=False)).equiv(ideal), \
                    "global folds must preserve the ideal unitary"
                assert folded.count_ops().get("rzz", 0) == scale * depth * fixture.n_edges, \
                    "scale the full logical circuit's RZZ count"
                assert folded.count_ops().get("measure", 0) == fixture.n_qubits
                measured = [(folded.find_bit(inst.qubits[0]).index,
                             folded.find_bit(inst.clbits[0]).index)
                            for inst in folded.data[-fixture.n_qubits:]
                            if inst.operation.name == "measure"]
                assert measured == [(q, q) for q in range(fixture.n_qubits)], \
                    "append measurements in logical-qubit order"
    return "factors 1, 3, 5: unitary, RZZ counts, and measurements passed"


def check_linear_intercepts(
    fn: Callable[..., np.ndarray], *, factors: Sequence[int] = (1, 3, 5),
) -> str:
    """Verify scale-zero intercepts for synthetic first-pass magnetizations.

    ``fn`` is the participant's ``zne_extrapolate`` function. Construct three
    cycle values with known intercepts and slopes at ``factors``, then check
    explicit two-factor/all-factor calls and the function's default factors.
    Return a success description or raise to the notebook's scoring loop. This
    deterministic numeric fixture submits no jobs and changes no globals.
    """
    factors = tuple(factors)
    intercept = np.array([0.3, -0.2, 0.55])
    slope = np.array([-0.02, 0.03, -0.04])
    per_scale = {scale: intercept + scale * slope for scale in factors}
    for scales in (factors[:2], factors):
        np.testing.assert_allclose(fn(per_scale, scales), intercept, rtol=0, atol=1e-8)
    np.testing.assert_allclose(fn(per_scale), intercept, rtol=0, atol=1e-8)
    return "per-cycle zero-noise intercepts passed"


def check_normalize_inverse_rzz(fn) -> float:
    """Check unitary preservation and positive RZZ angles without extra entanglers."""
    def verify():
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator

        circuit = QuantumCircuit(3)
        circuit.rx(0.2, 0)
        circuit.rzz(-PARAMS.theta_zz, 0, 1)
        circuit.rz(-0.1, 2)
        circuit.rzz(PARAMS.theta_zz, 1, 2)
        circuit.rzz(-0.21, 0, 2)
        actual = fn(circuit)
        assert Operator(actual).equiv(Operator(circuit)), "inverse normalization changed the unitary"
        entanglers = [inst.operation for inst in actual.data
                      if inst.operation.num_qubits == 2 and inst.operation.name != "barrier"]
        assert len(entanglers) == 3 and all(inst.name == "rzz" for inst in entanglers), "extra entanglers"
        assert all(float(inst.params[0]) > 0 for inst in entanglers), "negative-angle RZZ remains"
        return "positive-angle RZZs preserve the unitary"

    return _check("normalize_inverse_rzz", 5, verify)


def check_choose_layout(fn) -> float:
    """Check a SWAP-free placement on a deterministic fractional-RZZ target."""
    def verify():
        from qiskit.providers.fake_provider import GenericBackendV2
        from utils.floquet_helpers import _test_lattice

        lattice = _test_lattice()
        coupling = [list(edge) for a, b in lattice.edges for edge in ((a, b), (b, a))]
        backend = GenericBackendV2(
            lattice.n_qubits, basis_gates=["rz", "sx", "x", "rzz"],
            coupling_map=coupling, noise_info=False, seed=2026,
        )
        layout = list(fn(backend, lattice, lattice.color_groups(), limit=10))
        assert len(layout) == lattice.n_qubits, "layout has the wrong length"
        assert len(set(layout)) == len(layout), "layout repeats a physical qubit"
        assert all(isinstance(q, (int, np.integer)) and 0 <= q < backend.num_qubits for q in layout), "invalid qubit"
        edges = {tuple(sorted(pair)) for pair in backend.target["rzz"] if pair is not None}
        assert all(tuple(sorted((layout[i], layout[j]))) in edges for i, j in lattice.edges), "routing is required"
        return "every logical bond has native RZZ"

    return _check("choose_layout", 5, verify)


def check_mapomatic_layout(fn) -> float:
    """Check the lowest-error Mapomatic ``(layout, estimated_error)`` pair."""
    def verify():
        from unittest.mock import patch
        import mapomatic as mm
        from qiskit import QuantumCircuit
        from qiskit.providers.fake_provider import GenericBackendV2

        circuit = QuantumCircuit(3)
        circuit.rzz(0.2, 0, 1)
        circuit.rzz(0.2, 1, 2)
        coupling = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]
        backend = GenericBackendV2(4, basis_gates=["rz", "sx", "x", "rzz"],
                                   coupling_map=coupling, noise_info=False, seed=2026)
        layouts = [[0, 1, 2], [1, 2, 3]]
        ranked = [(layouts[0], 0.24), (layouts[1], 0.08)]
        with patch.object(mm, "matching_layouts", return_value=layouts) as matches, \
             patch.object(mm, "evaluate_layouts", return_value=ranked) as evaluates:
            actual = fn(circuit, backend)
        assert isinstance(actual, (tuple, list)) and len(actual) == 2, "return (layout, estimated_error)"
        layout, estimated_error = actual
        assert list(layout) == layouts[1], "did not choose the lowest estimated-error layout"
        assert np.isfinite(float(estimated_error)) and np.isclose(float(estimated_error), 0.08), \
            "estimated error does not match the chosen layout"
        assert matches.called and evaluates.called, "Mapomatic ranking was not used"
        return "lowest estimated-error layout and score selected"

    return _check("mapomatic_layout", 5, verify)


def check_build_estimator(fn, *, shots=8192, randomizations=128,
                          cal_randomizations=256) -> float:
    """Check TREX and twirling options without creating a live Runtime client."""
    def verify():
        from unittest.mock import patch

        class FakeEstimator:
            def __init__(self, mode):
                self.mode = mode
                self.options = SimpleNamespace(
                    max_execution_time=None, resilience_level=None, default_shots=None,
                    resilience=SimpleNamespace(
                        zne_mitigation=None, measure_mitigation=None,
                        measure_noise_learning=SimpleNamespace(num_randomizations=None),
                    ),
                    twirling=SimpleNamespace(
                        enable_measure=None, enable_gates=None, strategy=None,
                        num_randomizations=None, shots_per_randomization=None,
                    ),
                    dynamical_decoupling=SimpleNamespace(enable=None),
                )

        backend = SimpleNamespace(name="fixture_backend", target=SimpleNamespace(operation_names={"rzz"}))
        # functools.partial binds the configured organizer fallback without
        # hiding its globals from the no-QPU fixture.
        base_fn = fn.func if hasattr(fn, "func") else fn
        namespace = base_fn.__globals__
        sentinel = object()
        previous = namespace.get("Estimator", sentinel)
        namespace["Estimator"] = FakeEstimator
        try:
            with patch("qiskit_ibm_runtime.executor_estimator.Estimator", FakeEstimator):
                estimator = fn(backend, allowance=321)
        finally:
            if previous is sentinel:
                del namespace["Estimator"]
            else:
                namespace["Estimator"] = previous
        assert isinstance(estimator, FakeEstimator) and estimator.mode is backend, "wrong Estimator mode"
        options = estimator.options
        assert options.max_execution_time == 321 and options.default_shots == shots, "allowance or shots"
        assert options.resilience_level == 0, "resilience level"
        assert options.resilience.measure_mitigation is True, "TREX is disabled"
        assert options.resilience.zne_mitigation is False, "built-in ZNE should be disabled"
        assert options.resilience.measure_noise_learning.num_randomizations == cal_randomizations, "TREX calibration"
        assert options.twirling.enable_measure is True and options.twirling.enable_gates is False, "twirling switches"
        assert options.twirling.num_randomizations == randomizations, "twirling randomizations"
        assert options.twirling.strategy == "active-accum", "twirling strategy"
        assert options.twirling.shots_per_randomization == "auto", "twirling shots"
        assert options.dynamical_decoupling.enable is False, "dynamical decoupling"
        return "TREX options match the source workflow"

    return _check("build_estimator", 5, verify)


def check_make_observables(fn) -> float:
    """Check both normalized logical magnetization operators."""
    def verify():
        from utils.floquet_helpers import _test_lattice

        lattice = _test_lattice()
        actual = fn(lattice)
        expected = reference_make_observables(lattice)
        assert len(actual) == 2, "return [M, Mz2]"
        assert all(got.num_qubits == lattice.n_qubits and got.equiv(wanted)
                   for got, wanted in zip(actual, expected)), "operator coefficients or sites differ"
        return "M and degree-two Mz2 have normalized weights"

    return _check("make_observables", 5, verify)


def check_fold_cycles(fn) -> float:
    """Check factors 1/3/5 at depths 1/2 and exact entangler scaling."""
    def verify():
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        from utils.floquet_helpers import _test_lattice

        lattice = _test_lattice()
        groups = lattice.color_groups()
        for depth in (1, 2):
            ideal = QuantumCircuit(lattice.n_qubits)
            for _ in range(depth):
                reference_floquet_cycle(ideal, groups, PARAMS)
            for factor in (1, 3, 5):
                folded = fn(depth, lattice, groups, factor)
                assert Operator(folded).equiv(Operator(ideal)), "folding changed the ideal unitary"
                assert folded.count_ops().get("rzz", 0) == factor * depth * lattice.n_edges, "RZZ count"
        return "unitary and RZZ count match at factors 1, 3, 5"

    return _check("fold_cycles", 5, verify)


def _zne_fixture(kind: str):
    factors = np.asarray((1, 3, 5), dtype=float)
    if kind == "linear":
        intercept = np.asarray((0.31, -0.14, 0.47))
        slope = np.asarray((-0.025, 0.018, -0.014))
        values = intercept[:, None] + slope[:, None] * factors
    elif kind == "quadratic":
        intercept = np.asarray((0.41, 0.27, -0.13))
        slope = np.asarray((-0.03, 0.02, 0.016))
        curvature = np.asarray((0.004, -0.003, 0.002))
        values = intercept[:, None] + slope[:, None] * factors + curvature[:, None] * factors**2
    elif kind == "exponential":
        intercept = np.asarray((0.72, 0.48, 0.61))
        rate = np.asarray((0.13, 0.19, 0.09))
        values = intercept[:, None] * np.exp(-rate[:, None] * factors)
    else:
        raise ValueError(kind)
    run = {
        "cycles": np.asarray((1, 2, 3)),
        "factors": (1, 3, 5),
        "values": {"M": values, "Mz2": values * 0.8},
        "stds": {"M": np.full_like(values, 0.01), "Mz2": np.full_like(values, 0.01)},
    }
    return run, intercept


def _check_zne_extrapolator(name, fn, maximum, kind, chosen):
    def verify():
        run, expected = _zne_fixture(kind)
        _assert_array(fn(run, chosen_factors=chosen, observable="M"), expected, "zero-noise M", atol=2e-4)
        _assert_array(fn(run, chosen_factors=chosen, observable="Mz2"), expected * 0.8,
                      "zero-noise Mz2", atol=2e-4)
        try:
            fn(run, chosen_factors=(1, 7), observable="M")
        except ValueError:
            pass
        else:
            raise AssertionError("missing noise factor must raise ValueError")
        return f"{kind} intercept and missing-factor handling match"

    return _check(name, maximum, verify)


def check_zne_linear_extrapolator(fn) -> float:
    return _check_zne_extrapolator("zne_linear_extrapolator", fn, 2.5, "linear", (1, 3))


def check_zne_richardson_extrapolator(fn) -> float:
    return _check_zne_extrapolator("zne_richardson_extrapolator", fn, 2.5, "quadratic", (1, 3, 5))


def check_zne_exponential_extrapolator(fn) -> float:
    return _check_zne_extrapolator("zne_exponential_extrapolator", fn, 5, "exponential", (1, 3))


def check_select_zne_model(fn) -> float:
    """Check the source's minimum-RMSE choice, including its tie order."""
    def verify():
        target = np.asarray((0.52, 0.63, 0.71))
        factors = np.asarray((1, 3, 5), dtype=float)
        deviations = np.asarray((-0.02, -0.06, -0.20))
        values = target[:, None] + deviations[None, :]
        run = {
            "cycles": np.asarray((1, 2, 3)),
            "factors": tuple(int(f) for f in factors),
            "values": {"M": values, "Mz2": values * 0.9},
            "stds": {"M": np.full_like(values, 0.01), "Mz2": np.full_like(values, 0.01)},
        }
        selected = fn(run, np.r_[1.0, target])
        assert tuple(selected[0]) == (1, 3) and selected[1] == "linear", "wrong RMSE-minimizing method"
        return "linear (1,3) minimizes the fixture RMSE"

    return _check("select_zne_model", 5, verify)


def check_qmit_zne(fn) -> float:
    """Check factor selection, per-cycle errors, custom fits and fit failures."""
    def verify():
        from unittest.mock import patch

        run, _ = _zne_fixture("linear")
        run["cycles"] = np.asarray((2, 8, 30))
        run["stds"]["Mz2"] = np.asarray([
            [0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09],
        ])
        chosen = (5, 1)
        factors = np.asarray(chosen, dtype=float)
        values = run["values"]["Mz2"][:, [2, 0]]
        stds = run["stds"]["Mz2"][:, [2, 0]]
        calls, custom_calls = [], []

        def custom_factory(x, y):
            _assert_array(x, factors, "custom factors")
            custom_calls.append(np.asarray(y).copy())
            return (lambda x, amplitude: amplitude + 0 * x), [0.5], ([-1], [1])

        def fitter(observed, errors, x, *, models, extrapolated_noise_factor, custom_fit):
            index = len(calls)
            _assert_array(observed, values[index], "factor columns and cycle order")
            _assert_array(errors, stds[index], "per-cycle standard errors")
            _assert_array(x, factors, "requested factor order")
            assert models == ["custom"] and extrapolated_noise_factor == 0.0, "zero-noise model"
            assert custom_fit is not None and callable(custom_fit[0]), "custom fit not forwarded"
            calls.append(index)
            return np.asarray([[0.2 + 0.1 * index]]), np.asarray([[0.01]])

        target = "qiskit_mitigation.zne.extrapolation._fit_extrapolation_models"
        with patch(target, fitter):
            _assert_array(fn(run, chosen, "Mz2", "custom", custom_factory),
                          [0.2, 0.3, 0.4], "zero-noise output order")
        assert len(calls) == len(custom_calls) == 3, "fit each cycle exactly once"
        for got, expected in zip(custom_calls, values):
            _assert_array(got, expected, "custom observed values")
        for invalid in ((1, 7), (1, 1)):
            try:
                fn(run, invalid, "M", "linear")
            except ValueError:
                pass
            else:
                raise AssertionError("invalid factors must raise ValueError")
        broken = {**run, "values": {"M": run["values"]["M"].copy()}}
        broken["values"]["M"][0, 0] = np.nan
        try:
            fn(broken, (1, 3), "M", "linear")
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite measurements must raise ValueError")
        with patch(target, return_value=(np.asarray([[np.nan]]), np.asarray([[0.01]]))):
            try:
                fn(run, (1, 3), "M", "linear")
            except RuntimeError as exc:
                assert all(str(cycle) in str(exc) for cycle in run["cycles"]), "report failed cycle labels"
            else:
                raise AssertionError("non-finite fits must raise RuntimeError")
        return "factor order, errors, custom fits and failed cycle labels match"

    return _check("qmit_zne", 5, verify)


def _extension_fixture(fn, *, use_hardware, include_run21=True, factors=(1, 3)):
    """Invoke the size exercise with local callbacks that cannot contact a QPU."""
    from utils import floquet_helpers as fh

    lattice = fh.paper_colored_lattice(21)
    baseline = {"layout": list(range(21))} if include_run21 else None
    backend, ledger = object(), Path("fixture-ledger.json")
    cycles = np.asarray((2, 10, 30))
    saved_layout = list(range(80, 115))
    lookups, searches, measurements, charges = [], [], [], []
    returned_runs = {}

    def lookup(key, *, backend_name, ledger_path):
        assert use_hardware, "local track accessed the ledger"
        lookups.append((key, backend_name, ledger_path))
        return {"layout": saved_layout} if key == "trex-35q" else None

    def search(lat, groups, target):
        assert use_hardware, "local track searched hardware layouts"
        searches.append((lat, groups, target))
        return list(range(20, 20 + lat.n_qubits))

    def measure(lat, groups, target, layout, *, cycles, factors):
        assert use_hardware, "local track measured circuits"
        measurements.append((lat, groups, target, list(layout), tuple(cycles), tuple(factors)))
        run = {"n_qubits": lat.n_qubits, "layout": list(layout), "qpu_seconds": 7.0,
               "pub_order": [(int(cycle), factor) for cycle in cycles for factor in factors]}
        returned_runs[lat.n_qubits] = run
        return run

    def spent(path):
        assert use_hardware, "local track read QPU charges"
        charges.append(path)
        return 123.0  # Includes earlier experiments; deliberately differs from run charges.

    result = fn(
        lattice, baseline, cycles=cycles, selected_factors=factors, use_hardware=use_hardware,
        backend=backend, backend_name="fixture_backend", ledger_path=ledger,
        saved_run_settings=lookup, mapomatic_layout=search,
        measure_cycle_folded=measure, qpu_time_spent=spent,
    )
    return SimpleNamespace(
        result=result, lattice=lattice, baseline=baseline, backend=backend, ledger=ledger,
        cycles=cycles, factors=factors, saved_layout=saved_layout, lookups=lookups,
        searches=searches, measurements=measurements, charges=charges, runs=returned_runs,
    )


def check_extend_sizes(fn) -> float:
    """Award 5 points each for patches, layouts and guarded acquisition/accounting."""
    def patches():
        for include_baseline in (False, True):
            fixture = _extension_fixture(fn, use_hardware=False, include_run21=include_baseline)
            result = fixture.result
            assert set(result["lattices_by_size"]) == {21, 28, 35, 51}, "paper patch sizes"
            assert result["lattices_by_size"][21] is fixture.lattice, "preserve the 21q lattice"
            assert set(result["groups_by_size"]) == {21, 28, 35, 51}, "colour group sizes"
            for size, lat in result["lattices_by_size"].items():
                expected, groups = reference_lattice_groups(size)
                lat.validate()
                assert lat.n_qubits == size and lat.edges == expected.edges, "paper patch bonds"
                assert result["groups_by_size"][size] == groups, "ordered paper colours"
            assert set(result["runs_by_size"]) == ({21} if include_baseline else set()), "baseline runs"
            assert set(result["layout_by_size"]) == ({21} if include_baseline else set()), "baseline layouts"
            if include_baseline:
                assert result["runs_by_size"][21] is fixture.baseline, "preserve the baseline run"
                assert result["layout_by_size"][21] == fixture.baseline["layout"], "baseline layout"
            assert result["qpu_seconds"] == 0.0, "local track QPU time"
        return "all paper patches and optional baseline preserved"

    def layouts():
        fixture = _extension_fixture(fn, use_hardware=True)
        result = fixture.result
        assert fixture.lookups == [(f"trex-{size}q", "fixture_backend", fixture.ledger)
                                   for size in (28, 35, 51)], "size-specific saved settings"
        assert [lat.n_qubits for lat, _, _ in fixture.searches] == [28, 51], "search only unsaved layouts"
        for lat, groups, target in fixture.searches:
            assert lat is result["lattices_by_size"][lat.n_qubits], "layout patch"
            assert groups == result["groups_by_size"][lat.n_qubits] and target is fixture.backend, "layout groups/backend"
        assert result["layout_by_size"][35] == fixture.saved_layout, "reuse saved layout"
        for size in (28, 51):
            assert result["layout_by_size"][size] == list(range(20, 20 + size)), "store searched layout"
        return "saved layout reused; new layouts use their own patch"

    def acquisition():
        # Run the guard independently so it cannot earn acquisition credit merely
        # because the patch-construction check passed.
        _extension_fixture(fn, use_hardware=False)
        for factors in ((1, 3), (1, 3, 5)):
            fixture = _extension_fixture(fn, use_hardware=True, factors=factors)
            result = fixture.result
            assert [row[0].n_qubits for row in fixture.measurements] == [28, 35, 51], "measure each larger size once"
            assert set(result["runs_by_size"]) == {21, 28, 35, 51}, "store all runs"
            assert result["runs_by_size"][21] is fixture.baseline, "retain baseline run"
            for lat, groups, target, layout, cycles, chosen in fixture.measurements:
                size = lat.n_qubits
                assert lat is result["lattices_by_size"][size], "measurement patch"
                assert groups == result["groups_by_size"][size] and target is fixture.backend, "measurement groups/backend"
                assert layout == result["layout_by_size"][size], "measurement layout"
                assert cycles == tuple(fixture.cycles) and chosen == factors, "selected measurement plan"
                assert result["runs_by_size"][size] is fixture.runs[size], "preserve returned measurements"
            assert fixture.charges == [fixture.ledger] and result["qpu_seconds"] == 123.0, "cumulative ledger charge"
        return "hardware guard, selected plan, runs and cumulative QPU time match"

    earned = sum(_check(f"extend_sizes_{name}", 5, verify) for name, verify in (
        ("patches", patches), ("layouts", layouts), ("acquisition", acquisition),
    ))
    return _finish_check("extend_sizes", earned, 15)


# --------------------------------------------------------------------------- #
# Hardware comparison diagnostics and notebook exercise score summary
# --------------------------------------------------------------------------- #

def _walkthrough_target(
    n_qubits: int, reference_path: str | Path | None = None,
) -> tuple[np.ndarray, str]:
    """Use supplied references; missing release targets remain explicitly pending."""
    trace = reference_magnetization(n_qubits, reference_path=reference_path)
    source = ("digitized paper 51q experiment" if n_qubits == 51 and reference_path is None
              else f"corrected {n_qubits}q exact reference")
    return trace, source


def _magnetization_trace(result) -> np.ndarray:
    if hasattr(result, "magnetization"):
        trace = result.magnetization
    elif isinstance(result, Mapping):
        trace = result.get("magnetization", result.get("M"))
    else:
        trace = result
    return np.asarray(trace, dtype=float).reshape(-1)


def _is_measured_user_job(run, backend_name: str | None = None) -> bool:
    return bool(
        isinstance(run, Mapping)
        and run.get("provenance") == "user"
        and run.get("job_id")
        and run.get("job") is not None
        and run.get("backend") == (backend_name or SPEC["backend_name"])
    )


def _job_charges(run) -> dict[str, float]:
    """Return individually charged job IDs; multi-job runs need an itemized ledger."""
    total = float(run["qpu_seconds"])
    if not np.isfinite(total) or total <= 0:
        raise ValueError("measured job charge must be positive and finite")
    job_id = run["job_id"]
    if isinstance(job_id, str):
        ids = (job_id,)
    elif isinstance(job_id, (tuple, list)) and all(isinstance(item, str) for item in job_id):
        ids = tuple(job_id)
    else:
        raise ValueError("job_id must be a string or a sequence of strings")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("job IDs must be nonempty and unique within a run")
    if len(ids) == 1:
        return {ids[0]: total}
    itemized = run.get("job_charges")
    if not isinstance(itemized, Mapping) or set(itemized) != set(ids):
        raise ValueError("multiple job IDs require matching job_charges")
    charges = {job_id: float(itemized[job_id]) for job_id in ids}
    if any(not np.isfinite(value) or value <= 0 for value in charges.values()):
        raise ValueError("itemized job charges must be positive and finite")
    if not np.isclose(sum(charges.values()), total, rtol=0, atol=1e-6):
        raise ValueError("itemized job charges do not match run QPU time")
    return charges


def score_hardware_runs(
    runs_by_size, traces_by_size, qpu_seconds: float, *,
    reference_paths: Mapping[int, str | Path] | None = None,
    backend_name: str | None = None,
) -> dict[str, float]:
    """Hardware comparison diagnostics for measured jobs within the 1,800-second cap.

    The historical 30-credit comparison scale is separate from the 100-point
    exercise rubric. These values never award exercise or QESEM credit.

    Every size gets 60% of its weight for accuracy (full MAE <= .02, half <=
    .05) and 40% in proportion to trusted physical depth out of 30 cycles.
    Sparse traces compare only measured nonzero labels; depth stops at the
    first sampled failure without claiming agreement at unmeasured cycles.
    Archived organizer runs receive zero, even if their original job IDs and
    historical QPU charges are present.
    Boston is the production default; pass the configured backend for testing.
    Missing reference files leave that size pending, with no archive fallback.
    """
    scores = {part: 0.0 for part in HARDWARE_POINTS}
    try:
        elapsed = float(qpu_seconds)
    except (TypeError, ValueError):
        elapsed = float("nan")
    measured_runs = {
        n_qubits: run for n_qubits, run in
        (runs_by_size.items() if isinstance(runs_by_size, Mapping) else ())
        if _is_measured_user_job(run, backend_name)
    }
    if not measured_runs:
        print("hardware: 0/30 — organizer archives and runs without a measured user job earn no hardware points")
        return scores
    if not np.isfinite(elapsed) or elapsed <= 0:
        print("hardware: 0/30 — measured user job has no recorded total QPU time")
        return scores
    if elapsed > QPU_CAP_S:
        print(f"hardware: 0/30 — total QPU time exceeds {QPU_CAP_S:g} s")
        return scores

    ledger: dict[str, float] = {}
    try:
        for run in measured_runs.values():
            for job_id, charge in _job_charges(run).items():
                if job_id in ledger and not np.isclose(ledger[job_id], charge, rtol=0, atol=1e-6):
                    raise ValueError(f"conflicting charges for job {job_id}")
                ledger[job_id] = charge
    except (KeyError, TypeError, ValueError) as exc:
        print(f"hardware: 0/30 — job charge ledger is incomplete: {exc}")
        return scores
    distinct_charge = sum(ledger.values())
    if distinct_charge > elapsed + 1e-6:
        print(f"hardware: 0/30 — distinct job charges {distinct_charge:.1f} s "
              f"exceed reported total {elapsed:.1f} s")
        return scores

    for n_qubits, weight in HARDWARE_SIZE_POINTS.items():
        run = measured_runs.get(n_qubits)
        result = traces_by_size.get(n_qubits) if isinstance(traces_by_size, Mapping) else None
        if not isinstance(run, Mapping) or result is None:
            continue
        reference_path = (reference_paths or {}).get(n_qubits)
        try:
            target, source = _walkthrough_target(n_qubits, reference_path)
        except FileNotFoundError:
            print(f"hardware {n_qubits}q: pending/{weight} — supplied reference file is not yet available")
            continue
        measured = _magnetization_trace(result)
        result_cycles = (getattr(result, "cycles", None) if not isinstance(result, Mapping)
                         else result.get("cycles"))
        try:
            labels = _cycle_labels(result_cycles, N_CYCLES, n_points=measured.size)
            if "cycles" in run:
                run_cycles = np.asarray(run["cycles"])
                if (run_cycles.ndim != 1 or not run_cycles.size
                        or run_cycles.dtype.kind not in "iuf"):
                    raise ValueError("measured run cycle labels are missing or malformed")
                if run_cycles[0] != 0:
                    run_cycles = np.r_[0, run_cycles]
                expected = _cycle_labels(run_cycles, N_CYCLES)
                if not np.array_equal(labels, expected):
                    raise ValueError("result cycles disagree with the measured run")
        except (TypeError, ValueError) as exc:
            print(f"hardware {n_qubits}q: 0/{weight} — invalid cycle labels: {exc}")
            continue
        if len(target) != N_CYCLES + 1:
            print(f"hardware {n_qubits}q: 0/{weight} — expected full reference cycles 0..{N_CYCLES}")
            continue
        errors = np.abs(measured[1:] - target[labels[1:]])
        finite = np.isfinite(errors)
        mae = float(np.mean(errors)) if finite.all() else float("inf")
        accuracy_fraction = 1.0 if mae <= TOL_FULL else 0.5 if mae <= TOL_PASS else 0.0
        trusted_depth = 0
        for cycle, error in zip(labels[1:], errors):
            if np.isfinite(error) and error <= TOL_PASS:
                trusted_depth = int(cycle)
            else:
                break
        points = weight * (0.6 * accuracy_fraction + 0.4 * trusted_depth / N_CYCLES)
        part = "Part 4" if n_qubits == 21 else "Part 5"
        scores[part] += points
        depth_label = ("sampled trusted depth" if len(labels) != N_CYCLES + 1
                       else "trusted depth")
        print(f"hardware {n_qubits}q: {points:.2f}/{weight} — MAE {mae:.4f}, "
              f"{depth_label} {trusted_depth}/{N_CYCLES}; {source}")
    return scores


def summarize(local_scores, hardware_scores=None) -> float:
    """Total the 100-point exercise rubric; report hardware diagnostics separately.

    Part 5's score includes the separately maintained 10-point QESEM exercise.
    Historical hardware comparison credits never add to the exercise total.
    """
    hardware_scores = hardware_scores or {}
    total = 0.0
    for part, maximum in MAX_POINTS.items():
        local = float(local_scores.get(part, 0.0))
        local = min(maximum, max(0.0, local)) if np.isfinite(local) else 0.0
        hardware_maximum = HARDWARE_POINTS.get(part, 0)
        hardware = float(hardware_scores.get(part, 0.0))
        hardware = min(hardware_maximum, max(0.0, hardware)) if np.isfinite(hardware) else 0.0
        print(f"{part}: {local:g}/{maximum}")
        if hardware_maximum:
            print(f"  hardware diagnostic: {hardware:g}/{hardware_maximum} (outside exercise total)")
        total += local
    print(f"Total: {total:g}/{sum(MAX_POINTS.values())} (draft)")
    return total
