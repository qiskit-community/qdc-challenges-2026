"""Helper utilities for the QDC 2026 Floquet-dynamics challenge.

Everything here is *scaffolding* -- graph bookkeeping, measurement post-processing,
transpilation/mitigation glue, QPU-time budgeting, and the classical-reference
runner (local or Qiskit Serverless). The physics you implement yourself:
assembling the Floquet cycle circuit is Exercise 1.

The exact-statevector reference lives in the ``reference/`` package; this module
re-exports its plotting helpers so ``figure3`` etc. are one import away.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from json.decoder import JSONDecoder
from json.encoder import JSONEncoder
from pathlib import Path

import numpy as np

from reference import (  # exact-statevector reference (provided)
    FloquetParams,
    FloquetResult,
    FloquetSimulator,
    HeavyHexLattice,
    heavy_hex,
    heavy_hex_paper,
    paper_colored_21,
    paper_colored_lattice,
    PAPER_GROUPS_21,
    PAPER_GROUPS_28,
    PAPER_GROUPS_35,
    PAPER_GROUPS_51,
    PAPER_GROUPS_74,
    PAPER_PHYSICAL_LABELS_51,
    PAPER_PHYSICAL_LABELS_74,
    run_reference,
)
from reference.make_figure3 import (  # noqa: F401  (re-exported for the notebook)
    load_paper_51q,
    load_results,
)
from reference.plotting import (  # noqa: F401  (re-exported for the notebook)
    figure3,
    figure3_comparison,
    figure3_comparison_partial,
    plot_amplitude_scaling,
    plot_dynamics,
    plot_finite_size,
    plot_lattice,
)

__all__ = [
    "SPEC",
    "INSTANCE_ID",
    "REFERENCE_PATHS",
    "load_reference",
    "align_zne_reference",
    "trace_diagnostics",
    "hardware_diagnostics",
    "FloquetParams",
    "get_lattice",
    "heavy_hex_paper",
    "paper_colored_21",
    "paper_colored_lattice",
    "PAPER_GROUPS_21",
    "PAPER_GROUPS_28",
    "PAPER_GROUPS_35",
    "PAPER_GROUPS_51",
    "PAPER_GROUPS_74",
    "PAPER_PHYSICAL_LABELS_51",
    "PAPER_PHYSICAL_LABELS_74",
    "color_groups",
    "magnetization_from_counts",
    "magnetization_from_bitarray",
    "magnetization_z2_from_counts",
    "magnetization_z2_from_bitarray",
    "check_floquet_circuit",
    "QPUBudget",
    "fold_circuit_global",
    "run_statevector_reference",
    "reference_trace",
    "figure3",
    "figure3_comparison",
    "figure3_comparison_partial",
    "load_paper_51q",
    "load_results",
    "plot_dynamics",
    "plot_finite_size",
    "plot_amplitude_scaling",
    "plot_lattice",
]


# --------------------------------------------------------------------------- #
# Fixed challenge instance and classical references
# --------------------------------------------------------------------------- #

INSTANCE_ID = "floquet-paper-colored-2026"
N_CYCLES = 30
QPU_CAP_S = 30 * 60
SPEC = {
    "instance_id": INSTANCE_ID,
    "backend_name": "ibm_boston",
    "sizes": (21, 28, 35, 51),
    "n_cycles": N_CYCLES,
    "params": FloquetParams(),         # (pi/6, pi/27, pi/3)
    "factors": (1, 3, 5),
    "shots": 8192,
    "qpu_cap_seconds": QPU_CAP_S,
}

# Diagnostic tolerances on the mean absolute error (outside the 100 points).
TOL_PASS = 0.05
TOL_FULL = 0.02

_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
REFERENCE_PATHS = {
    n: _REFERENCE_DIR / f"traces_paper_colored_{n}q_v1.csv" for n in (21, 28, 35)
}
_PAPER_51Q: dict[int, FloquetResult] = {}


def load_reference(n_qubits: int, reference_path: str | Path | None = None) -> FloquetResult:
    """Load the exact paper-coloured trace (21/28/35) or the digitized 51q target.

    Pass ``reference_path`` to read a specific CSV instead of the release file.
    """
    if reference_path is None and n_qubits == 51:
        if 51 not in _PAPER_51Q:
            _PAPER_51Q[51], _ = load_paper_51q()
        return _PAPER_51Q[51]
    if reference_path is None:
        if n_qubits not in REFERENCE_PATHS:
            raise ValueError(f"No paper-coloured reference for N={n_qubits}; "
                             "supported sizes are 21, 28, 35, and 51")
        reference_path = REFERENCE_PATHS[n_qubits]
    path = Path(reference_path)
    if not path.is_file():
        raise FileNotFoundError(f"Reference trace for N={n_qubits} is not available: {path}")
    selected = [r for r in load_results(path) if r.n_qubits == n_qubits]
    if len(selected) != 1:
        raise ValueError(f"Expected one N={n_qubits} trace in {path}, found {len(selected)}")
    return selected[0]


def _magnetization_trace(result) -> np.ndarray:
    if hasattr(result, "magnetization"):
        trace = result.magnetization
    elif isinstance(result, Mapping):
        trace = result.get("magnetization", result.get("M"))
    else:
        trace = result
    return np.asarray(trace, dtype=float).reshape(-1)


def _is_integer_labels(labels: np.ndarray) -> bool:
    return bool(labels.ndim == 1 and labels.size and labels.dtype.kind in "iuf"
                and np.isfinite(labels).all() and np.equal(labels, np.floor(labels)).all())


def align_zne_reference(run, reference_trace) -> np.ndarray:
    """Align a labelled or dense reference with a run's measured nonzero cycles.

    ``reference_trace`` is a :class:`FloquetResult`, a mapping with ``cycles``, a
    dense array indexed by cycle, or an array already restricted to the run's
    cycles (optionally prefixed with cycle zero).
    """
    raw_cycles = np.asarray(run["cycles"])
    if (not _is_integer_labels(raw_cycles) or raw_cycles[0] <= 0
            or np.any(np.diff(raw_cycles.astype(float)) <= 0)):
        raise ValueError("run cycles must be strictly increasing positive integers")
    cycles = raw_cycles.astype(int)
    if hasattr(reference_trace, "cycles"):
        labels = np.asarray(reference_trace.cycles)
    elif isinstance(reference_trace, Mapping) and "cycles" in reference_trace:
        labels = np.asarray(reference_trace["cycles"])
    else:
        reference = np.asarray(reference_trace, dtype=float).reshape(-1)
        if reference.size > cycles[-1]:
            return reference[cycles]
        if reference.size == cycles.size + 1:
            return reference[1:]
        if reference.size == cycles.size:
            return reference
        raise ValueError("Reference and run have different cycle counts")
    values = _magnetization_trace(reference_trace)
    if (not _is_integer_labels(labels) or len(labels) != len(values)
            or len(np.unique(labels)) != len(labels) or not np.isin(cycles, labels).all()):
        raise ValueError("Reference does not contain every measured cycle")
    by_cycle = dict(zip(labels.astype(int), values))
    return np.asarray([by_cycle[cycle] for cycle in cycles], dtype=float)


# --------------------------------------------------------------------------- #
# Print-only hardware diagnostics (not part of the 100-point score)
# --------------------------------------------------------------------------- #

def _compare_trace(measured, reference: FloquetResult, cycles) -> dict:
    """MAE and trusted depth of a trace over its measured nonzero cycles.

    Any non-finite measured point makes the MAE infinite, so a missing cycle can
    never improve the comparison. The trusted depth is the last cycle before the
    first error above ``TOL_PASS``.
    """
    measured = np.asarray(measured, dtype=float).reshape(-1)
    labels = np.arange(measured.size) if cycles is None else np.asarray(cycles)
    if (not _is_integer_labels(labels) or labels[0] != 0
            or np.any(np.diff(labels.astype(float)) <= 0)):
        raise ValueError("cycles must be strictly increasing integers starting at 0")
    if labels.size != measured.size:
        raise ValueError(f"expected {labels.size} points for the labelled cycles, "
                         f"got {measured.size}")
    labels = labels.astype(int)
    target = align_zne_reference({"cycles": labels[1:]}, reference)
    errors = np.abs(measured[1:] - target)
    mae = float(errors.mean()) if errors.size and np.isfinite(errors).all() else float("inf")
    trusted = 0
    for cycle, error in zip(labels[1:], errors):
        if not (np.isfinite(error) and error <= TOL_PASS):
            break
        trusted = int(cycle)
    return {"mae": mae, "trusted_depth": trusted, "last_cycle": int(labels[-1])}


def _verdict(mae: float) -> str:
    if mae <= TOL_FULL:
        return f"within {TOL_FULL} (paper-level precision)"
    if mae <= TOL_PASS:
        return f"within {TOL_PASS}"
    return f"above {TOL_PASS}"


def trace_diagnostics(measured, reference: FloquetResult, *, label: str,
                      qpu_seconds: float | None = None, cycles=None) -> dict:
    """Print how a measured ``M`` trace compares with a reference. Awards no points."""
    metrics = _compare_trace(measured, reference, cycles)
    qpu = "" if qpu_seconds is None else f"  (QPU {float(qpu_seconds):.0f} s)"
    print(f"{label}: MAE = {metrics['mae']:.4f} ({_verdict(metrics['mae'])}), "
          f"trusted depth n* = {metrics['trusted_depth']}/{metrics['last_cycle']}{qpu}")
    return metrics


def hardware_diagnostics(traces_by_size, references_by_size, qpu_seconds: float) -> dict:
    """Print per-size comparisons of selected hardware traces. Awards no points.

    ``traces_by_size`` maps size to a result with ``cycles`` and ``magnetization``
    (or ``M``). Sizes without a reference are reported as pending.
    """
    report = {}
    for n_qubits, result in sorted(traces_by_size.items()):
        reference = references_by_size.get(n_qubits)
        if reference is None:
            print(f"{n_qubits}q: pending (no reference supplied)")
            continue
        cycles = (getattr(result, "cycles", None) if not isinstance(result, Mapping)
                  else result.get("cycles"))
        try:
            report[n_qubits] = trace_diagnostics(_magnetization_trace(result), reference,
                                                 label=f"{n_qubits}q", cycles=cycles)
        except ValueError as exc:
            print(f"{n_qubits}q: not compared ({exc})")
    elapsed = float(qpu_seconds)
    status = "within" if elapsed <= QPU_CAP_S else "OVER"
    print(f"total QPU time: {elapsed:.1f} s ({status} the {QPU_CAP_S} s cap)")
    return report


# --------------------------------------------------------------------------- #
# Lattice access
# --------------------------------------------------------------------------- #

def get_lattice(n_qubits: int) -> HeavyHexLattice:
    """Heavy-hex patch nearest ``n_qubits`` (snaps to a real patch; see reference)."""
    return heavy_hex(n_qubits)


def color_groups(lattice: HeavyHexLattice) -> list[list[tuple[int, int]]]:
    """The three edge-colour classes ``[E1, E2, E3]``.

    Every group is a set of disjoint bonds, so its ``RZZ`` rotations form one
    parallel layer -- this is the structure your Floquet cycle should exploit.
    """
    return lattice.color_groups()


# --------------------------------------------------------------------------- #
# Measurement post-processing (provided)
# --------------------------------------------------------------------------- #

def magnetization_from_counts(counts: dict[str, int], n_qubits: int) -> float:
    """M = (1/N) sum_q <Z_q> from a counts dict ``{bitstring: shots}``.

    Uses ``sum_q <Z_q> = N - 2 * <number of 1s>``, so ``M = 1 - 2 * mean_ones / N``.
    Bitstrings may contain spaces (Qiskit register separators); they are ignored.
    """
    total = 0
    ones = 0
    for bitstr, c in counts.items():
        b = bitstr.replace(" ", "")
        ones += b.count("1") * c
        total += c
    if total == 0:
        return float("nan")
    mean_ones = ones / total
    return 1.0 - 2.0 * mean_ones / n_qubits


def magnetization_from_bitarray(bit_array, n_qubits: int) -> float:
    """Same as :func:`magnetization_from_counts` but for a Qiskit ``BitArray``."""
    return magnetization_from_counts(bit_array.get_counts(), n_qubits)


def magnetization_z2_from_counts(counts: dict[str, int],
                                 lattice: HeavyHexLattice) -> float:
    """Coordination-two magnetization ``M_{z=2}`` from a counts dict.

    ``M_{z=2} = (1/|V_{z=2}|) sum_{q in V_{z=2}} <Z_q>``, averaging **only** over
    the degree-2 (coordination-two) qubits of ``lattice`` -- the paper's finite-
    size observable (Fig. 3b/c), and exactly the set the classical reference uses
    (``reference/simulator.py``). Build ``lattice`` with :func:`heavy_hex_paper`
    so ``V_{z=2}`` matches the provided reference (``|V_{z=2}|`` = 19/24/29 at
    21/28/35 qubits).

    Bit order: Qiskit is little-endian, so in a ``measure_all`` bitstring ``b``
    the rightmost character is qubit 0 and qubit ``q`` is ``b[n-1-q]``. Register
    separators (spaces) are stripped. ``<Z_q> = 1 - 2 P(bit q = 1)``.
    """
    deg = lattice.degrees()
    n = lattice.n_qubits
    z2 = [q for q in range(n) if deg[q] == 2]
    if not z2:
        return float("nan")

    total = 0
    ones = {q: 0 for q in z2}
    for bitstr, c in counts.items():
        b = bitstr.replace(" ", "")
        total += c
        for q in z2:
            if b[n - 1 - q] == "1":
                ones[q] += c
    if total == 0:
        return float("nan")

    z_sum = sum(1.0 - 2.0 * ones[q] / total for q in z2)
    return z_sum / len(z2)


def magnetization_z2_from_bitarray(bit_array, lattice: HeavyHexLattice) -> float:
    """Same as :func:`magnetization_z2_from_counts` but for a Qiskit ``BitArray``."""
    return magnetization_z2_from_counts(bit_array.get_counts(), lattice)


# --------------------------------------------------------------------------- #
# Circuit self-check (a free pre-grader for Exercise 1)
# --------------------------------------------------------------------------- #

def _test_lattice() -> HeavyHexLattice:
    """A tiny valid lattice for fast, exact circuit checks.

    A subdivided ``K_{1,3}`` (7 qubits): a degree-3 centre wired through three
    degree-2 flags to three leaves. It is the smallest heavy-hex fragment that
    needs **three** edge-colour classes -- essential here, because one Floquet
    cycle re-applies the ``RX``/``RZ`` layer *once per colour group* (paper Eq. 1,
    effective ``3 theta_x``). A 2-colourable lattice (e.g. an even ring) would let
    a 2-layer cycle pass, hiding the missing third sub-layer.
    """
    edges = ((0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6))
    colors = (0, 1, 2, 1, 0, 0)               # proper 3-colouring (forced at node 0)
    pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (-0.5, 0.87), 3: (-0.5, -0.87),
           4: (2.0, 0.0), 5: (-1.0, 1.73), 6: (-1.0, -1.73)}
    lat = HeavyHexLattice(n_qubits=7, edges=edges, edge_colors=colors, positions=pos)
    lat.validate()
    return lat


def check_floquet_circuit(
    floquet_cycle, params: FloquetParams | None = None, n_cycles: int = 5
) -> bool:
    """Behavioural self-check for your Exercise-1 ``floquet_cycle`` builder.

    Builds your circuit for ``n_cycles`` cycles on a tiny 6-qubit lattice, evolves
    ``|0...0>`` with Qiskit's exact ``Statevector``, and compares the resulting
    magnetization trace to the trusted reference engine. Returns ``True`` and
    prints ✅ on a match to ``1e-9`` (checks behaviour, not gate order, so it does
    not give the answer away).

    ``floquet_cycle`` must have signature ``floquet_cycle(qc, groups, params)``
    where ``groups`` is the output of :func:`color_groups`.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    params = params or FloquetParams()
    lat = _test_lattice()
    groups = lat.color_groups()

    # participant trace via exact statevector of their circuit
    got = [1.0]
    qc = QuantumCircuit(lat.n_qubits)
    for _ in range(n_cycles):
        floquet_cycle(qc, groups, params)
        sv = np.asarray(Statevector.from_instruction(qc).data)
        probs = np.abs(sv) ** 2
        ones = np.array([bin(s).count("1") for s in range(len(sv))])
        got.append(1.0 - 2.0 * float(probs @ ones) / lat.n_qubits)

    # trusted reference on the same lattice
    ref = FloquetSimulator(lat, params=params, device="cpu").run(n_cycles).magnetization

    ok = np.allclose(got, ref, atol=1e-9)
    if ok:
        print("✅ check_floquet_circuit passed: your U_F matches the reference "
              f"to 1e-9 over {n_cycles} cycles.")
    else:
        mae = float(np.mean(np.abs(np.array(got) - ref)))
        print(f"❌ check_floquet_circuit FAILED (mean |Δ| = {mae:.3e}). Check the "
              "layer order (RX → RZ → RZZ), the angles, and that every colour "
              "group's bonds get an RZZ.")
    return bool(ok)


# --------------------------------------------------------------------------- #
# QPU-time budgeting (hard cap: 30 minutes for the whole challenge)
# --------------------------------------------------------------------------- #

@dataclass
class QPUBudget:
    """Track cumulative Heron r3 QPU time against the 30-minute challenge cap.

    Record the *quantum* execution time reported for each job (e.g. the runtime
    ``usage`` / ``time_taken`` metadata) with :meth:`charge`; :meth:`remaining`
    tells you how much you have left before submissions are penalized.
    """

    cap_seconds: float = 30 * 60
    spent_seconds: float = 0.0

    def charge(self, seconds: float, label: str = "") -> float:
        self.spent_seconds += float(seconds)
        tag = f" [{label}]" if label else ""
        print(f"QPU charged {seconds:6.1f}s{tag} | "
              f"total {self.spent_seconds/60:5.2f} / {self.cap_seconds/60:.0f} min "
              f"({'OK' if self.within_budget else 'OVER BUDGET'})")
        return self.remaining

    @property
    def remaining(self) -> float:
        return max(0.0, self.cap_seconds - self.spent_seconds)

    @property
    def within_budget(self) -> bool:
        return self.spent_seconds <= self.cap_seconds

    @staticmethod
    def estimate(n_circuits: int, shots: int, per_shot_s: float = 2.5e-4,
                 overhead_s: float = 2.0) -> float:
        """Rough a-priori estimate: ``n_circuits * (overhead + shots*per_shot)``.

        ``per_shot_s`` and ``overhead_s`` are order-of-magnitude Heron r3 numbers
        for planning only -- always reconcile against the runtime's reported usage.
        """
        return n_circuits * (overhead_s + shots * per_shot_s)


# --------------------------------------------------------------------------- #
# Zero-noise-extrapolation helper (gate folding)
# --------------------------------------------------------------------------- #

def fold_circuit_global(circuit, scale_factor: int):
    """Global gate folding ``U -> U (U^dag U)^k`` for integer noise scale ``2k+1``.

    Multiplies the effective noise by an odd integer ``scale_factor`` while leaving
    the ideal unitary unchanged -- the standard input to zero-noise extrapolation.
    Returns a new circuit; requires ``scale_factor`` odd and >= 1.

    A ``barrier`` is inserted between every ``U``/``U^dag`` segment so a later
    transpiler pass cannot cancel the folds (``U^dag U -> I``) and silently undo
    the noise amplification. Fold the **logical** circuit (``rx``/``rz``/``rzz``,
    whose inverses are just negated-angle rotations that map cleanly to any
    backend basis), then transpile the result -- folding an already-ISA circuit
    would introduce inverse gates (e.g. ``sxdg``) that are not in the Heron basis.
    """
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError("scale_factor must be an odd integer >= 1 (1, 3, 5, ...)")
    if scale_factor == 1:
        return circuit.copy()

    base = circuit.copy()
    meas = None
    # strip measurements, fold the unitary part, then re-append
    if base.count_ops().get("measure"):
        from qiskit import QuantumCircuit
        meas = base.copy_empty_like()
        unitary = QuantumCircuit(*base.qregs)
        for inst in base.data:
            (meas if inst.operation.name == "measure" else unitary).append(inst)
        base = unitary

    folded = base.copy()
    inv = base.inverse()
    for _ in range((scale_factor - 1) // 2):
        folded.barrier()
        folded.compose(inv, inplace=True)
        folded.barrier()
        folded.compose(base, inplace=True)
    if meas is not None:
        folded.compose(meas, inplace=True)
    return folded


# --------------------------------------------------------------------------- #
# Classical reference runner (local, or Qiskit Serverless on the H100 node)
# --------------------------------------------------------------------------- #

def reference_trace(
    n_qubits: int, n_cycles: int = 25, params: FloquetParams | None = None,
    device: str = "auto", dtype: str = "complex128",
) -> FloquetResult:
    """One local exact-statevector reference run (thin wrapper over the package)."""
    return run_reference(n_qubits, n_cycles=n_cycles, params=params,
                         device=device, dtype=dtype)


def run_statevector_reference(
    sizes,
    n_cycles: int = 30,
    params: FloquetParams | None = None,
    *,
    local: bool = True,
    client=None,
    function_name: str = "floquet_statevector_engine",
    device: str = "auto",
    dtype: str = "complex128",
    poll_seconds: int = 10,
    timeout_seconds: int = 3600,
) -> dict[int, np.ndarray]:
    """Exact reference ``M(n)`` for one or more ladder sizes, locally or remotely.

    Mirrors the SKQD ``classically_diagonalize`` pattern: ``local=True`` runs each
    size on this machine; ``local=False`` submits one Qiskit Serverless job that
    fans out across sizes on the H100 node.

    Returns ``{n_qubits: magnetization_array}`` (index 0 is the initial state).
    """
    params = params or FloquetParams()
    sizes = [sizes] if isinstance(sizes, int) else list(sizes)

    if local:
        out: dict[int, np.ndarray] = {}
        for n in sizes:
            res = run_reference(n, n_cycles=n_cycles, params=params,
                               device=device, dtype=dtype)
            out[res.n_qubits] = res.magnetization
            print(f">>>>> local reference N={res.n_qubits} ({res.device}) done")
        return out

    if client is None:
        raise ValueError("local=False requires a QiskitServerless `client`.")

    data = JSONEncoder().encode([
        [int(n) for n in sizes], int(n_cycles),
        params.theta_x, params.theta_z, params.theta_zz, dtype, device,
    ])
    print(f">>>>> Sending Floquet reference to Serverless ({function_name}) ...")
    worker = client.load(function_name)
    job = worker.run(data=data)
    print(f">>>>> Serverless job: {job.job_id}")

    elapsed = 0
    while elapsed < timeout_seconds:
        status = job.status()
        if status in ("QUEUED", "INITIALIZING", "RUNNING"):
            print(f">>>>> [{elapsed}s] {status}")
            time.sleep(poll_seconds)
            elapsed += poll_seconds
        elif status == "ERROR":
            print(">>>>> Serverless ERROR:\n" + job.logs())
            raise RuntimeError("serverless job failed")
        elif status == "DONE":
            traces = JSONDecoder().decode(job.result()["outputs"])
            print(f">>>>> DONE: sizes {[t[0] for t in traces]}")
            return {int(nq): np.asarray(m, dtype=float) for nq, m in traces}
        else:
            break
    raise TimeoutError("serverless job did not finish in time")

def load_boston_data(
    n_qubits=21,
    run_dir="runs/v5_paper_colored/paper_v5_boston_02",
    bootstrap_samples=600,
):
    import json
    from pathlib import Path

    import numpy as np
    import utils.floquet_helpers as fh

    run_dir = Path(run_dir)
    if n_qubits not in (21, 28, 35, 51):
        raise ValueError("No Boston archive for this system size")
    if bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be at least 2")

    summary = json.loads((run_dir / f"folded_{n_qubits}_main.json").read_text())
    identity = summary["metadata"]["identity"]
    cycles = np.asarray(summary["cycles"], dtype=int)
    factors = tuple(int(f) for f in summary["factors"])

    if (
        identity["backend"] != "ibm_boston"
        or identity["shots"] != 16384
        or not np.array_equal(cycles, np.arange(1, 31))
        or factors != (1, 3, 5)
    ):
        raise ValueError(f"Archive is not the expected Boston 02 {n_qubits}q run")

    with np.load(run_dir / f"folded_{n_qubits}_main.npz", allow_pickle=False) as data:
        values = {
            obs: np.column_stack([data[f"{obs}_f{factor}"] for factor in factors])
            for obs in ("M", "Mz2")
        }

    lattice_boston = fh.paper_colored_lattice(n_qubits)
    degree_two = np.asarray(
        [degree == 2 for degree in lattice_boston.degrees()], dtype=float
    )
    weights = np.stack(
        [np.full(n_qubits, 1 / n_qubits), degree_two / degree_two.sum()]
    )

    if not np.array_equal(np.isfinite(values["M"]), np.isfinite(values["Mz2"])):
        raise ValueError("Archived observables have different cycle/factor coverage")
    pub_order = [(int(cycle), factor)
                 for i, cycle in enumerate(cycles)
                 for j, factor in enumerate(factors)
                 if np.isfinite(values["M"][i, j])]
    positions = {
        (int(cycle), factor): (i, j)
        for i, cycle in enumerate(cycles)
        for j, factor in enumerate(factors)
    }
    stds = {
        obs: np.full((len(cycles), len(factors)), np.nan)
        for obs in ("M", "Mz2")
    }

    rng = np.random.default_rng(20261005 + n_qubits)
    seen = set()
    job_ids = []

    for stem in summary["stems"]:
        manifest = json.loads((run_dir / f"{stem}_records.json").read_text())
        if manifest["model_signature"] != identity["model_signature"]:
            raise ValueError(f"Model mismatch in {stem}")
        job_ids.append(manifest["job_id"])

        with np.load(run_dir / f"{stem}_records.npz", allow_pickle=False) as data:
            calibration_draws = {}
            for cal_id in manifest["cal_ids"]:
                calibration = np.asarray(data[cal_id], dtype=float)
                indices = rng.integers(
                    len(calibration),
                    size=(bootstrap_samples, len(calibration)),
                )
                calibration_draws[cal_id] = np.exp(
                    calibration[indices].mean(axis=1) - 1
                )

            for entry in manifest["entries"]:
                tag = entry["tag"]
                key = (int(tag["cycle"]), int(tag["factor"]))
                if key not in positions or key in seen:
                    raise ValueError(f"Unexpected or duplicate PUB: {key}")
                seen.add(key)
                i, j = positions[key]

                records = np.asarray(data[entry["key"]], dtype=float)
                cal_id = entry["cal_id"]
                if cal_id is None:
                    denominator = 1.0
                    draw_denominator = 1.0
                else:
                    denominator = np.exp(
                        np.asarray(data[cal_id], dtype=float).mean(axis=0) - 1
                    )
                    draw_denominator = calibration_draws[cal_id]

                central = weights @ (records.mean(axis=0) / denominator)
                archived = np.array(
                    [values["M"][i, j], values["Mz2"][i, j]]
                )
                np.testing.assert_allclose(
                    central, archived, rtol=0, atol=1e-10
                )

                indices = rng.integers(
                    len(records), size=(bootstrap_samples, len(records))
                )
                sampled = records[indices].mean(axis=1) / draw_denominator
                se_m, se_mz2 = np.std(
                    sampled @ weights.T, axis=0, ddof=1
                )
                stds["M"][i, j] = se_m
                stds["Mz2"][i, j] = se_mz2

    if seen != set(pub_order):
        raise ValueError("The archive is missing cycle/factor PUBs")

    ledger = json.loads((run_dir / "ledger.json").read_text())
    charges = [ledger["jobs"][job_id]["qpu_seconds"] for job_id in job_ids]
    if any(charge is None for charge in charges):
        raise ValueError("A Boston job has no finalized QPU charge")

    return {
        "n_qubits": n_qubits,
        "cycles": cycles,
        "factors": factors,
        "pub_order": pub_order,
        "values": values,
        "stds": stds,
        "job_id": tuple(job_ids),
        "job": None,
        "qpu_seconds": float(sum(charges)),
        "backend": "ibm_boston",
        "layout": summary["metadata"]["layout"],
    }
