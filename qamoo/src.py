from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

NUM_ASSETS = 100  # 1 qubit per asset
LEVELS = (0, 1, 2, 3, 4)

# The 2 files this module reads, both beside this file.
_HERE = Path(__file__).resolve().parent
ANGLE_DIRECTORIES = (_HERE / "data" / "angles",)
INSTANCE_FILE = _HERE / "instance.npz"


def angle_file(p: int) -> Path:
    """The trained-angle CSV of depth p."""
    for directory in ANGLE_DIRECTORIES:
        candidate = directory / f"angles_p{p}.csv"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"angles_p{p}.csv was not found in any of "
        f"{[str(d) for d in ANGLE_DIRECTORIES]}"
    )


def reachable_pairs(n: int, k: int) -> list[tuple[int, int]]:
    """The qubit pairs a line swap network makes adjacent within k layers: the sparsity
    pattern of level K, derived and not stored.
    """
    from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

    distance = np.asarray(SwapStrategy.from_line(list(range(n))).distance_matrix)
    return [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if distance[i][j] is not None and distance[i][j] <= k
    ]


def ising_from_quadratic(quadratic, linear, constant):
    """Turn the binary QUBO x' Q x + c' x + d into the spin form h' z + 0.5 z' J z + offset.
    Returns (h, J, offset); the offset is also the objective of the uniform
    superposition.
    """
    from docplex.mp.model import Model
    from qiskit_addon_opt_mapper.converters import OptimizationProblemToQubo
    from qiskit_addon_opt_mapper.translators import from_docplex_mp, to_ising

    quadratic = np.asarray(quadratic, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64).ravel()
    if not np.allclose(quadratic, quadratic.T, atol=0.0, rtol=0.0):
        quadratic = 0.5 * (quadratic + quadratic.T)
    size = linear.size

    model = Model(name="ising_from_quadratic")
    x = model.binary_var_list(size, name="x")
    rows, columns = np.triu_indices(size, k=1)
    model.minimize(
        model.sum(
            float(quadratic[i, i] + linear[i]) * x[i]
            for i in range(size)
            if quadratic[i, i] != 0.0 or linear[i] != 0.0
        )
        + model.sum(
            2.0 * float(quadratic[r, c]) * x[r] * x[c]
            for r, c in zip(rows.tolist(), columns.tolist())
            if quadratic[r, c] != 0.0
        )
        + float(constant)
    )
    operator, offset = to_ising(
        OptimizationProblemToQubo().convert(from_docplex_mp(model))
    )
    model.end()

    field = np.zeros(size, dtype=np.float64)
    coupling = np.zeros((size, size), dtype=np.float64)
    for label, coefficient in zip(operator.paulis.to_labels(), operator.coeffs):
        where = [size - 1 - index for index, term in enumerate(label) if term == "Z"]
        value = float(np.real(coefficient))
        if len(where) == 1:
            field[where[0]] = value
        elif len(where) == 2:
            first, second = min(where), max(where)
            coupling[first, second] = coupling[second, first] = value
        else:
            raise RuntimeError(
                f"the addon returned a weight-{len(where)} Pauli term, {label}"
            )
    return field, coupling, float(offset)


_INSTANCE = None


def packaged_instance() -> dict:
    """The arrays of `instance.npz`, read once: the ~2,100 numbers of the instance that is
    provided.
    """
    global _INSTANCE
    if _INSTANCE is None:
        if not INSTANCE_FILE.exists():
            raise FileNotFoundError(
                f"{INSTANCE_FILE} is missing. It ships with this challenge, so "
                f"restore it from the repository."
            )
        with np.load(INSTANCE_FILE) as data:
            _INSTANCE = {name: np.asarray(data[name]) for name in data.files}
    return _INSTANCE


class Portfolio:
    """Portfolio at one sparsity level K, in physical qubit order. Reads mu and Sigma, derives the kept
    pairs from K, and checks the stored pair count against what K reaches.
    """

    def __init__(self, k: int, n: int = NUM_ASSETS):
        if k not in LEVELS:
            raise ValueError(f"the sparsity level K={k} is not one of {LEVELS}")
        self.k = int(k)
        self.n = int(n)
        blob = packaged_instance()

        self.permutation = np.asarray(blob[f"perm_{self.k}"], dtype=np.int64)
        self.mu = np.asarray(blob["mu"], dtype=np.float64)[self.permutation]
        if self.mu.size != self.n:
            raise RuntimeError(
                f"the instance holds {self.mu.size} assets and {self.n} are needed"
            )

        self.pairs = reachable_pairs(self.n, self.k)
        values = np.asarray(blob[f"pairs_{self.k}"], dtype=np.float64)
        if values.size != len(self.pairs):
            raise RuntimeError(
                f"K={self.k}: the instance holds {values.size} pair values and the "
                f"swap network reaches {len(self.pairs)} pairs"
            )

        self.sigma = np.zeros((self.n, self.n), dtype=np.float64)
        self.sigma[np.diag_indices(self.n)] = np.asarray(
            blob[f"diag_{self.k}"], dtype=np.float64
        )
        for (i, j), value in zip(self.pairs, values):
            self.sigma[i, j] = self.sigma[j, i] = value
        self.diagonal_shift = float(blob["diagonal_shift"][self.k])

        self.retained_pairs = len(self.pairs)
        self.max_qubit_distance = int(max(j - i for i, j in self.pairs))
        if self.max_qubit_distance != 2 * self.k + 1:
            raise RuntimeError(
                f"K={self.k}: the longest kept pair spans {self.max_qubit_distance} "
                f"physical qubits, and K swap layers allow {2 * self.k + 1}"
            )

        self.return_cost_scale = float(blob["return_cost_scale"][self.k])
        self.variance_scale = float(blob["variance_scale"][self.k])
        self.return_cost_lo = float(blob["return_cost_lo"][self.k])
        self.variance_lo = float(blob["variance_lo"][self.k])
        if min(self.return_cost_scale, self.variance_scale) <= 0.0:
            raise RuntimeError("a normalisation scale is not positive")

        zeros_matrix = np.zeros((self.n, self.n))
        zeros_vector = np.zeros(self.n)
        self.field_return, self.coupling_return, self.offset_return = (
            ising_from_quadratic(zeros_matrix, -self.mu, -self.return_cost_lo)
        )
        if np.any(self.coupling_return != 0.0):
            raise RuntimeError("the return part must carry no pair term")
        self.field_variance, self.coupling_variance, self.offset_variance = (
            ising_from_quadratic(self.sigma, zeros_vector, -self.variance_lo)
        )

        pair_i, pair_j = np.nonzero(np.triu(self.coupling_variance, 1))
        self.pair_i, self.pair_j = pair_i, pair_j
        if pair_i.size != self.retained_pairs:
            raise RuntimeError(
                f"K={self.k}: the Ising form holds {pair_i.size} pair terms and "
                f"{self.retained_pairs} are needed"
            )


def trained_weights(row: dict):
    """The multipliers (a, b) of 1 row, from its `cW0` and `cW1` columns. `cW1` depends on K
    and `cW0` does not.
    """
    a, b = float(row["cW0"]), float(row["cW1"])
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("the trained multipliers of this row are not both finite")
    if min(a, b) < 0.0:
        raise ValueError(f"a trained multiplier is negative: (a, b) = ({a}, {b})")
    return a, b


def trained_angles(row: dict, p: int):
    """The (betas, gammas) of 1 row, exactly as stored: no retraining and no rescaling."""
    try:
        betas = np.array([float(row[f"beta_{layer}"]) for layer in range(p)])
        gammas = np.array([float(row[f"gamma_{layer}"]) for layer in range(p)])
    except KeyError as exc:
        raise KeyError(
            f"the angle row holds no {exc.args[0]}, so it is not a depth-{p} row"
        ) from exc
    if not (np.all(np.isfinite(betas)) and np.all(np.isfinite(gammas))):
        raise ValueError("the trained angles of this row are not all finite")
    return betas, gammas


def load_slopes(p: int, k: int) -> list[dict]:
    """The 100 rows of the angle table of 1 level, sorted by weight setting."""
    rows = []
    with open(angle_file(p), newline="") as handle:
        for record in csv.DictReader(handle):
            if int(record["k"]) != int(k):
                continue
            row = {name: float(value) for name, value in record.items()}
            row["k"] = int(row["k"])
            row["weight_index"] = int(row["weight_index"])
            rows.append(row)
    rows.sort(key=lambda row: row["weight_index"])
    return rows


def spin_operator(field, coupling):
    """Field and coupling as a SparsePauliOp. The offset is dropped and added back when a
    shot is scored.
    """
    from qiskit.quantum_info import SparsePauliOp

    field = np.asarray(field, dtype=np.float64)
    coupling = np.asarray(coupling, dtype=np.float64)
    size = field.size
    terms = [
        ("Z", [qubit], float(field[qubit]))
        for qubit in range(size)
        if field[qubit] != 0.0
    ]
    rows, columns = np.triu_indices(size, k=1)
    terms += [
        ("ZZ", [int(row), int(column)], float(coupling[row, column]))
        for row, column in zip(rows.tolist(), columns.tolist())
        if coupling[row, column] != 0.0
    ]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=size).simplify()


def cost_operator(problem, weights):
    """The scalarised cost operator with the 2 weights left free."""
    return (
        weights[0] * spin_operator(problem.field_return, problem.coupling_return)
        + weights[1] * spin_operator(problem.field_variance, problem.coupling_variance)
    ).simplify()


def route_map(circuit):
    """Follow the swap gates and return the final mapping p2l and its inverse l2p."""
    size = circuit.num_qubits
    p2l = np.arange(size, dtype=np.int64)
    index_of = {bit: number for number, bit in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        if instruction.operation.name != "swap":
            continue
        first = index_of[instruction.qubits[0]]
        second = index_of[instruction.qubits[1]]
        p2l[first], p2l[second] = p2l[second], p2l[first]
    l2p = np.empty_like(p2l)
    l2p[p2l] = np.arange(size, dtype=np.int64)
    return p2l, l2p


def unpack_bits(packed_row, num_bits: int = NUM_ASSETS) -> np.ndarray:
    """1 weight setting of a result as a boolean array. Column j is logical qubit j."""
    from qiskit.primitives.containers import BitArray

    return BitArray(
        np.asarray(packed_row, dtype=np.uint8), num_bits=num_bits
    ).to_bool_array(order="little")


def selection_values(problem, bits) -> dict:
    """For every shot: the assets chosen, the expected return mu' x and the variance x'
    Sigma x.
    """
    x = np.asarray(bits, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != problem.n:
        raise ValueError(f"the shots are {x.shape} and (shots, {problem.n}) is needed")
    return {
        "chosen": x.sum(axis=1),
        "expected_return": x @ problem.mu,
        "variance": np.einsum("si,si->s", x @ problem.sigma, x),
    }


def objective_values(problem, values: dict, a: float, b: float) -> np.ndarray:
    """The scalarised f of every shot at (a, b): the same f the circuit encodes, so it
    compares with the offset.
    """
    return float(a) * (-values["expected_return"] - problem.return_cost_lo) + float(
        b
    ) * (values["variance"] - problem.variance_lo)


def pooled_front(problem, rows, bits_of, chunk: int = 25000) -> list[dict]:
    """The Pareto front over every shot of every setting, reduced in chunks."""
    front_ret = np.zeros(0, dtype=np.float64)
    front_var = np.zeros(0, dtype=np.float64)
    front_chosen = np.zeros(0, dtype=np.float64)
    front_weight = np.zeros(0, dtype=np.int64)
    per_setting, pool = None, 0

    for row in rows:
        index = int(row["weight_index"])
        bits = np.asarray(bits_of(index), dtype=bool)
        if bits.ndim != 2 or bits.shape[1] != problem.n:
            raise ValueError(
                f"the shots are {bits.shape} and (shots, {problem.n}) is needed"
            )
        if per_setting is None:
            per_setting = int(bits.shape[0])
        pool += int(bits.shape[0])

        for start in range(0, bits.shape[0], chunk):
            block = bits[start : start + chunk]
            values = selection_values(problem, block)
            ret = values["expected_return"]
            var = values["variance"]
            chosen = values["chosen"]
            # Reduce the block on its own first: a shot dominated inside its block is dominated
            # overall, so the merge below never sees it.
            inside = nondominated(ret, var)
            front_ret = np.concatenate([front_ret, ret[inside]])
            front_var = np.concatenate([front_var, var[inside]])
            front_chosen = np.concatenate([front_chosen, chosen[inside]])
            front_weight = np.concatenate(
                [front_weight, np.full(int(inside.sum()), index, dtype=np.int64)]
            )
            # Duplicates all survive domination, so drop them explicitly or the front fills up
            # with copies of the same portfolio.
            _unique, first = np.unique(
                np.column_stack([front_ret, front_var, front_chosen]),
                axis=0,
                return_index=True,
            )
            keep = np.zeros(front_ret.size, dtype=bool)
            keep[first] = True
            keep &= nondominated(front_ret, front_var)
            front_ret, front_var = front_ret[keep], front_var[keep]
            front_chosen, front_weight = front_chosen[keep], front_weight[keep]

    weight_of = {int(row["weight_index"]): row for row in rows}
    order = np.argsort(front_var)
    return [
        {
            "weight_index": int(front_weight[i]),
            "weight_return": float(weight_of[int(front_weight[i])]["weight_return"]),
            "weight_variance": float(
                weight_of[int(front_weight[i])]["weight_variance"]
            ),
            "best_return": float(front_ret[i]),
            "best_variance": float(front_var[i]),
            "best_chosen": int(front_chosen[i]),
            "shots": int(per_setting or 0),
            "pool": int(pool),
        }
        for i in order
    ]


def nondominated(returns, variances) -> np.ndarray:
    """The mask of points no other point beats on both objectives. An O(n log n) sweep;
    duplicates are kept.
    """
    returns = np.asarray(returns, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    if returns.size == 0:
        return np.zeros(0, dtype=bool)

    # A sweep in order of variance, so this is O(n log n) rather than the O(n^2) of comparing
    # every pair. That matters once the front is taken over every SHOT rather than 1 point per
    # weight setting: a few hundred thousand points is fine here and hopeless pairwise.
    #
    # Point i is dominated when some j satisfies either
    #     (A)  variance_j <  variance_i  and  return_j >= return_i,   or
    #     (B)  variance_j <= variance_i  and  return_j >  return_i,
    # which together are exactly "at least as good on both, strictly better on one". Each
    # case is strict in one objective, so 2 identical points satisfy neither and both survive.
    order = np.argsort(variances, kind="stable")
    sorted_variance = variances[order]
    sorted_return = returns[order]

    # the first index of each run of equal variance, and the run each point belongs to
    starts = np.flatnonzero(np.r_[True, sorted_variance[1:] != sorted_variance[:-1]])
    run = np.cumsum(np.r_[True, sorted_variance[1:] != sorted_variance[:-1]]) - 1

    # the largest return within each run, and the largest strictly BEFORE each run
    run_max = np.maximum.reduceat(sorted_return, starts)
    before = np.r_[-np.inf, np.maximum.accumulate(run_max)[:-1]]

    dominated = (before[run] >= sorted_return) | (
        np.maximum(before[run], run_max[run]) > sorted_return
    )
    keep = np.empty(returns.size, dtype=bool)
    keep[order] = ~dominated
    return keep


def weighted_sum_model(problem, a: float, b: float):
    """The weighted-sum objective as an unsolved docplex model. Returns (model, x).
    Unconstrained, so it can never be infeasible.
    """
    from docplex.mp.model import Model

    n = problem.n
    a, b = float(a), float(b)
    model = Model(name=f"portfolio_ws_K{problem.k}")
    x = model.binary_var_list(n, name="x")
    model.minimize(
        model.sum(
            (b * float(problem.sigma[i, i]) - a * float(problem.mu[i])) * x[i]
            for i in range(n)
        )
        + model.sum(
            2.0 * b * float(problem.sigma[i, j]) * x[i] * x[j] for i, j in problem.pairs
        )
        - a * problem.return_cost_lo
        - b * problem.variance_lo
    )
    return model, x


def weighted_sum_optimum(problem, a: float, b: float):
    """The exact optimum of the weighted sum at 1 (a, b). Returns (x, f). It reaches only
    SUPPORTED points of the front; the epsilon-constraint sweep reaches the rest.
    """
    import os

    os.environ.pop("MallocStackLogging", None)
    os.environ.pop("MallocStackLoggingNoCompact", None)

    model, x = weighted_sum_model(problem, a, b)
    model.parameters.mip.tolerances.mipgap = 0.0
    model.parameters.mip.tolerances.absmipgap = 0.0
    solution = model.solve(log_output=False)
    if solution is None:
        raise RuntimeError(f"K={problem.k}, a={a}, b={b}: CPLEX returned no solution")
    bits = np.rint([solution.get_value(v) for v in x]).astype(np.int8)
    value = float(solution.objective_value)
    model.end()
    return bits, value


def epsilon_constraint_optimum(problem, target: float):
    """The least normalised variance under a normalised return floor, both objectives in [0,
    1]. Returns (x, variance), or (None, None) when infeasible.
    """
    from docplex.mp.model import Model

    n = problem.n
    if float(target) > 1.0 + 1e-9:
        return None, None

    # The 2 spans, each the difference between the objective at x = 1 and at x = 0.
    return_span = float(-problem.return_cost_lo)  # mu' 1, minus 0
    variance_span = float(problem.variance_scale)  # 1' Sigma 1, minus 0

    model = Model(name=f"portfolio_eps_K{problem.k}")
    x = model.binary_var_list(n, name="x")
    model.minimize(
        model.sum(float(problem.sigma[i, i]) * x[i] for i in range(n))
        + model.sum(
            2.0 * float(problem.sigma[i, j]) * x[i] * x[j] for i, j in problem.pairs
        )
    )
    model.add_constraint(
        model.sum(float(problem.mu[i]) * x[i] for i in range(n))
        >= float(target) * return_span,
        ctname="normalised_return_floor",
    )
    model.parameters.mip.tolerances.mipgap = 0.0
    model.parameters.mip.tolerances.absmipgap = 0.0
    solution = model.solve(log_output=False)
    if solution is None:
        model.end()
        return None, None
    bits = np.rint([solution.get_value(v) for v in x]).astype(np.int8)
    value = float(solution.objective_value) / variance_span
    model.end()

    FLOOR_TOLERANCE = 1e-6
    achieved = float(bits @ problem.mu) / return_span
    if achieved < float(target) - FLOOR_TOLERANCE:
        raise RuntimeError(
            f"K={problem.k}: the returned portfolio earns {achieved:.9f} of the attainable "
            f"return and the floor was {float(target):.9f}"
        )
    return bits, value


def epsilon_constraint_front(problem, count: int = 100, targets=None) -> dict:
    """A sweep of evenly spaced normalised return floors. Returns a dict of arrays. CAUTION:
    the solutions are only weakly Pareto optimal, and `keep` drops the duplicates.
    """
    if targets is None:
        targets = np.linspace(0.0, 1.0, int(count))
    targets = np.asarray(targets, dtype=np.float64)

    rows, used = [], []
    for target in targets:
        bits, _variance = epsilon_constraint_optimum(problem, float(target))
        if bits is None:
            continue
        rows.append(bits)
        used.append(float(target))
    if not rows:
        raise RuntimeError(
            f"K={problem.k}: every epsilon-constraint solve was infeasible"
        )

    bits = np.array(rows, dtype=np.int8)
    values = selection_values(problem, bits.astype(bool))
    return {
        "bits": bits,
        "targets": np.array(used),
        "returns": values["expected_return"],
        "variances": values["variance"],
        "returns_normalised": values["expected_return"]
        / float(-problem.return_cost_lo),
        "variances_normalised": values["variance"] / float(problem.variance_scale),
        "chosen": values["chosen"],
        "keep": nondominated(values["expected_return"], values["variance"]),
    }


##  ---------- Plotting functions ----------
# Depth carries colour AND marker, so the depths stay apart for a colour blind reader
# and in print. The 3 hues are the first 3 slots of a palette validated on all pairs.
DEPTH_COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
DEPTH_MARKERS = {1: "o", 2: "s", 3: "^"}


def depth_style(p: int):
    """The (colour, marker) of depth p, fixed by depth and never by position."""
    return (
        DEPTH_COLOURS.get(p, f"C{(int(p) - 1) % 10}"),
        DEPTH_MARKERS.get(p, "D"),
    )


# The labels of the 2 exact reference fronts, held as constants so no notebook has to
# retype them, the epsilon character especially.
WS_LABEL = "Weighted sum CPLEX"  # the weighted-sum scalarisation, solved exactly
EPS_LABEL = "\u03b5-constrained CPLEX"  # the epsilon-constraint sweep

CURVE_STYLES = {
    "random": ("#767570", ".", ":", 1.4, 6),
    WS_LABEL: ("#0b0b0b", "x", "--", 1.4, 8),
    EPS_LABEL: ("#4a3aa7", "D", "-.", 1.6, 7),
}


def curve_style(label: str):
    """The (colour, marker, linestyle, linewidth, zorder) of 1 curve, keyed by its label, so
    adding or dropping a curve never repaints the others.
    """
    if label in CURVE_STYLES:
        return CURVE_STYLES[label]
    if label.startswith("p="):
        return (*depth_style(int(label.split("=")[1])), "-", 2.0, 2)
    return ("#e34948", "v", (0, (3, 1, 1, 1, 1, 1)), 1.4, 5)


def reference_sets(cplex_bits):
    """Normalise the `cplex_bits` argument into {label: bits}: an array, a dict, or None."""
    if cplex_bits is None:
        return {}
    if isinstance(cplex_bits, dict):
        return {label: bits for label, bits in cplex_bits.items() if bits is not None}
    return {WS_LABEL: cplex_bits}


def pareto_depths(
    problem,
    summaries: dict,
    cplex_bits=None,
    random_summary=None,
    title: str = "",
    show_points: bool = True,
):
    """The measured fronts of several depths of 1 level on 1 axis, with the random and exact
    references. Each legend entry carries its hypervolume fraction.
    """
    import matplotlib.pyplot as plt

    reference, box = reference_point(problem), box_volume(problem)
    figure, axes = plt.subplots(figsize=(7.0, 4.8))
    axes.grid(True, linewidth=0.4, alpha=0.3, zorder=0)
    axes.set_axisbelow(True)

    def overlay(ret, variance, colour, marker, style, label, width, order_z):
        front = nondominated(ret, variance)
        order = np.argsort(variance[front])
        fraction = hypervolume(ret, variance, reference) / box
        if show_points:
            # An unfilled marker takes its colour from `c`, a filled 1 from its edge, so
            # the hollow ring that keeps overlapping points readable only applies to the
            # filled markers the depths use.
            hollow = marker not in "x+.|_"
            axes.scatter(
                variance,
                ret,
                marker=marker,
                s=16,
                linewidths=0.7,
                alpha=0.45,
                zorder=order_z,
                **(
                    {"facecolors": "none", "edgecolors": colour}
                    if hollow
                    else {"c": colour}
                ),
            )
        axes.plot(
            variance[front][order],
            ret[front][order],
            color=colour,
            linestyle=style,
            marker=marker,
            markersize=5.5,
            linewidth=width,
            zorder=order_z + 1,
            label=f"{label} — {int(front.sum())} nondominated, hv {fraction:.1%}",
        )

    for p in sorted(summaries):
        colour, marker, style, width, order_z = curve_style(f"p={p}")
        ret, variance = summary_points(summaries[p])
        overlay(ret, variance, colour, marker, style, f"QAOA $p={p}$", width, order_z)

    if random_summary is not None:
        colour, marker, style, width, order_z = curve_style("random")
        ret, variance = summary_points(random_summary)
        # A pooled front is drawn from every shot, so name the pool rather than 1 setting's shots.
        drawn = random_summary[0].get("pool") or random_summary[0]["shots"]
        overlay(
            ret,
            variance,
            colour,
            marker,
            style,
            f"best of {drawn:,} random",
            width,
            order_z,
        )
    for label, bits in reference_sets(cplex_bits).items():
        colour, marker, style, width, order_z = curve_style(label)
        values = selection_values(problem, np.asarray(bits, dtype=bool))
        overlay(
            values["expected_return"],
            values["variance"],
            colour,
            marker,
            style,
            label,
            width,
            order_z,
        )

    axes.set_xlabel(r"variance  $x^\top \Sigma x$")
    axes.set_ylabel(r"expected return  $\mu^\top x$")
    axes.set_title(title or "the measured trade-off curve, per QAOA depth")
    axes.legend(loc="lower right", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    return figure


def results_table(title: str, headers, rows, note: str = "") -> str:
    """1 titled fixed-width text table, each cell formatted by its type."""
    headers = [str(h) for h in headers]

    def cell(value):
        if isinstance(value, str):
            return value
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return "—" if not np.isfinite(value) else f"{float(value):.4f}"
        return str(value)

    body = [[cell(v) for v in row] for row in rows]
    width = [
        max(len(headers[i]), *(len(r[i]) for r in body)) if body else len(headers[i])
        for i in range(len(headers))
    ]

    def line(values):
        out = values[0].ljust(width[0])
        for i in range(1, len(values)):
            out += "  " + values[i].rjust(width[i])
        return out.rstrip()

    parts = [title, "-" * max(len(title), len(line(headers)))]
    parts.append(line(headers))
    parts.append("  ".join("-" * w for w in width))
    parts.extend(line(r) for r in body)
    if note:
        parts.extend(["", note])
    return "\n".join(parts) + "\n"


def pareto_depth_table(problem, summaries: dict, cplex_bits=None, random_summary=None):
    """The numbers behind `pareto_depths`, 1 row per curve, matching what its legend prints."""
    reference, box = reference_point(problem), box_volume(problem)
    rows = []

    def row(label, ret, variance):
        front = nondominated(ret, variance)
        high, low = int(np.argmax(ret)), int(np.argmin(variance))
        rows.append(
            {
                "curve": label,
                "points": int(ret.size),
                "nondominated": int(front.sum()),
                "hypervolume": hypervolume(ret, variance, reference),
                "hv_fraction": hypervolume(ret, variance, reference) / box,
                "most_return": float(ret[high]),
                "variance_there": float(variance[high]),
                "least_variance": float(variance[low]),
                "return_there": float(ret[low]),
            }
        )

    for p in sorted(summaries):
        row(f"p={p}", *summary_points(summaries[p]))
    if random_summary is not None:
        row("random", *summary_points(random_summary))
    for label, bits in reference_sets(cplex_bits).items():
        values = selection_values(problem, np.asarray(bits, dtype=bool))
        row(label, values["expected_return"], values["variance"])
    return rows


def reference_point(problem):
    """The worst corner of the objective box, (0, variance_scale), fixed by the instance and
    not by the data.
    """
    return 0.0, float(problem.variance_scale)


def box_volume(problem) -> float:
    """The area of that box: the hypervolume a perfect front would fill."""
    return float(-problem.return_cost_lo) * float(problem.variance_scale)


def hypervolume(returns, variances, reference) -> float:
    """The dominated area from moocore, return maximised and variance minimised. Divide by
    `box_volume` before comparing levels.
    """
    import moocore

    points = np.column_stack(
        [np.asarray(returns, dtype=np.float64), np.asarray(variances, dtype=np.float64)]
    )
    return float(
        moocore.hypervolume(
            points,
            ref=[float(reference[0]), float(reference[1])],
            maximise=[True, False],
        )
    )


def summary_points(summary: list[dict]):
    """The (return, variance) arrays of a summary list."""
    return (
        np.array([item["best_return"] for item in summary]),
        np.array([item["best_variance"] for item in summary]),
    )


def hypervolume_figure(fractions: dict, title: str = ""):
    """Bars of the dominated fraction of the box, 1 group per level."""
    import matplotlib.pyplot as plt

    labels = list(fractions)
    levels = sorted({level for item in fractions.values() for level in item})
    colours = {
        "hardware": "crimson",
        **{
            label: curve_style(label)[0]
            for label in [*(f"p={p}" for p in DEPTH_COLOURS), *CURVE_STYLES]
        },
    }
    spot = np.arange(len(levels), dtype=np.float64)
    width = 0.8 / len(labels)

    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    for index, label in enumerate(labels):
        height = [fractions[label].get(level, 0.0) for level in levels]
        bars = axes.bar(
            spot + (index - (len(labels) - 1) / 2) * width,
            height,
            width,
            label=label,
            color=colours.get(label, f"C{index}"),
            alpha=0.85 if label != WS_LABEL else 0.6,
        )
        axes.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
    axes.set_xticks(spot, [f"K={level}" for level in levels])
    axes.set_ylabel("hypervolume / box area")
    axes.set_ylim(0.0, 1.0)
    axes.set_title(title or "the dominated fraction of the objective box")
    axes.legend(loc="lower right", fontsize=9)
    figure.tight_layout()
    return figure


K_ALPHA = {0.90: 1.7550, 0.95: 2.0627, 0.99: 2.6652}
CVAR_ALPHAS = (0.90, 0.95, 0.99)


def cvar_multiplier(alpha: float) -> float:
    """k_alpha of 1 confidence level, from `K_ALPHA`."""
    for known, value in K_ALPHA.items():
        if abs(float(alpha) - float(known)) < 1e-12:
            return float(value)
    raise KeyError(
        f"alpha={alpha} carries no tabulated multiplier; K_ALPHA holds {sorted(K_ALPHA)}"
    )


def cvar(returns, variances, alpha: float) -> np.ndarray:
    """-mu' x + k_alpha sqrt(x' Sigma x). LOWER is better. Not a weighted sum, but its
    minimiser always lies on the Pareto front.
    """
    returns = np.asarray(returns, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    if np.any(variances < 0.0):
        raise ValueError(
            "a variance is negative, so its standard deviation is not real"
        )
    return -returns + cvar_multiplier(alpha) * np.sqrt(variances)


def cvar_curves(
    problem,
    summaries: dict,
    cplex_bits=None,
    random_summary=None,
    include_shift: bool = True,
) -> dict:
    """The candidates of every solution set, ready to be priced. `include_shift` decides
    whether the variance carries the delta * I repair.
    """
    delta = 0.0 if include_shift else float(problem.diagonal_shift)
    curves = {}

    def add(label, ret, variance, chosen, indices):
        curves[label] = {
            "returns": np.asarray(ret, dtype=np.float64),
            "variances": np.asarray(variance, dtype=np.float64)
            - delta * np.asarray(chosen, dtype=np.float64),
            "chosen": np.asarray(chosen, dtype=np.float64),
            "indices": list(indices),
        }

    for p in sorted(summaries):
        ret, variance = summary_points(summaries[p])
        add(
            f"p={p}",
            ret,
            variance,
            [item["best_chosen"] for item in summaries[p]],
            [int(item["weight_index"]) for item in summaries[p]],
        )
    if random_summary is not None:
        ret, variance = summary_points(random_summary)
        add(
            "random",
            ret,
            variance,
            [item["best_chosen"] for item in random_summary],
            [int(item["weight_index"]) for item in random_summary],
        )
    for label, reference in reference_sets(cplex_bits).items():
        bits = np.asarray(reference, dtype=bool)
        values = selection_values(problem, bits)
        add(
            label,
            values["expected_return"],
            values["variance"],
            values["chosen"],
            range(bits.shape[0]),
        )
    return curves


def cvar_table(
    problem,
    summaries: dict,
    cplex_bits=None,
    random_summary=None,
    alphas=CVAR_ALPHAS,
    include_shift: bool = True,
    min_assets: int = 0,
) -> list[dict]:
    """The best CVaR of every (curve, alpha). `min_assets` drops the empty portfolio, which
    otherwise wins trivially. CAUTION: the sets differ in size.
    """
    rows = []
    for label, curve in cvar_curves(
        problem, summaries, cplex_bits, random_summary, include_shift
    ).items():
        ret, variance = curve["returns"], curve["variances"]
        # A negative variance has no standard deviation, and a portfolio below `min_assets`
        # is not being asked about. Neither is priced.
        keep = (variance >= 0.0) & (curve["chosen"] >= float(min_assets))
        front = nondominated(ret, variance) if keep.all() else np.zeros(ret.size, bool)
        if not keep.all() and keep.any():
            front[keep] = nondominated(ret[keep], variance[keep])
        for alpha in alphas:
            if not keep.any():
                rows.append(
                    {
                        "curve": label,
                        "alpha": float(alpha),
                        "k_alpha": cvar_multiplier(alpha),
                        "cvar": float("nan"),
                        "median_cvar": float("nan"),
                        "worst_cvar": float("nan"),
                        "weight_index": -1,
                        "expected_return": float("nan"),
                        "variance": float("nan"),
                        "std": float("nan"),
                        "num_assets": -1,
                        "on_front": False,
                        "candidates": 0,
                        "excluded": int(ret.size),
                    }
                )
                continue
            score = cvar(ret[keep], variance[keep], alpha)
            where = np.flatnonzero(keep)
            best = int(where[int(np.argmin(score))])
            rows.append(
                {
                    "curve": label,
                    "alpha": float(alpha),
                    "k_alpha": cvar_multiplier(alpha),
                    "cvar": float(score.min()),
                    "median_cvar": float(np.median(score)),
                    "worst_cvar": float(score.max()),
                    "weight_index": curve["indices"][best],
                    "expected_return": float(ret[best]),
                    "variance": float(variance[best]),
                    "std": float(np.sqrt(variance[best])),
                    "num_assets": int(curve["chosen"][best]),
                    "on_front": bool(front[best]),
                    "candidates": int(keep.sum()),
                    "excluded": int((~keep).sum()),
                }
            )
    return rows


def cvar_points(
    problem,
    summaries: dict,
    alpha: float,
    cplex_bits=None,
    random_summary=None,
    include_shift: bool = True,
    min_assets: int = 0,
) -> dict:
    """Every solution set in the (CVaR, return) plane, at 1 alpha. That front is a subset of
    the Pareto front.
    """
    curves = {}
    for label, curve in cvar_curves(
        problem, summaries, cplex_bits, random_summary, include_shift
    ).items():
        ret, variance = curve["returns"], curve["variances"]
        keep = (variance >= 0.0) & (curve["chosen"] >= float(min_assets))
        if not keep.any():
            continue
        curves[label] = {
            "returns": ret[keep],
            "cvars": cvar(ret[keep], variance[keep], alpha),
            "variances": variance[keep],
            "chosen": curve["chosen"][keep],
            "indices": [curve["indices"][i] for i in np.flatnonzero(keep)],
            "excluded": int((~keep).sum()),
        }
    return curves


def cvar_constraint_table(
    problem,
    summaries: dict,
    c_grid,
    alphas=CVAR_ALPHAS,
    cplex_bits=None,
    random_summary=None,
    include_shift: bool = True,
    min_assets: int = 0,
) -> dict:
    """CVaR as a budget c: what each c admits (count, mean and max return), with
    `binds_from`, `binds_to` and `feasible_from`.
    """
    c_grid = np.asarray(c_grid, dtype=np.float64)
    out = {}
    for alpha in alphas:
        points = cvar_points(
            problem,
            summaries,
            alpha,
            cplex_bits,
            random_summary,
            include_shift,
            min_assets,
        )
        for label, curve in points.items():
            ret, score = curve["returns"], curve["cvars"]
            mean = np.full(c_grid.size, np.nan)
            best = np.full(c_grid.size, np.nan)
            count = np.zeros(c_grid.size, dtype=np.int64)
            for index, budget in enumerate(c_grid):
                admitted = score <= budget
                count[index] = int(admitted.sum())
                if count[index]:
                    mean[index] = float(ret[admitted].mean())
                    best[index] = float(ret[admitted].max())
            feasible = np.flatnonzero(count > 0)
            out[(label, float(alpha))] = {
                "c": c_grid,
                "mean_return": mean,
                "max_return": best,
                "count": count,
                "total": int(ret.size),
                "binds_from": float(score.min()) if score.size else float("nan"),
                "binds_to": float(score.max()) if score.size else float("nan"),
                "feasible_from": (
                    float(c_grid[feasible[0]]) if feasible.size else float("nan")
                ),
            }
    return out


def cvar_constraint_panels(
    problem,
    summaries: dict,
    c_grid,
    quantity: str = "count",
    alphas=CVAR_ALPHAS,
    cplex_bits=None,
    random_summary=None,
    include_shift: bool = True,
    min_assets: int = 0,
    title: str = "",
    log_counts: bool = True,
):
    """1 panel per alpha of count or max return against the budget c.
    Where a budget admits nothing, the line is left blank rather than drawn at 0.
    """

    import matplotlib.pyplot as plt

    if quantity not in ("count", "max_return"):
        raise ValueError(f"quantity is 'count' or 'max_return', not {quantity!r}")
    table = cvar_constraint_table(
        problem,
        summaries,
        c_grid,
        alphas,
        cplex_bits,
        random_summary,
        include_shift,
        min_assets,
    )
    alphas = list(alphas)
    labels = []
    for label, _alpha in table:
        if label not in labels:
            labels.append(label)

    figure, row = plt.subplots(
        1, len(alphas), figsize=(4.6 * len(alphas), 4.2), sharey=True, sharex=True
    )
    row = np.atleast_1d(row)
    for axes, alpha in zip(row, alphas):
        axes.grid(True, linewidth=0.4, alpha=0.3)
        axes.set_axisbelow(True)
        for label in labels:
            one = table[(label, float(alpha))]
            colour, marker, style, width, order_z = curve_style(label)
            if quantity == "count":
                # 0 is "nothing admitted", which a log axis cannot draw and which should read as
                # a gap on a linear 1 as well.
                y = np.where(one["count"] > 0, one["count"], np.nan).astype(np.float64)
                legend = f"{label}  (of {one['total']})"
            else:
                y = one["max_return"]
                legend = label
            axes.plot(
                one["c"],
                y,
                color=colour,
                linestyle=style,
                linewidth=width,
                zorder=order_z,
                label=legend,
            )
        if quantity == "count" and log_counts:
            axes.set_yscale("log")
        axes.set_title(
            rf"$\alpha={alpha:g}$,  $k={cvar_multiplier(alpha):.4f}$", fontsize=10
        )
        axes.set_xlabel(r"risk budget $c$,  admitting $\mathrm{CVaR}_\alpha(x) \leq c$")
    row[0].set_ylabel(
        "portfolios admitted"
        if quantity == "count"
        else r"best available return  $\max\ \mu^\top x$"
    )
    row[-1].legend(loc="lower right", fontsize=8, framealpha=0.9)
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def cvar_figure(rows: list[dict], title: str = "", reference: str = WS_LABEL):
    """Bars of the best CVaR of each set, 1 group per alpha, with a dashed reference line in
    each.
    """
    import matplotlib.pyplot as plt

    labels, alphas = [], []
    for row in rows:  # keep the order `cvar_table` produced
        if row["curve"] not in labels:
            labels.append(row["curve"])
        if row["alpha"] not in alphas:
            alphas.append(row["alpha"])
    score = {(row["curve"], row["alpha"]): row["cvar"] for row in rows}

    spot = np.arange(len(alphas), dtype=np.float64)
    width = 0.8 / len(labels)
    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    axes.grid(True, axis="y", linewidth=0.4, alpha=0.3)
    axes.set_axisbelow(True)

    for index, label in enumerate(labels):
        colour = curve_style(label)[0]
        height = [
            score[(label, alpha)] if np.isfinite(score[(label, alpha)]) else 0.0
            for alpha in alphas
        ]
        bars = axes.bar(
            spot + (index - (len(labels) - 1) / 2) * width,
            height,
            width * 0.92,  # the gap is the 2px surface spacer between bars
            label=label,
            color=colour,
            alpha=0.6 if label == WS_LABEL else 0.9,
        )
        axes.bar_label(bars, fmt="%.2f", fontsize=7, padding=1, rotation=90)

    if reference in labels:
        for index, alpha in enumerate(alphas):
            axes.plot(
                [spot[index] - 0.45, spot[index] + 0.45],
                [score[(reference, alpha)]] * 2,
                color="#0b0b0b",
                linewidth=1.0,
                linestyle="--",
                zorder=5,
            )

    axes.set_xticks(
        spot,
        [rf"$\alpha={alpha:g}$,  $k={cvar_multiplier(alpha):.4f}$" for alpha in alphas],
    )
    axes.set_ylabel(
        r"CVaR $= -\mu^\top x + k_\alpha \sqrt{x^\top \Sigma x}$" "\n(lower is better)"
    )
    finite = [value for value in score.values() if np.isfinite(value)]
    if not finite:
        raise ValueError(
            "no set holds a priceable candidate, so there is nothing to draw"
        )
    axes.set_ylim(min(0.0, min(finite) * 1.18), max(finite) * 1.18)
    axes.set_title(title or "the best CVaR each solution set contains")
    axes.legend(loc="upper left", fontsize=8, ncols=min(3, len(labels)))
    figure.tight_layout()
    return figure
