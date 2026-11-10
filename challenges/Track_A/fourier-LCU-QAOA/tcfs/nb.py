"""Tables and figures of the notebook (v0.4, design Sec. 5.3).

Display only: every function prints a compact table and/or draws one
matplotlib figure, and returns the numbers behind them. Participant
functions passed in (``basis_circuit``, ``fourier_lcu``,
``evaluate_counts``, ``cvar``) are called inside ``try``: on an exception
one line is printed and the function returns ``None``.

Returned numbers are :class:`Report` dicts (or :class:`ReportList`). Their
content was printed when they were made, so Jupyter does not display them
a second time when a helper is the last line of a cell; ``print(r)`` or
``dict(r)`` shows them. One colour is used for ideal (exact) values and one
for hardware values throughout.

The static figures of the notebook's markdown (``instance_figure``,
``method_figure``, ``warm_start_figure``, ``fourier_figure``) are written
as PNG files by the notebook builder when ``path`` is given; they use only
kit code, never participant functions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import exact
from . import hardware as hw
from .cvar import cvar as _kit_cvar
from .instances import DensestKSubgraph

IDEAL_COLOR = "C0"
HARDWARE_COLOR = "C1"
ALPHA = 0.10
SLOT_MIN = 20          # one registration per 20 minutes (v0.4 event client only)
PLAN_BUDGET_S = 1800.0  # Exercise 2d: the 30 minutes of QPU time of one event account
PLAN_MIN_SHOTS = 1000   # a graded job has at least 1,000 shots in total
EXACT_MAX_N = 20       # the Part 1 helpers simulate exactly
FIG_DPI = 110          # the static PNG figures embedded in markdown
FIG_MAX_BYTES = 80_000  # larger PNGs are stored with a 256-colour palette


def _small(inst, name: str) -> bool:
    if inst.n <= EXACT_MAX_N:
        return True
    print(f"{name} simulates the instance exactly and is meant for the "
          f"12-qubit training instance (this one has n = {inst.n}).")
    return False


class Report(dict):
    """Numbers returned by a notebook helper (a plain dict that Jupyter
    does not display again: its content was printed already)."""

    def _ipython_display_(self):
        pass


class ReportList(list):
    """A list of rows returned by a notebook helper (not displayed again)."""

    def _ipython_display_(self):
        pass


# ------------------------------------------------------------------ helpers
def _plt():
    import matplotlib.pyplot as plt

    return plt


def _show(fig) -> None:
    """Display ``fig`` once in a Jupyter kernel, then close it (no second
    copy at the end of the cell); outside a kernel just close it."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
    except ImportError:
        ip = None
    if ip is not None and getattr(ip, "kernel", None) is not None:
        from IPython.display import display

        display(fig)
    _plt().close(fig)


def _save_png(fig, path) -> str:
    """Write ``fig`` to ``path`` as a PNG (``FIG_DPI``, white background,
    tight box) and close it. A file above ``FIG_MAX_BYTES`` is stored with
    a 256-colour palette instead (Pillow; kept as it is without Pillow)."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor="white",
                bbox_inches="tight")
    _plt().close(fig)
    data = buf.getvalue()
    if len(data) > FIG_MAX_BYTES:
        try:
            from PIL import Image
        except ImportError:
            Image = None
        if Image is not None:
            img = Image.open(io.BytesIO(data)).convert("RGB").quantize(
                colors=256, method=Image.Quantize.MEDIANCUT)
            out = io.BytesIO()
            img.save(out, format="PNG", optimize=True)
            data = min(data, out.getvalue(), key=len)
    Path(path).write_bytes(data)
    return str(path)


def _finish(fig, path):
    """Save ``fig`` as PNG to ``path`` when given, otherwise show it (both
    close it). Returns the path written, or None."""
    if path is None:
        _show(fig)
        return None
    return _save_png(fig, path)


class _Stop(Exception):
    pass


def _one_line(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text[:200] + ("..." if len(text) > 200 else "")


def _call(label: str, fn, *args):
    """Call participant code; print one line and stop on an exception."""
    try:
        return fn(*args)
    except NotImplementedError:
        print(f"{label} is not implemented yet (it raises NotImplementedError).")
    except Exception as exc:                     # noqa: BLE001 - participant code
        print(f"{label} raised {type(exc).__name__}: {_one_line(exc)}")
    raise _Stop


def _table(rows, columns, index=None, floatfmt="{:.4f}", na_rep="NaN") -> str:
    import pandas as pd

    df = pd.DataFrame(rows, columns=columns)
    if index is not None:
        df = df.set_index(index)
        df.index.name = None
    with pd.option_context("display.float_format", floatfmt.format,
                           "display.width", 120):
        return df.to_string(na_rep=na_rep)


def _tail_weights(values, weights, alpha: float) -> np.ndarray:
    """The part of each weight inside the best-``alpha`` tail of the
    distribution (the boundary sample counts with the part still missing):
    the weights that ``CVaR_alpha`` averages."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    order = np.argsort(-v, kind="stable")
    before = np.concatenate([[0.0], np.cumsum(w[order])[:-1]])
    tail = np.zeros_like(w)
    tail[order] = np.clip(alpha - before, 0.0, w[order])
    return tail


# ---------------------------------------------------------- the problem
def instance_figure(inst: DensestKSubgraph, path=None, seed: int = 3):
    """Draw the instance with one optimal k-subset highlighted (nodes and
    the edges inside it); save it as PNG to ``path`` when given (the
    notebook builder embeds that file), otherwise show it. Returns the
    subset, its number of edges and the path."""
    import networkx as nx

    plt = _plt()
    if inst.n <= EXACT_MAX_N:
        opt, subsets = inst.brute_force()
        subset = set(subsets[0])
    else:                                  # no enumeration at utility scale
        opt, subset = float("nan"), set()
    g = inst.graph
    pos = nx.kamada_kawai_layout(g) if inst.n <= 40 else nx.spring_layout(g, seed=seed)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    inside = [(i, j) for i, j in g.edges() if i in subset and j in subset]
    other = [e for e in g.edges() if e not in inside]
    nx.draw_networkx_edges(g, pos, edgelist=other, ax=ax, edge_color="0.7", width=1.2)
    nx.draw_networkx_edges(g, pos, edgelist=inside, ax=ax, edge_color=HARDWARE_COLOR,
                           width=3.0)
    colors = [HARDWARE_COLOR if v in subset else "white" for v in g.nodes()]
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, edgecolors="0.2",
                           node_size=420)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9)
    ax.set_axis_off()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        _show(fig)
    return Report({"path": None if path is None else str(path),
                   "subset": sorted(int(v) for v in subset),
                   "edges_inside": len(inside), "opt": float(opt),
                   "n": inst.n, "k": inst.k, "edges": len(inst.edges)})


def _example_instance() -> DensestKSubgraph:
    """The 4-node path with k = 2 of ``draw_example`` and ``method_figure``."""
    return DensestKSubgraph(n=4, k=2, edges=((0, 1, 1.0), (1, 2, 1.0),
                                             (2, 3, 1.0)), name="path-4")


# the Qiskit drawer sizes its gate boxes from a fixed character table, so
# larger fonts than its defaults (13, 8) fill the boxes without widening them
_CIRCUIT_STYLE = {"fontsize": 18, "subfontsize": 14}
_SYMBOLS = {"gamma": "γ", "theta": "θ", "beta": "β"}
_DIAGONAL = {"rz", "rzz"}


def _angle_label(expr):
    """Display name of a multiple of one parameter: ``3γ``, ``-γ/2``, ``θ``
    (None for anything else)."""
    from fractions import Fraction

    params = list(expr.parameters)
    if len(params) != 1:
        return None
    p = params[0]
    try:
        a = float(expr.gradient(p))
        offset = float(expr.bind({p: 0.0}))
    except (TypeError, ValueError):
        return None
    frac = Fraction(a).limit_denominator(16)
    if abs(offset) > 1e-12 or abs(float(frac) - a) > 1e-9 or frac == 0:
        return None
    sym = _SYMBOLS.get(p.name, p.name)
    num = abs(frac.numerator)
    return (("-" if frac < 0 else "") + ("" if num == 1 else str(num)) + sym
            + ("" if frac.denominator == 1 else f"/{frac.denominator}"))


def _pack_two_qubit(instructions, index):
    """Commuting diagonal gates reordered for a compact drawing: the
    two-qubit gates first, greedily in columns of disjoint wire spans, then
    the one-qubit gates in their order."""
    two = [ci for ci in instructions if len(ci.qubits) == 2]
    one = [ci for ci in instructions if len(ci.qubits) != 2]

    def span(ci):
        a, b = sorted(index[q] for q in ci.qubits)
        return a, b

    columns: list = []
    for ci in sorted(two, key=lambda ci: (span(ci)[1] - span(ci)[0], span(ci))):
        a, b = span(ci)
        for col in columns:
            if all(b < lo or a > hi for lo, hi in map(span, col)):
                col.append(ci)
                break
        else:
            columns.append([ci])
    return [ci for col in columns for ci in col] + one


def _drawing_copy(qc, blocks):
    """A copy of ``qc`` for drawing. ``blocks`` = [(label, size), ...] splits
    its instructions into consecutive layers, separated by barriers labelled
    with the name of the layer that follows. The drawing copy has no global
    phase, no gate of angle 0, each angle a*parameter written as a short
    name (``-3γ``), and the gates of an all-diagonal layer packed into as
    few columns as possible (they commute). The unitary is unchanged up to
    the global phase."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterExpression

    out = QuantumCircuit(qc.num_qubits)
    index = {q: i for i, q in enumerate(qc.qubits)}
    names: dict = {}

    def shown(x):
        if not isinstance(x, ParameterExpression):
            return x
        try:
            return float(x)                    # bound, or 0 * parameter
        except (TypeError, ValueError):
            pass
        label = _angle_label(x)
        return x if label is None else names.setdefault(label, Parameter(label))

    data, start = list(qc.data), 0
    if sum(size for _, size in blocks) != len(data):
        blocks = [("", len(data))]
    for label, size in blocks:
        layer = data[start:start + size]
        start += size
        if label:
            out.barrier(label=label)
        if layer and all(ci.operation.name in _DIAGONAL for ci in layer):
            layer = _pack_two_qubit(layer, index)
        for ci in layer:
            op = ci.operation
            params = [shown(x) for x in op.params]
            if (op.name in _DIAGONAL | {"rx", "ry"} and params
                    and all(isinstance(x, float) and abs(x) < 1e-12 for x in params)):
                continue                           # the identity: not drawn
            op = op.to_mutable()
            op.params = params
            out.append(op, [index[q] for q in ci.qubits])
    return out


def method_figure(path=None):
    """The method in one picture, on the 4-node path with k = 2 of
    ``draw_example``: the coherent penalty-QAOA circuit
    (``circuits.coherent_penalty_circuit``), whose penalty block acts on
    every pair of qubits, above the LCU basis circuit of paper Fig. 3
    (``circuits.penalty_basis_circuit``), where that block is one layer of
    ``rz(theta)``. Both are drawn at the same scale; commuting diagonal
    gates are packed into columns and zero angles left out. Saves a PNG to
    ``path`` when given, otherwise shows the figure. Returns the two-qubit
    gate counts of both circuits."""
    from qiskit.circuit import Parameter

    from . import circuits as circ

    plt = _plt()
    inst = _example_instance()
    gamma, theta, beta = Parameter("gamma"), Parameter("theta"), Parameter("beta")
    coherent = circ.coherent_penalty_circuit(inst, gamma, beta)
    basis = circ.penalty_basis_circuit(inst, gamma, theta, beta)
    size = {"warm": len(circ.warm_start_layer(inst).data),
            "cost": len(circ.cost_layer(inst, 1.0).data),
            "penalty": len(circ.penalty_layer(inst, 1.0).data),
            "lcu": len(circ.rz_phase_layer(inst.n, 1.0).data),
            "mixer": len(circ.warm_start_mixer_layer(inst, 1.0).data)}
    n2q = [sum(1 for ci in qc.data if _two_qubit(ci)) for qc in (coherent, basis)]
    pairs = inst.n * (inst.n - 1) // 2
    drawings = [
        _drawing_copy(coherent, [("", size["warm"]), ("cost", size["cost"]),
                                 ("penalty", size["penalty"]), ("mixer", size["mixer"])]),
        _drawing_copy(basis, [("", size["warm"]), ("cost", size["cost"]),
                              ("LCU phase", size["lcu"]), ("mixer", size["mixer"])])]
    titles = [f"Coherent penalty QAOA ({inst.n}-node path, k = {inst.k}, λ = {inst.lam:g}): "
              f"{n2q[0]} two-qubit gates, {pairs} of them for the penalty",
              f"LCU basis circuit (paper Fig. 3): {n2q[1]} two-qubit gates, "
              f"the penalty replaced by one layer of RZ(θ)"]
    natural = []                               # the drawer's own size, scale 1
    for d in drawings:
        f = d.draw("mpl", fold=-1, style=_CIRCUIT_STYLE)
        natural.append(tuple(f.get_size_inches()))
        plt.close(f)
    width, title_h = 8.6, 0.42
    s = width / max(w for w, _ in natural)     # one scale for both drawings
    height = sum(s * h + title_h for _, h in natural)
    fig = plt.figure(figsize=(width, height))
    top = height
    for (w, h), d, title in zip(natural, drawings, titles):
        top -= title_h + s * h
        ax = fig.add_axes((0.0, top / height, s * w / width, s * h / height))
        d.draw("mpl", fold=-1, style=_CIRCUIT_STYLE, ax=ax)
        ax.set_title(title, loc="left", fontsize=10)
    written = _finish(fig, path)
    return Report({"path": written, "n": inst.n, "k": inst.k, "lam": inst.lam,
                   "edges": len(inst.edges), "pairs": pairs,
                   "coherent_2q": n2q[0], "basis_2q": n2q[1]})


def warm_start_figure(inst: DensestKSubgraph, path=None):
    """Distribution of the Hamming weight W of the two candidate initial
    states: the uniform superposition (binomial, p = 1/2) and the warm start
    (binomial, p = k/n), with k marked and P(W = k) of both in the legend.
    Saves a PNG to ``path`` when given, otherwise shows the figure. Returns
    both distributions and both probabilities of W = k."""
    plt = _plt()
    n, k = inst.n, inst.k
    ws = np.arange(n + 1)
    p = k / n
    uniform = np.array([math.comb(n, w) for w in ws], dtype=float) / 2.0 ** n
    warm = np.array([math.comb(n, w) * p ** w * (1.0 - p) ** (n - w) for w in ws])
    shown = np.flatnonzero((uniform > 1e-4) | (warm > 1e-4))

    def percent(x):
        return f"{100 * x:.1f} %" if x >= 1e-3 else f"{100 * x:.2g} %"

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.bar(ws - 0.2, uniform, width=0.38, color="0.62",
           label=f"uniform, p = 1/2: P(W = k) = {percent(uniform[k])}")
    ax.bar(ws + 0.2, warm, width=0.38, color=IDEAL_COLOR,
           label=f"warm start, p = k/n: P(W = k) = {percent(warm[k])}")
    ax.axvline(k, color="0.3", linestyle="--", linewidth=1, zorder=0)
    ax.set_xlim(shown[0] - 0.7, shown[-1] + 0.7)
    top = max(uniform.max(), warm.max())
    ax.set_ylim(0, 1.45 * top)
    ax.text(k, 1.42 * top, f" k = {k}", ha="left", va="top", fontsize=9, color="0.3")
    if n <= 20:
        ax.set_xticks(ws)
    ax.set_xlabel("Hamming weight W of the measured bitstring")
    ax.set_ylabel("probability")
    ax.set_title(f"Initial state: each qubit is 1 with probability p (n = {n}, k = {k})",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    written = _finish(fig, path)
    return Report({"path": written, "n": n, "k": k,
                   "p_k_uniform": float(uniform[k]), "p_k_warm": float(warm[k]),
                   "uniform": uniform.tolist(), "warm": warm.tolist()})


def fourier_figure(inst: DensestKSubgraph, gamma: float, path=None):
    """The Fourier LCU of the penalty phase with the kit's reference
    (``tcfs.diagonal_lcu``), in two panels: (a) the target
    u(W) = exp(+i gamma lam (W - k)^2) at W = 0..n (markers) and the Fourier
    sum over b of c_b exp(i theta_b w) for continuous w (lines), which
    passes through the target at every integer; (b) the magnitudes |c_b|,
    with Gamma = (sum_b |c_b|)^2 and its bound n + 1. Saves a PNG to
    ``path`` when given, otherwise shows the figure. Returns Gamma, the
    bound, |c_b| and the largest deviation of the sum from u at the
    integers."""
    from . import diagonal_lcu as dlcu

    plt = _plt()
    n, k, lam = inst.n, inst.k, inst.lam
    phases = dlcu.cardinality_penalty_phases(gamma, lam, k, n)
    c = dlcu.fourier_coefficients(phases)
    thetas = dlcu.dft_angles(n + 1)
    Gamma = dlcu.gamma(c)
    ws = np.arange(n + 1)
    u = np.exp(1j * phases)
    w_cont = np.linspace(0.0, n, 60 * n + 1)
    fourier = np.exp(1j * np.outer(w_cont, thetas)) @ c
    at_integers = np.exp(1j * np.outer(ws, thetas)) @ c
    err = float(np.max(np.abs(at_integers - u)))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.2),
                                 gridspec_kw={"width_ratios": [1.7, 1.0]})
    ax.plot(w_cont, fourier.real, color=IDEAL_COLOR, linewidth=1.2,
            label="real part: sum")
    ax.plot(ws, u.real, "o", color=IDEAL_COLOR, markersize=5.5,
            markeredgecolor="white", label="real part: u(W)")
    ax.plot(w_cont, fourier.imag, color="0.6", linewidth=1.2, label="imaginary part: sum")
    ax.plot(ws, u.imag, "s", color="0.35", markersize=5,
            markeredgecolor="white", label="imaginary part: u(W)")
    lim = float(np.max(np.abs(np.concatenate([fourier.real, fourier.imag]))))
    ax.set_ylim(-1.05 * lim, 1.75 * lim)       # headroom for the legend
    ax.set_xlabel("Hamming weight W")
    ax.set_ylabel("real, imaginary part")
    if n <= 20:
        ax.set_xticks(ws)
    ax.set_title(r"(a) $u(W) = e^{i\gamma\lambda(W-k)^2}$ at integer W (markers) and "
                 r"$\Sigma_b\, c_b\, e^{i\theta_b w}$ (lines)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    bx.bar(np.arange(n + 1), np.abs(c), color=IDEAL_COLOR, width=0.7)
    bx.set_xlabel("branch b")
    bx.set_ylabel(r"$|c_b|$")
    if n <= 20:
        bx.set_xticks(np.arange(0, n + 1, 2 if n > 10 else 1))
    bx.set_title(rf"(b) $\Gamma = (\Sigma_b |c_b|)^2$ = {Gamma:.2f} ≤ n + 1 = {n + 1}",
                 fontsize=10)
    fig.text(0.01, 0.01, f"γ = {gamma:.4g}, λ = {lam:g}, k = {k}, n = {n}",
             fontsize=8, color="0.4")
    fig.tight_layout()
    written = _finish(fig, path)
    return Report({"path": written, "gamma": float(gamma), "Gamma": Gamma,
                   "bound": n + 1, "abs_c": np.abs(c).tolist(), "max_error": err})


def draw_example(fn, scale: float = 0.7, **draw_kwargs):
    """Call the participant's ``basis_circuit`` on a 4-node path with k = 2
    and return its matplotlib drawing (the four layers are legible at this
    size). Returns None and prints one line if the function fails."""
    from qiskit import QuantumCircuit

    path4 = _example_instance()
    try:
        qc = _call("basis_circuit(path instance)", fn, path4)
    except _Stop:
        return None
    if not isinstance(qc, QuantumCircuit):
        print(f"basis_circuit returned {type(qc).__name__}, not a QuantumCircuit.")
        return None
    try:
        return qc.draw("mpl", fold=-1, scale=scale, **draw_kwargs)
    except Exception as exc:                     # noqa: BLE001 - drawing backends
        print(f"the circuit could not be drawn: {_one_line(exc)}")
        return None


def operator_summary(objective, penalized):
    """Number of qubits, Z terms and ZZ terms of the objective operator and
    of the penalized operator (both from ``to_ising``), and the sentence
    that the penalty couples every pair of qubits."""
    from .mapping import z_terms, zz_terms

    rows, out = [], {}
    for name, op in (("objective f", objective), ("penalized f_lambda", penalized)):
        try:
            z, zz = z_terms(op), zz_terms(op)
            n = int(op.num_qubits)
        except Exception as exc:                 # noqa: BLE001 - participant object
            print(f"{name}: not a Pauli operator ({_one_line(exc)})")
            return None
        rows.append((name, n, len(z), len(zz)))
        out[name.split()[0]] = {"qubits": n, "Z_terms": len(z), "ZZ_terms": len(zz)}
    print(_table(rows, ["operator", "qubits", "Z terms", "ZZ terms"], index="operator"))
    n = rows[1][1]
    pairs = n * (n - 1) // 2
    print(f"The penalty couples every pair of qubits: the penalized operator has "
          f"{rows[1][3]} ZZ terms, all {pairs} = {n}*{n - 1}/2 pairs; the objective "
          f"has one per edge ({rows[0][3]}).")
    out["all_pairs"] = pairs
    return Report(out)


# ------------------------------------------------------------ Exercise 1b
def _circuit_distribution(circuit, values, n):
    """Measured-bit distribution of a participant's parametrized basis
    circuit at (gamma, theta, beta) (the machinery of the 1a check)."""
    from .checks import _bound_distribution, _measurement_positions, _named_parameters

    pos, problem = _measurement_positions(circuit, n)
    if problem:
        raise ValueError(problem)
    return _bound_distribution(circuit, _named_parameters(circuit), values, pos, n)


def _overhead_curve(inst, fourier_lcu, gamma: float, points: int = 1001):
    """The overhead Gamma returned by ``fourier_lcu`` over one period
    2 pi / lam of the penalty phase, centred on the multiple of 2 pi / lam
    nearest to ``gamma`` (there gamma * lam is a multiple of 2 pi, the
    penalty is the identity and Gamma = 1). Returns (gammas, Gammas,
    centre, period)."""
    period = 2.0 * np.pi / inst.lam
    centre = period * round(gamma / period)
    half = np.linspace(0.0, period / 2.0, points)
    gammas = centre + np.concatenate([-half[:0:-1], half])
    return (gammas, np.array([float(fourier_lcu(inst, float(g))[3]) for g in gammas]),
            centre, period)


# rows of lcu_report and the experiments of the paper's Table I they reproduce
_LCU_ROWS = (("coherent", "coherent circuit (penalty block)", "1"),
             ("ensemble", "LCU ensemble ({m} branches)", "2"),
             ("single", "single trained circuit (kit angles)", "4"))
_ROW_FIELDS = ("exp_H", "CVaR", "P_feasible", "P_optimal")


def lcu_report(inst: DensestKSubgraph, circuit, fourier_lcu, gamma: float,
               beta: float):
    """The Fourier LCU of the participant, run exactly: the n+1 branches of
    ``circuit`` at ``theta_b`` (``Statevector``) pooled with the
    participant's weights. Prints Gamma and its bound n+1, the smallest
    Gamma * p_ens(x) / p(x) over all bitstrings (>= 1 is the guarantee of
    paper Sec. II B, p from the coherent circuit), and a table of the mean
    of f_lam, its upper-tail CVaR at level 1/Gamma (Gamma of the ensemble,
    for all rows), P(feasible) and P(optimal) for the coherent circuit, the
    ensemble and a single trained circuit (the kit's angles, re-optimized in
    the setting of Table I experiment 4, ``event.TABLE_I["single"]``); at the
    Table I parameters (``event.TABLE_I["ensemble"]``) each row is followed
    by the paper's values (``grader.PAPER_TABLE_I``). One figure: (a) the
    weights q_b; (b) every bitstring as a point (p(x), p~(x)), all on or
    above the line p~ = p / Gamma; (c) Gamma of ``fourier_lcu`` over one
    period 2 pi / lam of gamma, with the bound n+1."""
    from .event import TABLE_I
    from .grader import PAPER_TABLE_I

    if not _small(inst, "lcu_report"):
        return None
    m = inst.n + 1
    try:
        out = _call(f"fourier_lcu(inst, {gamma:.4g})", fourier_lcu, inst, gamma)
        thetas, c, weights, Gamma = out
        thetas = np.asarray(thetas, dtype=float)
        q = np.asarray(weights, dtype=float)
        q = q / q.sum()
        Gamma = float(Gamma)
        if not (np.isfinite(Gamma) and Gamma >= 1.0 - 1e-9):
            print(f"Gamma = {Gamma} returned by fourier_lcu: the overhead "
                  f"(sum_b |c_b|)^2 is a number >= 1.")
            return None
        dists = []
        for t in thetas:
            dists.append(_call("the basis circuit", _circuit_distribution, circuit,
                               (gamma, float(t), beta), inst.n))
        s = TABLE_I["single"]
        single = _call("the basis circuit", _circuit_distribution, circuit,
                       (s["gamma"], s["theta"], s["beta"]), inst.n)
        gammas, Gammas, centre, period = _call(
            "fourier_lcu over one period of gamma", _overhead_curve, inst, fourier_lcu,
            gamma)
    except (_Stop, TypeError, ValueError) as exc:
        if not isinstance(exc, _Stop):
            print(f"fourier_lcu must return (thetas, c, weights, Gamma): {_one_line(exc)}")
        return None
    pooled = exact.pooled_distribution(np.array(dists), q)
    coherent = np.abs(exact.coherent_penalty_state(inst, beta, gamma)) ** 2
    support = coherent > 1e-12
    ratio = float(np.min(Gamma * pooled[support] / coherent[support]))
    alpha = 1.0 / max(Gamma, 1.0)
    flam = inst.penalized_table()
    verdict = ">= 1: the guarantee holds" if ratio >= 1 - 1e-9 else "below 1: check the weights"
    print(f"Gamma = {Gamma:.4f} (bound n+1 = {m}); smallest Gamma * "
          f"p_ens(x) / p(x) over all x = {ratio:.4f} ({verdict}).")
    at_paper = bool(np.allclose((gamma, beta), TABLE_I["ensemble"], atol=1e-3))
    rows, res, paper = [], {}, {}
    for key, label, expno in _LCU_ROWS:
        dist = {"coherent": coherent, "ensemble": pooled, "single": single}[key]
        met = exact.distribution_metrics(inst, dist)
        res[key] = {"exp_H": met["exp_H"], "CVaR": float(_kit_cvar(flam, dist, alpha)),
                    "P_feasible": met["P_feasible"], "P_optimal": met["P_optimal"]}
        rows.append((label.format(m=m), *(res[key][f] for f in _ROW_FIELDS)))
        if at_paper:
            paper[key] = dict(PAPER_TABLE_I[expno])
            rows.append((f"  paper, Table I exp. {expno}",
                         *(paper[key].get(f, np.nan) for f in _ROW_FIELDS)))
    print(_table(rows, ["", "mean f_lam", "CVaR_1/Gamma", "P(feasible)", "P(optimal)"],
                 index="", na_rep=""))
    print(f"CVaR_1/Gamma: the mean of f_lam over the best 1/Gamma = {100 * alpha:.2f} % of "
          f"the probability mass.")
    if at_paper:
        g3 = float(PAPER_TABLE_I["4"]["Gamma"])
        paper["single"]["alpha"] = 1.0 / g3
        res["single"]["CVaR_paper_level"] = float(_kit_cvar(flam, single, 1.0 / g3))
        print(f"Exp. 4 of the paper is at level 1/{g3:.4f} (Gamma of its exp. 3), where "
              f"this circuit has {res['single']['CVaR_paper_level']:.4f}.")
    else:
        print("(the paper's values are shown at the parameters of Table I, "
              "event.TABLE_I['ensemble'])")

    plt = _plt()
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(10.4, 3.3),
                                     gridspec_kw={"width_ratios": [1.0, 1.1, 1.1]})
    ax.bar(np.arange(m), q, color=IDEAL_COLOR, width=0.7)
    ax.set_xlabel("branch b")
    ax.set_ylabel("probability of branch b")
    ax.set_xticks(np.arange(0, m, 2 if m > 10 else 1))
    ax.set_title(r"(a) branch weights $q_b$", fontsize=10)

    p, pt = coherent[support], pooled[support]
    bx.loglog(p, pt, ".", color=IDEAL_COLOR, markersize=2.5, alpha=0.5,
              label=f"one bitstring x (all {support.sum()})")
    xs = np.array([p.min() / 2.0, p.max() * 2.0])
    bx.loglog(xs, xs / Gamma, color="0.15", linewidth=1.2,
              label=r"$\tilde p = p\,/\,\Gamma$")
    bx.set_xlim(*xs)
    bx.set_ylim(min(pt.min(), xs[0] / Gamma) / 2.0, pt.max() * 3.0)
    bx.set_xlabel(r"$p(x)$, coherent circuit")
    bx.set_ylabel(r"$\tilde p(x)$, LCU ensemble")
    bx.set_title(r"(b) every x lies on or above $\tilde p = p\,/\,\Gamma$", fontsize=10)
    bx.legend(frameon=False, fontsize=8, loc="lower right")

    at_centre = float(Gammas[len(Gammas) // 2])
    cx.plot(gammas, Gammas, color=IDEAL_COLOR, linewidth=0.9)
    cx.axhline(m, color="0.3", linestyle="--", linewidth=1)
    cx.text(gammas[0], m + 0.25, f" n + 1 = {m}", fontsize=8, color="0.3", va="bottom")
    cx.plot([gamma], [Gamma], "o", color="0.15", markeredgecolor="white", markersize=7)
    cx.plot([centre], [at_centre], "o", color="white", markeredgecolor="0.15",
            markersize=6)
    cx.annotate(f"this γ: Γ = {Gamma:.2f}", (gamma, Gamma),
                xytext=(gammas[-1], m + 0.25), fontsize=8, ha="right", va="bottom",
                arrowprops={"arrowstyle": "-", "color": "0.5", "linewidth": 0.8})
    cx.annotate(f" Γ = {at_centre:.0f} where γλ is\n a multiple of 2π", (centre, at_centre),
                xytext=(centre + 0.03 * period, at_centre), fontsize=8, ha="left",
                va="center")
    cx.set_ylim(0, m + 2.6)
    cx.set_xlim(gammas[0], gammas[-1])
    cx.set_xlabel("γ")
    cx.set_ylabel("Γ")
    cx.set_title("(c) Γ over one period 2π/λ of γ", fontsize=10)
    fig.tight_layout()
    _show(fig)
    return Report({"Gamma": Gamma, "bound": m, "alpha": alpha, "min_ratio": ratio,
                   "weights": q.tolist(), **res, "paper": paper if at_paper else None,
                   "Gamma_curve": {"gamma": gammas.tolist(), "Gamma": Gammas.tolist(),
                                   "centre": float(centre), "period": float(period),
                                   "max": float(Gammas.max()), "at_centre": at_centre}})


# ------------------------------------------------------------ Exercise 1c
_COHERENT_CACHE: dict = {}


def _two_qubit(ci) -> bool:
    return len(ci.qubits) == 2 and ci.operation.name not in ("barrier", "measure")


def _line_of(target, n: int):
    """A simple path of ``n`` working physical qubits in the Target's
    coupling graph (depth-first search), or None."""
    bad_q, bad_e = hw.faulty_from_target(target)
    nbrs: dict = {}
    for a, b in target.build_coupling_map().get_edges():
        if a in bad_q or b in bad_q or (a, b) in bad_e or (b, a) in bad_e:
            continue
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)

    if n > len(nbrs):
        return None
    budget = [20_000]                 # search steps: a lattice needs about n

    def extend(path, seen):
        if len(path) == n:
            return path
        for q in sorted(nbrs[path[-1]] - seen):
            budget[0] -= 1
            if budget[0] < 0:
                return None
            out = extend(path + [q], seen | {q})
            if out is not None:
                return out
        return None

    for start in sorted(nbrs, key=lambda q: (-len(nbrs[q]), q)):
        path = extend([start], {start})
        if path is not None or budget[0] < 0:
            return path
    return None


def coherent_swap_strategy_isa(inst: DensestKSubgraph, target,
                               seed_transpiler: int = 0):
    """ISA circuit of the coherent penalty-QAOA circuit, routed along a line
    of physical qubits with a swap strategy for commuting two-qubit gates:
    Qiskit's ``Commuting2qGateRouter`` with the ``qopt-best-practices``
    passes (``PrepareCostLayer``, ``SwapToFinalMapping``,
    ``QAOAConstructionPass``) as the ``pre_init`` stage of the level-3 preset
    pass manager. This is the router made for a block that couples every
    pair of qubits. Parameters of the result: the package's ``γ[0]`` and
    ``β[0]``. Returns None when ``qopt-best-practices`` is not installed or
    the Target has no line of ``inst.n`` working qubits."""
    try:
        from qopt_best_practices.transpilation.cost_layer import get_cost_layer
        from qopt_best_practices.transpilation.prepare_cost_layer import PrepareCostLayer
        from qopt_best_practices.transpilation.qaoa_construction_pass import (
            QAOAConstructionPass)
        from qopt_best_practices.transpilation.swap_cancellation_pass import (
            SwapToFinalMapping)
    except ImportError:
        return None
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import CXGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler import PassManager, generate_preset_pass_manager
    from qiskit.transpiler.passes import HighLevelSynthesis, InverseCancellation
    from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
        Commuting2qGateRouter, SwapStrategy)

    from . import mapping

    n, t0 = inst.n, inst.theta_init
    line = _line_of(target, n)
    if line is None:
        return None
    op = mapping.penalized_operator(inst)        # H1 + H2 up to a constant
    # all terms commute; PrepareCostLayer needs the ZZ terms first
    order = sorted(range(len(op)), key=lambda i: -int(op.paulis[i].z.sum()))
    op = SparsePauliOp(op.paulis[order], op.coeffs[order])
    init = QuantumCircuit(n)
    init.ry(t0, range(n))
    beta = Parameter("beta")
    mixer = QuantumCircuit(n)
    mixer.ry(-t0, range(n))
    mixer.rz(-2 * beta, range(n))
    mixer.ry(t0, range(n))
    pm = generate_preset_pass_manager(optimization_level=3, target=target,
                                      initial_layout=line,
                                      seed_transpiler=seed_transpiler)
    pm.pre_init = PassManager([
        PrepareCostLayer(),
        Commuting2qGateRouter(SwapStrategy.from_line(list(range(n))),
                              {(i, i + 1): i % 2 for i in range(n - 1)}),
        SwapToFinalMapping(),
        HighLevelSynthesis(basis_gates=["x", "cx", "sx", "rz", "id"]),
        InverseCancellation(gates_to_cancel=[CXGate()]),
        QAOAConstructionPass(num_layers=1, init_state=init, mixer_layer=mixer)])
    return pm.run(get_cost_layer(op))


def cost_comparison(inst: DensestKSubgraph, backend, isa):
    """Cost table of Exercise 1c: the participant's ISA circuit against the
    circuit with the coherent penalty block
    (``circuits.coherent_penalty_circuit``) for the same backend, transpiled
    (a) with the preset pass manager at optimization level 3, seed 0, and
    (b) with a swap strategy (:func:`coherent_swap_strategy_isa`; this row
    needs ``qopt-best-practices`` and is left out without it). Columns:
    two-qubit gates before transpilation, after transpilation, and
    two-qubit depth."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.transpiler import generate_preset_pass_manager

    from . import circuits as circ

    if not isinstance(isa, QuantumCircuit):
        print(f"isa is {type(isa).__name__}, not a QuantumCircuit.")
        return None
    if not _small(inst, "cost_comparison"):
        return None
    target = getattr(backend, "target", backend)
    key = (id(target), hw._instance_key(inst))
    if key not in _COHERENT_CACHE:
        qc = circ.coherent_penalty_circuit(inst, Parameter("gamma"), Parameter("beta"),
                                           measure=True)
        _COHERENT_CACHE[key] = generate_preset_pass_manager(
            optimization_level=3, target=target, seed_transpiler=0).run(qc)
        try:
            swap = coherent_swap_strategy_isa(inst, target)
        except Exception as exc:                 # noqa: BLE001 - optional row
            print(f"(swap-strategy row left out: {type(exc).__name__}: "
                  f"{_one_line(exc)})")
            swap = None
        _COHERENT_CACHE[key + ("swap",)] = swap
    coh, swap = _COHERENT_CACHE[key], _COHERENT_CACHE[key + ("swap",)]
    n_pairs = inst.n * (inst.n - 1) // 2

    def row(label, before, circuit):
        return (label, before, sum(1 for ci in circuit.data if _two_qubit(ci)),
                circuit.depth(_two_qubit))

    rows = [row("your ISA circuit (one LCU branch)", len(inst.edges), isa),
            row("coherent penalty circuit, preset pass manager",
                len(inst.edges) + n_pairs, coh)]
    out = {"isa": dict(zip(("before", "after", "depth_2q"), rows[0][1:])),
           "coherent": dict(zip(("before", "after", "depth_2q"), rows[1][1:]))}
    if swap is not None:
        # an edge and the penalty act on the same pair: one block per pair
        rows.append(row("coherent penalty circuit, swap strategy", n_pairs, swap))
        out["coherent_swap"] = dict(zip(("before", "after", "depth_2q"), rows[2][1:]))
    print(_table(rows, ["", "2q gates before", "2q gates after (cz)", "2q depth"],
                 index=""))
    if swap is None:
        print("(install qopt-best-practices to add the swap-strategy row)")
    else:
        print(f"The swap-strategy row applies the cost and penalty terms of a qubit pair "
              f"as one gate: {n_pairs} gates before transpilation, one per pair, against "
              f"{len(inst.edges)} + {n_pairs} in the row above.")
    return Report(out)


# ------------------------------------------------------------ Exercise 1d
def _hardware_rows(inst, counts_list, weights, evaluate_counts, cvar):
    """Pooled hardware metrics with the participant's functions."""
    opt = inst.brute_force()[0]
    vals, wts, feas = [], [], []
    for b, counts in enumerate(counts_list):
        out = _call("evaluate_counts", evaluate_counts, inst, counts)
        v, f, s = (np.asarray(x) for x in out)
        s = s.astype(float)
        vals.append(v.astype(float))
        feas.append(f.astype(bool))
        wts.append(float(weights[b]) * s / s.sum())
    v, f, w = np.concatenate(vals), np.concatenate(feas), np.concatenate(wts)
    w = w / w.sum()
    cv = float(_call("cvar", cvar, v, w, ALPHA))
    return {"P_feasible": float(w[f].sum()),
            "P_optimal": float(w[f & np.isclose(v, opt)].sum()),
            "CVaR": cv, "best": float(v.max()), "values": v, "weights": w}


def _ideal_rows(inst, dist):
    flam = inst.penalized_table()
    met = exact.distribution_metrics(inst, dist)
    return {"P_feasible": met["P_feasible"], "P_optimal": met["P_optimal"],
            "CVaR": float(_kit_cvar(flam, dist, ALPHA)),
            "best": float(flam[dist > 1e-12].max())}


def part1_results(inst: DensestKSubgraph, ensemble_counts, weights, single_counts,
                  evaluate_counts, cvar, ensemble=None, single=None):
    """Four-row table - ensemble ideal / hardware, single circuit ideal /
    hardware - with P(feasible), P(optimal), CVaR_0.10 and the best value
    found, and one histogram of f_lambda for the hardware ensemble against
    the ideal one, in which the best 10 % of the hardware probability mass
    (what CVaR_0.10 averages) is shaded and the hardware CVaR_0.10 and mean
    are marked. Hardware rows use the participant's ``evaluate_counts`` and
    ``cvar``; ideal rows are exact at ``ensemble`` = (gamma, beta) and
    ``single`` = {"gamma", "theta", "beta"} (default: ``event.TABLE_I``)."""
    from . import optimize
    from .event import TABLE_I

    if not _small(inst, "part1_results"):
        return None
    g_e, b_e = TABLE_I["ensemble"] if ensemble is None else ensemble
    s = TABLE_I["single"] if single is None else single
    try:
        q = np.asarray(weights, dtype=float)
        if len(ensemble_counts) != len(q):
            print(f"{len(ensemble_counts)} counts dictionaries for {len(q)} weights.")
            return None
        hw_e = _hardware_rows(inst, ensemble_counts, q, evaluate_counts, cvar)
        hw_s = _hardware_rows(inst, [single_counts], [1.0], evaluate_counts, cvar)
    except _Stop:
        return None
    except (TypeError, ValueError) as exc:
        print(f"the counts could not be evaluated: {_one_line(exc)}")
        return None
    _, thetas, q_ref, _ = optimize.penalty_lcu(inst, g_e)
    ideal_e = exact.pooled_distribution(
        np.array([np.abs(exact.penalty_basis_state(inst, b_e, g_e, t)) ** 2
                  for t in thetas]), q_ref)
    ideal_s = np.abs(exact.penalty_basis_state(inst, s["beta"], s["gamma"],
                                               s["theta"])) ** 2
    rows, res = [], {}
    for label, key, r in (("ensemble, ideal", "ensemble_ideal", _ideal_rows(inst, ideal_e)),
                          ("ensemble, hardware", "ensemble_hardware", hw_e),
                          ("single circuit, ideal", "single_ideal", _ideal_rows(inst, ideal_s)),
                          ("single circuit, hardware", "single_hardware", hw_s)):
        rows.append((label, r["P_feasible"], r["P_optimal"], r["CVaR"], r["best"]))
        res[key] = {k: r[k] for k in ("P_feasible", "P_optimal", "CVaR", "best")}
    print(_table(rows, ["", "P(feasible)", "P(optimal)", "CVaR_0.10", "best value"],
                 index=""))
    flam = inst.penalized_table()
    v_hw, w_hw = hw_e["values"], hw_e["weights"]
    mean_hw = float(np.dot(v_hw, w_hw))
    tail = _tail_weights(v_hw, w_hw, ALPHA)
    lo = int(np.floor(min(flam[ideal_e > 1e-9].min(), v_hw.min())))
    hi = int(np.ceil(max(flam.max(), v_hw.max())))
    lo = min(max(lo, hi - 40), int(np.floor(mean_hw)) - 1)   # the mean stays in view
    grid = np.arange(lo, hi + 1)
    p_ideal = np.array([ideal_e[np.isclose(flam, x)].sum() for x in grid])
    p_hw = np.array([w_hw[np.isclose(v_hw, x)].sum() for x in grid])
    p_tail = np.array([tail[np.isclose(v_hw, x)].sum() for x in grid])
    cv_hw = hw_e["CVaR"]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    headroom = 1.5                      # the top third of the axes holds the legend
    handles = [
        ax.bar(grid - 0.2, p_ideal, width=0.38, color=IDEAL_COLOR, label="ideal ensemble"),
        ax.bar(grid + 0.2, p_hw, width=0.38, color=HARDWARE_COLOR, alpha=0.4,
               label="hardware ensemble"),
        ax.bar(grid + 0.2, p_tail, width=0.38, color=HARDWARE_COLOR,
               label=f"hardware: best {100 * ALPHA:.0f} % of the mass"),
        ax.axvline(cv_hw, ymax=1.1 / headroom, color="0.15", linewidth=1.3,
                   label=f"hardware CVaR$_{{0.10}}$ = {cv_hw:.2f} (scored)"),
        ax.axvline(mean_hw, ymax=1.1 / headroom, color="0.15", linewidth=1.3,
                   linestyle="--", label=f"hardware mean = {mean_hw:.2f} (not scored)")]
    ax.set_xlabel(r"$f_\lambda(x)$")
    ax.set_ylabel("probability")
    ax.set_ylim(0, headroom * max(p_ideal.max(), p_hw.max()))
    ax.set_title(f"Scored: CVaR$_{{0.10}}$, the mean of the best {100 * ALPHA:.0f} % of "
                 f"the hardware mass", fontsize=10)
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    _show(fig)
    res["ensemble_hardware"]["mean"] = mean_hw
    return Report(res)


# -------------------------------------------------------------- Part 2
def _grid_heatmap(fig, ax, grid, gamma: float, beta: float) -> None:
    """The ideal <f> of the ``lagrangian_layers`` grid over (beta, gamma),
    with the chosen point."""
    gammas = sorted({r["gamma"] for r in grid})
    betas = sorted({r["beta"] for r in grid})
    z = np.full((len(gammas), len(betas)), np.nan)
    for r in grid:
        if r.get("exp_f") is not None:
            z[gammas.index(r["gamma"]), betas.index(r["beta"])] = r["exp_f"]
    dg = (gammas[-1] - gammas[0]) / max(1, len(gammas) - 1) / 2
    db = (betas[-1] - betas[0]) / max(1, len(betas) - 1) / 2
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="viridis",
                   extent=(betas[0] - db, betas[-1] + db,
                           gammas[0] - dg, gammas[-1] + dg))
    ax.plot([beta], [gamma], marker="x", color="white", markersize=10, mew=2)
    ax.set_xlabel("β")
    ax.set_ylabel("γ")
    fig.colorbar(im, ax=ax, label=r"ideal $\langle f\rangle$ at $\langle W\rangle$ = target")


def _theta_panels(aw, af, prof, info, k: int) -> None:
    """Ideal <W> and <f> against theta at the chosen (gamma, beta), with the
    target weight and the chosen theta marked (the last entry of ``prof``
    is the chosen theta)."""
    th, t = prof["theta"][:-1], float(info["theta"])
    target = float(info.get("target", k))
    for a, key, name in ((aw, "exp_W", r"ideal $\langle W\rangle$"),
                         (af, "exp_f", r"ideal $\langle f\rangle$")):
        a.plot(th, prof[key][:-1], color=IDEAL_COLOR, linewidth=1.4)
        a.axvline(t, color="0.6", linewidth=0.8, zorder=0)
        a.plot([t], [prof[key][-1]], "o", color=IDEAL_COLOR, markeredgecolor="white",
               markersize=7)
        a.set_ylabel(name)
        a.set_xlim(0.0, 2 * np.pi)
    aw.axhline(target, color="0.3", linestyle="--", linewidth=1, zorder=0)
    aw.text(0.03, target, f" {'k' if target == k else 'target'} = {target:g}", fontsize=8,
            color="0.3", va="bottom", transform=aw.get_yaxis_transform())
    aw.set_title(f"θ sets the mean weight; chosen θ = {t:.3f} (dot)", fontsize=10)
    ticks = np.arange(5) * np.pi / 2
    for a in (aw, af):
        a.set_xticks(ticks)
    aw.set_xticklabels([])
    af.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    af.set_xlabel("θ")


def expectations(info: dict, inst: DensestKSubgraph | None = None):
    """Print the parameters and ideal moments returned by
    ``hw.lagrangian_layers`` and draw the grid of ideal <f> (at the theta
    that puts the ideal <W> on the target) with the chosen point. With
    ``inst`` (the instance of ``info``) a second panel shows the ideal <W>
    and <f> as functions of theta over [0, 2 pi) at the chosen (gamma,
    beta) (``hw.theta_profile``), with the target weight and the chosen
    theta marked."""
    try:
        g, t, b = info["gamma"], info["theta"], info["beta"]
        print(f"layers = [(gamma, theta, beta)] = [({g:.4f}, {t:.4f}, {b:.4f})]")
        print(f"ideal <f> = {info['exp_f']:.2f}, <W> = {info['exp_W']:.2f} (target "
              f"{info['target']:g}), sd(W) = {info['sd_W']:.2f}, P(W = k) ~ "
              f"{info['p_feasible']:.3f} (Gaussian estimate)")
    except (KeyError, TypeError) as exc:
        print(f"info is not the dict returned by hw.lagrangian_layers ({exc}).")
        return None
    grid = info.get("grid") or []
    prof = None
    if inst is not None:
        try:          # 360 angles over [0, 2 pi), then the chosen theta
            prof = hw.theta_profile(inst, g, b,
                                    np.append(np.arange(360) * np.pi / 180, float(t)))
        except Exception as exc:                 # noqa: BLE001 - participant object
            print(f"the theta profile cannot be computed for this instance: "
                  f"{_one_line(exc)}")
    if grid or prof is not None:
        plt = _plt()
        if prof is None:
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            _grid_heatmap(fig, ax, grid, g, b)
        else:
            fig = plt.figure(figsize=(10.0 if grid else 5.0, 3.6))
            gs = fig.add_gridspec(2, 2 if grid else 1,
                                  width_ratios=[1.15, 1.0] if grid else None)
            if grid:
                ax = fig.add_subplot(gs[:, 0])
                _grid_heatmap(fig, ax, grid, g, b)
                ax.set_title("ideal ⟨f⟩ over the (β, γ) grid; x: the chosen point",
                             fontsize=10)
            aw = fig.add_subplot(gs[0, -1])
            _theta_panels(aw, fig.add_subplot(gs[1, -1], sharex=aw), prof, info,
                          inst.k)
        fig.tight_layout()
        _show(fig)
    out = {k: v for k, v in info.items() if k != "grid"}
    if prof is not None:
        out["theta_profile"] = {k: prof[k][:-1].tolist() for k in ("theta", "exp_W", "exp_f")}
        out["theta_profile"]["at_theta"] = {"exp_W": float(prof["exp_W"][-1]),
                                            "exp_f": float(prof["exp_f"][-1])}
    return Report(out)


def circuit_cost(isa, backend, shots: int = 2 ** 15):
    """One-row cost table of an ISA circuit on ``backend``: CZ count,
    two-qubit depth, depth, scheduled duration and the estimated QPU time
    of a job with this one circuit and ``shots`` shots."""
    from .event import job_usage_estimate

    target = getattr(backend, "target", backend)
    try:
        row = hw.circuit_cost(isa, target)
    except Exception as exc:                     # noqa: BLE001 - participant object
        print(f"the cost of this circuit cannot be computed: {_one_line(exc)}")
        return None
    row["job_usage_s"] = job_usage_estimate(1, shots, row["duration_us"] * 1e-6)
    print(_table([("ISA circuit", row["cz"], row["depth_2q"], row["depth"],
                   row["duration_us"], row["job_usage_s"])],
                 ["", "cz", "2q depth", "depth", "duration (us)",
                  f"QPU s per job ({shots} shots)"], index="", floatfmt="{:.1f}"))
    return Report(row)


def weight_histogram(inst: DensestKSubgraph, counts, ideal: dict | None = None):
    """Histogram of the Hamming weight W of the measured bitstrings, with k
    marked (and the ideal Gaussian of ``ideal`` = the ``lagrangian_layers``
    info when given). Prints the mean and spread of W, the feasible
    fraction, the mean f among feasible samples and the best feasible f."""
    try:
        bits, cnt = hw.bits_from_counts(counts, inst.n)
    except Exception as exc:                     # noqa: BLE001 - participant object
        print(f"counts must be a Qiskit counts dictionary of {inst.n}-bit strings "
              f"({_one_line(exc)}).")
        return None
    f, w = inst.evaluate_bits(bits)
    st = hw.weight_stats(inst, counts)
    feas = w == inst.k
    mean_f = float(np.dot(f[feas], cnt[feas]) / cnt[feas].sum()) if feas.any() else float("nan")
    best = float(f[feas].max()) if feas.any() else float("nan")
    print(f"mean W {st.mean:.2f} (k = {inst.k}), sd(W) {st.sd:.2f}, feasible fraction "
          f"{st.feasible:.4f}, mean f of feasible samples {mean_f:.2f}, best feasible "
          f"f {best:.0f}")
    grid = np.arange(int(w.min()), int(w.max()) + 1)
    prob = np.array([cnt[w == x].sum() for x in grid]) / cnt.sum()
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.bar(grid, prob, color=HARDWARE_COLOR, label="measured")
    if ideal is not None and "exp_W" in ideal and "sd_W" in ideal:
        xs = np.linspace(grid[0] - 1, grid[-1] + 1, 200)
        mu, sd = float(ideal["exp_W"]), float(ideal["sd_W"])
        ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
                color=IDEAL_COLOR, label="ideal (Gaussian)")
    ax.axvline(inst.k, color="0.3", linestyle="--", linewidth=1, label="k")
    ax.set_xlabel("Hamming weight W")
    ax.set_ylabel("fraction of shots")
    ax.legend(frameon=False)
    fig.tight_layout()
    _show(fig)
    return Report({"mean_W": st.mean, "sd_W": st.sd, "F": st.feasible,
                   "mean_f_feasible": mean_f, "best_feasible": best,
                   "shots": int(cnt.sum())})


def _whole(v) -> int | None:
    """``v`` as an int when it is a whole number >= 1 (a whole float
    counts), else None."""
    if isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v) if v >= 1 else None
    if isinstance(v, (float, np.floating)) and math.isfinite(v) \
            and float(v).is_integer() and v >= 1:
        return int(v)
    return None


def _plan_row(item, i: int):
    """Row ``i`` of a plan as ``(label, circuits, shots, repeat, problems)``
    for display (a dict takes the defaults circuits 1, shots 2^15, repeat
    1); ``problems`` lists what the grader of Exercise 2d refuses in it.
    Raises TypeError or ValueError for a row that cannot be read."""
    if isinstance(item, dict):
        unknown = sorted(set(map(str, item)) - {"label", "circuits", "shots", "repeat"})
        label = item.get("label", f"job {i}")
        values = (item.get("circuits", 1), item.get("shots", 2 ** 15),
                  item.get("repeat", 1))
        problems = [f"the keys {unknown} are not read"] if unknown else []
        if "label" not in item:
            problems.append("no label")
    else:
        label, circ_, shots, *rest = item
        if len(rest) > 1:
            raise ValueError(f"row {i} holds {3 + len(rest)} values")
        values = (circ_, shots, rest[0] if rest else 1)
        problems = []
    if not isinstance(label, str) or not label.strip():
        problems.append("the label must be a non-empty text")
    whole = [_whole(v) for v in values]
    for name, v, w in zip(("circuits", "shots", "repeat"), values, whole):
        if w is None:
            problems.append(f"{name} must be a whole number of at least 1, got {v!r}")
    c, s, r = (int(v) for v in values)
    if None not in whole and c * s < PLAN_MIN_SHOTS:
        problems.append(f"{c} x {s} = {c * s} shots per job, fewer than "
                        f"{PLAN_MIN_SHOTS:,}")
    return str(label), c, s, r, problems


def plan(rows, budget_s: float = PLAN_BUDGET_S):
    """Exercise 2d: the QPU-time plan. ``rows`` is a list of planned jobs,
    each a tuple ``(label, circuits, shots[, repeat])`` or a dict
    ``{"label", "circuits", "shots", "repeat"}`` (``circuits`` = circuits or
    parameter sets per job, ``shots`` per circuit, ``repeat`` how many such
    jobs, default 1). Prints the estimated QPU seconds per row
    (``event.job_usage_estimate``), the number of jobs, the total and what
    is left of ``budget_s`` (default 1,800 s, the planning budget of the
    exercise), and a note for every row that the grader refuses (no label,
    a number that is not a whole number of at least 1, fewer than 1,000
    shots per job). Returns ``{"jobs", "qpu_s", "budget_s",
    "left_after_s", "rows", "refused"}``, or None when a row cannot be read.

    The v0.4 form ``plan(campaign, jobs)`` (a ``tcfs.event.Campaign``) is
    still accepted: it plans against the campaign's budget and counts
    20-minute registration slots."""
    if callable(getattr(rows, "budget", None)):
        return _plan_campaign(rows, budget_s)
    from .event import job_usage_estimate

    table, refused = [], []
    try:
        for i, item in enumerate(rows, 1):
            label, c, s, r, problems = _plan_row(item, i)
            each = job_usage_estimate(c, s)
            table.append((label, r, c, s, each, r * each))
            if problems:
                name = label.strip() or "without a label"
                refused.append(f"row {i} ({name}): " + "; ".join(problems))
    except (TypeError, ValueError) as exc:
        print("each planned job is (label, circuits, shots[, repeat]) or "
              f"{{'label', 'circuits', 'shots', 'repeat'}}: {_one_line(exc)}")
        return None
    budget_s = float(budget_s)
    if not table:
        print("The plan is empty: add one row per kind of job, (label, circuits or "
              "parameter sets per job, shots per circuit, number of such jobs). "
              f"Planning budget {budget_s:.0f} s.")
        return Report({"jobs": 0, "qpu_s": 0.0, "budget_s": budget_s,
                       "left_after_s": budget_s, "rows": [], "refused": []})
    total = float(sum(row[5] for row in table))
    n_jobs = int(sum(row[1] for row in table))
    print(_table(table, ["job", "repeat", "circuits", "shots", "QPU s each", "QPU s"],
                 index="job", floatfmt="{:.1f}"))
    print(f"Planned: {n_jobs} jobs, {total:.1f} s of QPU time; planning budget "
          f"{budget_s:.0f} s, left after the plan {budget_s - total:.1f} s.")
    for line in refused:
        print("The grader refuses " + line + ".")
    if total > budget_s:
        print("The grader refuses a plan above the planning budget: plan fewer "
              "jobs, circuits or shots.")
    return Report({"jobs": n_jobs, "qpu_s": total, "budget_s": budget_s,
                   "left_after_s": budget_s - total, "rows": table,
                   "refused": refused})


def _plan_campaign(campaign, jobs):
    """``plan(campaign, jobs)`` of notebook v0.4: the QPU-time and clock
    plan against a ``tcfs.event.Campaign``. ``jobs`` is a list of planned
    jobs, each a dict ``{"label", "circuits", "shots", "repeat"}`` or a
    tuple ``(label, circuits, shots[, repeat])`` (``circuits`` = circuits
    or parameter sets per job, ``shots`` per circuit). Prints the estimated
    QPU seconds per job (``event.job_usage_estimate``), the total, the
    number of 20-minute registration slots and what is left of the
    campaign's budget."""
    from .event import job_usage_estimate

    rows = []
    try:
        for item in jobs:
            if isinstance(item, dict):
                label = str(item.get("label", f"job {len(rows) + 1}"))
                circ_, shots = int(item.get("circuits", 1)), int(item.get("shots", 2 ** 15))
                rep = int(item.get("repeat", 1))
            else:
                label, circ_, shots, *rest = item
                circ_, shots = int(circ_), int(shots)
                rep = int(rest[0]) if rest else 1
            each = job_usage_estimate(circ_, shots)
            rows.append((label, rep, circ_, shots, each, rep * each))
    except (TypeError, ValueError) as exc:
        print("each planned job is {'label', 'circuits', 'shots', 'repeat'} or "
              f"(label, circuits, shots[, repeat]): {_one_line(exc)}")
        return None
    budget, spent, left = campaign.budget()
    if not rows:
        print("The plan is empty: add one row per kind of job, (label, circuits or "
              f"parameter sets per job, shots per circuit, number of such jobs). Budget "
              f"{budget:.0f} s, spent {spent:.1f} s, left {left:.1f} s.")
        return Report({"jobs": 0, "qpu_s": 0.0, "slots": 0, "hours": 0.0,
                       "left_after_s": left, "rows": []})
    total = float(sum(r[5] for r in rows))
    n_jobs = int(sum(r[1] for r in rows))
    rate = SLOT_MIN
    print(_table(rows, ["job", "repeat", "circuits", "shots", "QPU s each", "QPU s"],
                 index="job", floatfmt="{:.1f}"))
    print(f"Planned: {n_jobs} jobs, {total:.1f} s of QPU time; {n_jobs} registration "
          f"slots = {n_jobs * rate / 60:.1f} h at one per {rate:g} min. Budget "
          f"{budget:.0f} s, spent {spent:.1f} s, left after the plan "
          f"{left - total:.1f} s.")
    return Report({"jobs": n_jobs, "qpu_s": total, "slots": n_jobs,
                   "hours": n_jobs * rate / 60, "left_after_s": left - total,
                   "rows": rows})


# -------------------------------------------------------------- Part 3
def frontier_costs(inst: DensestKSubgraph, layout, backend, shots: int = 2 ** 15,
                   layers=None):
    """Cost of the Part 3 options on ``backend``: the basis circuit at depth
    p = 1, 2, 3 (``hw.hardware_basis_circuit_p`` with ``swap_last=True``
    through ``hw.resynth_pass_manager``) and a job with the full LCU
    ensemble (n+1 p = 1 circuits): CZ count, two-qubit depth, scheduled
    duration and estimated QPU seconds per job with ``shots`` shots per
    circuit. ``layers``: one (gamma, theta, beta) layer repeated p times
    (default: ``hw.lagrangian_layers(inst)``)."""
    from .event import job_usage_estimate

    target = getattr(backend, "target", backend)
    if layers is None:
        layers, _ = hw.lagrangian_layers(inst)
    layer = tuple(layers[0])
    pm = hw.resynth_pass_manager(target, layout.physical)
    rows, res, p1 = [], {}, None
    for p in (1, 2, 3):
        qc, _ = hw.hardware_basis_circuit_p(inst, layout, [layer] * p, swap_last=True)
        isa = pm.run(qc)
        c = hw.circuit_cost(isa, target)
        usage = job_usage_estimate(1, shots, c["duration_us"] * 1e-6)
        rows.append((f"p = {p}", 1, c["cz"], c["depth_2q"], c["duration_us"], usage))
        res[f"p{p}"] = {**c, "job_usage_s": usage}
        if p == 1:
            p1 = c
    m = inst.n + 1
    usage = job_usage_estimate(m, shots, p1["duration_us"] * 1e-6)
    rows.append(("LCU ensemble, p = 1", m, p1["cz"], p1["depth_2q"],
                 p1["duration_us"], usage))
    res["ensemble"] = {**p1, "circuits": m, "job_usage_s": usage}
    print(_table(rows, ["", "circuits", "cz per circuit", "2q depth",
                        "duration (us)", f"QPU s per job ({shots} shots each)"],
                 index="", floatfmt="{:.1f}"))
    return Report(res)
