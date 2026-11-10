"""Axis-T training toolbox: objectives, restarts, initializers, policies,
surrogate sub-instances.

Everything here trains on the *exact* statevector objective (``tcfs.exact``),
which is practical up to ~24 qubits: the 12-qubit worked example and
<= 24-qubit surrogate sub-instances of a larger instance (axis T of Part 3).
Maximization throughout. The same objectives score hardware counts through
:func:`objective_from_counts`, so a team's own training loop can swap the
exact evaluator for a Sampler without touching the optimizer code.

Parameter layout. A flat vector ``x`` is the concatenation of the layer
tuples in circuit order, in the circuit builders' argument order
(``tcfs.exact`` module docstring):

    penalty basis   (gamma, theta, beta)          per layer
    coherent        (gamma, beta)                 per layer
    XY basis        (gamma, alpha, theta, chi)    per layer

Periodic bounds per angle (``layer_bounds``): gamma (-pi, pi], theta
[0, 2 pi), beta (-pi/2, pi/2], alpha / chi [0, 2 pi). beta, theta, alpha,
chi are periodic for every instance; gamma is 2 pi-periodic only for integer
edge weights (and an integer lam), so :func:`random_restarts` wraps its
answer only when wrapping leaves the objective unchanged.

What the 12-qubit numbers showed (exact calculations on the seed-42
3-regular instance, k = 4, alpha = 0.10, target p = 2 CVaR 3.8355, Q 0.898,
from p = 1 CVaR 3.3192, Q 0.578):

    joint random restarts (30)          median 3.567, 1 of 30 hits 3.835
    warm start from p = 1 (small layer) stalls at 3.33 (0 of 30)
    scaled copy / INTERP from p = 1     3.33 - 3.40
    alpha schedule 0.5 -> 0.25 -> 0.10  median 3.33, best 3.834
    layer-wise (freeze 1, train 2)      3.730 layer-2-only (9 of 30), 3.8355
                                        after the joint polish
    training on <H> instead of CVaR     Q = 0.000 (CVaR 2.124)

The objective matters as much as the optimizer. One exact p = 2 CVaR
evaluation costs ~1.3 ms at 12 qubits (p = 1: ~1.0 ms, p = 3: ~5 ms).
"""

from __future__ import annotations

import time
from collections import namedtuple

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from . import exact
from .instances import DensestKSubgraph

TWO_PI = 2.0 * np.pi
ANGLE_NAMES = {
    "penalty": ("gamma", "theta", "beta"),
    "coherent": ("gamma", "beta"),
    "xy": ("gamma", "alpha", "theta", "chi"),
}
_ANGLE_BOUNDS = {
    "gamma": (-np.pi, np.pi),
    "theta": (0.0, TWO_PI),
    "beta": (-np.pi / 2, np.pi / 2),
    "alpha": (0.0, TWO_PI),
    "chi": (0.0, TWO_PI),
}
KINDS = ("cvar", "exp_H", "tail_proxy", "pooled_cvar")
POLICIES = ("restarts", "layerwise", "interp", "warm", "ramp", "schedule")
ALIAS_WEIGHT = 10.0   # pooled_cvar pays ALIAS_WEIGHT * max(0, gamma_min - Gamma)


# ------------------------------------------------------ parameter layout
def _width(family: str) -> int:
    if family not in ANGLE_NAMES:
        raise ValueError(f"family must be one of {tuple(ANGLE_NAMES)}, "
                         f"got {family!r}")
    return len(ANGLE_NAMES[family])


def layers_from_flat(x, family: str = "penalty") -> list[tuple]:
    """Flat vector -> list of layer tuples (circuit order)."""
    w = _width(family)
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0 or x.size % w:
        raise ValueError(f"a {family} parameter vector has a multiple of {w} "
                         f"entries (>= {w}), got {x.size}")
    return [tuple(float(a) for a in x[i:i + w]) for i in range(0, x.size, w)]


def flat_from_layers(layers) -> np.ndarray:
    """List of layer tuples -> flat vector (inverse of layers_from_flat)."""
    return np.array([float(a) for layer in layers for a in layer])


def depth_of(x, family: str = "penalty") -> int:
    """p = number of layers encoded in the flat vector ``x``."""
    return len(layers_from_flat(x, family))


def layer_bounds(family: str = "penalty", p: int = 1) -> list[tuple]:
    """Per-angle periodic bounds [(lo, hi), ...] for a depth-p flat vector
    (plain 2-tuples, so they also feed ``optimize.grid_then_cobyla``)."""
    _width(family)
    if p < 1:
        raise ValueError("p must be >= 1")
    return [_ANGLE_BOUNDS[a] for _ in range(p) for a in ANGLE_NAMES[family]]


def wrap_periodic(x, bounds) -> np.ndarray:
    """Wrap every entry into its period (hi - lo): symmetric intervals
    (lo < 0) are taken as (lo, hi], the others as [lo, hi). Only valid where
    the angle really is periodic (see the module docstring on gamma)."""
    out = np.array(x, dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        if (lo < out[i] <= hi) if lo < 0 else (lo <= out[i] < hi):
            continue                      # already inside: left bit-exact
        v = lo + (out[i] - lo) % (hi - lo)
        if lo < 0 and v <= lo:
            v = hi
        elif lo >= 0 and v >= hi:      # float rounding of a tiny negative
            v = lo
        out[i] = v
    return out


# ------------------------------------------------------------ distributions
def distribution(inst: DensestKSubgraph, x, family: str = "penalty",
                 lcu_layers=()) -> tuple[np.ndarray, float]:
    """(exact output distribution over statevector indices, Gamma).

    With ``lcu_layers`` (penalty family only) the distribution is the pooled
    layered-LCU one (``exact.layered_lcu_distribution``; those layers' theta
    is ignored) and Gamma its sampling overhead; otherwise Gamma = 1.0 (one
    circuit, nothing pooled)."""
    layers = layers_from_flat(x, family)
    lcu_layers = _lcu_tuple(lcu_layers)
    if lcu_layers:
        if family != "penalty":
            raise ValueError("lcu_layers needs family='penalty'")
        dist, info = exact.layered_lcu_distribution(inst, layers, lcu_layers)
        return dist, float(info.Gamma)
    state_fn = {"penalty": exact.penalty_basis_state_p,
                "coherent": exact.coherent_penalty_state_p,
                "xy": exact.xy_basis_state_p}[family]
    return np.abs(state_fn(inst, layers)) ** 2, 1.0


def _lcu_tuple(lcu_layers) -> tuple:
    return tuple(sorted({int(l) for l in lcu_layers}))


class _Tables:
    """Per-instance tables + a presorted upper-tail CVaR (same arithmetic
    as ``cvar.cvar``, without the per-call argsort of 2^n values)."""

    def __init__(self, inst: DensestKSubgraph):
        self.flam = inst.penalized_table()
        self.f = inst.objective_table()
        self.w = inst.hamming_table().astype(float)
        self.order = np.argsort(self.flam, kind="stable")[::-1]
        self.v_sorted = self.flam[self.order]

    def cvar(self, dist: np.ndarray, alpha: float) -> float:
        w = dist[self.order]
        w = w / w.sum()
        cum = np.cumsum(w)
        j = min(int(np.searchsorted(cum, alpha, side="left")), len(w) - 1)
        frac = alpha - w[:j].sum()
        return float((np.dot(self.v_sorted[:j], w[:j])
                      + frac * self.v_sorted[j]) / alpha)

    def moments(self, dist: np.ndarray) -> dict:
        m1 = float(np.dot(dist, self.w))
        return {"exp_f": float(np.dot(dist, self.f)), "exp_W": m1,
                "var_W": float(np.dot(dist, self.w ** 2)) - m1 ** 2}


def _tables(inst) -> _Tables:
    if "policy_tables" not in inst._cache:
        inst._cache["policy_tables"] = _Tables(inst)
    return inst._cache["policy_tables"]


def _alias_penalty(Gamma: float, gamma_min: float) -> float:
    return ALIAS_WEIGHT * max(0.0, gamma_min - Gamma)


# --------------------------------------------------------- objective factory
def make_objective(inst: DensestKSubgraph, kind: str = "cvar",
                   family: str = "penalty", alpha: float = 0.10,
                   lcu_layers=(), gamma_min: float = 2.0,
                   opt_value: float | None = None,
                   baseline: float | None = None):
    """Exact objective ``fun(x) -> float`` (to be maximized).

    kind:

    - ``"cvar"``: CVaR_alpha of f_lam under the exact distribution (the
      numerator of the challenge's Q; with ``lcu_layers`` the pooled one,
      no alias guard). The 12-q p = 2 optimum is 3.8355 (Q 0.898).
    - ``"exp_H"``: <f_lam>, the paper's Exp.-1 objective. At 12 q, training
      p = 2 on <H> lands at CVaR 2.124, i.e. Q = 0 (below the warm-start
      baseline B): it buys feasibility with objective.
    - ``"tail_proxy"``: the kit's light-cone transfer objective
      (``hardware.transfer_objective``, reused, mu = nu = 1) on the exact
      moments of the distribution, <f> - (<W> - k)^2 - lam * max(0, sigma_W
      - sigma_alpha)^2 with sigma_alpha = 1 / (alpha sqrt(2 pi)); it
      recovered ~0.3-0.6 of the direct CVaR optimum at 12-24 q.
    - ``"pooled_cvar"``: needs ``lcu_layers``; CVaR of the pooled
      layered-LCU distribution minus 10 * max(0, gamma_min - Gamma), so an
      optimizer cannot escape to the gamma*lam = 2 pi m alias where the
      penalty is the identity and Gamma = 1 (an unguarded p = 1
      re-optimization lands at Gamma 1.02).

    ``opt_value`` / ``baseline`` are accepted so the keyword set of
    :func:`evaluate` can be forwarded unchanged (``train`` does); the
    objective does not need them - Q is an increasing affine map of the CVaR
    clipped to [0, 1], so maximizing the CVaR maximizes Q.
    """
    del opt_value, baseline
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    _width(family)
    lcu_layers = _lcu_tuple(lcu_layers)
    if lcu_layers and family != "penalty":
        raise ValueError("lcu_layers needs family='penalty'")
    if kind == "pooled_cvar" and not lcu_layers:
        raise ValueError("kind='pooled_cvar' needs non-empty lcu_layers "
                         "(use kind='cvar' for a single circuit)")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    tab = _tables(inst)

    if kind == "cvar":
        def fun(x):
            return tab.cvar(distribution(inst, x, family, lcu_layers)[0],
                            alpha)
    elif kind == "pooled_cvar":
        def fun(x):
            dist, G = distribution(inst, x, family, lcu_layers)
            return tab.cvar(dist, alpha) - _alias_penalty(G, gamma_min)
    elif kind == "exp_H":
        def fun(x):
            return float(np.dot(distribution(inst, x, family,
                                              lcu_layers)[0], tab.flam))
    else:  # tail_proxy
        from .hardware import transfer_objective

        def fun(x):
            e = tab.moments(distribution(inst, x, family, lcu_layers)[0])
            return float(transfer_objective(e, inst.k, inst.lam, alpha))
    fun.kind, fun.family, fun.alpha, fun.lcu_layers = (kind, family, alpha,
                                                       lcu_layers)
    return fun


def objective_info(inst: DensestKSubgraph, x, family: str = "penalty",
                   lcu_layers=(), gamma_min: float = 2.0) -> dict:
    """Gamma and the alias flags of the LCU layers at ``x``: ``Gamma``
    (product over LCU layers, 1.0 without any), ``layer_gammas``,
    ``alias_flags`` (``exact.gamma_alias_flags`` per LCU layer), ``alias``
    (any flagged), ``alias_penalty`` (what "pooled_cvar" subtracts)."""
    layers = layers_from_flat(x, family)
    lcu_layers = _lcu_tuple(lcu_layers)
    if not lcu_layers:
        return {"Gamma": 1.0, "layer_gammas": (), "alias_flags": [],
                "alias": False, "alias_penalty": 0.0}
    if family != "penalty":
        raise ValueError("lcu_layers needs family='penalty'")
    info = exact.layered_lcu(inst, layers, lcu_layers)
    flags = [exact.gamma_alias_flags(inst, layers[l][0]) for l in lcu_layers]
    return {"Gamma": float(info.Gamma), "layer_gammas": info.layer_gammas,
            "alias_flags": flags, "alias": any(f["alias"] for f in flags),
            "alias_penalty": _alias_penalty(info.Gamma, gamma_min)}


def _references(inst, alpha, opt_value, baseline) -> tuple[float, float]:
    if opt_value is None or baseline is None:
        if inst.n > 20:
            raise ValueError("n > 20: pass opt_value (e.g. tcfs.hardware."
                             "milp_optimum) and baseline (inst.baseline_B or "
                             "tcfs.hardware.baseline_B_mc) explicitly")
        opt_value = inst.brute_force()[0] if opt_value is None else opt_value
        baseline = inst.baseline_B(alpha) if baseline is None else baseline
    return float(opt_value), float(baseline)


def evaluate(inst: DensestKSubgraph, x, family: str = "penalty",
             alpha: float = 0.10, lcu_layers=(),
             opt_value: float | None = None,
             baseline: float | None = None) -> dict:
    """Exact metrics at ``x``: CVaR, Q = clip((CVaR - B)/(OPT - B), 0, 1),
    F (feasible mass), exp_H, P_opt (mass on optimal feasible strings),
    Gamma (1.0 for a single circuit), alias. OPT and B default to
    ``inst.brute_force()[0]`` and ``inst.baseline_B(alpha)`` for n <= 20;
    above that they must be given."""
    opt, B = _references(inst, alpha, opt_value, baseline)
    lcu_layers = _lcu_tuple(lcu_layers)
    tab = _tables(inst)
    dist, G = distribution(inst, x, family, lcu_layers)
    cv = tab.cvar(dist, alpha)
    feas = inst.hamming_table() == inst.k
    optimal = feas & np.isclose(tab.f, opt)
    alias = (objective_info(inst, x, family, lcu_layers)["alias"]
             if lcu_layers else False)
    return {"CVaR": cv, "Q": float(np.clip((cv - B) / (opt - B), 0, 1)),
            "F": float(dist[feas].sum()),
            "exp_H": float(np.dot(dist, tab.flam)),
            "P_opt": float(dist[optimal].sum()), "Gamma": G, "alias": alias,
            "OPT": opt, "B": B}


def objective_from_counts(inst: DensestKSubgraph, counts, kind: str = "cvar",
                          alpha: float = 0.10) -> float:
    """The same objectives from measured counts (any n, no 2^n tables).

    ``counts`` is a Qiskit counts dict, or - for a pooled LCU submission - a
    list of ``(counts_dict, weight)`` pairs; each shot of circuit j then
    weighs weight_j / shots_j, exactly as the grader pools declared weights.
    kind "cvar" is the grader's score: CVaR_alpha of the raw bitstrings'
    f_lam, the quantity inside Q (``hardware.metrics_from_counts``); "exp_H"
    is the sample mean of f_lam; "tail_proxy" uses the sample moments.
    ("pooled_cvar" has no counts form: the alias guard is a property of the
    declaration, pass the list form with kind "cvar".)
    """
    from .cvar import cvar
    from .hardware import bits_from_counts, metrics_from_counts, \
        transfer_objective

    if kind not in ("cvar", "exp_H", "tail_proxy"):
        raise ValueError("kind must be 'cvar', 'exp_H' or 'tail_proxy' for "
                         "counts")
    if isinstance(counts, dict):
        if kind == "cvar":
            return float(metrics_from_counts(inst, counts, alpha)["CVaR"])
        pairs = [(counts, 1.0)]
    else:
        pairs = list(counts)
    f_all, w_all, wt_all = [], [], []
    for cnt_dict, weight in pairs:
        bits, cnt = bits_from_counts(cnt_dict, inst.n)
        f, w = inst.evaluate_bits(bits)
        f_all.append(f)
        w_all.append(w.astype(float))
        wt_all.append(float(weight) * cnt / cnt.sum())
    f, w, wt = map(np.concatenate, (f_all, w_all, wt_all))
    flam = f - inst.lam * (w - inst.k) ** 2
    if kind == "cvar":
        return float(cvar(flam, wt, alpha))
    wt = wt / wt.sum()
    if kind == "exp_H":
        return float(np.dot(wt, flam))
    m1 = float(np.dot(wt, w))
    e = {"exp_f": float(np.dot(wt, f)), "exp_W": m1,
         "var_W": float(np.dot(wt, w ** 2)) - m1 ** 2}
    return float(transfer_objective(e, inst.k, inst.lam, alpha))


# --------------------------------------------------------------- optimizers
def _cobyla(fun, x0, maxiter, rhobeg):
    res = minimize(lambda p: -fun(p), np.asarray(x0, dtype=float),
                   method="COBYLA",
                   options={"maxiter": int(maxiter), "rhobeg": rhobeg})
    return np.asarray(res.x, dtype=float), float(-res.fun), int(res.nfev)


def _safe_wrap(fun, x, value, bounds, rtol=1e-9):
    """Wrap x into bounds only if that leaves fun unchanged (gamma is not
    periodic for non-integer weights)."""
    xw = wrap_periodic(x, bounds)
    if np.allclose(xw, x):
        return xw
    vw = fun(xw)
    return xw if abs(vw - value) <= rtol * max(1.0, abs(value)) else x


def random_starts(rng: np.random.Generator, bounds, m: int) -> list:
    """m uniform random points in the box ``bounds``."""
    lo, hi = np.array(bounds, dtype=float).T
    return [rng.uniform(lo, hi) for _ in range(int(m))]


def random_restarts(fun, bounds, n_starts: int = 30,
                    rng: np.random.Generator | None = None,
                    maxiter: int = 400, rhobeg: float = 0.3, starts=None,
                    tol_same: float = 1e-3) -> dict:
    """Multi-start COBYLA maximization.

    ``starts`` (a list of vectors) replaces the ``n_starts`` uniform random
    starts drawn from ``rng``. Each start's result is at least its own
    starting value. Returns best_value, best_x (wrapped where that is
    value-preserving), all_values (sorted, best first), start_values (in
    start order), n_starts, n_within_tol (results within ``tol_same`` of the
    best - the "hit rate"), nfev_total, wall_s."""
    t0 = time.perf_counter()
    if starts is None:
        if rng is None:
            raise ValueError("pass rng (or explicit starts)")
        starts = random_starts(rng, bounds, n_starts)
    results, start_values, nfev = [], [], 0
    for x0 in starts:
        x0 = np.asarray(x0, dtype=float)
        v0 = float(fun(x0))
        x, v, nf = _cobyla(fun, x0, maxiter, rhobeg)
        nfev += nf + 1
        if v < v0:
            x, v = x0, v0
        results.append((v, x))
        start_values.append(v0)
    if not results:
        raise ValueError("no starts")
    results.sort(key=lambda r: -r[0])
    best_v, best_x = results[0]
    best_x = _safe_wrap(fun, best_x, best_v, bounds)
    vals = [r[0] for r in results]
    return {"best_value": best_v, "best_x": best_x, "all_values": vals,
            "start_values": start_values, "n_starts": len(vals),
            "n_within_tol": int(sum(v >= best_v - tol_same for v in vals)),
            "nfev_total": nfev, "wall_s": time.perf_counter() - t0}


def polish(fun, x0, maxiter: int = 1500, rhobeg: float = 0.1):
    """Joint COBYLA polish from x0; returns (x, value), never worse than x0."""
    x0 = np.asarray(x0, dtype=float)
    v0 = float(fun(x0))
    if maxiter <= 0:
        return x0, v0
    x, v, _ = _cobyla(fun, x0, maxiter, rhobeg)
    return (x, v) if v >= v0 else (x0, v0)


# -------------------------------------------------------------- initializers
def _as_schedule(x, family) -> np.ndarray:
    return np.asarray(layers_from_flat(x, family), dtype=float)


def interp_init(x_p, family: str = "penalty") -> np.ndarray:
    """INTERP (Zhou et al., PRX 10, 021067 (2020)): a depth-(p+1) start from
    a trained depth-p schedule, per angle type

        u'_i = (i-1)/p * u_{i-1} + (p-i+1)/p * u_i,  i = 1..p+1,
        u_0 = u_{p+1} = 0,

    so the first and last angles are kept and p = 1 -> 2 is a plain copy.
    Plain linear interpolation of the raw values (wrap first if a schedule
    straddles a period boundary). At 12 q, INTERP from the p = 1 optimum
    polishes to 3.34 (the p = 2 optimum is 3.8355)."""
    u = _as_schedule(x_p, family)
    p = len(u)
    pad = np.vstack([np.zeros((1, u.shape[1])), u, np.zeros((1, u.shape[1]))])
    out = np.array([(i - 1) / p * pad[i - 1] + (p - i + 1) / p * pad[i]
                    for i in range(1, p + 2)])
    return out.ravel()


def linear_ramp_init(p: int, family: str = "penalty", gamma_max: float = 0.5,
                     beta_max: float = 0.5, theta: float | None = None,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """TQA-style linear ramp (Sack & Serbyn 2021; LR-QAOA 2405.09169):
    gamma_i = gamma_max * (i+1)/p, beta_i = beta_max * (1 - i/p),
    i = 0..p-1. The LCU angle theta is ``theta`` in every layer, or uniform
    random per layer from ``rng`` (0.0 if neither is given). XY family: the
    ramped "beta" is the Y angle theta of R(alpha, theta, chi), alpha = 0,
    and chi plays the free phase (``theta`` argument / random)."""
    width = _width(family)
    if p < 1:
        raise ValueError("p must be >= 1")

    def free_phase():
        if theta is not None:
            return float(theta)
        return float(rng.uniform(0.0, TWO_PI)) if rng is not None else 0.0

    out = []
    for i in range(p):
        g, b = gamma_max * (i + 1) / p, beta_max * (1 - i / p)
        if family == "penalty":
            out.append((g, free_phase(), b))
        elif family == "coherent":
            out.append((g, b))
        else:
            out.append((g, 0.0, b, free_phase()))
    x = flat_from_layers(out)
    assert x.size == p * width
    return x


def warm_from_lower_depth(x_p, family: str = "penalty",
                          rng: np.random.Generator | None = None,
                          noise: float = 0.1,
                          copy_theta: bool = True) -> np.ndarray:
    """Append one near-identity layer: gamma, beta ~ N(0, noise) and the
    previous layer's theta (``copy_theta``; else uniform). The appended layer
    ends in M(beta ~ 0), so the distribution starts at the depth-p optimum.
    XY family: gamma and the Y angle ~ N(0, noise), (alpha, chi) copied.
    At 12 q this start stalls at 3.33 from the p = 1 optimum (0 of 30 reach
    the p = 2 optimum): the optimizer stays in the old basin."""
    rng = np.random.default_rng() if rng is None else rng
    last = layers_from_flat(x_p, family)[-1]
    g = rng.normal(0.0, noise)
    if family == "penalty":
        t = last[1] if copy_theta else rng.uniform(0.0, TWO_PI)
        new = (g, t, rng.normal(0.0, noise))
    elif family == "coherent":
        new = (g, rng.normal(0.0, noise))
    else:
        a, c = ((last[1], last[3]) if copy_theta
                else tuple(rng.uniform(0.0, TWO_PI, 2)))
        new = (g, a, rng.normal(0.0, noise), c)
    return np.concatenate([np.asarray(x_p, dtype=float), new])


def scaled_copy_init(x1, p: int, scale: float,
                     family: str | None = None) -> np.ndarray:
    """p copies of a depth-1 layer with gamma and beta (XY: the Y angle)
    scaled by ``scale``, the LCU / phase angles copied. ``family`` is
    inferred from len(x1) (3 penalty, 2 coherent, 4 XY) when omitted. At
    12 q the scaled copies (0.5, 0.75, 1.0) of the p = 1 optimum polish to
    3.33 - 3.40 at p = 2."""
    x1 = np.asarray(x1, dtype=float).ravel()
    if family is None:
        family = {3: "penalty", 2: "coherent", 4: "xy"}.get(x1.size)
        if family is None:
            raise ValueError("x1 must be a depth-1 vector (2, 3 or 4 angles)")
    (layer,) = layers_from_flat(x1, family)
    layer = np.array(layer)
    scaled = {"penalty": (0, 2), "coherent": (0, 1), "xy": (0, 2)}[family]
    layer[list(scaled)] *= scale
    return np.tile(layer, p)


# ------------------------------------------------------------------ policies
LayerStep = namedtuple("LayerStep",
                       "layer_only_best polished n_within_tol layer")


class _Counter:
    """Counts objective evaluations across the per-depth objectives."""

    def __init__(self):
        self.n = 0

    def wrap(self, fun):
        def counted(x):
            self.n += 1
            return fun(x)
        counted.__dict__.update(getattr(fun, "__dict__", {}))
        return counted


def _depth_objective(inst, d, family, kind, alpha, lcu_layers, obj_kw,
                     counter=None):
    """Objective at depth d: LCU layers >= d are dropped; "pooled_cvar"
    without any LCU layer yet falls back to "cvar"."""
    lcu = tuple(l for l in sorted(set(lcu_layers)) if l < d)
    k = "cvar" if (kind == "pooled_cvar" and not lcu) else kind
    fun = make_objective(inst, kind=k, family=family, alpha=alpha,
                         lcu_layers=lcu, **obj_kw)
    return counter.wrap(fun) if counter is not None else fun


_OBJ_KEYS = ("gamma_min", "opt_value", "baseline")


def layerwise_train(inst: DensestKSubgraph, p: int, family: str = "penalty",
                    kind: str = "cvar", alpha: float = 0.10,
                    rng: np.random.Generator | None = None,
                    starts_per_layer: int = 30, maxiter: int = 400,
                    polish_maxiter: int = 1500, x_init=None, lcu_layers=(),
                    **objective_kw) -> dict:
    """Layer-wise training: train layer 0 by random restarts (or start from
    ``x_init`` at depth p0), then for every new layer freeze the earlier
    ones, random-restart the new layer only, and polish all layers jointly.

    At 12 q from the p = 1 optimum (3.3192): layer 2 alone reaches 3.730 in
    9 of 30 starts and the joint polish 3.8355 = the p = 2 optimum (Q 0.898)
    - the only policy that hit it reliably. Returns x, value, per_layer (a
    ``LayerStep(layer_only_best, polished, n_within_tol, layer)`` per trained
    layer), nfev, wall_s."""
    t0 = time.perf_counter()
    rng = np.random.default_rng() if rng is None else rng
    counter = _Counter()
    obj_kw = {k: v for k, v in objective_kw.items() if k in _OBJ_KEYS}
    extra = set(objective_kw) - set(_OBJ_KEYS)
    if extra:
        raise TypeError(f"unexpected keyword(s) {sorted(extra)}")
    width = _width(family)
    one = layer_bounds(family, 1)
    per_layer = []
    if x_init is None:
        fun = _depth_objective(inst, 1, family, kind, alpha, lcu_layers,
                               obj_kw, counter)
        r = random_restarts(fun, one, starts_per_layer, rng, maxiter=maxiter)
        x, v = polish(fun, r["best_x"], polish_maxiter)
        per_layer.append(LayerStep(r["best_value"], v, r["n_within_tol"], 0))
    else:
        x = np.asarray(x_init, dtype=float)
        p0 = depth_of(x, family)
        if p0 > p:
            raise ValueError(f"x_init has depth {p0} > p = {p}")
        fun = _depth_objective(inst, p0, family, kind, alpha, lcu_layers,
                               obj_kw, counter)
        v = float(fun(x))
        if p0 == p:
            x, v = polish(fun, x, polish_maxiter)
    for d in range(depth_of(x, family), p):
        fun = _depth_objective(inst, d + 1, family, kind, alpha, lcu_layers,
                               obj_kw, counter)
        frozen = x.copy()

        def layer_only(y, frozen=frozen, fun=fun):
            return fun(np.concatenate([frozen, y]))

        r = random_restarts(layer_only, one, starts_per_layer, rng,
                            maxiter=maxiter)
        x, v = polish(fun, np.concatenate([frozen, r["best_x"]]),
                      polish_maxiter)
        per_layer.append(LayerStep(r["best_value"], v, r["n_within_tol"], d))
    x = _safe_wrap(fun, x, v, layer_bounds(family, len(x) // width))
    return {"x": x, "value": float(v), "per_layer": per_layer,
            "nfev": counter.n, "wall_s": time.perf_counter() - t0}


def alpha_schedule_train(inst: DensestKSubgraph, x0, alphas=(0.5, 0.25, 0.10),
                         family: str = "penalty", maxiter: int = 300,
                         **kw) -> tuple[np.ndarray, float]:
    """CVaR-alpha schedule: COBYLA on CVaR_{alphas[0]}, then each next alpha
    from the previous answer (rhobeg 0.3 first, 0.1 after). ``kw`` goes to
    :func:`make_objective` (kind, lcu_layers, gamma_min). Returns (x, value
    at alphas[-1]). At 12 q (30 random p = 2 starts, 0.5 -> 0.25 -> 0.10):
    median 3.33, best 3.834, 0 of 30 at the optimum - no better than plain
    restarts."""
    counter = kw.pop("_counter", None)
    x = np.asarray(x0, dtype=float)
    v = None
    for i, a in enumerate(alphas):
        fun = make_objective(inst, family=family, alpha=a, **kw)
        if counter is not None:
            fun = counter.wrap(fun)
        x, v = polish(fun, x, maxiter, 0.3 if i == 0 else 0.1)
    return x, float(v)


def train(inst: DensestKSubgraph, p: int, policy: str = "layerwise",
          rng: np.random.Generator | None = None, family: str = "penalty",
          kind: str = "cvar", alpha: float = 0.10, n_starts: int = 30,
          **kw) -> dict:
    """Single entry point, for the notebook and a team's own training loop.

    policy:
      "restarts"   joint random restarts at depth p
      "layerwise"  :func:`layerwise_train` (``n_starts`` per layer)
      "interp"     restarts at p = 1, then INTERP + polish per extra layer
      "warm"       restarts at p = 1, then ``n_starts`` warm starts
                   (:func:`warm_from_lower_depth`) per extra layer
      "ramp"       ``n_starts`` random linear ramps at depth p
      "schedule"   ``n_starts`` random starts through :func:`alpha_schedule_train`
    Every policy ends with a joint polish of its best point.

    kw: lcu_layers, gamma_min, opt_value, baseline (objective / metrics);
    maxiter (400), polish_maxiter (1500), x_init (layerwise), alphas
    (schedule), noise (warm). Returns x, layers, value (objective ``kind``
    at x), metrics (:func:`evaluate`), policy, wall_s, nfev."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    t0 = time.perf_counter()
    rng = np.random.default_rng() if rng is None else rng
    lcu_layers = tuple(kw.pop("lcu_layers", ()))
    obj_kw = {k: kw.pop(k) for k in _OBJ_KEYS if k in kw}
    maxiter = kw.pop("maxiter", 400)
    polish_maxiter = kw.pop("polish_maxiter", 1500)
    x_init = kw.pop("x_init", None)
    alphas = kw.pop("alphas", (0.5, 0.25, alpha))
    noise = kw.pop("noise", 0.1)
    if kw:
        raise TypeError(f"unexpected keyword(s) {sorted(kw)}")
    width = _width(family)
    # fail fast: the metrics need OPT and B (see evaluate)
    opt, B = _references(inst, alpha, obj_kw.get("opt_value"),
                         obj_kw.get("baseline"))
    obj_kw.update(opt_value=opt, baseline=B)
    counter = _Counter()

    def fun_at(d):
        return _depth_objective(inst, d, family, kind, alpha, lcu_layers,
                                obj_kw, counter)

    fun_p = fun_at(p)
    bounds_p = layer_bounds(family, p)
    if policy == "layerwise":
        r = layerwise_train(inst, p, family, kind, alpha, rng, n_starts,
                            maxiter, polish_maxiter, x_init, lcu_layers,
                            **obj_kw)
        x, v = r["x"], r["value"]
        counter.n += r["nfev"]
    elif policy in ("interp", "warm"):
        f1 = fun_at(1)
        r = random_restarts(f1, layer_bounds(family, 1), n_starts, rng,
                            maxiter=maxiter)
        x, v = polish(f1, r["best_x"], polish_maxiter)
        for d in range(2, p + 1):
            fd = fun_at(d)
            if policy == "interp":
                x, v = polish(fd, interp_init(x, family), polish_maxiter)
            else:
                starts = [warm_from_lower_depth(x, family, rng, noise)
                          for _ in range(n_starts)]
                r = random_restarts(fd, layer_bounds(family, d),
                                    starts=starts, maxiter=maxiter)
                x, v = polish(fd, r["best_x"], polish_maxiter)
    else:
        if policy == "restarts":
            r = random_restarts(fun_p, bounds_p, n_starts, rng,
                                maxiter=maxiter)
            x = r["best_x"]
        elif policy == "ramp":
            (glo, ghi), (blo, bhi) = (_ANGLE_BOUNDS["gamma"],
                                      _ANGLE_BOUNDS["beta"])
            starts = [linear_ramp_init(p, family, rng.uniform(glo, ghi),
                                       rng.uniform(blo, bhi), rng=rng)
                      for _ in range(n_starts)]
            r = random_restarts(fun_p, bounds_p, starts=starts,
                                maxiter=maxiter)
            x = r["best_x"]
        else:  # schedule
            best = None
            skw = {k: v for k, v in obj_kw.items() if k == "gamma_min"}
            for x0 in random_starts(rng, bounds_p, n_starts):
                xs, _ = alpha_schedule_train(
                    inst, x0, alphas, family, maxiter, kind=kind,
                    lcu_layers=lcu_layers, _counter=counter, **skw)
                vs = fun_p(xs)
                if best is None or vs > best[1]:
                    best = (xs, vs)
            x = best[0]
        x, v = polish(fun_p, x, polish_maxiter)
    x = _safe_wrap(fun_p, np.asarray(x, dtype=float), v, bounds_p)
    assert x.size == p * width
    metrics = evaluate(inst, x, family, alpha, lcu_layers, opt, B)
    return {"x": x, "layers": layers_from_flat(x, family), "value": float(v),
            "metrics": metrics, "policy": policy,
            "wall_s": time.perf_counter() - t0, "nfev": counter.n}


# ---------------------------------------------------------------- surrogates
def surrogate_instance(inst: DensestKSubgraph, n_sub: int,
                       seed: int | None = None,
                       start: int | None = None) -> DensestKSubgraph:
    """A connected induced sub-instance on n_sub nodes for exact training.

    Grown from ``start`` (else a node drawn with ``seed``) by region growth:
    each step adds the frontier node with the most edges into the region
    (ties: earliest discovered), which keeps the region compact and its
    interior degrees close to the parent's. Nodes are relabelled 0..n_sub-1
    in growth order, weights kept, k_sub = max(1, round-half-up(k n_sub /
    n)), and lam follows the kit's rule on the sub-graph
    (``DensestKSubgraph.from_graph``). Check the regime with
    :func:`surrogate_report` before trusting transferred angles."""
    g = inst.graph
    if not 1 <= n_sub <= inst.n:
        raise ValueError(f"n_sub must be in [1, {inst.n}]")
    if start is None:
        start = int(np.random.default_rng(seed).integers(inst.n))
    if start not in g:
        raise ValueError(f"start node {start} not in the graph")
    region, inside = [start], {start}
    seen = {start: 0}                       # node -> discovery rank
    links: dict = {}
    for v in g[start]:
        seen.setdefault(v, len(seen))
        links[v] = links.get(v, 0) + 1
    while len(region) < n_sub:
        if not links:
            raise ValueError(f"the component of node {start} has only "
                             f"{len(region)} nodes (< n_sub = {n_sub})")
        best = min(links, key=lambda v: (-links[v], seen[v]))
        del links[best]
        region.append(best)
        inside.add(best)
        for v in g[best]:
            if v not in inside:
                seen.setdefault(v, len(seen))
                links[v] = links.get(v, 0) + 1
    relabel = {v: i for i, v in enumerate(region)}
    sub = nx.Graph()
    sub.add_nodes_from(range(n_sub))
    for a, b, data in g.subgraph(region).edges(data=True):
        sub.add_edge(relabel[a], relabel[b], weight=data.get("weight", 1.0))
    k_sub = max(1, min(n_sub, int(np.floor(inst.k * n_sub / inst.n + 0.5))))
    return DensestKSubgraph.from_graph(
        sub, k=k_sub, name=f"{inst.name}-sub{n_sub}-start{start}")


def _regime(inst: DensestKSubgraph) -> dict:
    deg = [d for _, d in inst.graph.degree()]
    return {"n": inst.n, "k": inst.k, "k_over_n": inst.k / inst.n,
            "lam": inst.lam, "mean_degree": float(np.mean(deg)),
            "P_feasible": inst.p_feasible_warm_start}


def surrogate_report(parent: DensestKSubgraph, sub: DensestKSubgraph,
                     alpha: float = 0.10) -> dict:
    """Does the surrogate sit in the parent's regime? ``parent`` / ``sub``:
    n, k, k_over_n, lam, mean_degree, P_feasible (warm start); regime_match
    is True when both P_feasible are on the same side of alpha. A warning:
    P_feasible is 0.24 at 12 q (0.17 at 24 q) but 0.082 at 106 q,
    below alpha = 0.10, so at scale the top-10 % tail necessarily contains
    infeasible strings while a small surrogate's tail can be all feasible."""
    a, b = _regime(parent), _regime(sub)
    return {"parent": a, "surrogate": b, "alpha": alpha,
            "regime_match": bool((a["P_feasible"] >= alpha)
                                 == (b["P_feasible"] >= alpha))}
