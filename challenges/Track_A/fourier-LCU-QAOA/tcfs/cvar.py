"""Weighted CVaR estimators (paper Sec. II B and Ref. [31]; spec Sec. 4).

For a maximization problem the relevant quantity is the *upper-tail* CVaR:
the weighted mean of the best alpha-fraction of the probability mass. The
paper's guarantee (Sec. II B): with X ~ coherent distribution p and
X~tilde ~ LCU-sampler distribution p~, for any f,

    CVaR_lower_{1/Gamma}(f(X~))  <=  E[f(X)]  <=  CVaR_upper_{1/Gamma}(f(X~)).

The same routine serves exact distributions (weights = probabilities) and
finite weighted samples (weights = q_j / shots_j per shot, or occurrence
counts); weights only need to be nonnegative, they are normalized internally.
"""

from __future__ import annotations

import numpy as np


def cvar(values, weights, alpha: float, tail: str = "upper") -> float:
    """CVaR_alpha of a weighted value distribution.

    Parameters
    ----------
    values : array of objective values (any order, repeats allowed)
    weights : nonnegative weights (probabilities or sample weights)
    alpha : tail mass in (0, 1]; alpha = 1 recovers the mean
    tail : "upper" (best values, maximization) or "lower"

    Fractional boundary weight is handled exactly: the sample straddling the
    alpha quantile contributes only the mass needed to fill the tail.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape or v.ndim != 1:
        raise ValueError("values and weights must be equal-length 1-D arrays")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(w))):
        raise ValueError("values and weights must be finite (no NaN/inf)")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    total = w.sum()
    if total <= 0:
        raise ValueError("total weight must be positive")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    order = np.argsort(v, kind="stable")
    if tail == "upper":
        order = order[::-1]
    elif tail != "lower":
        raise ValueError("tail must be 'upper' or 'lower'")

    v, w = v[order], w[order] / total
    cum = np.cumsum(w)
    # index of the sample that fills the tail
    j = int(np.searchsorted(cum, alpha, side="left"))
    j = min(j, len(v) - 1)
    inner = w[:j].sum()
    frac = alpha - inner  # mass taken from the boundary sample
    return float((np.dot(v[:j], w[:j]) + frac * v[j]) / alpha)


def cvar_from_counts(counts: dict, objective, alpha: float,
                     tail: str = "upper", weight: float = 1.0) -> float:
    """CVaR from a Qiskit counts dict; ``objective(bitstring) -> value``.

    ``weight`` scales all counts (used when pooling branches with LCU
    weights q_j; pool by concatenating values/weights across branches and
    calling :func:`cvar` directly).
    """
    values = [objective(b) for b in counts]
    weights = [weight * c for c in counts.values()]
    return cvar(np.array(values), np.array(weights), alpha, tail)
