"""The paper's experiment protocols, exactly (Sec. VI A and VI B).

All runs are statevector-exact (the paper's 12-qubit methodology): grid
search followed by COBYLA fine-tuning, maximization throughout.

Penalty track (Sec. VI A)          XY track (Sec. VI B)
  1 coherent QAOA, <H>               1 coherent XY-QAOA, CVaR_{p0}
  2 LCU evaluated at (b*, g*)        2 sampled LCU ensemble at (b*, g*)
  3 LCU re-optimized (b, g)          3 ensemble re-optimized (b, g)
  4 single basis circuit (b, g, t)   4 single basis circuit (g, th, ch)
  5 coherent re-opt, CVaR_{1/G3}     5 coherent re-opt, CVaR_{p0/G3}
                                     6 coherent, CVaR_{1/G3-penalty}
                                     7 single basis, CVaR_{1/G3-penalty}

Implementation note (ensemble re-optimization, XY Exp. 3): the paper fixes
the 10^6-candidate pool and the sampled circuits "for all LCU-based
experiments" while noting Gamma depends on beta. We therefore keep the
sampled circuit angles fixed and reweight them by |a_beta_new|/|a_beta_0|
(self-normalized importance sampling); Gamma(beta) is re-estimated on a
fixed Haar pool. This is an implementation decision the paper leaves open.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from . import diagonal_lcu as dlcu
from . import exact, schur_weyl
from .cvar import cvar
from .instances import DensestKSubgraph


# ------------------------------------------------------------ optimizer core
def maximize(fun, x0, bounds=None, maxiter: int = 200, rhobeg: float = 0.3):
    """COBYLA maximization of ``fun`` starting at x0."""
    res = minimize(lambda p: -fun(p), np.asarray(x0, dtype=float),
                   method="COBYLA",
                   options={"maxiter": maxiter, "rhobeg": rhobeg})
    return res.x, -res.fun


def grid_then_cobyla(fun, bounds, n_grid: int = 21, maxiter: int = 150,
                     rhobeg: float = 0.25, top_k: int = 3):
    """Grid search then COBYLA fine-tune (the paper's protocol).

    The p = 1 landscapes here are highly multi-modal, so the ``top_k`` best
    grid points are each polished and the overall best result is returned;
    COBYLA can never lose to its own starting point.
    """
    axes = [np.linspace(lo, hi, n_grid) for lo, hi in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    vals = np.array([fun(p) for p in pts])
    order = np.argsort(vals)[::-1][:top_k]
    best_x, best_v = pts[order[0]], float(vals[order[0]])
    for i in order:
        x, v = maximize(fun, pts[i], maxiter=maxiter, rhobeg=rhobeg)
        if v > best_v:
            best_x, best_v = x, float(v)
    return best_x, best_v


def _row(name, inst, dist, Gamma=None, eta=None):
    m = exact.distribution_metrics(inst, dist)
    flam = inst.penalized_table()
    row = {"experiment": name, "exp_H": m["exp_H"],
           "Gamma": np.nan if Gamma is None else Gamma,
           "eta": np.nan if eta is None else eta,
           "CVaR": np.nan if eta is None else cvar(flam, dist, eta),
           "P_feasible": m["P_feasible"], "P_optimal": m["P_optimal"],
           "exp_H_feasible": m["exp_H_feasible"]}
    return row


# ----------------------------------------------------------- penalty helpers
def penalty_lcu(inst, gamma: float):
    """(coefficients c, angles theta_j, q_j, Gamma) for the cardinality
    penalty at angle gamma."""
    phases = dlcu.cardinality_penalty_phases(gamma, inst.lam, inst.k, inst.n)
    c = dlcu.fourier_coefficients(phases)
    return (c, dlcu.dft_angles(inst.n + 1), dlcu.branch_probabilities(c),
            dlcu.gamma(c))


def penalty_pooled_dist(inst, beta: float, gamma: float):
    """(pooled p~, Gamma) of the ancilla-free penalty LCU (all n+1 branches)."""
    _, thetas, q, G = penalty_lcu(inst, gamma)
    dists = np.array([np.abs(exact.penalty_basis_state(inst, beta, gamma, t))
                      ** 2 for t in thetas])
    return exact.pooled_distribution(dists, q), G


# -------------------------------------------------------- penalty experiments
def run_penalty_experiments(inst: DensestKSubgraph, n_grid: int = 61):
    """Experiments 1-5 of Sec. VI A. Returns (rows, info)."""
    flam = inst.penalized_table()
    # beta has period pi in the rz(-2 beta) convention: the half-range keeps
    # the grid spacing identical to a (-pi, pi) grid on the rz angle itself
    bounds = [(-np.pi / 2, np.pi / 2), (-np.pi, np.pi)]  # (beta, gamma)

    def coherent_dist(p):
        return np.abs(exact.coherent_penalty_state(inst, p[0], p[1])) ** 2

    # 1: coherent, maximize <H>
    (b1, g1), _ = grid_then_cobyla(lambda p: np.dot(coherent_dist(p), flam),
                                   bounds, n_grid)
    rows = [_row("1 Coherent QAOA", inst, coherent_dist([b1, g1]))]

    # 2: LCU evaluated at (b*, g*)
    p2, G2 = penalty_pooled_dist(inst, b1, g1)
    rows.append(_row("2 Fourier-based LCU", inst, p2, Gamma=G2, eta=1 / G2))

    # 3: LCU re-optimized on CVaR_{1/Gamma(gamma)}
    def lcu_cvar(p):
        dist, G = penalty_pooled_dist(inst, p[0], p[1])
        return cvar(flam, dist, 1 / G)

    (b3, g3), _ = grid_then_cobyla(lcu_cvar, bounds, n_grid)
    p3, G3 = penalty_pooled_dist(inst, b3, g3)
    rows.append(_row("3 Fourier-based LCU re-optimized", inst, p3,
                     Gamma=G3, eta=1 / G3))

    # 4: single LCU basis circuit over (beta, gamma, theta), CVaR_{1/G3}
    def basis_dist(p):
        return np.abs(exact.penalty_basis_state(inst, p[0], p[1], p[2])) ** 2

    def basis_cvar(p):
        return cvar(flam, basis_dist(p), 1 / G3)

    starts = [np.array([b3, g3, t]) for t in dlcu.dft_angles(inst.n + 1)]
    cands = [maximize(basis_cvar, s, maxiter=150) for s in starts]
    (x4, v4) = max(cands, key=lambda t: t[1])
    rows.append(_row("4 Single LCU basis circuit", inst, basis_dist(x4),
                     Gamma=G3, eta=1 / G3))

    # 5: coherent re-optimized on CVaR_{1/G3}
    (b5, g5), _ = grid_then_cobyla(
        lambda p: cvar(flam, coherent_dist(p), 1 / G3), bounds, n_grid)
    rows.append(_row("5 Coherent QAOA (CVaR)", inst,
                     coherent_dist([b5, g5]), Gamma=G3, eta=1 / G3))

    info = {"opt_coherent": (b1, g1), "opt_lcu": (b3, g3), "Gamma3": G3,
            "opt_basis": tuple(x4), "opt_coherent_cvar": (b5, g5)}
    return rows, info


# ------------------------------------------------------------ XY experiments
def run_xy_experiments(inst: DensestKSubgraph, gamma3_penalty: float,
                       seed: int = 7, num_candidates: int = 100_000,
                       num_circuits: int = 400, n_grid: int = 21):
    """Experiments 1-7 of Sec. VI B. ``gamma3_penalty`` is the penalty-track
    Gamma from Experiment 3 (for the cross-comparison objectives 6-7)."""
    rng = np.random.default_rng(seed)
    flam = inst.penalized_table()
    p0 = inst.p_feasible_warm_start
    n = inst.n
    bounds_bg = [(0.0, np.pi / 2), (-np.pi, np.pi)]  # (beta, gamma)

    # the paper's secondary objective (Sec. VI B): P(optimal) with weight
    # 1e-5 breaks the degeneracy of the tiny-tail CVaR = OPT plateau
    opt_mask = (inst.hamming_table() == inst.k) & np.isclose(
        inst.objective_table(), inst.brute_force()[0])

    def score(dist, eta):
        return cvar(flam, dist, eta) + 1e-5 * float(dist[opt_mask].sum())

    def coh_dist(p):
        return np.abs(exact.coherent_xy_state(inst, p[0], p[1])) ** 2

    # 1: coherent XY-QAOA, CVaR_{p0}
    (b1, g1), _ = grid_then_cobyla(lambda p: score(coh_dist(p), p0),
                                   bounds_bg, n_grid)
    rows = [_row("1 Coherent XY-QAOA (CVaR)", inst, coh_dist([b1, g1]),
                 eta=p0)]

    # 2: sampled Fourier LCU ensemble at (b*, g*)
    angles, G2 = schur_weyl.importance_sample_circuit_angles(
        n, b1, rng, num_candidates, num_circuits)
    a0 = np.abs(schur_weyl.a_beta(n, b1, *angles))  # sampling weights at b1
    dists = np.array([np.abs(exact.xy_basis_state(inst, g1, al, th, ch)) ** 2
                      for al, th, ch in angles.T])
    p2 = dists.mean(axis=0)  # circuits drawn ~ q  =>  equal-weight pooling
    rows.append(_row("2 Fourier-based LCU", inst, p2, Gamma=G2,
                     eta=p0 / G2))

    # fixed Haar pool for Gamma(beta) re-estimation
    pool = schur_weyl.haar_euler_samples(rng, 20_000)

    def gamma_of_beta(beta):
        return float(np.abs(schur_weyl.a_beta(n, beta, *pool)).mean() ** 2)

    # 3: re-optimize (beta, gamma) over the fixed ensemble, reweighted
    def ensemble(p):
        beta, gam = p
        if abs(gam - ensemble.gam) > 1e-15:
            ensemble.dists = np.array(
                [np.abs(exact.xy_basis_state(inst, gam, al, th, ch)) ** 2
                 for al, th, ch in angles.T])
            ensemble.gam = gam
        w = np.abs(schur_weyl.a_beta(n, beta, *angles)) / a0
        dist = exact.pooled_distribution(ensemble.dists, w)
        return dist, gamma_of_beta(beta)

    ensemble.gam, ensemble.dists = g1, dists

    def ens_cvar(p):
        dist, G = ensemble(p)
        return score(dist, p0 / G)

    (b3, g3), _ = grid_then_cobyla(ens_cvar, bounds_bg, max(9, n_grid // 2))
    p3, G3 = ensemble([b3, g3])
    rows.append(_row("3 Fourier-based LCU re-optimized", inst, p3, Gamma=G3,
                     eta=p0 / G3))

    # 4: single basis circuit over (gamma, theta, chi), CVaR_{p0/G3}
    def basis_dist(p):
        return np.abs(exact.xy_basis_state(inst, p[0], 0.0, p[1], p[2])) ** 2

    def opt_basis(eta):
        best = max(
            ((th, ch) for th in np.linspace(0, np.pi, 7)
             for ch in np.linspace(0, 4 * np.pi, 9)),
            key=lambda tc: score(basis_dist([g3, *tc]), eta))
        return maximize(lambda p: score(basis_dist(p), eta),
                        [g3, *best], maxiter=200)

    x4, _ = opt_basis(p0 / G3)
    rows.append(_row("4 Single LCU basis circuit", inst, basis_dist(x4),
                     Gamma=G3, eta=p0 / G3))

    # 5: coherent re-opt, CVaR_{p0/G3}
    (b5, g5), _ = grid_then_cobyla(
        lambda p: score(coh_dist(p), p0 / G3), bounds_bg, n_grid)
    rows.append(_row("5 Coherent XY-QAOA (CVaR)", inst, coh_dist([b5, g5]),
                     Gamma=G3, eta=p0 / G3))

    # 6-7: cross-comparison at the penalty track's CVaR level 1/Gamma3-penalty
    eta_p = 1 / gamma3_penalty
    (b6, g6), _ = grid_then_cobyla(
        lambda p: score(coh_dist(p), eta_p), bounds_bg, n_grid)
    rows.append(_row("6 Coherent XY-QAOA (CVaR 1/G3p)", inst,
                     coh_dist([b6, g6]), Gamma=gamma3_penalty, eta=eta_p))

    x7, _ = opt_basis(eta_p)
    rows.append(_row("7 Single LCU basis circuit (CVaR 1/G3p)", inst,
                     basis_dist(x7), Gamma=gamma3_penalty, eta=eta_p))

    info = {"opt_coherent": (b1, g1), "ensemble_angles": angles,
            "opt_ensemble": (b3, g3), "Gamma3": G3, "opt_basis": tuple(x4),
            "opt_basis_crosscmp": tuple(x7)}
    return rows, info
