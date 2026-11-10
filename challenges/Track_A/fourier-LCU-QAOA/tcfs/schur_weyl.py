"""Permutation-invariant Fourier LCU and the XY-mixer (paper Sec. IV, App. A-D).

Schur-Weyl: (C^2)^{(x)n} decomposes into spin-j sectors, j in
J_n = {n/2, n/2-1, ..., j_min}, with multiplicities
m_j = C(n, n/2-j) - C(n, n/2-j-1).

The fully connected XY-mixer H_XY = J_x^2 + J_y^2 (J_mu = sum_i sigma_mu^i)
acts on the spin-j block as A_j(beta) = sum_m e^{-i*4*beta*(j(j+1)-m^2)}
|j,m><j,m|.  Its exact continuous LCU over collective rotations
R(alpha, theta, chi) = e^{-i*alpha*Z/2} e^{-i*theta*Y/2} e^{-i*chi*Z/2} is

    U_XY(beta) = Int a_beta(g) R(g)^{(x)n} dmu(g),

    a_beta(alpha, theta, chi) =
        sum_j (2j+1) sum_m e^{-i*4*beta*(j(j+1)-m^2)} e^{i*m*(alpha+chi)}
              d^j_{mm}(theta),

with the Haar measure  dmu = sin(theta)/(16*pi^2) dalpha dtheta dchi over
alpha in [0, 2pi), theta in [0, pi], chi in [0, 4pi).  The QPD cost is
Gamma_XY = (Int |a| dmu)^2 <= (n+1)(n+2)(n+3)/6.

Wigner-d diagonals are computed exactly by eigendecomposition of the spin-j
S_y operator: d^j_{mm}(theta) = <j,m| e^{-i*theta*S_y} |j,m>.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import comb


def spin_sectors(n: int) -> list[float]:
    """J_n = {n/2, n/2 - 1, ..., 1/2 or 0}."""
    jmin = 0.0 if n % 2 == 0 else 0.5
    return [n / 2 - i for i in range(int(n / 2 - jmin) + 1)]


def multiplicity(n: int, j: float) -> int:
    """m_j = C(n, n/2 - j) - C(n, n/2 - j - 1)  (paper App. A)."""
    a = comb(n, int(round(n / 2 - j)), exact=True)
    kk = int(round(n / 2 - j - 1))
    b = comb(n, kk, exact=True) if kk >= 0 else 0
    return a - b


@lru_cache(maxsize=None)
def _sy_eig(two_j: int):
    """Eigendecomposition of S_y in the spin-j basis, m ascending -j..j.

    Returns (eigenvalues ~ m' in ascending order, |V|^2 weight matrix W with
    W[m_idx, k] = |<j, m | v_k>|^2) so that
    d^j_{mm}(theta) = sum_k W[m_idx, k] * exp(-i*theta*lambda_k).
    """
    j = two_j / 2
    dim = two_j + 1
    m = -j + np.arange(dim)  # ascending
    # ladder operators: S+|j,m> = sqrt(j(j+1) - m(m+1)) |j,m+1>
    sp = np.zeros((dim, dim))
    for a in range(dim - 1):
        sp[a + 1, a] = np.sqrt(j * (j + 1) - m[a] * (m[a] + 1))
    sy = (sp - sp.T) / 2j
    lam, vec = np.linalg.eigh(sy)
    return lam, np.abs(vec) ** 2


def wigner_d_diag(j: float, thetas: np.ndarray) -> np.ndarray:
    """d^j_{mm}(theta) for all m = -j..j (rows) and each theta (columns).

    Exact: d^j_{mm}(theta) = <j,m|e^{-i*theta*S_y}|j,m>
                           = sum_k |V_{mk}|^2 e^{-i*theta*lambda_k}.
    The result is real; the imaginary part (numerical noise) is discarded.
    """
    lam, w2 = _sy_eig(int(round(2 * j)))
    phases = np.exp(-1j * np.outer(lam, np.atleast_1d(thetas)))
    return np.real(w2 @ phases)


def a_beta(n: int, beta: float, alphas, thetas, chis) -> np.ndarray:
    """The XY-mixer LCU coefficient function a_beta(alpha, theta, chi).

    Vectorized over equal-length arrays of Euler angles. (Paper App. D.)
    """
    alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
    thetas = np.atleast_1d(np.asarray(thetas, dtype=float))
    chis = np.atleast_1d(np.asarray(chis, dtype=float))
    out = np.zeros(alphas.shape, dtype=complex)
    ac = alphas + chis
    for j in spin_sectors(n):
        d = wigner_d_diag(j, thetas)              # (2j+1, S)
        m = -j + np.arange(int(round(2 * j)) + 1)  # ascending
        phase_beta = np.exp(-1j * 4 * beta * (j * (j + 1) - m ** 2))
        phase_ac = np.exp(1j * np.outer(m, ac))    # (2j+1, S)
        out += (2 * j + 1) * np.einsum("m,ms,ms->s", phase_beta, phase_ac, d)
    return out


def haar_euler_samples(rng: np.random.Generator, size: int):
    """Haar-uniform Euler angles: alpha ~ U[0,2pi), cos(theta) ~ U[-1,1],
    chi ~ U[0,4pi)."""
    alphas = rng.uniform(0, 2 * np.pi, size)
    thetas = np.arccos(rng.uniform(-1, 1, size))
    chis = rng.uniform(0, 4 * np.pi, size)
    return alphas, thetas, chis


def estimate_gamma_xy(n: int, beta: float, rng: np.random.Generator,
                      num_samples: int = 100_000) -> float:
    """Monte-Carlo estimate of Gamma_XY(beta) = (E_Haar |a_beta|)^2."""
    al, th, ch = haar_euler_samples(rng, num_samples)
    return float(np.abs(a_beta(n, beta, al, th, ch)).mean() ** 2)


def gamma_xy_bound(n: int) -> float:
    """Worst-case bound Gamma_XY <= (n+1)(n+2)(n+3)/6 (paper App. C)."""
    return (n + 1) * (n + 2) * (n + 3) / 6


def importance_sample_circuit_angles(n: int, beta: float,
                                     rng: np.random.Generator,
                                     num_candidates: int,
                                     num_circuits: int):
    """The paper's XY ensemble protocol (Sec. VI B).

    Draw ``num_candidates`` Haar-uniform Euler triples, weight them by
    |a_beta|, then draw ``num_circuits`` triples (with replacement) from the
    normalized discrete distribution. Returns (angles (3, num_circuits),
    Gamma estimate from the candidate pool).
    """
    al, th, ch = haar_euler_samples(rng, num_candidates)
    wts = np.abs(a_beta(n, beta, al, th, ch))
    gamma_est = float(wts.mean() ** 2)
    q = wts / wts.sum()
    idx = rng.choice(num_candidates, size=num_circuits, p=q)
    return np.vstack([al[idx], th[idx], ch[idx]]), gamma_est


def euler_zyz(u: np.ndarray) -> tuple[float, float, float, float]:
    """ZYZ Euler angles of a 2x2 unitary: u = e^{i*delta} R(alpha, theta, chi).

    Returns (alpha, theta, chi, delta). Used for the paper's App. G warm-start
    initialization of the XY basis circuit: map the optimized per-qubit
    penalty-track unitary onto (theta, chi) with alpha dropped (it acts
    immediately before measurement).
    """
    delta = np.angle(np.linalg.det(u)) / 2
    su = u * np.exp(-1j * delta)
    theta = 2 * np.arctan2(np.abs(su[1, 0]), np.abs(su[0, 0]))
    if np.abs(su[0, 0]) < 1e-12:      # theta = pi: only alpha - chi defined
        alpha = 2 * np.angle(su[1, 0])
        chi = 0.0
    elif np.abs(su[1, 0]) < 1e-12:    # theta = 0: only alpha + chi defined
        alpha = 2 * np.angle(su[1, 1])
        chi = 0.0
    else:
        s, d = 2 * np.angle(su[1, 1]), 2 * np.angle(su[1, 0])
        alpha, chi = (s + d) / 2, (s - d) / 2
    return alpha, theta, chi, delta


# --------------------------------------------------------------- verification
def su2_matrix(alpha: float, theta: float, chi: float) -> np.ndarray:
    """R(alpha, theta, chi) = e^{-i*alpha*Z/2} e^{-i*theta*Y/2} e^{-i*chi*Z/2}."""
    ca, sa = np.exp(-1j * alpha / 2), np.exp(1j * alpha / 2)
    cc, sc = np.exp(-1j * chi / 2), np.exp(1j * chi / 2)
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[ca * cc * ct, -ca * sc * st],
                     [sa * cc * st, sa * sc * ct]])


def xy_mixer_matrix(n: int, beta: float) -> np.ndarray:
    """Dense e^{-i*beta*(J_x^2 + J_y^2)} for small n (verification only)."""
    from scipy.linalg import expm

    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]])
    jx = sum(_embed(x, i, n) for i in range(n))
    jy = sum(_embed(y, i, n) for i in range(n))
    return expm(-1j * beta * (jx @ jx + jy @ jy))


def _embed(op: np.ndarray, i: int, n: int) -> np.ndarray:
    """op on qubit i (statevector index convention: qubit 0 = least significant,
    i.e. the *last* kron factor)."""
    mats = [np.eye(2, dtype=complex)] * n
    mats[n - 1 - i] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def xy_lcu_matrix_quadrature(n: int, beta: float) -> np.ndarray:
    """Reconstruct U_XY(beta) from the continuous LCU by exact quadrature.

    Uniform grids in alpha (period 2pi) and chi (period 4pi) integrate the
    trigonometric polynomials exactly once the grid exceeds the bandwidth;
    Gauss-Legendre in cos(theta) handles the polar integral. Small n only.
    """
    n_grid = 2 * n + 5
    alphas = 2 * np.pi * np.arange(n_grid) / n_grid
    chis = 4 * np.pi * np.arange(n_grid) / n_grid
    nodes, weights = np.polynomial.legendre.leggauss(n + 4)  # over cos(theta)
    thetas = np.arccos(nodes)

    dim = 1 << n
    total = np.zeros((dim, dim), dtype=complex)
    for th, w_th in zip(thetas, weights):
        for al in alphas:
            a_vals = a_beta(n, beta, np.full(n_grid, al),
                            np.full(n_grid, th), chis)
            for ch, a_val in zip(chis, a_vals):
                r = su2_matrix(al, th, ch)
                rn = r
                for _ in range(n - 1):
                    rn = np.kron(rn, r)
                total += a_val * w_th * rn
    # measure: dmu = sin(th)/(16 pi^2) dal dth dch; with substitution
    # u = cos(th): Int sin(th) dth -> Int du (Gauss-Legendre weights), and the
    # uniform alpha/chi rules carry factors (2pi/n_grid) and (4pi/n_grid).
    return total * (2 * np.pi / n_grid) * (4 * np.pi / n_grid) / (16 * np.pi ** 2)
