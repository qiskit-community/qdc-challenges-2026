"""Fourier-based LCU for diagonal unitaries (paper Sec. III).

Target: a diagonal unitary applying the phase e^{i*phi(w)} to every basis
state |x> with g(x) = w, where g: {0,1}^n -> {0,...,m}. With the DFT angles
theta_j = 2*pi*j/(m+1) and coefficients

    c_j = 1/(m+1) * sum_k e^{i*phi(k)} e^{-i*theta_j*k},

the inverse DFT gives  e^{i*phi(w)} = sum_j c_j e^{i*theta_j*w}, i.e. the LCU

    U = sum_j c_j V(theta_j),     V(theta)|x> = e^{i*theta*g(x)}|x>.

Because |e^{i*phi}| = 1, Parseval gives ||c||_2 = 1 and hence the sampling
overhead Gamma = ||c||_1^2 <= m+1.

Sign map to the paper: the paper writes U_f(gamma) = sum_x e^{-i*gamma*f(g(x))},
i.e. phi(k) = -gamma*f(k). We keep phi explicit so every sign is visible.

For the Hamming-weight case g(x) = 1^T x, V(theta) is a single layer of RZ
gates up to the global phase e^{-i*theta*n/2}:  RZ(theta)^{(x)n}|x> =
e^{i*theta*(1^T x - n/2)}|x>.  ``circuits.rz_phase_layer`` restores the global
phase so the LCU identity holds exactly at the operator level.
"""

from __future__ import annotations

import numpy as np


def fourier_coefficients(phases: np.ndarray) -> np.ndarray:
    """DFT coefficients c_j for target diag phases e^{i*phi(k)}, k = 0..m.

    Returns c with c[j] = 1/(m+1) sum_k e^{i*phases[k]} e^{-2i*pi*j*k/(m+1)}.
    """
    u = np.exp(1j * np.asarray(phases, dtype=float))
    return np.fft.fft(u) / len(u)


def dft_angles(m_plus_1: int) -> np.ndarray:
    """theta_j = 2*pi*j/(m+1), j = 0..m."""
    return 2.0 * np.pi * np.arange(m_plus_1) / m_plus_1


def reconstruct_phases(c: np.ndarray) -> np.ndarray:
    """Inverse map: sum_j c_j e^{i*theta_j*k}; equals e^{i*phi(k)} exactly."""
    return np.fft.ifft(c) * len(c)


def gamma(c: np.ndarray) -> float:
    """Sampling overhead Gamma = ||c||_1^2."""
    return float(np.abs(c).sum() ** 2)


def branch_probabilities(c: np.ndarray) -> np.ndarray:
    """Ancilla-free sampling distribution q_j = |c_j| / ||c||_1 (Sec. II B)."""
    a = np.abs(c)
    return a / a.sum()


# ----------------------------------------------------------- penalty phases
def cardinality_penalty_phases(gamma_angle: float, lam: float, k: int,
                               n: int) -> np.ndarray:
    """Phases phi(w) = +gamma*lam*(w-k)^2 for w = 0..n.

    The penalized Hamiltonian H|x> = f_lam(x)|x> contains the penalty term
    h2(x) = -lam*(1^T x - k)^2, so the separator factor is
    e^{-i*gamma*h2(x)} = e^{+i*gamma*lam*(w-k)^2} = e^{i*phi(w)}.
    """
    w = np.arange(n + 1)
    return gamma_angle * lam * (w - k) ** 2


def indicator_window_phases(gamma_angle: float, lam: float, lo: int, hi: int,
                            n: int) -> np.ndarray:
    """Phases for the indicator-window penalty  -lam*1[w not in [lo, hi]].

    (Paper Sec. III; the Hamming-weight window variant, an unscored kit
    capability since 2026-09-29.)  The separator factor
    is e^{+i*gamma*lam*1[w not in [lo, hi]]}.
    """
    w = np.arange(n + 1)
    outside = (w < lo) | (w > hi)
    return gamma_angle * lam * outside.astype(float)
