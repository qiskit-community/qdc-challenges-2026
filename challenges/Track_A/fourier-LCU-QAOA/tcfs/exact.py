"""Fast exact statevector engine + distribution metrics.

The 12-qubit validation study needs thousands of circuit evaluations inside
optimization loops. All Part-1 circuits share the structure

    warm start -> diagonal phases -> single-qubit product layer(s),

so states are evolved directly as numpy vectors: diagonal unitaries multiply
elementwise, and a product single-qubit layer is n reshaped 2x2 contractions.
``tests/test_exact.py`` asserts this engine agrees with
``qiskit.quantum_info.Statevector`` on the actual circuits to 1e-12.

Index convention: statevector index b encodes x_i = (b >> i) & 1.

Depth p (the ``*_p`` functions). A *layer* is a tuple of angles in the
**circuit builder's** argument order (``tcfs.circuits`` uses the same tuples):

- penalty basis layer    ``(gamma, theta, beta)``:  e^{-i*gamma*H1} (graph
  part only), then V(theta)|x> = e^{i*theta*(1^T x)}|x> (the LCU RZ layer
  including its global phase), then the warm-start mixer M(beta);
- coherent penalty layer ``(gamma, beta)``:  e^{-i*gamma*H} with the full
  penalized H, then M(beta);
- XY basis layer         ``(gamma, alpha, theta, chi)``:  e^{-i*gamma*H1},
  then R(alpha, theta, chi)^{(x)n}.

``layers`` is a sequence of such tuples, p = len(layers), and the warm start
is applied once, before the first layer. The p = 1 functions below keep their
historical ``(inst, beta, gamma, theta)`` argument order; a one-layer ``*_p``
call reproduces them bit for bit.

Layered (partial-depth) Fourier LCU: in a depth-p penalty basis circuit the
cardinality penalty of the layers in ``lcu_layers`` is realized by the Fourier
LCU over the n+1 DFT branches theta_j = 2*pi*j/(n+1) (that layer's own theta
is ignored), while the other layers keep their trained theta. Branches are
index tuples b = (j_l for l in sorted(lcu_layers)) with product coefficients
c_b = prod_l c_{l, j_l}; linearity gives exactly

    sum_b c_b |psi_b> = |psi_coh>,

where |psi_coh> has every LCU layer replaced by the coherent penalty layer
(gamma_l, beta_l) (``coherent_equivalent_state``). The ancilla-free sampler
pools |psi_b|^2 with q_b = |c_b| / ||c||_1 at overhead
Gamma = ||c||_1^2 = prod_l Gamma_l.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse.linalg import expm_multiply

from . import diagonal_lcu as dlcu
from .instances import DensestKSubgraph


# ------------------------------------------------------------------- engine
def warm_start_state(inst: DensestKSubgraph) -> np.ndarray:
    """Exact amplitudes of the warm-start product state."""
    t = inst.theta_init
    w = inst.hamming_table()
    return np.cos(t / 2) ** (inst.n - w) * np.sin(t / 2) ** w


def apply_diag_phases(state: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Apply the diagonal unitary e^{i*phases[b]} elementwise."""
    return state * np.exp(1j * phases)


def apply_uniform_1q(state: np.ndarray, u: np.ndarray, n: int) -> np.ndarray:
    """Apply the same 2x2 unitary u to every qubit."""
    psi = state.reshape((2,) * n)
    for axis in range(n):
        psi = np.tensordot(u, psi, axes=([1], [axis]))
        psi = np.moveaxis(psi, 0, axis)
    return psi.reshape(-1)


def ry_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz_matrix(phi: float) -> np.ndarray:
    return np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)])


def ws_mixer_matrix(inst: DensestKSubgraph, beta: float) -> np.ndarray:
    """RY(t) RZ(-2 beta) RY(-t), t = theta_init (matrix form of the Fig. 3
    mixer, in the paper's / the authors' released-code convention: the mixer
    is e^{+i beta Z} in the warm-start frame)."""
    t = inst.theta_init
    return ry_matrix(t) @ rz_matrix(-2 * beta) @ ry_matrix(-t)


# ------------------------------------------------- circuit-equivalent states
def coherent_penalty_state(inst, beta: float, gamma: float) -> np.ndarray:
    """State of the fully coherent penalty-QAOA circuit (exact)."""
    psi = warm_start_state(inst).astype(complex)
    psi = apply_diag_phases(psi, -gamma * inst.penalized_table())
    return apply_uniform_1q(psi, ws_mixer_matrix(inst, beta), inst.n)


def penalty_basis_state(inst, beta: float, gamma: float,
                        theta: float) -> np.ndarray:
    """State of the Fig. 3 LCU basis circuit: cost, RZ(theta) layer, mixer."""
    psi = warm_start_state(inst).astype(complex)
    phases = -gamma * inst.objective_table() + theta * inst.hamming_table()
    psi = apply_diag_phases(psi, phases)
    return apply_uniform_1q(psi, ws_mixer_matrix(inst, beta), inst.n)


def xy_basis_state(inst, gamma: float, alpha: float, theta: float,
                   chi: float) -> np.ndarray:
    """State of the Fig. 5 basis circuit: cost then R(alpha,theta,chi)^{(x)n}."""
    from .schur_weyl import su2_matrix

    psi = warm_start_state(inst).astype(complex)
    psi = apply_diag_phases(psi, -gamma * inst.objective_table())
    return apply_uniform_1q(psi, su2_matrix(alpha, theta, chi), inst.n)


@lru_cache(maxsize=4)
def _xy_hamiltonian_sparse(n: int):
    """Sparse H_XY = J_x^2 + J_y^2 = 2n*I + 2*sum_{i<j}(X_i X_j + Y_i Y_j)."""
    from qiskit.quantum_info import SparsePauliOp

    terms = [("", [], 2.0 * n)]
    for i in range(n):
        for j in range(i + 1, n):
            terms.append(("XX", [i, j], 2.0))
            terms.append(("YY", [i, j], 2.0))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).to_matrix(
        sparse=True).tocsc()


def coherent_xy_state(inst, beta: float, gamma: float) -> np.ndarray:
    """warm start -> e^{-i*gamma*H1} -> exact e^{-i*beta*H_XY} (no Trotter)."""
    psi = warm_start_state(inst).astype(complex)
    psi = apply_diag_phases(psi, -gamma * inst.objective_table())
    return expm_multiply(-1j * beta * _xy_hamiltonian_sparse(inst.n), psi)


# ----------------------------------------------------------- depth-p layers
def penalty_basis_layer(inst, psi: np.ndarray, gamma: float, theta: float,
                        beta: float) -> np.ndarray:
    """One penalty basis layer: e^{-i*gamma*H1}, V(theta), M(beta)."""
    phases = -gamma * inst.objective_table() + theta * inst.hamming_table()
    psi = apply_diag_phases(psi, phases)
    return apply_uniform_1q(psi, ws_mixer_matrix(inst, beta), inst.n)


def coherent_penalty_layer(inst, psi: np.ndarray, gamma: float,
                           beta: float) -> np.ndarray:
    """One coherent penalty layer: e^{-i*gamma*H} (penalized H), M(beta)."""
    psi = apply_diag_phases(psi, -gamma * inst.penalized_table())
    return apply_uniform_1q(psi, ws_mixer_matrix(inst, beta), inst.n)


def xy_basis_layer(inst, psi: np.ndarray, gamma: float, alpha: float,
                   theta: float, chi: float) -> np.ndarray:
    """One XY basis layer: e^{-i*gamma*H1}, R(alpha, theta, chi)^{(x)n}."""
    from .schur_weyl import su2_matrix

    psi = apply_diag_phases(psi, -gamma * inst.objective_table())
    return apply_uniform_1q(psi, su2_matrix(alpha, theta, chi), inst.n)


def _as_layers(layers, width: int, kind: str) -> list[tuple]:
    out = [tuple(float(a) for a in layer) for layer in layers]
    if not out:
        raise ValueError("layers must contain at least one layer (p >= 1)")
    for layer in out:
        if len(layer) != width:
            raise ValueError(f"a {kind} layer has {width} angles, got "
                             f"{len(layer)}: {layer}")
    return out


def penalty_basis_state_p(inst, layers) -> np.ndarray:
    """Depth-p penalty basis state; ``layers`` = [(gamma, theta, beta), ...]."""
    psi = warm_start_state(inst).astype(complex)
    for layer in _as_layers(layers, 3, "penalty basis"):
        psi = penalty_basis_layer(inst, psi, *layer)
    return psi


def coherent_penalty_state_p(inst, layers) -> np.ndarray:
    """Depth-p coherent penalty-QAOA state; ``layers`` = [(gamma, beta), ...]."""
    psi = warm_start_state(inst).astype(complex)
    for layer in _as_layers(layers, 2, "coherent penalty"):
        psi = coherent_penalty_layer(inst, psi, *layer)
    return psi


def xy_basis_state_p(inst, layers) -> np.ndarray:
    """Depth-p XY basis state; ``layers`` = [(gamma, alpha, theta, chi), ...]."""
    psi = warm_start_state(inst).astype(complex)
    for layer in _as_layers(layers, 4, "XY basis"):
        psi = xy_basis_layer(inst, psi, *layer)
    return psi


# ------------------------------------------------- layered (partial) LCU
def penalty_lcu_coefficients(inst, gamma: float):
    """(c, q, Gamma) of the cardinality-penalty Fourier LCU of one layer at
    angle gamma: c_j with sum_j c_j e^{i*theta_j*w} = e^{+i*gamma*lam*(w-k)^2},
    q_j = |c_j| / ||c||_1, Gamma = ||c||_1^2 (branch angles: ``dlcu.dft_angles
    (n+1)``)."""
    phases = dlcu.cardinality_penalty_phases(gamma, inst.lam, inst.k, inst.n)
    c = dlcu.fourier_coefficients(phases)
    return c, dlcu.branch_probabilities(c), dlcu.gamma(c)


@dataclass(frozen=True)
class LayeredLCU:
    """Branch structure of a layered (partial-depth) Fourier LCU.

    ``branches[b]`` holds one DFT index per LCU layer, in the order of
    ``lcu_layers`` (sorted); ``c[b]`` = prod_l c_{l, branches[b][l]};
    ``layer_gammas[l]`` is the overhead Gamma_l of the l-th LCU layer (same
    order), ``Gamma`` = prod_l Gamma_l = ||c||_1^2.
    """

    lcu_layers: tuple
    branches: list
    c: np.ndarray
    q: np.ndarray
    Gamma: float
    thetas: np.ndarray
    layer_gammas: tuple


def _lcu_index(layers, lcu_layers) -> tuple:
    p = len(layers)
    idx = tuple(sorted({int(l) for l in lcu_layers}))
    for l in idx:
        if not 0 <= l < p:
            raise ValueError(f"LCU layer index {l} out of range for p = {p}")
    return idx


def layered_lcu(inst, layers, lcu_layers) -> LayeredLCU:
    """Product-coefficient record of the layered LCU (see module docstring).

    ``layers`` are penalty basis layers (gamma, theta, beta); the layers in
    ``lcu_layers`` contribute their gamma only. There are (n+1)^|lcu_layers|
    branches, ordered as ``itertools.product`` (last LCU layer fastest).
    """
    layers = _as_layers(layers, 3, "penalty basis")
    idx = _lcu_index(layers, lcu_layers)
    m1 = inst.n + 1
    c = np.ones(1, dtype=complex)
    layer_gammas = []
    for l in idx:
        c_l, _, g_l = penalty_lcu_coefficients(inst, layers[l][0])
        c = np.multiply.outer(c, c_l).reshape(-1)
        layer_gammas.append(g_l)
    Gamma = float(np.prod(layer_gammas))
    Gamma_c = dlcu.gamma(c)
    assert abs(Gamma - Gamma_c) <= 1e-10 * max(1.0, Gamma), (Gamma, Gamma_c)
    branches = list(itertools.product(range(m1), repeat=len(idx)))
    return LayeredLCU(lcu_layers=idx, branches=branches, c=c,
                      q=dlcu.branch_probabilities(c), Gamma=Gamma,
                      thetas=dlcu.dft_angles(m1),
                      layer_gammas=tuple(layer_gammas))


def layered_lcu_states(inst, layers, lcu_layers) -> list[np.ndarray]:
    """Branch states |psi_b>, in ``layered_lcu(...).branches`` order: the
    depth-p penalty basis state with each LCU layer's theta replaced by the
    branch's DFT angle (non-LCU layers keep their own theta). Shared prefixes
    are evolved once; the arithmetic per branch is exactly that of
    ``penalty_basis_state_p``."""
    layers = _as_layers(layers, 3, "penalty basis")
    idx = set(_lcu_index(layers, lcu_layers))
    thetas = dlcu.dft_angles(inst.n + 1)
    states = [warm_start_state(inst).astype(complex)]
    for l, (g, t, b) in enumerate(layers):
        if l in idx:
            states = [penalty_basis_layer(inst, psi, g, tj, b)
                      for psi in states for tj in thetas]
        else:
            states = [penalty_basis_layer(inst, psi, g, t, b)
                      for psi in states]
    return states


def layered_lcu_distribution(inst, layers, lcu_layers):
    """(pooled p~, info): p~ = sum_b q_b |psi_b|^2, info = ``layered_lcu``."""
    info = layered_lcu(inst, layers, lcu_layers)
    dists = np.array([np.abs(psi) ** 2 for psi in
                      layered_lcu_states(inst, layers, lcu_layers)])
    return pooled_distribution(dists, info.q), info


def coherent_equivalent_state(inst, layers, lcu_layers) -> np.ndarray:
    """The state the layered LCU represents exactly, sum_b c_b |psi_b>: LCU
    layers become coherent penalty layers (gamma, beta) (theta dropped), the
    others stay penalty basis layers (gamma, theta, beta)."""
    layers = _as_layers(layers, 3, "penalty basis")
    idx = set(_lcu_index(layers, lcu_layers))
    psi = warm_start_state(inst).astype(complex)
    for l, (g, t, b) in enumerate(layers):
        psi = (coherent_penalty_layer(inst, psi, g, b) if l in idx
               else penalty_basis_layer(inst, psi, g, t, b))
    return psi


def gamma_alias_flags(inst, gamma: float, tol: float = 1e-6) -> dict:
    """Flag the trivial LCU: gamma*lam within ``tol`` of 2*pi*m makes every
    penalty phase gamma*lam*(w-k)^2 a multiple of 2*pi (the penalty is the
    identity, Gamma = 1). m = 0 (gamma = 0) is flagged too. Not flagged, but
    visible in ``Gamma``: for odd n, gamma*lam = pi*(2m+1) reduces the penalty
    to the linear phase (-1)^(w-k), a single DFT branch (Gamma = 1)."""
    x = gamma * inst.lam
    m = int(np.round(x / (2 * np.pi)))
    _, _, G = penalty_lcu_coefficients(inst, gamma)
    return {"Gamma": G, "alias": bool(abs(x - 2 * np.pi * m) <= tol),
            "nearest_m": m}


# ------------------------------------------------------------------- pooling
def pooled_distribution(dists: np.ndarray, q: np.ndarray) -> np.ndarray:
    """p~ = sum_j q_j p_j (ancilla-free LCU sampler, paper Sec. II B)."""
    q = np.asarray(q, dtype=float)
    return np.tensordot(q / q.sum(), np.asarray(dists), axes=1)


# ------------------------------------------------------------------- metrics
def distribution_metrics(inst: DensestKSubgraph, dist: np.ndarray,
                         opt_value: float | None = None) -> dict:
    """<H>, P(feasible), P(optimal feasible), <H | feasible> for a distribution
    over statevector indices."""
    flam = inst.penalized_table()
    w = inst.hamming_table()
    feas = w == inst.k
    opt = inst.brute_force()[0] if opt_value is None else opt_value
    p_feas = float(dist[feas].sum())
    optimal = feas & np.isclose(inst.objective_table(), opt)
    out = {
        "exp_H": float(np.dot(dist, flam)),
        "P_feasible": p_feas,
        "P_optimal": float(dist[optimal].sum()),
        "exp_H_feasible": (float(np.dot(dist[feas], flam[feas]) / p_feas)
                           if p_feas > 0 else np.nan),
    }
    return out
