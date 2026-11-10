"""Qiskit circuits for all constructions (paper Figs. 3 and 5).

Global phases are tracked exactly (via ``QuantumCircuit.global_phase``) so
that every circuit implements its target unitary *exactly*, not merely up to
phase. This matters for LCU: the coefficients c_j multiply operators, so a
dropped global phase silently corrupts the linear combination.

Gate-order pitfalls (both flagged in the paper's figures):
- Warm-start mixer (Fig. 3): as a matrix, M(beta) = RY(t) RZ(-2 beta) RY(-t);
  in *circuit order* that is ry(-t), rz(-2 beta), ry(t) - the paper's and
  the authors' released-code convention (beta values are directly comparable).
- XY basis layer (Fig. 5): R(alpha, theta, chi) = RZ(alpha) RY(theta) RZ(chi)
  as matrices, so the circuit applies rz(chi), ry(theta), rz(alpha) - the
  alpha rotation lands immediately before measurement, which is why alpha is
  irrelevant at p = 1 (paper Sec. VI B).

Depth p (the ``*_p`` builders) uses the layer tuples of ``tcfs.exact``, in
this module's argument order: penalty basis ``(gamma, theta, beta)``, coherent
penalty ``(gamma, beta)``, XY basis ``(gamma, alpha, theta, chi)``. The warm
start is applied once; ``layers`` is a sequence of tuples, p = len(layers). A
one-layer ``*_p`` circuit has exactly the ops and global phase of the p = 1
builder.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from .instances import DensestKSubgraph


# ------------------------------------------------------------------ operators
def hamiltonian_pauli(inst: DensestKSubgraph,
                      include_penalty: bool = True) -> SparsePauliOp:
    """H with H|x> = f_lam(x)|x> (or f(x) if ``include_penalty=False``).

    Uses x_i = (1 - z_i)/2 and, for the penalty,
    (1^T x - k)^2 = ((n/2 - k) I - sum_i Z_i / 2)^2.
    """
    n, k, lam = inst.n, inst.k, inst.lam
    terms = []
    const = 0.0
    for i, j, w in inst.edges:  # w * x_i x_j
        const += w / 4
        terms.append((("Z", [i], -w / 4)))
        terms.append((("Z", [j], -w / 4)))
        terms.append((("ZZ", [i, j], w / 4)))
    if include_penalty:
        c0 = n / 2 - k
        const -= lam * (c0 ** 2 + n / 4)
        for i in range(n):
            terms.append((("Z", [i], lam * c0)))
        for i in range(n):
            for j in range(i + 1, n):
                terms.append((("ZZ", [i, j], -lam / 2)))
    op = SparsePauliOp.from_sparse_list(
        [(p, q, c) for p, q, c in terms], num_qubits=n)
    return (op + SparsePauliOp.from_sparse_list([("", [], const)],
                                                num_qubits=n)).simplify()


# -------------------------------------------------------------------- layers
def warm_start_layer(inst: DensestKSubgraph) -> QuantumCircuit:
    """|psi_0^k> = prod_i (sqrt(1-k/n)|0> + sqrt(k/n)|1>) via RY(theta_init)."""
    qc = QuantumCircuit(inst.n, name="warm_start")
    qc.ry(inst.theta_init, range(inst.n))
    return qc


def cost_layer(inst: DensestKSubgraph, gamma) -> QuantumCircuit:
    """Exactly e^{-i*gamma*H1} with H1|x> = f(x)|x> (graph-local RZZ/RZ).

    Per edge: e^{-i*gamma*w*x_i*x_j} with x_i x_j = (1 - z_i - z_j + z_i z_j)/4,
    realized as rzz(gamma*w/2), rz(-gamma*w/2) on both qubits, and a global
    phase -gamma*w/4.
    """
    qc = QuantumCircuit(inst.n, name="cost")
    for i, j, w in inst.edges:
        qc.rzz(gamma * w / 2, i, j)
        qc.rz(-gamma * w / 2, i)
        qc.rz(-gamma * w / 2, j)
        qc.global_phase -= gamma * w / 4
    return qc


def rz_phase_layer(n: int, theta) -> QuantumCircuit:
    """Exactly V(theta)|x> = e^{i*theta*(1^T x)}|x>: an RZ layer plus the
    global phase e^{i*theta*n/2} (paper Sec. III)."""
    qc = QuantumCircuit(n, name="lcu_rz")
    qc.rz(theta, range(n))
    qc.global_phase += theta * n / 2
    return qc


def warm_start_mixer_layer(inst: DensestKSubgraph, beta) -> QuantumCircuit:
    """Warm-start single-qubit mixer (paper Fig. 3, Ref. [37]).

    Matrix per qubit: RY(theta_init) RZ(-2 beta) RY(-theta_init) =
    exp(+i*beta * n_hat . sigma) with n_hat the warm-start axis (paper /
    authors' convention: rz(-2*beta)).
    """
    t = inst.theta_init
    qc = QuantumCircuit(inst.n, name="ws_mixer")
    qc.ry(-t, range(inst.n))
    qc.rz(-2 * beta, range(inst.n))
    qc.ry(t, range(inst.n))
    return qc


def collective_rotation_layer(n: int, alpha, theta, chi) -> QuantumCircuit:
    """R(alpha, theta, chi)^{(x)n} = (RZ(alpha) RY(theta) RZ(chi))^{(x)n}.

    Circuit order per qubit: rz(chi), ry(theta), rz(alpha). Note RZ(phi) =
    e^{-i*phi*Z/2} exactly in Qiskit, so no global-phase correction is needed.
    """
    qc = QuantumCircuit(n, name="collective_R")
    qc.rz(chi, range(n))
    qc.ry(theta, range(n))
    qc.rz(alpha, range(n))
    return qc


def _append_penalty(qc: QuantumCircuit, inst: DensestKSubgraph,
                    gamma) -> None:
    """Append e^{+i*gamma*lam*(w-k)^2} to ``qc`` in place (ops + global phase).

    In place rather than via ``compose`` so ``coherent_penalty_circuit`` keeps
    its global phase bit for bit (``compose`` re-wraps a sub-circuit's phase
    into [0, 2*pi) first, which changes the last bits of the sum).
    """
    n, k, lam = inst.n, inst.k, inst.lam
    # e^{+i*g*(w-k)^2}, g = gamma*lam, with (w-k)^2 expanded in z-variables:
    # (w-k)^2 = (c0 - sum_i z_i/2)^2, c0 = n/2 - k
    #         = [c0^2 + n/4] - c0*sum_i z_i + (1/2)*sum_{i<j} z_i z_j
    g = gamma * lam
    c0 = n / 2 - k
    qc.global_phase += g * (c0 ** 2 + n / 4)
    qc.rz(2 * g * c0, range(n))            # e^{-i*g*c0*z_i} per qubit
    for i in range(n):
        for j in range(i + 1, n):
            qc.rzz(-g, i, j)               # e^{+i*(g/2)*z_i*z_j}


def penalty_layer(inst: DensestKSubgraph, gamma) -> QuantumCircuit:
    """Exactly e^{-i*gamma*h2} = e^{+i*gamma*lam*(1^T x - k)^2}: the O(n^2)
    all-to-all RZZ block of the coherent penalty circuit (global phase
    included). cost_layer(gamma) followed by this is e^{-i*gamma*H}."""
    qc = QuantumCircuit(inst.n, name="penalty")
    _append_penalty(qc, inst, gamma)
    return qc


# ------------------------------------------------------------- full circuits
def penalty_basis_circuit(inst: DensestKSubgraph, gamma, theta, beta,
                          measure: bool = False) -> QuantumCircuit:
    """Fig. 3: warm start -> e^{-i*gamma*H1} -> RZ LCU layer -> ws mixer."""
    qc = QuantumCircuit(inst.n)
    qc.compose(warm_start_layer(inst), inplace=True)
    qc.compose(cost_layer(inst, gamma), inplace=True)
    qc.compose(rz_phase_layer(inst.n, theta), inplace=True)
    qc.compose(warm_start_mixer_layer(inst, beta), inplace=True)
    if measure:
        qc.measure_all()
    return qc


def xy_basis_circuit(inst: DensestKSubgraph, gamma, alpha, theta, chi,
                     measure: bool = False) -> QuantumCircuit:
    """Fig. 5: warm start -> e^{-i*gamma*H1} -> R(alpha, theta, chi)^{(x)n}."""
    qc = QuantumCircuit(inst.n)
    qc.compose(warm_start_layer(inst), inplace=True)
    qc.compose(cost_layer(inst, gamma), inplace=True)
    qc.compose(collective_rotation_layer(inst.n, alpha, theta, chi),
               inplace=True)
    if measure:
        qc.measure_all()
    return qc


def coherent_penalty_circuit(inst: DensestKSubgraph, gamma, beta,
                             measure: bool = False) -> QuantumCircuit:
    """The fully coherent penalty-QAOA circuit e^{-i*beta*HM} e^{-i*gamma*H}.

    The penalty needs O(n^2) all-to-all RZZ gates - this is the expensive
    circuit the LCU replaces. Built explicitly for gate counting and exact
    verification. Realizes e^{-i*gamma*(-lam*(1^T x - k)^2)} =
    e^{+i*gamma*lam*(w - k)^2} term by term.
    """
    qc = QuantumCircuit(inst.n)
    qc.compose(warm_start_layer(inst), inplace=True)
    qc.compose(cost_layer(inst, gamma), inplace=True)
    _append_penalty(qc, inst, gamma)
    qc.compose(warm_start_mixer_layer(inst, beta), inplace=True)
    if measure:
        qc.measure_all()
    return qc


def parametrized_penalty_basis(inst: DensestKSubgraph):
    """(circuit, (gamma, theta, beta)) with free Parameters, for hardware."""
    g, t, b = Parameter("gamma"), Parameter("theta"), Parameter("beta")
    return penalty_basis_circuit(inst, g, t, b, measure=True), (g, t, b)


# ---------------------------------------------------------------- depth p
def _check_layers(layers, width: int, kind: str) -> list[tuple]:
    out = [tuple(layer) for layer in layers]
    if not out:
        raise ValueError("layers must contain at least one layer (p >= 1)")
    for layer in out:
        if len(layer) != width:
            raise ValueError(f"a {kind} layer has {width} angles, got "
                             f"{len(layer)}: {layer}")
    return out


def penalty_basis_circuit_p(inst: DensestKSubgraph, layers,
                            measure: bool = False) -> QuantumCircuit:
    """Depth-p Fig. 3 circuit: warm start, then per layer (gamma, theta,
    beta): e^{-i*gamma*H1} -> V(theta) (RZ layer + global phase) -> M(beta)."""
    qc = QuantumCircuit(inst.n)
    qc.compose(warm_start_layer(inst), inplace=True)
    for gamma, theta, beta in _check_layers(layers, 3, "penalty basis"):
        qc.compose(cost_layer(inst, gamma), inplace=True)
        qc.compose(rz_phase_layer(inst.n, theta), inplace=True)
        qc.compose(warm_start_mixer_layer(inst, beta), inplace=True)
    if measure:
        qc.measure_all()
    return qc


def coherent_penalty_circuit_p(inst: DensestKSubgraph, layers,
                               measure: bool = False) -> QuantumCircuit:
    """Depth-p coherent penalty QAOA: warm start, then per layer (gamma,
    beta): e^{-i*gamma*H} (cost + penalty block) -> M(beta)."""
    qc = QuantumCircuit(inst.n)
    qc.compose(warm_start_layer(inst), inplace=True)
    for gamma, beta in _check_layers(layers, 2, "coherent penalty"):
        qc.compose(cost_layer(inst, gamma), inplace=True)
        _append_penalty(qc, inst, gamma)
        qc.compose(warm_start_mixer_layer(inst, beta), inplace=True)
    if measure:
        qc.measure_all()
    return qc


def xy_basis_circuit_p(inst: DensestKSubgraph, layers,
                       measure: bool = False) -> QuantumCircuit:
    """Depth-p Fig. 5 circuit: warm start, then per layer (gamma, alpha,
    theta, chi): e^{-i*gamma*H1} -> R(alpha, theta, chi)^{(x)n}."""
    qc = QuantumCircuit(inst.n)
    qc.compose(warm_start_layer(inst), inplace=True)
    for gamma, alpha, theta, chi in _check_layers(layers, 4, "XY basis"):
        qc.compose(cost_layer(inst, gamma), inplace=True)
        qc.compose(collective_rotation_layer(inst.n, alpha, theta, chi),
                   inplace=True)
    if measure:
        qc.measure_all()
    return qc


def parametrized_penalty_basis_p(inst: DensestKSubgraph, p: int):
    """(measured depth-p penalty basis circuit, params) with free Parameters
    gamma_0, theta_0, beta_0, gamma_1, ...; ``params`` is that flat tuple in
    layer order, so ``qc.assign_parameters(dict(zip(params,
    np.ravel(layers))))`` binds a list of (gamma, theta, beta) layers."""
    if int(p) < 1:
        raise ValueError(f"depth p must be >= 1, got {p}")
    layers = [(Parameter(f"gamma_{l}"), Parameter(f"theta_{l}"),
               Parameter(f"beta_{l}")) for l in range(int(p))]
    params = tuple(a for layer in layers for a in layer)
    return penalty_basis_circuit_p(inst, layers, measure=True), params
