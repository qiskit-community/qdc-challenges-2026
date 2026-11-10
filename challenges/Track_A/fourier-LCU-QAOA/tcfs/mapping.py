"""The Map step of the notebook with ``qiskit-addon-opt-mapper`` (notebook
v0.4, design Sec. 3.5 and 5.4).

The notebook models densest-k-subgraph as an ``OptimizationProblem``,
turns the cardinality constraint into the quadratic penalty with
``EqualityToPenalty(penalty=lam)`` and derives Ising operators with
``to_ising``. This module holds the same construction for the tests.

Conventions of ``to_ising`` (opt-mapper 0.1.0), verified in
``tests/test_mapping.py``: it returns ``(op, offset)`` for the problem to
*minimize*, and for a maximization problem it negates the objective, so
``op + offset * I`` has diagonal ``-f(x)``. Hence ``-op`` is the kit's cost
operator ``circuits.hamiltonian_pauli(inst, include_penalty=False)`` up to
its constant term, and ``-op`` of the penalized problem is
``hamiltonian_pauli(inst)`` (the operator of f_lam) up to a constant.
Qubit i is variable x_i (declared in index order).
"""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp

from .instances import DensestKSubgraph


def dks_problem(inst: DensestKSubgraph, constrained: bool = True):
    """The ``OptimizationProblem`` of the notebook: maximize sum_(i,j) w_ij
    x_i x_j over binary x_0 .. x_{n-1}, with the constraint sum_i x_i = k
    when ``constrained``."""
    from qiskit_addon_opt_mapper import OptimizationProblem

    problem = OptimizationProblem("densest-k-subgraph")
    for i in range(inst.n):
        problem.binary_var(f"x{i}")
    problem.maximize(quadratic={(f"x{i}", f"x{j}"): w for i, j, w in inst.edges})
    if constrained:
        problem.linear_constraint({f"x{i}": 1 for i in range(inst.n)}, "==",
                                  inst.k, name="cardinality")
    return problem


def ising(inst: DensestKSubgraph, penalized: bool = False):
    """``to_ising`` of the objective (``penalized=False``) or of the problem
    with the constraint turned into ``lam * (sum_i x_i - k)^2`` by
    ``EqualityToPenalty(penalty=inst.lam)``: ``(op, offset)`` with
    ``op + offset * I`` = -f (or -f_lam) on the diagonal."""
    from qiskit_addon_opt_mapper.converters import EqualityToPenalty
    from qiskit_addon_opt_mapper.translators import to_ising

    if not penalized:
        return to_ising(dks_problem(inst, constrained=False))
    converted = EqualityToPenalty(penalty=inst.lam).convert(
        dks_problem(inst, constrained=True))
    return to_ising(converted)


def cost_operator(inst: DensestKSubgraph) -> SparsePauliOp:
    """H1 = f as a Pauli operator without its constant term: minus the
    ``to_ising`` operator of the objective (what the notebook calls
    ``cost_op``). ``PauliEvolutionGate(cost_operator(inst), time=gamma)``
    is the cost layer e^{-i gamma H1} up to a global phase."""
    op, _ = ising(inst, penalized=False)
    return (-op).simplify()


def penalized_operator(inst: DensestKSubgraph) -> SparsePauliOp:
    """f_lam = f - lam (W - k)^2 as a Pauli operator without its constant
    term (minus the ``to_ising`` operator of the penalized problem)."""
    op, _ = ising(inst, penalized=True)
    return (-op).simplify()


def zz_terms(op: SparsePauliOp, atol: float = 1e-12) -> list[tuple[int, int]]:
    """Sorted qubit pairs (i, j), i < j, that carry a Z_i Z_j term with a
    nonzero coefficient (terms with X or Y are ignored)."""
    pairs = set()
    for pauli, c in zip(op.paulis, op.coeffs):
        if abs(c) <= atol or pauli.x.any():
            continue
        qs = [int(q) for q in pauli.z.nonzero()[0]]
        if len(qs) == 2:
            pairs.add((min(qs), max(qs)))
    return sorted(pairs)


def z_terms(op: SparsePauliOp, atol: float = 1e-12) -> list[int]:
    """Sorted qubits that carry a single-qubit Z term with a nonzero
    coefficient."""
    out = set()
    for pauli, c in zip(op.paulis, op.coeffs):
        if abs(c) <= atol or pauli.x.any():
            continue
        qs = pauli.z.nonzero()[0]
        if len(qs) == 1:
            out.add(int(qs[0]))
    return sorted(out)
