"""
Transpiler pass to rewrite Clifford-angle RZZ gates in terms of CZ, RZ and Z.

`transpile(basis_gates=[..., "rzz", ...], optimization_level=2)` collects adjacent two-qubit
gates into KAK blocks and re-synthesizes them. Where the collected block is a Clifford -- the
three-CNOT SWAP runs at the start and end of `triangle_circ`, for instance -- the synthesizer
emits RZZ at a multiple of pi/2, because that is the natural KAK form when `rzz` is available.

Those RZZ are not fractional gates: RZZ(pi/2) is CZ plus two virtual RZ, and RZZ(pi) is purely
local. Leaving them as RZZ mixes Clifford and fractional two-qubit gates in the same box, which
samplomatic >= 0.21 handles differently from earlier versions under `local_pauli` twirling
(`builders/get_builder.py` excludes |angle| == pi/2 from the gate-dependent twirl set, routing
those qubits through Pauli-past-Clifford propagation instead).

Rewriting them costs nothing in two-qubit gate count -- one RZZ becomes one CZ -- and pays only
in virtual RZ, so the optimization_level=2 gate count is preserved while the genuinely fractional
RZZ are left untouched.

Identities used, all exact up to global phase, with k = theta / (pi/2) mod 4:

    k = 0   ->  identity
    k = 1   ->  CZ . RZ(+pi/2) . RZ(+pi/2)
    k = 2   ->  Z . Z                        (local, removes a two-qubit gate)
    k = 3   ->  CZ . RZ(-pi/2) . RZ(-pi/2)

Run it directly after the optimization_level=2 transpile and before the boxing pass manager.
Placing it later is not sufficient: nothing downstream re-fuses the gates, but the boxing pass
manager reads the gate names to decide the twirl strategy.

Usage:
    pm = PassManager([DeCliffordRZZ()])
    circuit = pm.run(circuit)
    print(pm.passes()[0][0].max_deviation)   # audit how far any angle was snapped
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import CZGate, RZGate, ZGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass

HALF_PI = np.pi / 2


class DeCliffordRZZ(TransformationPass):
    """
    Replace RZZ gates whose angle is a multiple of pi/2 with CZ, RZ and Z.

    Fractional RZZ and RZZ carrying an unbound `Parameter` are left alone.

    Angles within `atol` of a multiple of pi/2 are snapped to it, so the pass can change the
    unitary by up to `atol` radians per gate. `conversions` and `max_deviation` record how many
    gates were rewritten and the largest snap applied, so a run can be audited: a `max_deviation`
    of ~0 means every rewritten gate was already exactly Clifford.

    Args:
        atol: Absolute tolerance, in radians, for treating an angle as a multiple of pi/2.
    """

    def __init__(self, atol: float = 0.01):
        super().__init__()
        self.atol = atol
        self.conversions = 0
        self.max_deviation = 0.0

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in list(dag.op_nodes()):
            if node.op.name != "rzz":
                continue

            param = node.op.params[0]
            if getattr(param, "parameters", None):
                continue

            theta = float(param)
            k = int(round(theta / HALF_PI))
            deviation = abs(theta - k * HALF_PI)
            if deviation > self.atol:
                continue

            self.conversions += 1
            self.max_deviation = max(self.max_deviation, deviation)

            # RZZ(theta + 2*pi) = -RZZ(theta), so k is only meaningful mod 4
            mini = DAGCircuit()
            register = QuantumRegister(2)
            mini.add_qreg(register)

            k %= 4
            if k == 2:
                mini.apply_operation_back(ZGate(), [register[0]])
                mini.apply_operation_back(ZGate(), [register[1]])
            elif k in (1, 3):
                angle = HALF_PI if k == 1 else -HALF_PI
                mini.apply_operation_back(CZGate(), [register[0], register[1]])
                mini.apply_operation_back(RZGate(angle), [register[0]])
                mini.apply_operation_back(RZGate(angle), [register[1]])

            dag.substitute_node_with_dag(node, mini)

        return dag


def declifford_rzz(circuits, atol: float = 0.01):
    """Convenience wrapper applying the pass through a pass manager."""
    return PassManager([DeCliffordRZZ(atol=atol)]).run(circuits)
