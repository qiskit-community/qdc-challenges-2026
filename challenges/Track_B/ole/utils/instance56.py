"""The paper's 56-qubit OLE instance (Barron et al., arXiv:2607.25998), and relabeling helpers.

The specification in ../data/ole56_instance.json was extracted from the published circuits
(Quantum Advantage Tracker, operator_loschmidt_echo_56x1488, Apache-2.0). A circuit built from it
matches those circuits exactly, up to global phase, for every published scattering strength.
Qubit labels are the physical Heron qubits of the published ibm_boston layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from qiskit import QuantumCircuit

SPEC = json.loads((Path(__file__).resolve().parents[1] / "data" / "ole56_instance.json").read_text())
NUM_DEVICE_QUBITS = 156


def instance_edges() -> list[tuple[int, int]]:
    """All couplers of the instance, as sorted pairs."""
    return sorted((min(a, b), max(a, b)) for group in SPEC["edge_groups"] for a, b in group)


def relabel(circuit: QuantumCircuit, placement: dict[int, int]) -> QuantumCircuit:
    """Move a circuit on the published qubits to other physical qubits, `placement[old] = new`."""
    moved = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    for inst in circuit.data:
        qubits = [placement.get(circuit.find_bit(q).index, circuit.find_bit(q).index) for q in inst.qubits]
        clbits = [circuit.find_bit(c).index for c in inst.clbits]
        moved.append(inst.operation, qubits, clbits)
    return moved
