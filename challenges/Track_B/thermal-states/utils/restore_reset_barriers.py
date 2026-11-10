"""
Transpiler pass to restore pre_reset_{n}/post_reset_{n} labelled barriers in a circuit
whose barriers were stripped by samplomatic's boxing pass manager.

samplomatic's generate_boxing_pass_manager(remove_barriers='after_stratification') appends
stock qiskit RemoveBarriers, which is label-blind (dag.remove_all_ops_named("barrier")).
That un-fences the reset windows: system-qubit idle then merges with the surrounding idle,
roughly doubling the longest unrefocused interval and growing with the ZNE noise factor.

Restoring all-layout-width barriers around each reset window re-establishes the fence, so DD
refocuses a bounded interval. (The labels themselves are only needed by a DD pass that matches
on them; stock PadDynamicalDecoupling fills every idle window regardless. They are kept because
they make the windows locatable for diagnostics.)

This pass locates reset windows structurally -- by finding contiguous runs of Reset
instructions rather than by label -- and wraps each in fresh labelled barriers spanning
the requested qubits. Run it on the template circuit AFTER samplomatic build(), BEFORE
ALAPScheduleAnalysis and the DD pass.

Usage:
    pm = PassManager([
        RestoreResetBarriers(barrier_qubits=layout),
        ALAPScheduleAnalysis(durations),
        PadDynamicalDecoupling(durations, dd_sequence, spacing=..., skip_reset_qubits=True),
    ])
    template_t = pm.run(template)
"""

from __future__ import annotations

from typing import Iterable, Sequence

from qiskit.circuit import Barrier, QuantumCircuit
from qiskit.circuit.reset import Reset
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass


class RestoreResetBarriers(TransformationPass):
    """
    Wrap each contiguous run of Reset instructions in pre_reset_{n}/post_reset_{n} barriers.

    Reset windows are found structurally. Walking the circuit in order, a "run" is a maximal
    set of Reset instructions uninterrupted by any non-Reset, non-barrier operation. Each run
    is assumed to be one logical reset window and is numbered sequentially from 0, matching the
    `step` numbering the circuit generator originally used.

    Args:
        barrier_qubits: Qubit indices the inserted barriers should span. Pass the full layout
            (system + bath) to reproduce the original all-qubit fence, which is what both
            restores reset DD and prevents system-qubit idle from merging with surrounding
            idle. If None, the barriers span every qubit in the circuit.
        start_index: First integer used in the barrier labels.
    """

    def __init__(
        self,
        barrier_qubits: Sequence[int] | None = None,
        start_index: int = 0,
    ):
        super().__init__()
        self.barrier_qubits = None if barrier_qubits is None else list(barrier_qubits)
        self.start_index = start_index

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        circuit = dag_to_circuit(dag)
        restored = restore_reset_barriers(
            circuit,
            barrier_qubits=self.barrier_qubits,
            start_index=self.start_index,
        )
        return circuit_to_dag(restored)


def _reset_runs(circuit: QuantumCircuit) -> list[tuple[int, int]]:
    """
    Return (first_idx, last_idx) instruction-index bounds of each contiguous run of Resets.

    Barriers do not break a run: samplomatic emits its own dressing barriers inside the reset
    region, and the bath resets are one logical window regardless.

    Delays on qubits already reset within the current run also do not break it. The ZNE noise
    amplification appends `delay(int(2*prob*reset_duration))` after each reset, inside the
    original pre/post_reset barrier pair, so those delays belong to the window. Restricting
    absorption to qubits that already reset in this run keeps unrelated delays from extending it.
    """
    runs = []
    current_first = None
    current_last = None
    current_qubits: set = set()

    def close():
        nonlocal current_first, current_last, current_qubits
        if current_first is not None:
            runs.append((current_first, current_last))
        current_first = None
        current_last = None
        current_qubits = set()

    for idx, instruction in enumerate(circuit.data):
        op = instruction.operation
        if isinstance(op, Reset):
            if current_first is None:
                current_first = idx
            current_last = idx
            current_qubits.update(instruction.qubits)
        elif isinstance(op, Barrier):
            continue
        elif (
            current_first is not None
            and op.name == "delay"
            and set(instruction.qubits) <= current_qubits
        ):
            current_last = idx
        else:
            close()

    close()

    return runs


def restore_reset_barriers(
    circuit: QuantumCircuit,
    barrier_qubits: Iterable[int] | None = None,
    start_index: int = 0,
) -> QuantumCircuit:
    """
    Insert pre_reset_{n}/post_reset_{n} barriers around each contiguous run of Resets.

    Returns a new circuit; the input is not modified.
    """
    runs = _reset_runs(circuit)

    if barrier_qubits is None:
        fence = list(range(circuit.num_qubits))
    else:
        fence = list(barrier_qubits)

    open_before = {first: n for n, (first, _) in enumerate(runs, start=start_index)}
    close_after = {last: n for n, (_, last) in enumerate(runs, start=start_index)}

    new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    new_circuit.global_phase = circuit.global_phase
    fence_qubits = [new_circuit.qubits[q] for q in fence]

    for idx, instruction in enumerate(circuit.data):
        if idx in open_before:
            new_circuit.barrier(fence_qubits, label="pre_reset_{:d}".format(open_before[idx]))

        new_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)

        if idx in close_after:
            new_circuit.barrier(fence_qubits, label="post_reset_{:d}".format(close_after[idx]))

    return new_circuit
