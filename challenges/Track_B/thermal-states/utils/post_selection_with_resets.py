"""
`AddPostSelectionMeasures` variant that tolerates mid-circuit `reset` and `delay`.

The upstream pass rejects any op that is not a standard gate, barrier, measure, or control-flow
op. The Gibbs circuit needs two more: `reset` on the bath qubits between Trotter steps, and
`delay` from the ZNE idle stretching. So the pass cannot be used as-is.

Both are treated exactly like a gate: they mark their qubit as *not* terminated by a measurement,
so a measure preceding a reset is correctly not considered terminal. All the real work -- the
xslow pulse sequence, the copied `_ps` classical registers, the barrier placement -- is left to
the upstream pass.

This works around a library gap; it is not a behavioural change to the post selection itself.
"""

from __future__ import annotations

from qiskit.circuit import ControlFlowOp, Clbit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError

from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
    AddPostSelectionMeasures,
)


NON_TERMINATING_OPS = {"reset", "delay"}


class AddPostSelectionMeasuresWithResets(AddPostSelectionMeasures):
    """Add post selection measurements, allowing mid-circuit resets and delays."""

    def _find_terminal_measurements(self, dag: DAGCircuit) -> dict[Qubit, Clbit]:
        terminal_measurements: dict[Qubit, Clbit | None] = {}

        for node in dag.topological_op_nodes():
            name = node.op.name
            if node.is_standard_gate() or name in NON_TERMINATING_OPS:
                for qarg in node.qargs:
                    terminal_measurements[qarg] = None
            elif name == "barrier":
                continue
            elif name == "measure":
                terminal_measurements[node.qargs[0]] = node.cargs[0]
            elif isinstance(node.op, ControlFlowOp):
                all_terminal_measurements: list[dict[Qubit, Clbit]] = []
                for block in node.op.blocks:
                    block_dag = circuit_to_dag(block)
                    clbit_map = {
                        block_clbit: clbit
                        for block_clbit, clbit in zip(block_dag.clbits, node.cargs)
                    }
                    qubit_map = {
                        block_qubit: qubit
                        for block_qubit, qubit in zip(block_dag.qubits, node.qargs)
                    }
                    block_terminal = self._find_terminal_measurements(block_dag)
                    all_terminal_measurements.append({
                        qubit_map[q]: (None if c is None else clbit_map[c])
                        for q, c in block_terminal.items()
                    })

                if len(all_terminal_measurements) == 1:
                    terminal_measurements.update(all_terminal_measurements[0])
                else:
                    for qubit in node.qargs:
                        clbits = {d.get(qubit) for d in all_terminal_measurements}
                        if len(clbits) == 1:
                            terminal_measurements[qubit] = next(iter(clbits))
                        else:
                            terminal_measurements[qubit] = None
            else:
                raise TranspilerError("``'{}'`` is not supported.".format(name))

        return terminal_measurements
