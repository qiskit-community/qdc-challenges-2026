import numpy as np
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import CZGate, RZZGate, XGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass


class CZReplacementPass(TransformationPass):
    """
    A Qiskit transformation pass that, with probability `p`, replaces each CZ gate with three consecutive CZs and
    Rzz(theta) with Rzz(theta) Rzz(-theta) Rzz(theta).

    """
    def __init__(self, p, seed):
        super().__init__()
        self.p = p  # Probability of replacing CZ with three CZs
        np.random.seed(seed)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for run in dag.collect_runs(["cz"]):
            for node in run:
                if np.random.rand() < self.p:
                    new_dag = DAGCircuit()
                    qreg = QuantumRegister(2)
                    new_dag.add_qreg(qreg)
                    new_dag.apply_operation_back(CZGate(), [qreg[0],qreg[1]])
                    new_dag.apply_operation_back(CZGate(), [qreg[0],qreg[1]])
                    new_dag.apply_operation_back(CZGate(), [qreg[0],qreg[1]])
                    dag.substitute_node_with_dag(node, new_dag)
        for run in dag.collect_runs(["rzz"]):
            for node in run:
                if np.random.rand() < self.p:
                    new_dag = DAGCircuit()
                    qreg = QuantumRegister(2)
                    new_dag.add_qreg(qreg)
                    new_dag.apply_operation_back(RZZGate(node.params[0]), [qreg[0],qreg[1]])

                    new_dag.apply_operation_back(XGate(), [qreg[0]])
                    new_dag.apply_operation_back(RZZGate(node.params[0]), [qreg[0],qreg[1]])
                    new_dag.apply_operation_back(XGate(), [qreg[0]])

                    new_dag.apply_operation_back(RZZGate(node.params[0]), [qreg[0],qreg[1]])
                    dag.substitute_node_with_dag(node, new_dag)

        return dag


def noise_amplification(circs, prob, seed):
    return PassManager([CZReplacementPass(prob, seed)]).run(circs)


class RemoveGlobalPhasePass(TransformationPass):
    def run(self, dag):
        for node in dag.op_nodes():
            if node.op.name == "global_phase":
                dag.remove_op_node(node)
        return dag
