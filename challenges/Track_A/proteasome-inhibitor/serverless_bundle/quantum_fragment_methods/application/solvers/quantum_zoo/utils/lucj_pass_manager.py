# (C) Copyright IBM 2024-2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

"""LUCJ pass manager implementation for hardware-aware transpilation."""

from __future__ import annotations

import copy
import warnings
from collections import OrderedDict
from typing import Any, Literal, cast

import rustworkx
from qiskit.circuit import Instruction
from qiskit.providers import BackendV2
from qiskit.transpiler import (
    PassManager,
    StagedPassManager,
    generate_preset_pass_manager,
)
from qiskit.transpiler.basepasses import BasePass, TransformationPass
from qiskit.transpiler.passes import Decompose, VF2PostLayout
from qiskit.dagcircuit import DAGCircuit
import numpy as np
from rustworkx import NoEdgeBetweenNodes, PyGraph

VF2_CALL_LIMIT = 30_000_000


def _min_or_none(values):
    present = [value for value in values if value is not None]
    return min(present) if present else None


class MergeOrbitalRotations(TransformationPass):
    """Merge consecutive orbital rotation gates."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for run in dag.collect_runs(["orb_rot_jw"]):
            node = run[0]
            norb = node.op.norb
            combined_mat_a = np.eye(norb, dtype=complex)
            combined_mat_b = np.eye(norb, dtype=complex)
            tol = max(getattr(n.op, "tol", 1e-12) for n in run)
            max_givens = _min_or_none(getattr(n.op, "max_givens", None) for n in run)
            max_layers = _min_or_none(getattr(n.op, "max_layers", None) for n in run)
            optimize_kwargs = getattr(run[0].op, "optimize_kwargs", {})
            for n in run:
                combined_mat_a = getattr(n.op, "orbital_rotation_a", getattr(n.op, "orbital_rotation", np.eye(norb))) @ combined_mat_a
                combined_mat_b = getattr(n.op, "orbital_rotation_b", getattr(n.op, "orbital_rotation", np.eye(norb))) @ combined_mat_b

            op_cls = node.op.__class__
            try:
                new_op = op_cls(
                    norb,
                    (combined_mat_a, combined_mat_b),
                    tol=tol,
                    max_givens=max_givens,
                    max_layers=max_layers,
                    **optimize_kwargs,
                )
            except Exception:
                new_op = op_cls(
                    norb,
                    (combined_mat_a, combined_mat_b),
                    tol=tol,
                )
            dag.replace_block_with_op(run, new_op)

        for run in dag.collect_runs(["prep_slater_jw"]):
            node = run[0]
            if len(run) > 1:
                pass
        return dag


def pre_init_passes():
    """Yield transpiler passes recommended for the Qiskit transpiler pre_init stage."""
    yield Decompose(
        [
            "dc_trotter_split_op_jw",
            "df_trotter_jw",
            "hartree_fock_jw",
            "num_num_ansatz_jw",
            "ucj_jw",
            "ucj_spinless_jw",
        ]
    )
    yield MergeOrbitalRotations()


def _create_two_linear_chains(
    pairs_aa: list[tuple[int, int]], pairs_bb: list[tuple[int, int]]
) -> PyGraph:
    graph = rustworkx.PyGraph()
    for pairs in [pairs_aa, pairs_bb]:
        uniq: OrderedDict = OrderedDict()
        for q0, q1 in pairs:
            if q0 not in uniq:
                uniq[q0] = None
            if q1 not in uniq:
                uniq[q1] = None
        for q in uniq.keys():
            node_idx = graph.add_node(q)
            uniq[q] = node_idx
        for q0, q1 in pairs:
            node_idx0, node_idx1 = uniq[q0], uniq[q1]
            graph.add_edge(node_idx0, node_idx1, None)
    return graph


def _get_layout_graph_and_allowed_pairs_ab(
    norb: int,
    backend_coupling_graph: PyGraph,
    connectivity: Literal["heavy-hex", "square"],
    pairs_aa: list[tuple[int, int]],
    pairs_ab: list[tuple[int, int]],
    pairs_bb: list[tuple[int, int]],
) -> tuple[PyGraph, list[tuple[int, int]]]:
    isomorphic = False
    graph = _create_two_linear_chains(pairs_aa=pairs_aa, pairs_bb=pairs_bb)
    num_nodes = graph.num_nodes()
    graph_new = copy.deepcopy(graph)

    while (not isomorphic) and pairs_ab:
        graph_new = copy.deepcopy(graph)
        if connectivity == "heavy-hex":
            for i, (a, b) in enumerate(sorted(pairs_ab, key=lambda x: x[0])):
                new_node = num_nodes + i
                graph_new.add_node(new_node)
                graph_new.add_edge(a, new_node, None)
                graph_new.add_edge(new_node, b + norb, None)
        elif connectivity == "square":
            for a, b in sorted(pairs_ab, key=lambda x: x[0]):
                graph_new.add_edge(a, b + norb, None)
        else:
            raise ValueError(f"connectivity={connectivity} not supported.")

        isomorphic = rustworkx.is_subgraph_isomorphic(
            backend_coupling_graph,
            graph_new,
            call_limit=VF2_CALL_LIMIT,
            id_order=False,
            induced=False,
        )

        if not isomorphic:
            warnings.warn(
                f"Backend cannot accommodate pairs_ab={pairs_ab}.\n"
                f"Removing interaction {pairs_ab[-1]} from the end."
            )
            del pairs_ab[-1]

    return graph_new, pairs_ab


def _collect_two_q_quantum_gates(operations_in_target: list[Instruction]) -> list[str]:
    excluded_insts = ["barrier", "reset", "measurement"]
    two_q_gate_names = []
    for inst in operations_in_target:
        if inst.num_qubits == 2 and inst.name not in excluded_insts:
            two_q_gate_names.append(inst.name)
    return two_q_gate_names


def _make_backend_cmap_pygraph(
    backend: BackendV2,
    two_qubit_error_threshold: float,
    readout_error_threshold: float,
) -> PyGraph:
    target = backend.target
    two_q_gate_names = _collect_two_q_quantum_gates(target.operations)
    graph = copy.deepcopy(backend.coupling_map.graph)

    if not graph.is_symmetric():
        graph.make_symmetric()
    backend_coupling_graph = graph.to_undirected()

    edge_list = backend_coupling_graph.edge_list()
    removed_edge = []
    for edge in edge_list:
        if set(edge) in removed_edge:
            continue
        try:
            backend_coupling_graph.remove_edge(edge[0], edge[1])
            removed_edge.append(set(edge))
        except NoEdgeBetweenNodes:
            pass

    node_indices = backend_coupling_graph.node_indices()
    for node_id in node_indices:
        meas_prop = target["measure"][(node_id,)]
        if meas_prop is None:
            continue
        if meas_prop.error is not None and meas_prop.error >= readout_error_threshold:
            backend_coupling_graph.remove_node(node_id)

    edge_list = backend_coupling_graph.edge_list()
    for gate in two_q_gate_names:
        for edge in edge_list:
            if edge in target[gate]:
                found_edge = edge
            else:
                found_edge = edge[::-1]

            gate_prop = target[gate].get(found_edge)
            if gate_prop is None:
                continue

            if gate_prop.error is not None and gate_prop.error >= two_qubit_error_threshold:
                backend_coupling_graph.remove_edge(edge[0], edge[1])

    return backend_coupling_graph


def _get_placeholder_layout_and_allowed_interactions(
    backend: BackendV2,
    norb: int,
    connectivity: Literal["heavy-hex", "square"],
    pairs_aa: list[tuple[int, int]],
    pairs_ab: list[tuple[int, int]],
    pairs_bb: list[tuple[int, int]],
    two_qubit_error_threshold: float,
    readout_error_threshold: float,
) -> tuple[list[int], list[tuple[int, int]]]:
    backend_coupling_graph = _make_backend_cmap_pygraph(
        backend=backend,
        two_qubit_error_threshold=two_qubit_error_threshold,
        readout_error_threshold=readout_error_threshold,
    )

    (layout_graph, allowed_pairs_ab) = _get_layout_graph_and_allowed_pairs_ab(
        norb=norb,
        backend_coupling_graph=backend_coupling_graph,
        connectivity=connectivity,
        pairs_aa=pairs_aa,
        pairs_ab=pairs_ab,
        pairs_bb=pairs_bb,
    )
    num_allowed_pairs_ab = len(allowed_pairs_ab)
    if num_allowed_pairs_ab == 0:
        raise RuntimeError(
            "No alpha-beta interaction can be accommodated. Terminating."
        )

    isomorphic_mappings = rustworkx.vf2_mapping(
        backend_coupling_graph,
        layout_graph,
        subgraph=True,
        id_order=False,
        induced=False,
        call_limit=VF2_CALL_LIMIT,
    )

    mapping = next(isomorphic_mappings, None)
    if mapping is None:
        raise RuntimeError("No layout is found.")

    initial_layout = [-1] * (2 * norb + num_allowed_pairs_ab)
    for key, value in mapping.items():
        initial_layout[value] = key

    if -1 in initial_layout[:-num_allowed_pairs_ab]:
        raise ValueError(
            f"Not all qubits in placeholder initial_layout properly set: {initial_layout[:-num_allowed_pairs_ab]}"
        )

    return initial_layout[:-num_allowed_pairs_ab], allowed_pairs_ab


def generate_lucj_pass_manager(
    backend: BackendV2,
    norb: int,
    connectivity: Literal["heavy-hex", "square"] = "heavy-hex",
    interaction_pairs: tuple[list[tuple[int, int]], list[tuple[int, int]] | None]
    | tuple[list[tuple[int, int]], list[tuple[int, int]] | None, list[tuple[int, int]]] = None,
    two_qubit_error_threshold: float = 1.0,
    readout_error_threshold: float = 0.10,
    **qiskit_pm_kwargs,
) -> tuple[StagedPassManager, list[tuple[int, int]]]:
    """Generate a Qiskit preset pass manager for the LUCJ ansatz."""
    if "initial_layout" in qiskit_pm_kwargs:
        warnings.warn("Argument ``initial_layout`` is ignored.")
        del qiskit_pm_kwargs["initial_layout"]

    if "layout_method" in qiskit_pm_kwargs:
        warnings.warn("Argument ``layout_method`` is ignored.")
        del qiskit_pm_kwargs["layout_method"]

    if interaction_pairs is None:
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = None
        pairs_bb = pairs_aa
    elif len(interaction_pairs) == 2:
        pairs_aa, pairs_ab = interaction_pairs
        pairs_bb = pairs_aa
    elif len(interaction_pairs) == 3:
        pairs_aa, pairs_ab, pairs_bb = interaction_pairs
    else:
        raise ValueError("``interaction_pairs`` must be either length-2 or length-3 tuple.")

    if pairs_aa is not None:
        for p0, p1 in pairs_aa:
            if not ((0 <= p0 < norb) and (0 <= p1 < norb)):
                raise ValueError(f"Interaction pair {(p0, p1)} is not in range norb [0, {norb}).")

    if connectivity.lower() not in ["heavy-hex", "square"]:
        raise ValueError(f"connectivity={connectivity} not supported.")
    connectivity = cast(Literal["heavy-hex", "square"], connectivity.lower())

    if pairs_ab is None:
        if connectivity == "heavy-hex":
            resolved_pairs_ab = [(p, p) for p in range(norb) if p % 4 == 0]
        elif connectivity == "square":
            resolved_pairs_ab = [(p, p) for p in range(norb)]
    else:
        resolved_pairs_ab = list(pairs_ab)

    (placeholder_initial_layout, allowed_pairs_ab) = (
        _get_placeholder_layout_and_allowed_interactions(
            backend=backend,
            norb=norb,
            connectivity=connectivity,
            pairs_ab=resolved_pairs_ab,
            two_qubit_error_threshold=two_qubit_error_threshold,
            readout_error_threshold=readout_error_threshold,
            pairs_aa=pairs_aa,
            pairs_bb=pairs_bb,
        )
    )

    pm = generate_preset_pass_manager(
        backend=backend, initial_layout=placeholder_initial_layout, **qiskit_pm_kwargs
    )
    pm.pre_init = PassManager(list(pre_init_passes()))

    def _custom_apply_post_layout_condition(property_set: dict[str, Any]) -> bool:
        return property_set["post_layout"] is not None

    pm.routing.append(
        VF2PostLayout(
            target=backend.target, strict_direction=False, call_limit=VF2_CALL_LIMIT
        )
    )
    return pm, allowed_pairs_ab
