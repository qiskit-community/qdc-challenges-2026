"""Provided native-RZZ compilation and placement tools for the Floquet challenge.

The notebook supplies its accepted circuit-building and normalization functions.
Importing these helpers neither connects to a backend nor submits a QPU job.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

if TYPE_CHECKING:
    from qiskit.transpiler import StagedPassManager


def make_pm(layout: Sequence[int], backend: Any) -> StagedPassManager:
    """Build the challenge's fixed-layout pass manager without routing.

    Args:
        layout: Logical-to-physical qubit placement, in logical-qubit order.
        backend: Backend exposing the native fractional-RZZ target.

    Returns:
        A level-one preset pass manager with routing disabled and transpiler
        seed 12345. Creating it does not run a circuit or submit a job.
    """
    return generate_preset_pass_manager(
        backend=backend,
        optimization_level=1,
        initial_layout=layout,
        routing_method="none",
        seed_transpiler=12345,
    )


def check_isa(
    isa: QuantumCircuit,
    expected_rzz: int,
    backend: Any,
    layout: Sequence[int],
    *,
    theta_zz: float,
) -> dict[str, int | list[int]]:
    """Audit native RZZ counts, angles, and physical placement after compilation.

    Args:
        isa: Compiled circuit whose qubit indices identify physical qubits.
        expected_rzz: RZZ count before compilation, including intentional folds.
        backend: Backend whose RZZ target entries define supported physical edges.
        layout: Physical qubits in the selected logical-to-physical placement.
        theta_zz: Positive native angle required for every RZZ, compared with
            ``numpy.isclose`` using its default tolerances.

    Returns:
        RZZ, SX, and X counts, circuit depth, and sorted selected physical qubits.

    Raises:
        ValueError: Compilation changed the entangler count/type, used an
            unsupported edge or angle, left the selected patch, or added routing
            gates. The input circuit and backend are not modified.
    """
    entanglers = [
        inst for inst in isa.data
        if inst.operation.num_qubits == 2 and inst.operation.name != "barrier"
    ]
    if len(entanglers) != expected_rzz or any(
        inst.operation.name != "rzz" for inst in entanglers
    ):
        raise ValueError(
            f"Expected {expected_rzz} RZZ gates, got {len(entanglers)} entanglers"
        )
    allowed = {tuple(sorted(qubits)) for qubits in backend.target["rzz"]
               if qubits is not None}
    patch = set(layout)
    for inst in entanglers:
        angle = float(inst.operation.params[0])
        physical = tuple(sorted(isa.find_bit(qubit).index for qubit in inst.qubits))
        if (not np.isclose(angle, theta_zz)
                or physical not in allowed or not set(physical) <= patch):
            raise ValueError(
                f"Off-patch, unsupported, or wrong-angle RZZ: {physical}, {angle}"
            )
    counts = dict(isa.count_ops())
    if any(counts.get(gate, 0) for gate in ("swap", "cz", "cx", "ecr")):
        raise ValueError("Routing or extra entanglers found")
    return {
        "rzz": len(entanglers),
        "sx": counts.get("sx", 0),
        "x": counts.get("x", 0),
        "depth": isa.depth(),
        "active_physical": sorted(patch),
    }


def compile_native(
    qc: QuantumCircuit,
    backend: Any,
    layout: Sequence[int],
    pm: StagedPassManager | None = None,
    *,
    normalize: Callable[[QuantumCircuit], QuantumCircuit],
    theta_zz: float,
) -> QuantumCircuit:
    """Normalize inverse RZZ gates, compile without routing, and audit the ISA.

    Args:
        qc: Logical circuit to prepare and compile.
        backend: Verified backend with a fractional-RZZ target.
        layout: Fixed logical-to-physical placement on native RZZ edges.
        pm: Reusable pass manager; when omitted, ``make_pm`` creates one.
        normalize: Participant implementation accepted by the preceding exercise
            check, or its organizer fallback. It should return a new circuit.
        theta_zz: Positive native RZZ angle used by the post-compilation audit.

    Returns:
        Compiled circuit with its layout metadata, after checking that every
        prepared RZZ remains native and compilation adds no entangler overhead.

    Raises:
        RuntimeError: Backend or layout is missing.
        ValueError: The native-ISA audit fails.

    This helper does not submit jobs. Its treatment of the logical input follows
    the supplied normalizer; the notebook's implementation returns a new circuit.
    """
    if backend is None or layout is None:
        raise RuntimeError("A verified fractional-RZZ target and fixed layout are required")
    prepared = normalize(qc)
    isa = (pm or make_pm(layout, backend)).run(prepared)
    check_isa(
        isa, prepared.count_ops().get("rzz", 0), backend, layout,
        theta_zz=theta_zz,
    )
    return isa


def two_qubit_gate_counts(circuit: QuantumCircuit) -> dict[str, int]:
    """Count two-qubit operations by gate name for compilation comparisons.

    Directive operations, including barriers, are excluded. The returned mapping
    includes routing gates when present, so summing its values measures the full
    two-qubit gate overhead. The circuit is not modified.
    """
    counts: dict[str, int] = {}
    for inst in circuit.data:
        gate = inst.operation
        if gate.num_qubits == 2 and not getattr(gate, "_directive", False):
            counts[gate.name] = counts.get(gate.name, 0) + 1
    return counts


def mapomatic_layout(
    lattice: Any,
    groups: Sequence[Sequence[tuple[int, int]]],
    backend: Any,
    *,
    logical_circuit: Callable[..., QuantumCircuit],
    n_cycles: int,
) -> list[int]:
    """Select a calibrated native-RZZ layout for a larger measured paper patch.

    Args:
        lattice: Patch exposing ``n_qubits`` and logical ``edges``.
        groups: Ordered bond groups used by the accepted circuit builder.
        backend: Live target and calibration data used by Mapomatic.
        logical_circuit: Notebook circuit builder, using the accepted Floquet
            cycle implementation and the notebook's drive parameters.
        n_cycles: Depth of the measured circuit used for layout ranking.

    Returns:
        Logical-to-physical qubit placement from the first Mapomatic score after
        filtering out layouts whose bonds lack native RZZ support. The selected
        calibration-dependent score is printed for notebook analysis.

    Raises:
        ValueError: Mapomatic finds no compatible native-RZZ layout.

    Circuit translation and calibration analysis are local; no job is submitted.
    """
    import mapomatic as mm
    from qiskit import transpile

    circuit = transpile(
        logical_circuit(n_cycles, lattice, groups, measured=True),
        basis_gates=["rz", "sx", "x", "rzz"],
        optimization_level=0,
    )
    layouts = mm.matching_layouts(circuit, backend, strict_direction=False)
    rzz_edges = {tuple(sorted(edge)) for edge in backend.target["rzz"]
                 if edge is not None}
    layouts = [
        layout for layout in layouts
        if all(tuple(sorted((layout[a], layout[b]))) in rzz_edges
               for a, b in lattice.edges)
    ]
    scores = mm.evaluate_layouts(circuit, layouts, backend)
    if not scores:
        raise ValueError(f"Mapomatic found no {lattice.n_qubits}q layout")
    layout, score = scores[0]
    print(f"{lattice.n_qubits}q Mapomatic score: {score:.6f}")
    return list(layout)
