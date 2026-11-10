---
title: Qiskit SDK
short_title: Qiskit SDK
tool_type: sdk
owner: Qiskit SDK team
last_verified: 2026-10-20
---

# Qiskit SDK

The core `qiskit` package: the circuit library, the transpiler with its layout, routing and scheduling passes, and saving circuits with QPY. In the IBM Quantum documentation these are the Qiskit SDK guides on circuits and operators and on transpilation. `qiskit-ibm-transpiler` adds AI-powered passes that plug into the same pass managers.

Package versions this document was checked against:

| Package | Version checked | Notes |
|---|---|---|
| `qiskit` | 2.5.2 | Requires Python 3.10 or later. Qiskit 2.6.0 (16 October 2026) is now the newest release; read [Saving circuits with QPY](#saving-circuits-with-qpy) before you mix 2.5 and 2.6 |
| `qiskit-ibm-transpiler` | 0.18.0 |  |

## Circuit library

The circuit library holds the standard gates and building blocks, such as `PauliEvolutionGate` for Hamiltonian evolution, `RZZGate` for Ising couplings and the `qaoa_ansatz` function for QAOA circuits with a custom mixer and initial state.

- **When to use:** you need standard gates, Hamiltonian-evolution blocks or a QAOA circuit.
- **Tags:** family: general, qaoa, trotter, floquet; step: map; runs: local; status: stable
- **Key APIs:** `qiskit.circuit.library.RZZGate`, `RXGate`, `PauliEvolutionGate`, `qaoa_ansatz(cost_operator, reps=1, initial_state=None, mixer_operator=None)`
- **Watch out:**
  - The `QAOAAnsatz` class builds on `NLocal` and `BlueprintCircuit`, which are deprecated since Qiskit 2.1 and due for removal in 3.0, so it warns; use the `qaoa_ansatz` function.
  - `qaoa_ansatz` names its parameters `β[i]` and `γ[i]`, and `circuit.parameters` lists every β before every γ, so a value array ordered (γ, β) silently swaps the angles.
- **Related:** [Fractional gates](ibm-quantum-compute-client.md#fractional-gates) run `RZZ` and `RX` rotations as single native gates on Heron QPUs.

**Docs:**
- [Synthesize unitary operations](https://quantum.cloud.ibm.com/docs/en/guides/synthesize-unitary-operators)
- [Circuit library overview](https://quantum.cloud.ibm.com/docs/en/guides/circuit-library)
- [Circuit library API reference](https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit_library)

## Transpiler: preset and custom pass managers

Every circuit must be rewritten into the target's instruction set architecture (ISA) before it runs. `generate_preset_pass_manager` gives you the standard pipeline at optimization levels 0 to 3. For more control, build a `PassManager` from individual passes, or write your own `TransformationPass` that edits the circuit's DAG. Typical custom passes rewrite gates into a form a later step expects, fold gates for zero-noise extrapolation, or simplify a symbolic global phase.

- **When to use:** always: every circuit must be transpiled, and a custom pass manager helps when a preset level damages a hand-built structure or lacks a pass you need.
- **Tags:** family: general; step: optimize; runs: local; status: stable; updated: 2026-07
- **Recent changes:**
  - Qiskit 2.5 replaces `ConsolidateBlocks` + `UnitarySynthesis` with `TwoQubitPeepholeOptimization` at levels 2 and 3. Transpilation is typically 2–3x faster.
  - Qiskit 2.5 adds `QuantumCircuit.estimate_fidelity(target)` and a default `seed_transpiler` through the `QISKIT_TRANSPILER_SEED` environment variable.
  - Qiskit 2.3 made control flow and boxes Rust-native: after you edit a box's annotations in place, write the box back with `DAGCircuit.substitute_node`.
- **Key APIs:** `qiskit.transpiler.generate_preset_pass_manager`, `PassManager`, `qiskit.transpiler.basepasses.TransformationPass`, `Collect2qBlocks`, `ConsolidateBlocks`, `UnitarySynthesis`, `QuantumCircuit.estimate_duration`, `QuantumCircuit.estimate_fidelity`
- **Related:** [Calibrated instructions](ibm-quantum-compute-client.md#specially-calibrated-instructions-and-calibration-sets) for the runtime's own transpiler passes, [Dynamical decoupling passes](#dynamical-decoupling-passes) for the scheduling and padding passes.

**Docs:**
- [Introduction to transpilation](https://quantum.cloud.ibm.com/docs/en/guides/transpile)
- [Install and use transpiler plugins](https://quantum.cloud.ibm.com/docs/en/guides/transpiler-plugins)
- [Transpile with pass managers](https://quantum.cloud.ibm.com/docs/en/guides/transpile-with-pass-managers)
- [Configure the preset pass managers](https://quantum.cloud.ibm.com/docs/en/guides/defaults-and-configuration-options)
- [Write a custom transpiler pass](https://quantum.cloud.ibm.com/docs/en/guides/custom-transpiler-pass)
- [Transpiler passes index](https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler_passes)
- [AI-powered transpiler passes (beta)](https://quantum.cloud.ibm.com/docs/en/guides/ai-transpiler-passes)
- Release notes: [Qiskit 2.3](https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/2.3), [2.4](https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/2.4), [2.5](https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/2.5)

## AI-powered transpiler passes (qiskit-ibm-transpiler)

`qiskit-ibm-transpiler` provides reinforcement-learning transpiler passes that drop into a `PassManager`: `AIRouting` does layout and routing in one pass, and the synthesis passes re-synthesize collected Clifford, linear-function, permutation and Pauli-network blocks, keeping a block only when it has fewer CNOTs. Since releases 0.16 through 0.18 the models run on your machine after a download from Hugging Face (`qiskit-gym` models for synthesis, a Rust routing engine for routing). The cloud `TranspilerService` is a Premium Plan feature that is being migrated; the README recommends local mode. The passes are in beta.

- **When to use:** a preset pass manager leaves too many SWAPs or CNOTs in a large circuit and you can spare a separate environment for the AI passes.
- **Tags:** family: general; step: optimize; runs: local, account; status: beta; updated: 2026-03
- **Current version:** 0.18.0 (10 March 2026); Python 3.10 to 3.13.
- **Key APIs:** `qiskit_ibm_transpiler.ai.routing.AIRouting(backend=..., optimization_level=1|2|3, layout_mode="optimize"|"improve"|"keep")`, `ai.collection.CollectCliffords`, `CollectLinearFunctions`, `CollectPermutations`, `CollectPauliNetworks`, `ai.synthesis.AICliffordSynthesis`, `AILinearFunctionSynthesis`, `AIPermutationSynthesis`, `AIPauliNetworkSynthesis`, `generate_ai_pass_manager(coupling_map=..., ai_optimization_level=..., optimization_level=..., ai_layout_mode=...)`
- **Watch out:**
  - 0.18.0 does not import on Qiskit 2.5 or later, 2.6 included (`ModuleNotFoundError: qiskit.synthesis.linear.linear_matrix_utils`, a private module removed in 2.5; issue #250, open since 4 October 2026), although its `qiskit<3` pin lets pip install it next to them. The workaround in the issue is `qiskit<2.5`.
  - It pins `qiskit-serverless~=0.30`, which pins `qiskit-ibm-runtime<0.46`, and `networkx==2.8.5`, so pip resolves runtime 0.45.1 for it: it cannot share an environment with runtime 0.50 ([Sampler and Estimator](ibm-quantum-compute-client.md#sampler-and-estimator-client-side-primitives)). It also pulls in torch and ray (128 packages in a fresh environment). Give it its own environment and move routed circuits over as QPY, which a newer Qiskit reads.
  - Block sizes are capped: nine qubits for Clifford and linear-function blocks, six for Pauli networks, and 27, 33 or 65 for permutations. A block must lie on a connected subgraph of the coupling map, or the pass warns and leaves it unchanged.

**Docs:**
- [AI-powered transpiler passes](https://quantum.cloud.ibm.com/docs/en/guides/ai-transpiler-passes)
- [qiskit-ibm-transpiler API reference](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-transpiler)
- [AIRouting API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-transpiler/ai-ai-routing)
- [TranspilerService API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-transpiler/transpiler-service-transpiler-service)
- [Release notes](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-transpiler/release-notes)
- [README on GitHub](https://github.com/Qiskit/qiskit-ibm-transpiler)
- [Issue #250: ImportError with Qiskit 2.5](https://github.com/Qiskit/qiskit-ibm-transpiler/issues/250)
- [Tutorial: compilation methods for Hamiltonian simulation circuits](https://quantum.cloud.ibm.com/docs/en/tutorials/compilation-methods-for-hamiltonian-simulation-circuits)

## Swap strategies for commuting gates

A QAOA cost layer is made of commuting two-qubit gates, so it can be routed with a fixed *swap strategy*, such as a line of swaps, instead of the general-purpose router. On large problems this gives short, predictable depth.

- **When to use:** your two-qubit layer consists of commuting gates (QAOA cost layers, Ising interactions) and you want short, predictable routing.
- **Tags:** family: qaoa, multi-objective, trotter, floquet; step: optimize; runs: local; status: stable; updated: 2026-07
- **Recent changes:** Qiskit 2.5 fixes `Commuting2qGateRouter` adding the global phase twice.
- **Key APIs:** `qiskit.transpiler.passes.routing.commuting_2q_gate_routing.SwapStrategy` (`SwapStrategy.from_line`), `qiskit.transpiler.passes.Commuting2qGateRouter`
- **Related:** [Quantum optimization best practices (qopt-best-practices)](ecosystem.md#quantum-optimization-best-practices-qopt-best-practices) packages swap-strategy routing for QAOA.

**Docs:**
- [Commuting2qGateRouter (documents SwapStrategy)](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.Commuting2qGateRouter)
- [ElidePermutations](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.ElidePermutations)
- [Tutorial: advanced techniques for QAOA](https://quantum.cloud.ibm.com/docs/en/tutorials/advanced-techniques-for-qaoa)

## Qubit selection and layout

Which physical qubits you use matters as much as how you compile.

* **Calibration data.** A common first step is to drop the qubits and couplers whose reported error is above a threshold before you search for a layout; [Calibration data and layer fidelity](hardware.md#calibration-data-and-layer-fidelity) says where the backend reports those errors and what they mean.
* **Subgraph search.** `rustworkx.vf2_mapping(device_graph, pattern, subgraph=True, induced=False)` finds every placement of your interaction graph on the device; with the defaults (`subgraph=False`, `induced=True`) it finds no placements. `VF2PostLayout` does it inside the transpiler, and [mapomatic](ecosystem.md#layout-ranking-with-mapomatic) does it for a transpiled circuit across one or more backends. Pass the device as an undirected graph, for example `backend.coupling_map.graph.to_undirected(multigraph=False)`, because a `PyDiGraph` and a `PyGraph` cannot be matched (`TypeError`). Each mapping maps device qubits to pattern nodes, so invert it to build an `initial_layout`.
* **Fresh benchmarks.** Calibration data typically comes from procedures that run about once a day, and qubits drift on shorter timescales. [Device benchmarking](ecosystem.md#device-benchmarking-qiskit-device-benchmarking) re-measures the calibration data for your qubits. Refresh close to the time you execute, because the device keeps drifting.
* **Pictures.** `plot_gate_map` draws the coupling map, `plot_error_map` overlays the calibrated errors, and `plot_circuit_layout` shows where a transpiled circuit landed.

- **When to use:** before every hardware run, to choose the physical qubits that carry your circuit.
- **Tags:** family: general, qaoa, multi-objective, floquet, echoes; step: optimize; runs: local, account; status: stable; updated: 2026-06
- **Recent changes:** `VF2Layout` and `VF2PostLayout` score partial layouts while they search (Qiskit 2.3). They no longer fail when a gate error is reported as 1.0 (2.4.2).
- **Key APIs:** `rustworkx.vf2_mapping`, `qiskit.transpiler.passes.VF2PostLayout`, `qiskit.visualization.plot_gate_map`, `plot_error_map`, `plot_circuit_layout`
- **Related:** [qopt-best-practices](ecosystem.md#quantum-optimization-best-practices-qopt-best-practices) for `BackendEvaluator`, [Specially calibrated instructions and calibration sets](ibm-quantum-compute-client.md#specially-calibrated-instructions-and-calibration-sets) for calibration sets, whose instruction properties feed `backend.target`.

**Docs:**
- [plot_error_map](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.visualization.plot_error_map)
- [VF2PostLayout](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.VF2PostLayout)
- [Tutorial: refresh backend properties with real-time benchmarking](https://quantum.cloud.ibm.com/docs/en/tutorials/refresh-backend-properties-with-real-time-benchmarking)
- [Tutorial: real-time benchmarking for qubit selection](https://quantum.cloud.ibm.com/docs/en/tutorials/real-time-benchmarking-for-qubit-selection)

## Dynamical decoupling passes

Dynamical decoupling (DD) fills idle periods on qubits with pulse sequences that multiply to the identity, which suppresses decoherence and crosstalk. To put it in the circuit yourself, schedule with `ALAPScheduleAnalysis` and pad with `PadDynamicalDecoupling`. The pass needs a scheduled, transpiled circuit, so run it after the preset pass manager. The primitives can also insert DD for you ([Dynamical decoupling options](ibm-quantum-compute-client.md#dynamical-decoupling-options)).

- **When to use:** your circuit leaves qubits idle, for example in sparse layers or while other qubits are measured or reset, and you want to choose the sequence and where it goes, or the circuit has mid-circuit measurements, resets or feed-forward.
- **Tags:** family: general, lcu, thermal, detection; step: optimize; noise: suppression; runs: local; status: stable
- **Key APIs:** `qiskit.transpiler.passes.ALAPScheduleAnalysis`, `qiskit.transpiler.passes.PadDynamicalDecoupling`, `qiskit.transpiler.passes.BasisTranslator`
- **Watch out:**
  - When you pad yourself, use only gates in `backend.target`, and run the padded circuit. IBM targets (Heron and Nighthawk) have no `y` gate, so padding with `YGate` raises `TranspilerError`; pad with X–X, or follow the guide: add `YGate` to the target with the X gate's duration and error (`target.add_instruction(YGate(), properties)`), pad with XY4, then rerun `BasisTranslator` so that no `y` remains.
  - In dynamic circuits, use `stretch` delays for feed-forward windows.
  - Samplomatic's boxing pass uses barriers to decide which gates share a box, then removes them by default; see [Pauli twirling with Samplomatic](noise-management.md#pauli-twirling-with-samplomatic) for the `remove_barriers` settings that keep them.
- **Related:** [Dynamic circuits and resets](ibm-quantum-compute-client.md#dynamic-circuits-and-resets), [Dynamical decoupling options](ibm-quantum-compute-client.md#dynamical-decoupling-options)

**Docs:**
- [Create a pass manager for dynamical decoupling](https://quantum.cloud.ibm.com/docs/en/guides/dynamical-decoupling-pass-manager)
- [PadDynamicalDecoupling](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.PadDynamicalDecoupling)

## Saving circuits with QPY

Save circuits with QPY. QPY is backward compatible but not forward compatible: `load` reads older formats, not newer ones.

- **When to use:** always, and especially before long runs or when you share circuits between Qiskit versions.
- **Tags:** family: general; step: develop; runs: local; status: stable; updated: 2026-10
- **Recent changes:** Qiskit 2.3 through 2.5 write QPY format 17 by default (`qpy.dump` can emit versions 13 to 17). Qiskit 2.6.0 (16 October 2026) writes format 18 by default, which older Qiskit versions cannot read, so pass `version=17` to `qpy.dump` when files must load elsewhere.
- **Key APIs:** `qiskit.qpy.dump(circuits, file, version=17)`, `qiskit.qpy.load`, `qpy.QPY_VERSION`
- **Watch out:** `qpy.load` always returns a list, even for one circuit.
- **Related:** [Saving results and job IDs](ibm-quantum-compute-client.md#saving-results-and-job-ids)

**Docs:**
- [Save circuits to disk (QPY)](https://quantum.cloud.ibm.com/docs/en/guides/save-circuits)
- [QPY API and format version history](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qpy)
