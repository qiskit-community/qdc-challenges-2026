---
title: Samplomatic and qiskit-mitigation
short_title: Samplomatic and mitigation
tool_type: noise-management
owner: Noise-management team
last_verified: 2026-10-20
---

# Samplomatic and qiskit-mitigation

IBM's client-side packages for building circuit variants, mitigating results and designing noise-learning experiments. Samplomatic builds the randomized variants behind twirling and noise injection, `qiskit-mitigation` turns `Executor` results into mitigated expectation values and filters shots with error checks, and `qiskit-noise-learning` builds learning experiments of your own. The primitives' built-in options for the same techniques are in [IBM Quantum Compute client](ibm-quantum-compute-client.md).

Package versions this document was checked against:

| Package | Version checked | Notes |
|---|---|---|
| `samplomatic` | 0.21.0 | Requires Qiskit 2.3 or later |
| `qiskit-mitigation` | 0.1.1 | Its programs need runtime 0.50; requires `samplomatic>=0.21` |
| `qiskit-noise-learning` | 0.1.0 | Requires Python 3.11 or later, Qiskit 2.5 or later and `samplomatic>=0.21` |

## Pauli twirling with Samplomatic

Samplomatic lets you state in the circuit itself how it should be randomized. You group gates and measurements into Qiskit `box` instructions and annotate them (`Twirl`, `InjectNoise`, `ChangeBasis`, `Tag`). `samplomatic.build` then turns the boxed circuit into a parameterized template plus a samplex that draws the random parameters, either on your machine or inside `Executor`. `generate_boxing_pass_manager` adds the boxes for you after transpilation: it stratifies two-qubit gates into boxes that each hold a single layer, groups measurements into boxes of their own, and absorbs each single-qubit gate into the nearest eligible box to its right.

- **When to use:** you want to choose exactly which gates and measurements are randomized, and with which twirling group, or to inject noise for PEA, PEC or noise learning.
- **Tags:** family: general, trotter, floquet, echoes, thermal, verifiable; step: optimize; noise: suppression; runs: local, qpu; status: beta; updated: 2026-08
- **Recent changes:** Samplomatic 0.14 to 0.21 added:
  - `generate_boxing_pass_manager` takes keyword-only arguments (0.14).
  - The `Tag` annotation and the `local_c1` twirling group (0.17).
  - Twirling of parametric `RZZ` gates with the new `local_pauli` group (0.19).
  - Twirling of resets and mid-circuit measurements (0.20).
  - `asap` and `alap` boxing strategies, delay and stretch support, and the `samplomatic.quantum_program` module (0.21).
- **Key APIs:** `samplomatic.build(circuit)` returning `(template, samplex)`, `samplomatic.transpiler.generate_boxing_pass_manager(enable_gates=..., enable_measures=..., twirling_group=..., twirling_strategy=..., measure_annotations="twirl" | "change_basis" | "all", inject_noise_targets=..., inject_noise_strategy=..., remove_barriers=..., boxing_strategy=...)`, `samplomatic.Twirl(group=...)`, `samplomatic.InjectNoise(ref=..., modifier_ref=...)`, `samplomatic.ChangeBasis`, `samplomatic.Tag`, `samplomatic.utils.find_unique_box_instructions`, `samplomatic.utils.get_annotation`, `samplex.inputs()`, `samplex.sample`, `samplex.parameters`
- **Watch out:**
  - Samplex parameter values follow the order of `samplex.parameters`; build your arrays from that list.
  - Stratification walks the circuit in topological order and puts each two-qubit gate into the earliest box whose qubits are free; barriers, existing boxes and measurements act as delimiters, and no commutation analysis is done. For circuits with repeated layers (Trotter steps, ansatz layers, error-amplification sequences), put barriers at the layer boundaries in the logical circuit before transpiling; they survive routing and keep gates of different layers out of the same box.
  - `remove_barriers` decides when those barriers disappear: `"after_stratification"` (the default) uses them for grouping and then removes them so single-qubit gates can still be absorbed; `"immediately"` ignores them; `"never"` keeps them, which traps single-qubit gates between a barrier and a box and leaves them untwirled; `"finally"` keeps them in the output for visualization at the same cost.
  - Boxes are only as wide as the gates they hold unless `twirling_strategy` extends them (`"active_accum"`, `"active_circuit"`, `"all"`); idle qubits outside a box are not twirled. Note the spelling: Samplomatic writes `active_accum`, the primitives' `options.twirling.strategy` writes `active-accum`.
  - Hook the boxing pass into a preset pass manager with `preset_pm.post_scheduling = generate_boxing_pass_manager(...)` so it runs on the ISA circuit.
- **Related:** [Executor and QuantumProgram](ibm-quantum-compute-client.md#executor-and-quantumprogram) runs the samplex items; [Twirling options in the primitives](ibm-quantum-compute-client.md#twirling-options-in-the-primitives) gives the same twirling groups without editing the circuit.

**Docs:**
- [Samplomatic documentation](https://quantum.cloud.ibm.com/docs/en/addons/samplomatic)
- [Transpiler guide (boxing pass manager)](https://quantum.cloud.ibm.com/docs/en/addons/samplomatic/guides/transpiler)
- [Samplex inputs and outputs](https://quantum.cloud.ibm.com/docs/en/addons/samplomatic/guides/samplex-io)
- [Dressed boxes](https://quantum.cloud.ibm.com/docs/en/addons/samplomatic/guides/dressed-boxes)
- [API reference](https://quantum.cloud.ibm.com/docs/en/api/samplomatic/index)
- [Changelog](https://github.com/Qiskit/samplomatic/blob/main/CHANGELOG.md)

## Post-selection on non-Markovian error checks

Heron devices show non-Markovian noise that learned noise models miss. A cheap remedy: after the final measurement, apply a slow X pulse (`xslow`, or a train of N `rx(π/N)` gates) to each qubit and measure again. A qubit that fails to flip marks the shot as unreliable, and you discard it. The same check can run before the circuit (flip |0⟩ to |1⟩ and back, then measure) and on idle spectator qubits next to the data qubits. `PostSelector` builds the keep-mask from the check registers; its `node` strategy drops a shot if any check fails and its `edge` strategy drops it only when both qubits of a coupling-map edge fail. Masks for pre- and post-circuit checks can be combined with AND. Expect to lose many shots: the guide's 20-qubit example keeps between 0.03% and 28% of the shots depending on the strategy. The strictest filter gave the best value there, and some of the lenient ones did worse than the raw circuit, so compare strategies on your own problem.

- **When to use:** you want a cheap filter for shots hit by non-Markovian errors, without changing the body of your circuit.
- **Tags:** family: detection, verifiable, thermal, echoes; step: post-process; noise: detection; runs: local, qpu; status: beta; updated: 2026-09
- **Recent changes:** the checks now live in `qiskit_mitigation.postselection` (September 2026) and in `qiskit_addon_utils.noise_management.bit_flip_checks` (0.4, July 2026); the older `qiskit_addon_utils.noise_management.post_selection` module still works. `XSlowGate` became a runtime instruction in 0.50 (`qiskit_ibm_runtime.circuit.library.XSlowGate`, [Calibrated instructions](ibm-quantum-compute-client.md#specially-calibrated-instructions-and-calibration-sets)), and `NoiseLearnerV3Options` gained `bit_flip_checks` (pre- and post-circuit options), which replaces its `post_selection` option; `post_selection` is deprecated since runtime 0.49 and will be removed.
- **Key APIs:** `qiskit_mitigation.postselection.passes.AddPostCircuitNonMarkovianErrorChecks("xslow")` (or `"rx"`), `AddPreCircuitNonMarkovianErrorChecks`, `AddSpectatorPreCircuitNonMarkovianErrorChecks(coupling_map, ...)`, `AddSpectatorPostCircuitNonMarkovianErrorChecks(coupling_map, ...)`, `qiskit_mitigation.postselection.PostSelector.from_circuit(circuit, coupling_map)`, `PostSelector.compute_mask(result, strategy="edge", mode="post")` (`mode` is `"post"`, `"pre"` or `"both"`; the default strategy is `"node"`), the same classes in `qiskit_addon_utils.noise_management.bit_flip_checks` (`PostSelector`, `XSlowGate`, and in `bit_flip_checks.passes` the passes `AddPostCircuitBitFlipChecks`, `AddPreCircuitBitFlipChecks`, `AddSpectatorPostCircuitBitFlipChecks`, `AddSpectatorPreCircuitBitFlipChecks`), the older `qiskit_addon_utils.noise_management.post_selection.PostSelector` with `post_selection.transpiler.passes.AddPostSelectionMeasures(x_pulse_type=...)` and `AddSpectatorMeasures(coupling_map)`, and `NoiseLearnerV3Options.bit_flip_checks` to apply the same filter to noise-learning circuits (`NoiseLearnerV3Options.post_selection` still works but is deprecated since runtime 0.49)
- **Watch out:**
  - `xslow` exists only where the backend calibrates it; use `rx` for local tests. The passes are supported only on Heron QPUs with fractional gates.
  - The checks add classical registers (`meas_pre`, `meas_ps`, `spec_pre`, `spec` and `spec_ps` next to `meas`), and they are placed on the qubits that have a final measurement. A grader or results parser that expects the original registers may not accept them, so keep the checks out of the jobs you submit to such code unless it handles them.
  - `samplomatic.build` has failed on boxed circuits that combine these checks with twirled measurements (which TREX needs); add the checks to the built template instead. The [shaded-light-cone](qiskit-addons.md#shaded-light-cones-qiskit-addon-slc) and [PNA](qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) tutorials do exactly that: `PassManager([AddSpectatorMeasures(coupling_map), AddPostSelectionMeasures(x_pulse_type="rx")]).run(template_circuit)`, and then build the `PostSelector` from that final template.
  - A qubit that is measured and then reset still counts as measured at the end, so the post-circuit pass adds a check to it, and that check flags every shot in which the measurement read 1. To keep the checks off such qubits, write their mid-circuit outcomes to a register of their own whose name ends in a suffix you add to `ignore_creg_suffixes` (keep `"_pre"`), for example `AddPostCircuitNonMarkovianErrorChecks("xslow", ignore_creg_suffixes=["_pre", "_mid"])`. Then build the `PostSelector` from a copy of the circuit without those mid-circuit measurements and their register: `PostSelector.from_circuit` raises `ValueError` for a register that has no `_ps` copy.
  - `PostSelector.from_circuit` does not look inside measurement boxes; build the selector from an unboxed copy with the same registers.
  - `compute_mask` takes a dict of register name to boolean array, which is what `Executor` results already are; the mask has the arrays' shape without the last (clbit) axis. Requesting `mode="pre"` on a circuit without pre-circuit checks raises `ValueError`.

**Docs:**
- [Guide: post-selection with non-Markovian error checks](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-mitigation/guides/postselection-with-non-markovian-error-checks)
- [qiskit-mitigation postselection API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-mitigation/postselection)
- [qiskit-addon-utils bit_flip_checks API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-utils/noise-management-bit-flip-checks)
- [qiskit-addon-utils post-selection transpiler passes](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-utils/noise-management-post-selection-transpiler-passes)
- [Tutorial: PEC with shaded light cones (checks on a built template)](https://quantum.cloud.ibm.com/docs/en/tutorials/pec-with-shaded-lightcones)

## qiskit-mitigation: TREX, ZNE, PEA and PEC

A new IBM package of error-mitigation building blocks for the directed execution model. Each `MitigationTask` (`GateFolding` for ZNE, `PEA`, `PEC`) prepares items in a `QuantumProgram` for `Executor` and post-processes the results into mitigated expectation values (a `PubResult` whose data holds the expectation values `evs` and their standard deviations). `TREX` is a separate helper: `TREX().prepare(num_randomizations, quantum_program)` adds readout-calibration items to the same program, and you pass it to a task with `prepare(..., trex=trex)`. The package also computes postselected noise channels, from circuit symmetries or spacetime checks, that reduce PEC's sampling overhead, and it hosts the [non-Markovian error checks](#post-selection-on-non-markovian-error-checks). The client-side Estimator in `qiskit-ibm-runtime` 0.50 is built on this package.

* **TREX** (twirled readout error extinction) twirls the final measurements and rescales expectation values by a learned readout factor.
* **ZNE** (zero-noise extrapolation) measures at several amplified noise levels, for example by gate folding, and extrapolates to zero noise.
* **PEA** (probabilistic error amplification) is a ZNE variant that amplifies noise by injecting errors drawn from a learned noise model.
* **PEC** (probabilistic error cancellation) samples circuits that cancel a learned noise model on average. It is unbiased, but its shot cost grows with the sampling overhead γ².

- **When to use:** you build your own mitigation workflow on `Executor`, or want to see and control what the Estimator's mitigation does.
- **Tags:** family: general, lcu, trotter, floquet, echoes, verifiable, thermal; step: post-process; noise: mitigation-expval; runs: local, qpu; status: beta; updated: 2026-09
- **Recent changes:** first released on 13 September 2026 (0.1.0). Version 0.1.1 followed on 24 September with guides for ZNE by gate folding and for PEC with TREX. `qiskit-ibm-runtime` 0.50 depends on it.
- **Key APIs:** `qiskit_mitigation.GateFolding`, `PEA`, `PEC`, `TREX`, `MitigationTask.prepare(circuit, observables, parameters, custom_boxing_options=..., shots_per_randomization=64, num_randomizations=128, trex=..., quantum_program=program)`, `GateFolding.prepare(..., folding_method="random"|"front"|"back", noise_factors=(1, 3, 5))`, `GateFolding.postprocess(results, extrapolator=[...], custom_fit=...)`, `PEA.prepare(..., noise_maps=..., noise_factors=(1, 3, 5))`, `PEC.prepare(..., noise_maps=..., scale_randomizations_by_gamma=True, noise_gain="auto", max_sampling_overhead=100)`, `PEC.gamma` (set by `prepare`), `PEC.find_unique_layers(circuit, custom_boxing_options=...)`, `qiskit_mitigation.find_combined_unique_layers(circuits, custom_boxing_options=...)`, `qiskit_mitigation.utils.load_tasks_from_result(result, return_trex=False)`, `qiskit_mitigation.utils.executor_expectation_values`, `TREX.trex_factors_each_term(measurement_noise_map, observables)`, `qiskit_mitigation.zne.gate_folding.SUPPORTED_FOLDED_GATES` (documented under `qiskit_mitigation.zne`), `qiskit_mitigation.noise.create_postselected_noise_mask(noisy_circuit, noise_maps, detectors)` (for a boxed circuit with `InjectNoise` annotations, the learned maps keyed by `InjectNoise.ref` and the Pauli detectors you post-select on, it returns the `local_scales` dictionary, with 0 for every generator a detector would flag and 1 for the rest, and the sampling cost γ² of canceling only the surviving generators; bind the scales as samplex inputs the way the shaded-light-cone tutorial does)
- **Watch out:**
  - Pass your runtime `QuantumProgram` through `quantum_program=`. Otherwise the tasks build a Samplomatic one, which the local `Executor` cannot run. For local tests the guides run `Executor(AerSimulator(), options={"simulator": {"layer_noise_model": ...}})`, box with `{"measure_annotations": "change_basis", "add_tags": "unique_box"}` so the simulator can map the noise onto the boxes, and flag the program with `program._semantic_role = "simulation"`.
  - The task constructors take no options. Pass randomizations and noise factors to `prepare`, which defaults to 128 randomizations of 64 shots and, for `GateFolding` and `PEA`, noise factors (1, 3, 5).
  - `GateFolding` folds only the two-qubit gate types in `SUPPORTED_FOLDED_GATES` (ECR, CX, CY, CZ, SWAP). Fractional `rzz` layers need hand-built folding.
  - `postprocess` tries the extrapolators you list (`linear`, `polynomial_degree_k` for k from 1 to 7, `exponential`, `double_exponential`, `custom` with `custom_fit=(function, initial_guess, bounds)` forwarded to `scipy.optimize.curve_fit`, and `fallback`) in order and returns the first physically valid value with finite variance, so the order matters. `fallback` returns the value at the smallest noise factor; without it, and without any valid fit, the result with the smallest `std` is returned. Under Pauli noise an observable decays exponentially with the noise factor, so the guides put `exponential` first. You can rerun `postprocess` with other extrapolators without re-executing.
  - ZNE and PEC trade bias for variance: extrapolation amplifies shot noise (Richardson extrapolation through factors 1, 3 and 5 roughly doubles the standard error at equal shots), and PEC's reported standard errors can miss the spread between randomizations. Estimate error bars by resampling shots or randomizations; with `broadcast_obs_and_params=True`, `postprocess` also returns `twirl_stds`.
  - Post-selected PEC needs the same `InjectNoise` boxing as the noise learning, and `PEC.prepare` cannot apply per-generator scales in 0.1.1 (see [Shaded light cones](qiskit-addons.md#shaded-light-cones-qiskit-addon-slc)).
  - PEC scales the number of randomizations by about γ² by default, capped by `max_sampling_overhead` (default 100). When γ² exceeds that cap, `noise_gain="auto"` keeps a fraction 1 − log(max_overhead)/log(γ²) of the noise instead of canceling all of it. Check `task.gamma` and `task.gamma**2` after `prepare` and before you execute, and set `scale_randomizations_by_gamma=False` to choose the count yourself. The injected noise is sampled once per randomization, so PEC and PEA want many randomizations with few shots each.

**Docs:**
- [Qiskit Mitigation overview](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-mitigation)
- [Guide: ZNE by gate folding with custom extrapolation](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-mitigation/guides/gate-folding-with-custom-extrapolation)
- [Guide: PEA with noise learning](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-mitigation/guides/pea-with-noise-learning)
- [Guide: PEC with TREX](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-mitigation/guides/pec-with-trex)
- [Best practices for digital zero-noise extrapolation (arXiv:2307.05203)](https://arxiv.org/abs/2307.05203)
- [API reference](https://quantum.cloud.ibm.com/docs/en/api/qiskit-mitigation)
- [ZNE API (GateFolding, PEA)](https://quantum.cloud.ibm.com/docs/en/api/qiskit-mitigation/zne)
- [PEC API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-mitigation/pec)
- [Error mitigation and suppression techniques](https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques)

## Custom noise learning (qiskit-noise-learning)

`qiskit-noise-learning` is IBM's toolkit for noise-learning experiments that you design yourself, as opposed to the managed [NoiseLearnerV3](ibm-quantum-compute-client.md#noiselearnerv3). Its docs warn of breaking changes between 0.x releases.

- **When to use:** you design your own noise-learning experiment, or want to study one without QPU time.
- **Tags:** family: trotter, floquet, echoes, verifiable; step: execute; noise: characterization; runs: local, qpu; status: alpha; updated: 2026-10
- **Current version:** 0.1.0 (22 October 2026), the first PyPI release.
- **Key APIs:** `qiskit_noise_learning.protocols.prepare_vanilla_program(backend, instructions, num_randomizations=32, shots_per_randomization=128, fragment_depths=(0, 1, 2, 4, 16, 32))`, which returns a `QuantumProgram` for `Executor`; `process_vanilla_result(result)`, which returns the learned `PauliLindbladMap`s, the gate names and the fit
- **Watch out:**
  - Pin it, for example `qiskit-noise-learning==0.1.*`.
  - IBM's qiskit-noise-learning pages (installation, API reference and guides) still describe the pre-release API: they install from GitHub `main`, simulate with `aer_executor.AerExecutor` and call `protocols.prepare_learning_program` and `process_learning_result`. PyPI 0.1.0 has no `aer_executor` or `noise_learner` module, and its protocol functions are `prepare_vanilla_program` and `process_vanilla_result`.
  - The qiskit-mitigation PEA and PEC guides learn a synthetic noise model locally with its `NoiseLearner` and `AerExecutor`, which were removed in October 2026, before 0.1.0; commit `a9511b4` (8 October 2026) still has both. Neither class works with `Executor` in local mode.
  - The program is a plain Samplomatic `QuantumProgram`, so the local `Executor` raises an `AttributeError` about `_semantic_role` until you set `program._semantic_role = "simulation"`.
- **Related:** [NoiseLearnerV3](ibm-quantum-compute-client.md#noiselearnerv3) learns the noise models that [qiskit-mitigation](#qiskit-mitigation-trex-zne-pea-and-pec) consumes.

**Docs:**
- [Qiskit noise learning (alpha)](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-noise-learning)
