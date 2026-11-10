---
title: Qiskit addons
short_title: Qiskit addons
tool_type: addons
owner: Qiskit addons team
last_verified: 2026-10-20
---

# Qiskit addons

The Qiskit addons are separate packages, each with its own release cycle, that IBM documents alongside the SDK. This document covers the addons, and the IBM packages documented with them, for mapping problems (optimization mapper, Qiskit Fermions), shrinking circuits (operator backpropagation, AQC-Tensor, circuit cutting), detecting errors (Paulice), turning shot data into expectation values (`qiskit-addon-utils`), mitigating errors in observables (shaded light cones, propagated noise absorption and Pauli propagation) and in counts (M3), post-processing samples (SQD) and combining Trotter expectation values (multi-product formulas).

Package versions this document was checked against:

| Package | Version checked | Notes |
|---|---|---|
| `qiskit-addon-utils` | 0.4.0 | 0.4 includes post-selection on slow-X error checks ([Post-selection on non-Markovian error checks](noise-management.md#post-selection-on-non-markovian-error-checks)) |
| `qiskit-addon-slc` | 0.1.0 (GitHub commit `7156757`) | PyPI 0.1.0 lacks some newer APIs that are only on GitHub ([Shaded light cones](#shaded-light-cones-qiskit-addon-slc)), so install from GitHub when an import fails; needs PySCF (`pyscf>=2.10`) |
| `qiskit-addon-opt-mapper` | 0.1.0 |  |
| `qiskit-paulice` | 0.2.1 | Its public API lacks `stabilizers=`, `dope()` and `estimate_fault_rates()`, which are only on GitHub `main` ([Spacetime Pauli checks](#spacetime-pauli-checks-qiskit-paulice)). Requires `qiskit-aer` 0.17.1 or later |
| `qiskit-addon-sqd` | 0.14.0 | Trim SQD (`TrimPolicy`) is new in 0.14. Needs PySCF (`pyscf>=2.9`) for `qiskit_addon_sqd.fermion`; pip leaves PySCF out on Windows |
| `mthree` | 3.0.0 | mthree 3.0.0 has wheels for Python 3.9–3.12 only; on 3.13 and 3.14, pip builds it from source and needs a C compiler |
| `qiskit-fermions` | 0.3.0 | Requires Python 3.11 or later |
| `qiskit-addon-obp` | 0.3.0 |  |
| `qiskit-addon-aqc-tensor` | 0.3.1 |  |
| `qiskit-addon-cutting` | 0.10.0 |  |
| `qiskit-addon-pna` / `pauli-prop` | 0.2.0 / 0.2.1 |  |
| `qiskit-addon-mpf` | 0.3.0 |  |

## Optimization mapper (qiskit-addon-opt-mapper)

The successor to the modeling layer of `qiskit-optimization`. You build an `OptimizationProblem` directly or import one from docplex. Converters turn equality constraints into penalties or the whole problem into a QUBO, and `to_ising` produces the `SparsePauliOp` for a QAOA cost layer.

- **When to use:** you need to turn a binary optimization model, with or without constraints, into a QUBO or into an Ising `SparsePauliOp` for a QAOA cost layer.
- **Tags:** family: qaoa, multi-objective; step: map; runs: local; status: stable; updated: 2025-11
- **Current version:** 0.1.0 (3 November 2025), the only release so far. Unreleased work on GitHub `main` adds a `BinaryToLinearBinary` converter and a faster `to_ising`.
- **Key APIs:** `OptimizationProblem`, `translators.from_docplex_mp`, `translators.to_ising`, `converters.EqualityToPenalty`, `converters.OptimizationProblemToQubo`
- **Watch out:**
  - `to_ising` already applies the objective's sense. For a `maximize` problem it returns the negated objective, whose ground state is the optimum, plus a constant offset. Negate it only if you want an operator equal to the objective itself.
  - `to_ising` raises `OptimizationError` if the problem still has constraints or any integer or continuous variable, so run the converters first. Variables map to qubits in order: the i-th variable becomes the i-th qubit.

**Docs:**
- [Documentation home](https://qiskit.github.io/qiskit-addon-opt-mapper/)
- [GitHub repository](https://github.com/Qiskit/qiskit-addon-opt-mapper)
- [Migrate from qiskit-optimization](https://qiskit.github.io/qiskit-addon-opt-mapper/how_tos/00_migrate_from_qiskit_optimization.html)
- [Converters for optimization problems](https://qiskit.github.io/qiskit-addon-opt-mapper/how_tos/02_converters_for_optimization_problems.html)
- [Translate to and from docplex](https://qiskit.github.io/qiskit-addon-opt-mapper/how_tos/04_translate_docplex.html)
- [Tutorial: quantum approximate multi-objective optimization](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-multi-objective-optimization)
- [Tutorial: warm-start QAOA with the Optimization Mapper](https://quantum.cloud.ibm.com/docs/en/tutorials/warm-start-qaoa)

## Qiskit Fermions (qiskit-fermions)

The `qiskit-fermions` addon represents fermionic operators and circuits and maps them to qubits with any fermion-to-qubit encoding. It works with ffsim operators: `UCJ(ucj_op)` wraps an ffsim operator as a gate in a `FermionicCircuit`, and `generate_preset_jw_pass_manager` lowers it through Jordan-Wigner.

- **When to use:** you build fermionic operators or circuits and want to choose the fermion-to-qubit encoding, or you want to transpile an ffsim LUCJ operator as one fermionic gate.
- **Tags:** family: sqd; step: map; runs: local; status: beta; updated: 2026-10
- **Recent changes:**
  - 0.1.0 (14 August 2026) was the first release.
  - 0.2.0 (11 September 2026) made `UCJ` and `UCC` take an ffsim operator as their only argument and removed the parameterization methods such as `UCJ.from_t_amplitudes` without a deprecation period; build the operator with `ffsim.UCJOpSpinBalanced.from_t_amplitudes` instead. It also added fermionic product formulas and the `FermionicTrotterization` pass.
  - 0.3.0 (22 October 2026) adds a ternary-tree fermion-to-qubit mapping and drops Python 3.10.
- **Key APIs:** `qiskit_fermions.circuit.library.UCJ`, `qiskit_fermions.transpiler.presets.generate_preset_jw_pass_manager`
- **Watch out:**
  - Every release so far requires `qiskit~=2.5.0`, so installing it moves Qiskit 2.6 back to 2.5. Give it its own environment if you need Qiskit 2.6.
  - `UCJ` and `UCC` need the `ffsim` extra (`pip install "qiskit-fermions[ffsim]"`), which pip skips on Windows.
  - Only Jordan-Wigner lowering of `UCJ` works so far: the synthesis plugins for `OrbitalRotation`, `InitializeModes` and `PrepareSlaterDeterminant` assume a one-to-one occupation-basis encoding, and plugins for other encodings do not ship yet. For a Jordan-Wigner circuit alone, ffsim's `ffsim.qiskit.UCJOpSpinBalancedJW` is enough.
- **Related:** [Chemistry: PySCF and ffsim](ecosystem.md#chemistry-pyscf-and-ffsim), [SQD](#sample-based-quantum-diagonalization-qiskit-addon-sqd)

**Docs:**
- [Qiskit Fermions overview](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-fermions)
- [Guide: transpile an LUCJ ansatz](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-fermions/guides/lucj)

## Operator backpropagation (qiskit-addon-obp)

Operator backpropagation (OBP) makes a deep circuit shallower by moving the last slices of the circuit into the observable: more Pauli terms to measure, fewer gates to run. It suits time-evolution circuits.

- **When to use:** your circuit is too deep for the hardware and you measure low-weight observables, as in time evolution, and can afford to measure more Pauli terms.
- **Tags:** family: trotter, floquet, echoes; step: optimize; noise: mitigation-expval; runs: local; status: stable; updated: 2025-05
- **Current version:** 0.3.0 (22 May 2025).
- **Key APIs:** `qiskit_addon_utils.slicing.slice_by_gate_types`, `combine_slices`, `qiskit_addon_obp.backpropagate`, `qiskit_addon_obp.utils.simplify.OperatorBudget`
- **Related:** [AQC-Tensor](#aqc-tensor-qiskit-addon-aqc-tensor) shortens the start of the circuit instead; [Exact and tensor-network references](simulators.md#exact-and-tensor-network-references) for reference values.

**Docs:**
- [Operator backpropagation overview](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-obp)
- [OBP quickstart](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-obp/guides/quickstart)
- [Tutorial: operator backpropagation](https://quantum.cloud.ibm.com/docs/en/tutorials/operator-back-propagation)

## AQC-Tensor (qiskit-addon-aqc-tensor)

AQC-Tensor makes a deep circuit shallower by replacing its first part with a shallower parametrized ansatz trained against a tensor-network simulation. It suits time-evolution circuits.

- **When to use:** your circuit is too deep for the hardware, as in time evolution, and its first part can be simulated with tensor networks on your machine.
- **Tags:** family: trotter, floquet; step: optimize; noise: suppression; runs: local, serverless; status: stable; updated: 2026-07
- **Recent changes:** AQC-Tensor 0.3 (May to July 2026) adds Python 3.14 support and quimb-backend gradients for multi-parameter gates.
- **Key APIs:** `qiskit_addon_aqc_tensor.ansatz_generation.parametrize_circuit`, `objective.MaximizeStateFidelity`
- **Watch out:** in a [managed workflow](managed-workflows.md#external-dependencies), `qiskit-addon-aqc-tensor[quimb-jax]` (with `quimb` and `jax`) and `cotengrust` are already in the runtime image at pinned versions; bring any other version as a wheel.
- **Related:** [Operator backpropagation](#operator-backpropagation-qiskit-addon-obp) shortens the end of the circuit instead; [Exact and tensor-network references](simulators.md#exact-and-tensor-network-references) for reference values; [Managed Workflows](managed-workflows.md#write-upload-and-run-a-workflow) for deploying the workflow.

**Docs:**
- [AQC-Tensor documentation](https://qiskit.github.io/qiskit-addon-aqc-tensor/)
- [AQC-Tensor API reference](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-aqc-tensor)
- [Tutorial: AQC for time evolution](https://quantum.cloud.ibm.com/docs/en/tutorials/approximate-quantum-compilation-for-time-evolution)
- [Guide: deploy a Qiskit Function template for AQC + Trotter dynamics](https://quantum.cloud.ibm.com/docs/en/guides/function-template-aqc-trotter)

## Circuit cutting (qiskit-addon-cutting)

Circuit cutting cuts two-qubit gates or wires, runs the resulting smaller subcircuits, and reconstructs expectation values classically from a quasi-probability decomposition. Use it to fit a circuit onto fewer qubits, or to avoid a deep SWAP chain for a long-range gate. Every cut multiplies the sampling overhead: 9 for a cut CNOT or CZ (less for a small-angle `RZZ`), 16 for a wire cut.

- **When to use:** your circuit needs more qubits than the QPU has, or a few distant two-qubit gates, and you need expectation values and can pay the shot overhead.
- **Tags:** family: lcu, trotter; step: optimize; runs: local; status: stable; updated: 2025-03
- **Current version:** 0.10.0 (28 March 2025), with Qiskit SDK 2.0 support and `SparsePauliOp` utilities. Its pin `qiskit>=1.3.1,<3` admits Qiskit 2.5 and 2.6; gate cuts, wire cuts, `find_cuts` and reconstruction from Aer `SamplerV2` results ran under 2.5.2 and 2.6.0 when this document was checked. Unreleased `main` drops Python 3.9 and the `qiskit-ibm-runtime` dependency.
- **Key APIs:** `qiskit_addon_cutting.partition_problem`, `cut_gates`, `find_cuts`, `OptimizationParameters`, `DeviceConstraints`, `cut_wires`, `expand_observables`, `generate_cutting_experiments(circuits, observables, num_samples)`, `reconstruct_expectation_values`, `instructions.CutWire`, `qpd.TwoQubitQPDGate`
- **Watch out:**
  - Two cut CZ gates already cost 81 times the shots. Check `np.prod([b.overhead for b in bases])`, or the `sampling_overhead` that `find_cuts` reports, before you submit.
  - `partition_problem` needs a circuit without classical bits, `PauliList` observables on all its qubits, and bound parameters (symbolic parameters cannot be cut, issue #539).
  - Transpile each subexperiment yourself, run it with the Sampler ([Sampler and Estimator](ibm-quantum-compute-client.md#sampler-and-estimator-client-side-primitives)), and pass the results to `reconstruct_expectation_values` in the order they were generated, as a dictionary keyed by partition label.

**Docs:**
- [Circuit cutting documentation home](https://qiskit.github.io/qiskit-addon-cutting/)
- [Explanation: overview of circuit cutting and sampling overheads](https://qiskit.github.io/qiskit-addon-cutting/explanation/index.html)
- [Tutorials](https://qiskit.github.io/qiskit-addon-cutting/tutorials/index.html)
- [How-to guides](https://qiskit.github.io/qiskit-addon-cutting/how-tos/index.html)
- [API reference](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-cutting/qiskit-addon-cutting)
- [Release notes](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-cutting/release-notes)
- [Qiskit addons overview](https://quantum.cloud.ibm.com/docs/en/guides/addons)

## Spacetime Pauli checks (qiskit-paulice)

Qiskit Paulice adds hardware-efficient Pauli checks to Clifford circuits. Ancilla qubits next to the circuit's qubits measure products of Paulis that should not change (their back-propagated product is a stabilizer of the ideal state), so a flipped check reveals an error. Each check is spread across the payload in space and time, which avoids the SWAP-heavy, time-localized measurements of coherent Pauli checks on limited connectivity such as heavy-hex.

The library picks the checks that catch the most noise under a noise model of your device, scored either by `gamma` (the sampling overhead of the postselected inverse noise channel) or by `LER` (a Monte Carlo logical error rate), and returns a list of `CheckedCircuit`s with increasing numbers of checks, from none up to one per target qubit. Each one provides a post-selection function that maps a shot to its syndrome vector; keep the shots whose syndrome is all zero. The method gives single-shot access to the state, so it works in sampling and in expectation-value workflows, and it has been used on Clifford-dominated circuits of up to 50 qubits and 2450 entangling gates.

- **When to use:** your circuit is Clifford, or Clifford with a few non-Clifford gates, and you want low-overhead error detection with ancilla checks.
- **Tags:** family: verifiable, detection; step: map, post-process; noise: detection; runs: local; status: stable; updated: 2026-10
- **Recent changes:** a new library (first release 26 May 2026). `layout.get_check_qubits` finds target and ancilla qubit pairs on the device (0.2.0). Version 0.2.1 fixed circuits defined on an incomplete Clifford basis (the checked circuit now comes back in the internal Clifford basis instead of failing) and circuits with wires of two-qubit depth 1. GitHub `main` (21 September to 1 October 2026) adds checks for state stabilizers (`add_pauli_checks(..., stabilizers=...)`), `CheckedCircuit.dope()` and `CheckedCircuit.estimate_fault_rates()`; these are not in 0.2.1.
- **Key APIs:** `qiskit_paulice.add_pauli_checks(circuit, target_qubits, noise_model, cost="gamma", method="windowed", ancilla_qubits=..., seed=...)` (the circuit needs at least one terminal measurement; the checks get their own registers, `checks_q` and `checks_c` by default), `qiskit_paulice.CheckedCircuit` (`circuit`, `target_qubits`, `check_qubits`, `check_support`, `cost`, `cost_metric`, `uncovered_paulis`), `CheckedCircuit.get_postselection_method()`, `qiskit_paulice.layout.get_check_qubits(backend_or_coupling_map, layout)`, `qiskit_paulice.layout.get_low_overhead_ancillas(coupling_map, layout)`, `qiskit_paulice.noise_models.NoiseModel.from_backend(backend, layout, uniform_gate_noise=True)` (`uniform_gate_noise=True` gives one depolarizing rate for every entangling gate, as the guides use; the default `False` gives a per-edge model from the calibration data), `NoiseModel.from_pauli_lindblad_maps`
- **Watch out:**
  - Every check must commute with the measured observable, so a circuit measured in several bases needs its own set of checks per basis; the Paulice guide, which estimates a fidelity from randomly sampled stabilizers, finds one check set per stabilizer.
  - If your code calls `stabilizers=`, `dope()` or `estimate_fault_rates()`, the PyPI build (0.2.1) fails with `TypeError` or `AttributeError` until you replace it with GitHub `main`. The package compiles a Rust extension, so install a Rust toolchain first (for example with rustup). Then run `pip install --force-reinstall --no-deps git+https://github.com/Qiskit/qiskit-paulice.git`: `main` still reports version 0.2.1, so without `--force-reinstall` pip keeps the release you have, and `--no-deps` keeps your other pins.
  - Checks are added in `target_qubits` order and the ancillas are appended after the payload, so transpile the checked circuit with `initial_layout = layout + ancilla_qubits` (truncate the ancilla list for a circuit with fewer checks) and `optimization_level=0`, as the guides do.
  - The post-selection function reads a bitstring by the checked circuit's classical bits, clbit 0 rightmost, so decode the bitstring of all registers joined in the circuit's own order: with `Sampler`, `pub_result.join_data().get_bitstrings()`, leaving `names` unset. `join_data(names=[...])` joins the registers in the order you give, and a wrong order decodes the wrong bits without an error. An array input is indexed by qubit instead. Check the decoding on a noiseless simulator, where every shot must pass.
  - `CheckedCircuit` is a frozen dataclass, so wrap edited circuits with `dataclasses.replace` to keep the decoder.
  - Check picking is deterministic only with the defaults `cost="gamma"` and `method="windowed"`; `cost="LER"`, `method="genetic"` and `method="windowed_genetic"` have randomness that `seed` does not control. Idling noise is not part of `NoiseModel.from_backend` and is ignored when checks are picked.
  - Since 1 October 2026, `FaultRates.harmless_rate` on GitHub `main` is a fraction of all shots. Divide it by `acceptance_rate` for the rate among accepted shots. Code written before that change may treat it as the rate among accepted shots; check which normalization your code assumes.

**Docs:**
- [Qiskit Paulice overview](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-paulice)
- [Quickstart](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-paulice/guides/quickstart)
- [Low-overhead error detection using spacetime codes](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-paulice/guides/low-overhead-error-detection-using-spacetime-codes)
- [Tutorial: spacetime codes](https://quantum.cloud.ibm.com/docs/en/tutorials/spacetime-codes)
- [API reference](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-paulice)
- [GitHub repository](https://github.com/Qiskit/qiskit-paulice)
- [Paper: low-overhead error detection with spacetime codes](https://arxiv.org/abs/2504.15725)

## Expectation values from shot data

`Executor` and the Sampler return bitstrings, not expectation values. Helper functions turn shot data, including twirled and post-selected data, into expectation values. `executor_expectation_values` takes a boolean array whose last two axes are shots and bits, a mapping from measurement bases to observables, and optional extras: axes to average over (usually the twirling axis), measurement flips to undo measurement twirling, PEC signs, a post-selection mask and TREX rescale factors. It returns one (expectation value, variance) pair per observable. For plain counts, `BitArray.expectation_values` evaluates diagonal observables directly and `sampled_expectation_value` does the same for a counts dictionary.

- **When to use:** you have raw bitstrings from `Executor` or the Sampler and need expectation values, with twirl flips undone and post-selection applied.
- **Tags:** family: general, lcu, thermal, verifiable, detection, echoes; step: post-process; runs: local; status: stable; updated: 2026-09
- **Recent changes:** `qiskit_addon_utils.exp_vals.executor_expectation_values` accepts an `(observables, bases)` tuple (0.4, July 2026). `qiskit_mitigation.utils.executor_expectation_values` does the same job within the mitigation workflow (September 2026). `sampled_expectation_value` accepts a `SparseObservable` since Qiskit 2.3.
- **Key APIs:** `qiskit_addon_utils.exp_vals.executor_expectation_values(bool_array, basis_mapping, meas_basis_axis, avg_axis=..., measurement_flips=..., pauli_signs=..., postselect_mask=..., rescale_factors=...)`, `exp_vals.get_measurement_bases(observables)`, `exp_vals.map_observable_isa_to_canonical`, `qiskit_mitigation.utils.executor_expectation_values`, `qiskit.primitives.BitArray.expectation_values`, `BitArray.get_counts`, `qiskit.result.sampled_expectation_value`
- **Watch out:**
  - Qiskit bitstrings are little-endian: qubit 0 is the rightmost character. `executor_expectation_values` expects the least significant bit at index 0 of the bits axis, and `BitArray.from_bool_array` takes an `order` argument.
  - The returned variances ignore covariances between the terms of an observable.
  - With PEC and `gamma_factor=None`, the rescaling is derived from the count of positive minus negative samples and divides by zero when they are equal.
  - `rescale_factors` differs between the two packages: the utils version takes nested lists ordered like `basis_mapping`, the mitigation version takes a dictionary keyed by Pauli term.
  - `BitArray.expectation_values` only takes diagonal observables (I, Z, 0, 1) with as many qubits as the array has bits.

**Docs:**
- [qiskit-addon-utils exp_vals API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-utils/exp-vals)
- [qiskit-mitigation utils API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-mitigation/utils)
- [BitArray API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.BitArray)

## Shaded light cones (qiskit-addon-slc)

PEC gives unbiased results, but its sampling cost grows quickly with circuit size. Shaded light cones bound how much each learned error term can change your observable: a forward bound propagates the error to the end of the circuit and bounds its commutator with the observable, a backward bound propagates it to the start and bounds its effect on the initial state, and merging the two with the learned rates gives a bias contribution per term. PEC then cancels only the errors that matter, truncating the least impactful terms until your `bias_tolerance` is reached or your `sampling_cost_budget` is filled, which cuts the sampling overhead at a controlled bias. The bounds are classical, run on pauli-prop ([PNA and Pauli propagation](#propagated-noise-absorption-and-pauli-propagation)), and can be computed before the noise learning runs.

- **When to use:** PEC suits your problem, but its sampling overhead is too large for your circuit.
- **Tags:** family: trotter, floquet, echoes; step: post-process; noise: mitigation-expval; runs: local, qpu; status: stable; updated: 2026-10
- **Recent changes:** new bound APIs (`LightCone`, `compute_bounds`, `time_evolved_norm_forward`, a `max_num_boxes` argument), IBM documentation pages, and a PEC tutorial built on `NoiseLearnerV3`, Samplomatic and `Executor`. These APIs are not in the only PyPI release (0.1.0): until 0.2.0 is released, install from GitHub (`pip install git+https://github.com/Qiskit/qiskit-addon-slc.git`); this cheat sheet was checked against commit `7156757` (14 October 2026), which `@7156757` pins. Use a fresh environment: the commit also reports version 0.1.0, so pip keeps a PyPI 0.1.0 that is already installed. The upcoming release also deprecates the `atol` argument of `compute_forward_bounds` in favor of `atol_simplify` and `atol_eigenvalue`, reworks the parallel bound computation, lets you interrupt it with `KeyboardInterrupt` and keep the results so far, and adds `globals.PROGRESS_POLLING_PERIOD` and `globals.ZERO_ATOL`.
- **Key APIs:** `qiskit_addon_slc.bounds.compute_forward_bounds(circuit, noise_model_paulis, observable, *, evolution_max_terms=1_000_000, eigval_max_qubits=14, atol_simplify=..., atol_eigenvalue=..., num_processes=..., timeout=..., max_num_boxes=...)`, `compute_backward_bounds`, `tighten_with_speed_limit(bounds, circuit, noise_model_paulis, observable)`, `merge_bounds(circuit, forward_bounds, backward_bounds, noise_rates)`, `compute_local_scales(circuit, bounds, noise_rates, *, sampling_cost_budget=inf, bias_tolerance=0.0)` (set one of the two, not both; returns the `local_scales` dict, the sampling cost γ² and the residual bias; `apply_in_post=True` instead ranks the terms for applying the scales to existing full-PEC data), `compute_bounds`, `LightCone`, `qiskit_addon_slc.utils.generate_noise_model_paulis`, `qiskit_addon_slc.utils.map_modifier_ref_to_ref`, `qiskit_addon_slc.visualization.draw_shaded_lightcone`, `qiskit_addon_utils.noise_management.gamma_from_noisy_boxes(noise_models, box_id_to_noise_id, noise_scales_each_box=None)`
- **Watch out:**
  - Bounds are computed for a circuit with bound parameters on fixed qubits, so a new angle or a new placement needs new bounds. Cache them.
  - `PEC.prepare` in qiskit-mitigation 0.1.1 cannot apply per-term local scales (it fails with one noise modifier per box); you need your own task that applies the scales to the noise modifiers. The tutorial does it with Samplomatic directly: `samplomatic.build` the boxed circuit, bind `local_scales` and `noise_scales.<ref> = -1` as samplex inputs, append the item to a `QuantumProgram(noise_maps=...)`, run it with `Executor`, and pass `gamma_from_noisy_boxes(noise_maps, id_map, local_scales)` as `gamma_factor` to `qiskit_addon_utils.exp_vals.expectation_values.executor_expectation_values`.
  - `compute_forward_bounds` and `tighten_with_speed_limit` raise `NotImplementedError` for an observable with more than one Pauli term; call them once per Pauli.
  - The bounds assume `InjectNoise` annotations with `site="after"` and the boxing option `inject_noise_strategy="individual_modification"`. Set `inject_noise_site="after"` explicitly, because Samplomatic's default changed.
  - A `timeout` fills the layers not reached with the trivial bound 2.0, which is safe but loose; more processes or a longer timeout tighten the bounds. `compute_backward_bounds` removes every `Measure` before it runs. `merge_bounds` with `noise_rates=None` assumes uniform rates, so recompute it once the rates are learned.
  - The tutorial sets `multiprocessing.set_start_method("spawn")` and `OMP_NUM_THREADS=1` before the bound computation. Windows is not supported.

**Docs:**
- [Shaded light cones overview](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-slc)
- [Quickstart](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-slc/guides/quickstart)
- [Tutorial: PEC with shaded light cones](https://quantum.cloud.ibm.com/docs/en/tutorials/pec-with-shaded-lightcones)
- [bounds API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-slc/bounds)
- [Release notes](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-slc/release-notes)
- [GitHub repository](https://github.com/Qiskit/qiskit-addon-slc)

## Propagated noise absorption and Pauli propagation

**PNA** (`qiskit-addon-pna`) inverts the learned noise of each entangling layer, propagates the anti-noise generators forward through the rest of the circuit with Pauli propagation, and absorbs them into the observable; you then measure the modified observable on the unmodified noisy circuit. It works best on near-Clifford circuits. Its bias comes from three truncations: of each propagated anti-noise generator (`max_err_terms`), of the growing observable (`max_obs_terms`), and of the terms you finally measure, since the new observable has more Pauli terms and larger coefficients and therefore needs more shots. **pauli-prop** is the Rust-accelerated Pauli-propagation engine underneath shaded light cones and PNA. You can also use it on its own to estimate expectation values classically, in the Heisenberg or the Schrödinger frame, truncating by coefficient tolerance or by term count, or to train circuit parameters. It is single-threaded.

- **When to use:** you want to move learned noise into the observable instead of canceling it in the circuit, or to estimate Pauli expectation values classically, for example to train circuit parameters before you run them.
- **Tags:** family: trotter, floquet, echoes, qaoa, multi-objective; step: map, post-process; noise: mitigation-expval; runs: local; status: stable; updated: 2026-08
- **Recent changes:** pauli-prop 0.2 (March 2026) propagates through noisy circuits that contain Aer `PauliLindbladError` instructions (`propagate_through_circuit`), and `propagate_through_rotation_gates` accepts `RotationGates` or `NoisyRotationGates`. Both packages now have IBM documentation pages and tutorials. This cheat sheet was checked against qiskit-addon-pna 0.2.0 (November 2025) and pauli-prop 0.2.1 (20 August 2026), both PyPI wheels.
- **Key APIs:** `qiskit_addon_pna.generate_noise_mitigating_observable(noisy_circuit, observable, refs_to_noise_model_map=None, *, max_err_terms, max_obs_terms, search_step=4, num_processes=1, batch_size=1, inject_noise_before=True, mp_start_method="spawn")`; `pauli_prop.evolve_through_cliffords`, `propagate_through_circuit(operator, circuit, max_terms, atol, frame)` (`frame` is `"h"` or `"s"`), `propagate_through_operator`, `propagate_through_rotation_gates`, `circuit_to_rotation_gates`, `RotationGates` (and `pauli_prop.propagation.NoisyRotationGates`)
- **Watch out:**
  - `generate_noise_mitigating_observable` uses `multiprocessing`, so call it under an `if __name__ == "__main__":` guard.
  - Give it either a boxed circuit with `InjectNoise` annotations plus `refs_to_noise_model_map` keyed by `InjectNoise.ref` (what `NoiseLearnerV3Results.to_dict` returns), or an unboxed circuit with Aer `PauliLindbladError` instructions next to each layer and no map.
  - `max_obs_terms` must exceed the number of terms in the observable, and the observable must have real coefficients.
  - The tutorial transpiles with `optimization_level=0` and a fixed `initial_layout` so the two-qubit layers that the boxing and noise learning rely on survive.

**Docs:**
- [PNA overview](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-pna)
- [PNA API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-pna/qiskit-addon-pna)
- [Tutorial: propagated noise absorption](https://quantum.cloud.ibm.com/docs/en/tutorials/propagated-noise-absorption)
- [Pauli propagation overview](https://quantum.cloud.ibm.com/docs/en/addons/pauli-prop)
- [Simulate noisy expectation values with Pauli propagation](https://quantum.cloud.ibm.com/docs/en/addons/pauli-prop/guides/simulate-noisy-expectation-values)

## Measurement mitigation with M3 (mthree)

Matrix-free measurement mitigation calibrates readout errors on the qubits you measure and corrects counts into quasi-probabilities or expectation values. It works in the subspace of observed bitstrings, so it scales to many qubits, and it supports mid-circuit measurements. `M3Mitigation` wraps an IBM backend in a `runningman.RunningManBackend` and submits its calibration circuits through the `runningman` package.

- **When to use:** your result is counts or quasi-probabilities on many qubits and readout errors matter.
- **Tags:** family: qaoa, multi-objective, lcu, sqd, verifiable, floquet; step: post-process; noise: mitigation-sampling; runs: local, qpu; status: stable; updated: 2025-03
- **Current version:** 3.0.0 (March 2025).
- **Key APIs:** `mthree.M3Mitigation(backend, iter_threshold=4096)`, `M3Mitigation.system.set_suppression_options(dynamical_decoupling=..., twirling=...)`, `M3Mitigation.cals_from_system(qubits, method="balanced")` (other keyword arguments: `shots`, `cals_file`, `async_cal`, default `True`, `runtime_mode` for a `Batch` or `Session`, `initial_reset`, `rep_delay`), `cals_to_file` and `cals_from_file`, `M3Mitigation.apply_correction(counts, qubits, method="auto", return_mitigation_overhead=False)`, `mthree.utils.final_measurement_mapping(circuit)`, `mthree.utils.expval` and `expval_and_stddev`
- **Watch out:**
  - The calibration is a QPU job of its own, submitted asynchronously by default (`async_cal=True`); save it with `cals_file` or `cals_to_file` if you want to reuse it.
  - If your jobs use measurement twirling, switch it on for the calibration too: call `mit.system.set_suppression_options(twirling={"enable_measure": True})` before `cals_from_system`, or the correction is biased. The call exists only when M3 wraps a real IBM backend (`mit.system` is then a `runningman` backend). On a fake backend, `mit.system` is the fake backend itself and has no such method, so calibrate without twirling and leave measurement twirling off in the local Sampler too. `M3Mitigation` does not accept `GenericBackendV2`, because it needs `backend.configuration()`, so test M3 on a fake backend such as `FakeBoston`.
  - Correct each circuit with its own final layout.

**Docs:**
- [M3 documentation](https://qiskit.github.io/qiskit-addon-mthree/)
- [Basic usage](https://qiskit.github.io/qiskit-addon-mthree/basic.html)
- [Expectation values](https://qiskit.github.io/qiskit-addon-mthree/expvals.html)
- [M3 with mid-circuit measurements](https://qiskit.github.io/qiskit-addon-mthree/tutorials/04_dynamic_bv.html)

## Sample-based quantum diagonalization (qiskit-addon-sqd)

SQD turns noisy hardware samples into an accurate energy estimate. It recovers configurations with the right particle numbers, subsamples them into batches, projects the Hamiltonian into the sampled determinant subspace, and diagonalizes it with a selected-CI solver, repeating until the energy and orbital occupancies converge. `diagonalize_fermionic_hamiltonian` is the entry point: it takes the one- and two-body integrals, the `BitArray` of samples, `samples_per_batch`, `norb` and `nelec`. The subspace dimension is set indirectly through `samples_per_batch`, the batch diagonalizations are independent of each other, and the fermionic solver handles about 25 spatial orbitals and 10 electrons with subspace dimensions near 10^7 on 10 to 30 cores.

- **When to use:** you want ground-state energies of a chemistry Hamiltonian from noisy hardware samples.
- **Tags:** family: sqd; step: post-process; noise: mitigation-sampling; runs: local, serverless; status: stable; updated: 2026-10
- **Recent changes:**
  - 0.12 added `max_dim` to limit the SCI subspace and `subsampling.postselect_by_hamming_right_and_left`, deprecating `postselect_and_subsample`.
  - 0.13 (August 2026) lets `diagonalize_fermionic_hamiltonian` run collectively under MPI, hands orbital optimization over to ffsim (`optimize_orbitals` and `rotate_integrals` are deprecated) and supports Python 3.13 and 3.14.
  - 0.14 (8 October 2026) adds Trim SQD (`qiskit_addon_sqd.trim.TrimPolicy`, passed as `policy=`) and `subsampling.partition_subsample`, which draws the disjoint batches it needs.
- **Key APIs:** `qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian(hcore, eri, bit_array, samples_per_batch, norb, nelec, num_batches=..., max_iterations=..., sci_solver=..., policy=..., symmetrize_spin=..., max_dim=..., carryover_threshold=..., callback=..., seed=...)`, `fermion.solve_sci_batch` (wrap it with `functools.partial` to pass `spin_sq` and `max_cycle`), `fermion.solve_fermion`, `qiskit_addon_sqd.trim.TrimPolicy(trim_ratio=0.1, carryover_threshold=1e-4, carryover_ratio=0.1, max_strings_per_trim=None, max_carryover=None)`, `subsampling.partition_subsample`, `subsampling.postselect_by_hamming_right_and_left`, `counts.generate_bit_array_uniform` for tests without a QPU
- **Watch out:**
  - Cap the subspace size (`max_dim`, and the trim limits under Trim SQD), or the diagonalization can run for hours. Otherwise the dimension is only set indirectly, through `samples_per_batch` and `num_batches`, so run a convergence study over `samples_per_batch` on a small case first.
  - Trim SQD costs `num_batches + 1` diagonalizations per iteration and only helps when the batches differ, so `TrimPolicy` draws disjoint batches itself; if you build batches by hand for a screening solver, use `partition_subsample` rather than independent `subsample` calls.
  - The APIs are single-threaded by contract. Only `diagonalize_fermionic_hamiltonian` may be called collectively under MPI, and only the eigensolver step is distributed; the rest of the loop runs on rank 0, and a custom collective `sci_solver` must abort every process on an error.
  - The batch solver calls are embarrassingly parallel, and the docs recommend running them in parallel; the tutorial's loop over batches is serial.
  - For very large runs, the C++ `qiskit-addon-sqd-hpc` library (GitHub only, no Python package) handles the post-processing steps. Windows is unsupported because pyscf does not install there.
- **Related:** [Managed Workflows](managed-workflows.md#write-upload-and-run-a-workflow) to run the post-processing next to the QPU; [Qiskit Function templates](managed-workflows.md#qiskit-function-templates) for ready-made SQD workflows.

**Docs:**
- [SQD overview](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-sqd)
- [SQD quickstart](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-sqd/guides/quickstart)
- [Bounding the subspace dimension](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-sqd/guides/choose-subspace-dimension)
- [Tutorial: sample-based quantum diagonalization](https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-quantum-diagonalization)
- [HPC acceleration](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-addon-sqd/guides/hpc-acceleration)
- [Tutorial: sample-based Krylov quantum diagonalization of a fermionic lattice model](https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-krylov-quantum-diagonalization)
- [Trim SQD guide](https://qiskit.github.io/qiskit-addon-sqd/guides/trim_sqd.html)
- [SQD release notes](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-sqd/release-notes)
- [Tutorial: implicit-solvent calculations with SQD in a managed workflow](https://quantum.cloud.ibm.com/docs/en/tutorials/implicit-solvent-calculations)

## Multi-product formulas (qiskit-addon-mpf)

A multi-product formula (MPF) combines expectation values from several Trotter circuits with different step counts k_j, ⟨A⟩ = Σ_j x_j ⟨A⟩_{k_j}, with coefficients chosen so that the leading Trotter-error terms cancel: deep-circuit accuracy from shallower circuits, at the cost of running several and amplifying their noise by the coefficients. Static coefficients depend only on the k_j and the formula's order and symmetry; dynamic ones also depend on the Hamiltonian, state and time and come from tensor-network overlaps (quimb or TeNPy). The addon only produces the coefficients; you run the circuits with the Estimator and combine the numbers.

- **When to use:** Trotter error, not qubit count, limits a time-evolution expectation value, and depth is the bottleneck.
- **Tags:** family: trotter, lcu; step: post-process; runs: local; status: stable; updated: 2025-03
- **Current version:** 0.3.0 (20 March 2025): Qiskit SDK 2.0 support, Python 3.13, a `conserve` option for TeNPy. Its pin is `qiskit>=1.2,<3`; static and dynamic coefficients (quimb TEBD, TeNPy layer models) and all three cost problems ran under Qiskit 2.5.2 and 2.6.0 with cvxpy 1.9.3 when this document was checked. Install `qiskit-addon-mpf[quimb]` or `[tenpy]`.
- **Key APIs:** `qiskit_addon_mpf.static.setup_static_lse(trotter_steps, order=..., symmetric=...)`, `costs.LSE.solve()`, `costs.setup_exact_problem`, `costs.setup_sum_of_squares_problem(lse, max_l1_norm=...)`, `costs.setup_frobenius_problem`, `dynamic.setup_dynamic_lse`, `backends.quimb_tebd.TEBDEvolver`, `backends.tenpy_layers.LayerModel.from_quantum_circuit`, `LayerwiseEvolver`, `backends.HAS_QUIMB`, `HAS_TENPY`
- **Watch out:**
  - The L1 norm of the coefficients amplifies shot noise and hardware error: in IBM's 50-qubit tutorial, the exact static coefficients (L1 norm 4.7) gave the worst estimate and the largest error bar, while the norm-capped (`max_l1_norm=2`) and dynamic coefficients both beat a deeper single circuit with 10 Trotter steps. Keep t/k_min at or below about 1 and mitigate each circuit ([qiskit-mitigation](noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec)).
  - Under TeNPy's default spin conservation, `LayerModel.from_quantum_circuit` rejects a layer of `rxx` or `ryy` gates (`ValueError: operator 'Sx' not existent`): pass `conserve="None"`, or fuse each `rxx`/`ryy` pair into an `XXPlusYYGate` as the tutorial does.
- **Related:** [Operator backpropagation](#operator-backpropagation-qiskit-addon-obp) and [AQC-Tensor](#aqc-tensor-qiskit-addon-aqc-tensor) shorten the circuit instead; [Exact and tensor-network references](simulators.md#exact-and-tensor-network-references) for reference values.

**Docs:**
- [MPF documentation home](https://qiskit.github.io/qiskit-addon-mpf/)
- [Tutorial: multi-product formulas to reduce Trotter error](https://quantum.cloud.ibm.com/docs/en/tutorials/multi-product-formula)
- [How to choose the Trotter steps](https://qiskit.github.io/qiskit-addon-mpf/how_tos/choose_trotter_steps.html)
- [Static MPFs API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-mpf/static)
- [Dynamic MPFs API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-mpf/dynamic)
- [Cost functions API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-mpf/costs)
- [Backends API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-mpf/backends)
- [Release notes](https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-mpf/release-notes)
