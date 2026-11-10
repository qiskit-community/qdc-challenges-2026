---
title: IBM Quantum Compute client (qiskit-ibm-runtime)
short_title: Compute client
tool_type: client
owner: IBM Quantum Compute client team
last_verified: 2026-10-20
---

# IBM Quantum Compute client (qiskit-ibm-runtime)

`qiskit-ibm-runtime` is the package that connects Qiskit to IBM QPUs. The Qiskit Runtime service is now called IBM Quantum Compute Service (July 2026; no code changes), and the docs call `qiskit-ibm-runtime` the IBM Quantum Compute client. The package name is unchanged, and since 0.49 `QiskitRuntimeService` is also available as `IBMQuantumComputeService`. This document covers execution with `Executor` and the Sampler and Estimator primitives, including their dynamical-decoupling and twirling options, execution modes, QPU-time budgeting, saving results, noise learning, the hardware features you reach through the client (fractional gates, specially calibrated instructions, dynamic circuits), local testing with fake backends and the NEAT debugging tool.

Package versions this document was checked against:

| Package | Version checked | Notes |
|---|---|---|
| `qiskit-ibm-runtime` | 0.50.0 | Requires Python 3.10 or later and Qiskit 2.3 or later; installs `samplomatic>=0.21`, `qiskit-mitigation>=0.1.1` and `ibm-quantum-schemas>=0.12.20260921` automatically. Runtime 0.49 has `SamplerV2` but not the client-side `executor_sampler.Sampler` ([Sampler and Estimator](#sampler-and-estimator-client-side-primitives)). Keep a notebook's 0.49 pin for that notebook. To use 0.50 in an environment built from a requirements file that pins 0.49 and an older `ibm-quantum-schemas`, do not only edit the runtime pin, because 0.50 needs the newer schemas release: run `pip install -r requirements.txt`, then `pip install qiskit-ibm-runtime==0.50.0` |

## Fractional gates

Heron QPUs run *fractional gates*: a native `RZZ(θ)` for 0 < θ ≤ π/2 and `RX(θ)` for any angle. One fractional `RZZ` replaces a two-CZ decomposition, and a fractional `RX` replaces the RZ and SX decomposition of an arbitrary single-qubit rotation, which cuts its duration and error by up to a factor of two. Both shorten circuits that have many small rotation angles. Load the backend with `use_fractional_gates=True` to add them to the target; `service.backends(use_fractional_gates=True)` lists the backends that support them.

- **When to use:** your circuit has many small-angle `RZZ` or `RX` rotations (Trotter steps, QAOA cost layers, variational ansatzes with many parameters) that a Heron QPU can run as single native gates.
- **Tags:** family: qaoa, lcu, trotter, floquet, thermal; step: optimize; runs: qpu, local; status: experimental; updated: 2026-07
- **Recent changes:** asking for fractional gates on a backend that lacks them now raises an error (0.44 and 0.45). The target carries the `rzz` angle bounds (0.48), so the preset pass managers keep bound angles in range. `service.least_busy(use_fractional_gates=True)` considers only backends that support them and returns the least busy one (0.48 or later).
- **Key APIs:** `QiskitRuntimeService.backend(name, use_fractional_gates=True)`, `service.backends(use_fractional_gates=True)`, with `RZZGate` and `RXGate` from the [circuit library](qiskit-sdk.md#circuit-library)
- **Watch out:**
  - Fractional gates are experimental, and the behavior of the `use_fractional_gates` flag may change. IBM's fractional-gates guide lists Pauli gate twirling, PEC and ZNE with PEA as unsupported with them; measurement twirling with TREX works. The Estimator's gate-folding ZNE is accepted but does not fold `rzz` ([Sampler and Estimator](#sampler-and-estimator-client-side-primitives)).
  - The `local_pauli` twirling group twirls fractional `RZZ` gates only partially. It is available in Samplomatic and as `options.twirling.group` in the client-side primitives. It uses only the Paulis that commute with the gate, so a coherent ZZ over-rotation stays coherent.
  - Angles of parametric `RZZ` gates must be bound inside (0, π/2], or the job fails. The transpiler assumes the parameters will receive in-range values at runtime, so the check happens only when the PUB runs.
  - For a negative angle, put an X on one of the two qubits before and after the gate: RZZ(−θ) = (X⊗I)·RZZ(θ)·(X⊗I) for θ > 0. The runtime's `FoldRzzAngle` pass does this rewrite for bound angles ([Calibrated instructions](#specially-calibrated-instructions-and-calibration-sets)).
  - The transpiler has limited ability to use `RZZ` in its optimization passes, so take more care with hand-built structure when you use fractional gates.
  - The fractional-gates guide says the error reported for a fractional gate in the target is a copy of the non-fractional counterpart's error, because error reporting for fractional gates is not yet supported; the QPU-information guide, however, documents an angle-averaged `rzz` error for Heron. Treat the `rzz` error with caution either way.
  - Fake backends leave fractional gates out by default; [Fake backends and local testing mode](#fake-backends-and-local-testing-mode) shows how to add them and how to test without an account.
- **Related:** [Choosing error detection and mitigation](../guides/techniques.md#choosing-error-detection-and-mitigation) has a fractional `RZZ` column for every mitigation method; [Twirling options in the primitives](#twirling-options-in-the-primitives) covers the `local_pauli` group.

**Docs:**
- [Fractional gates](https://quantum.cloud.ibm.com/docs/en/guides/fractional-gates)

## Specially calibrated instructions and calibration sets

Beyond the standard gates, IBM QPUs calibrate extra instructions that you add from `qiskit_ibm_runtime.circuit.library`. They are calibrated alongside the default instruction set, so you add them to a circuit like any other instruction. Support depends on the QPU, so check `backend.supported_instructions` before you use one; `service.backends(filters=lambda b: "measure_2" in b.supported_instructions)` finds the backends that support a given instruction, and `backend.configuration().instruction_signatures` shows their signatures.

| Instruction | What it does | Processors | Runtime version |
|---|---|---|---|
| `MidCircuitMeasure` (`measure_2`) | Shorter measurement for mid-circuit use | Heron, Nighthawk r2 | 0.43 or later |
| `MidCircuitReset` (`reset_2`) | Shorter conditional reset that uses `measure_2` | Heron | 0.49 or later |
| `MeasureReset` (`measure_reset`, `measure_reset_2`) | Measure and reset in one instruction; pass `name="measure_reset_2"` for the shorter variant | Heron, Nighthawk r2 | 0.50 or later |
| `XSlowGate` (`xslow`) | X gate with an extended duration, used by post-selection checks | all | 0.50 or later |

On Heron the reset in `measure_reset` is conditional on the measurement result; on Nighthawk r2 it is unconditional. `xslow` is the default `x_pulse_type` of the post-selection passes in qiskit-mitigation.

You can also load an *alternative calibration* by passing `calibration_id` when you get the backend; its instruction properties then apply to both the transpiler and your jobs, and `backend.calibration_id` tells you which one is loaded. The documented alternative calibrations slow down `cz`, and availability depends on the QPU. The guide recommends:

* the default calibration for workloads whose bitstrings can be classically distilled (peaked circuits, SQD) and for error-detection workloads;
* `slow_gates_dc` (Heron) for large-scale observable estimation with heuristic mitigation such as ZNE;
* `slow_gates_ac` (Heron), with all jobs in one session, for workflows that learn noise for PEA or PEC, because its AC two-level-system bias keeps the error rates stable over a long session;
* `slow_gates` (Nighthawk) for observable estimation with ZNE, PEA or PEC.

If your challenge notebook names a calibration ID, use that one, and check that the notebook allows a different calibration before you switch; a session uses QPU time for as long as it stays open, and the Open Plan has no sessions. Compare `backend.target["cz"][pair].duration` between the default and the slow backend to see what the change costs, and transpile against a backend loaded with the same `calibration_id` you run on.

The runtime also provides transpiler passes:
* `ConvertToMidCircuitMeasure` (0.44) and `ConvertToMidCircuitResetAndMeasure` (0.49) replace non-terminal measurements and resets with `measure_2` and `reset_2` where the target supports them.
* `FoldRzzAngle` folds out-of-range fractional `RZZ` angles. Since 0.48 the preset pass managers do this through the target's angle bounds.
* `ConvertISAToClifford` rounds an ISA circuit to a Clifford one for cheap local tests.

- **When to use:** you need faster mid-circuit measurements or resets, slow-X error checks, or a more stable CZ calibration for noise learning.
- **Tags:** family: thermal, detection, verifiable, echoes; step: optimize; runs: qpu; status: stable; updated: 2026-09
- **Recent changes:** `MidCircuitReset` (0.49), `MeasureReset` and `XSlowGate` (0.50), and the `ConvertToMidCircuit*` passes.
- **Key APIs:** `QiskitRuntimeService.backend(name, calibration_id=...)`, `backend.calibration_id`, `backend.supported_instructions`, `backend.configuration().instruction_signatures`, `qiskit_ibm_runtime.circuit.library` (`MidCircuitMeasure`, `MidCircuitReset`, `MeasureReset`, `XSlowGate`), `qiskit_ibm_runtime.transpiler.passes` (`ConvertToMidCircuitMeasure`, `ConvertToMidCircuitResetAndMeasure`, `FoldRzzAngle`, `ConvertISAToClifford`)
- **Watch out:** Aer cannot simulate these instructions. A local run fails with `AerError: unknown instruction: measure_2` (likewise `reset_2`, `measure_reset` and `xslow`), even on `FakeBoston`, whose target lists `measure_2` and `reset_2`, and no fake backend in runtime 0.50 has `measure_reset` or `xslow`. Test with plain `measure` and `reset`, and apply `ConvertToMidCircuitMeasure` or `ConvertToMidCircuitResetAndMeasure` only to the copy you submit.
- **Related:** [Dynamic circuits and resets](#dynamic-circuits-and-resets), [Fractional gates](#fractional-gates)

**Docs:**
- [Specially calibrated instructions and calibration sets](https://quantum.cloud.ibm.com/docs/en/guides/custom-calibration)
- [circuit.library API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/circuit-library)
- [FoldRzzAngle](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/transpiler-passes-fold-rzz-angle)
- [ConvertISAToClifford](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/transpiler-passes-convert-isa-to-clifford)

## Dynamic circuits and resets

Dynamic circuits measure qubits mid-circuit and act on the results in real time (`if_test`, `switch`, classical expressions). Resets let you reuse qubits or re-prepare ancillas. `backend.supported_instructions` shows whether a backend lists `if_else` and `store`.

- **When to use:** you measure or reset qubits mid-circuit, reuse qubits, or condition later gates on measurement results.
- **Tags:** family: thermal, detection; step: map; runs: qpu, local; status: stable; updated: 2026-09
- **Recent changes:** the `Store` instruction runs on IBM QPUs (May 2026): compute a classical expression mid-circuit, save it to a variable, and condition later gates on the variable instead of on a deeply nested expression. `ibm_phoenix`, the first Nighthawk r2 QPU (September 2026), supports mid-circuit reset and dynamic circuits with a fast reset ([QPUs at a glance](hardware.md#qpus-at-a-glance)).
- **Key APIs:** `QuantumCircuit.if_test`, `QuantumCircuit.switch`, `QuantumCircuit.reset`, `QuantumCircuit.store`, `qiskit.circuit.classical.expr` (`expr.lift`, `expr.bit_xor` for parities)
- **Watch out:**
  - In a dynamic circuit, only `rz` gates may carry unbound parameters. Bind the others before you submit, or the job fails with error 1553 ("Parametric instructions other than RZ gates are not supported with dynamic circuits").
  - Many mid-circuit measurements or resets can need a longer repetition delay; a timing error (8038) means `rep_delay` is too short for them. Increase `rep_delay` or reduce the number of measurements and resets per circuit.
  - If `if_else` is missing from `backend.target.operation_names`, the long-range-entanglement tutorial adds it with `backend.target.add_instruction(IfElseOp, name="if_else")` before transpiling.
  - Not every fake backend's target has `reset` and `if_else` ([QPUs at a glance](hardware.md#qpus-at-a-glance) says which lack them). Check `backend.target.operation_names` before you test a reset or feed-forward circuit locally.
  - The primitives' built-in dynamical decoupling does not apply to dynamic circuits; pad them yourself ([Dynamical decoupling passes](qiskit-sdk.md#dynamical-decoupling-passes)).
- **Related:** [Specially calibrated instructions and calibration sets](#specially-calibrated-instructions-and-calibration-sets) for `measure_2`, `reset_2` and `measure_reset`.

**Docs:**
- [Classical feedforward and control flow](https://quantum.cloud.ibm.com/docs/en/guides/classical-feedforward-and-control-flow)
- [Execute dynamic circuits](https://quantum.cloud.ibm.com/docs/en/guides/execute-dynamic-circuits)
- [Deferred timing resolution using stretch](https://quantum.cloud.ibm.com/docs/en/guides/stretch)
- [Tutorial: long-range entanglement with dynamic circuits](https://quantum.cloud.ibm.com/docs/en/tutorials/long-range-entanglement)
- [IBM Quantum Compute Service changelog](https://quantum.cloud.ibm.com/docs/en/guides/changelog-quantum-compute-service)

## Dynamical decoupling options

The primitives can insert dynamical decoupling (DD) for you: enable `options.dynamical_decoupling` and choose a sequence.

- **When to use:** your circuit leaves qubits idle, for example in sparse layers, and has no mid-circuit measurements, resets or feed-forward.
- **Tags:** family: general; step: execute; noise: suppression; runs: qpu, local; status: stable; updated: 2026-09
- **Recent changes:** the client-side primitives of runtime 0.50 (24 September 2026) insert the sequences on your machine, in local mode too, and build Y pulses from X and RZ gates; the legacy `SamplerV2` and `EstimatorV2` let the service insert them.
- **Key APIs:** `options.dynamical_decoupling.enable`, `options.dynamical_decoupling.sequence_type`
- **Watch out:**
  - The primitives' `sequence_type` options (`"XX"`, `"XpXm"` and `"XY4"`) work on any IBM backend.
  - The built-in DD option cannot be combined with dynamic circuits. For circuits with mid-circuit measurements, resets or feed-forward, pad the circuit yourself ([Dynamical decoupling passes](qiskit-sdk.md#dynamical-decoupling-passes)).
- **Related:** [Twirling options in the primitives](#twirling-options-in-the-primitives), the other suppression option of the primitives.

**Docs:**
- [Error mitigation and suppression techniques](https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques)

## Executor and QuantumProgram

`Executor` is the low-level primitive behind the *directed execution model*. You give it a `QuantumProgram` made of circuits, plus optional Samplomatic *samplex* items: objects that describe exactly which randomized variants to generate (twirls, injected noise, basis changes). It runs exactly what you specified and returns raw shot data, with no implicit error suppression or mitigation, so every processing step stays in your code.

- **When to use:** you want full control over what runs: explicit twirling and noise injection, raw shot data and your own mitigation.
- **Tags:** family: general, lcu, trotter, floquet, echoes, thermal, verifiable; step: execute; runs: qpu, local; status: beta; updated: 2026-09
- **Recent changes:**
  - The model was announced in January 2026, and `Executor` and `QuantumProgram` shipped in `qiskit-ibm-runtime` 0.47 (May 2026). IBM still labels all of the model's components beta.
  - Version 0.50 (September 2026) adds local mode: pass a fake backend or an `AerSimulator` as `mode`, and tune the simulation with `options.simulator`.
  - Version 0.50 also adds `dry_run=True`, which runs a job against a mock twin of the backend instead of a QPU (see [Budgeting QPU time](#budgeting-qpu-time)). It moves the base classes of `QuantumProgram` and its results into `samplomatic.quantum_program`; importing `CircuitItem`, `SamplexItem`, `DataTree` and the chunk-timing classes from `qiskit_ibm_runtime.quantum_program` and `qiskit_ibm_runtime.results` is deprecated.
  - Since 0.49 the classes in `options_models` are Pydantic models: use `options.model_dump()` instead of `asdict`.
  - `QuantumProgram.meas_level = "both"` (0.48) returns classified and kerneled data together, and `Executor` and `NoiseLearnerV3` expose a `mode` property (0.50).
- **Key APIs:** `Executor(mode=backend)`, `Executor.run(program, dry_run=False)`, `qiskit_ibm_runtime.QuantumProgram(shots=...)`, `QuantumProgram.append_circuit_item(circuit, circuit_arguments=...)`, `QuantumProgram.append_samplex_item(circuit, samplex=..., samplex_arguments=..., shape=...)`, `ExecutorOptions`, `job.result()[i]["<register>"]`
- **Watch out:**
  - Create programs with `qiskit_ibm_runtime.QuantumProgram` (now a subclass of the Samplomatic class). The local `Executor` fails on a plain Samplomatic `QuantumProgram` with `AttributeError: '_semantic_role'`.
  - A program has one shot value for all of its items. A different shot count means a second program, and therefore a second job.
  - Results arrive per program item as arrays of raw bits keyed by classical register name, with shape `(randomizations, parameter sets, shots, bits)` for samplex items; the right-most axis runs over classical bits in little-endian order. Measurement-twirl flips come back under `measurement_flips.<register>`. Undo them with an XOR before you compute expectation values.
  - The Executor options guide was written for runtime 0.47. In 0.50, set `meas_level` on `QuantumProgram` instead of `execution.meas_type`, and set `max_execution_time` at the top level of the options (`environment.max_execution_time` is deprecated).
- **Related:** [Pauli twirling with Samplomatic](noise-management.md#pauli-twirling-with-samplomatic) builds the samplex items; [Sampler and Estimator: client-side primitives](#sampler-and-estimator-client-side-primitives) wraps `Executor` in the PUB interface.

**Docs:**
- [Directed execution model overview](https://quantum.cloud.ibm.com/docs/en/guides/directed-execution-model)
- [Executor quickstart](https://quantum.cloud.ibm.com/docs/en/guides/get-started-with-executor)
- [Executor inputs and outputs](https://quantum.cloud.ibm.com/docs/en/guides/executor-input-output)
- [Executor broadcasting](https://quantum.cloud.ibm.com/docs/en/guides/executor-broadcasting)
- [Executor options](https://quantum.cloud.ibm.com/docs/en/guides/executor-options)
- [Executor examples](https://quantum.cloud.ibm.com/docs/en/guides/executor-examples)
- [Migrate from Sampler to Executor](https://quantum.cloud.ibm.com/docs/en/guides/migrate-from-sampler-to-executor)
- [Executor API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/executor)
- [IBM Quantum Compute client release notes](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/release-notes)

## Sampler and Estimator: client-side primitives

`qiskit-ibm-runtime` 0.50 adds a client-side `Sampler` and `Estimator` built on `Executor`. They keep the familiar PUBs (primitive unified blocs: a circuit with its parameter values or observables, plus optional shots or precision), options and result shapes. The pre- and post-processing (twirling, TREX, ZNE, PEA and PEC bookkeeping) now runs on your machine. You can inspect that work and test the mitigation options locally on a fake backend. Switching usually needs only a new import.

- **When to use:** you want the familiar PUB interface: bitstrings from the Sampler, or expectation values with built-in mitigation from the Estimator.
- **Tags:** family: general; step: execute; noise: mitigation-expval; runs: qpu, local; status: stable; updated: 2026-09
- **Recent changes:**
  - Client-side primitives arrived in 0.50 (24 September 2026).
  - The server-side `SamplerV2` and `EstimatorV2` are deprecated and will be removed no sooner than three months after that release. Later, the top-level `Sampler` and `Estimator` imports will point to the client-side classes.
  - Differences to know:
    - `job.primitive_id` reads `executor`, and `job.inputs` returns Executor inputs; read the options a job ran with from `job.result().metadata["options"]`.
    - PEA and PEC no longer learn noise for you: learn it with `NoiseLearnerV3` (which has no local mode) and pass `options.resilience.layer_noise_model`. The older `NoiseLearner` does not work with the client-side Estimator.
    - One job cannot mix different `shots` or `precision` values: group PUBs by value, one job per value, and submit the groups in a batch.
    - Some input validation moved to the server and now raises `RuntimeError` instead of `IBMInputValueError`. `options.seed_estimator` is gone (setting it raises a `ValidationError`), and `measure_noise_learning.shots_per_randomization` is no longer supported.
    - Circuit metadata is copied into the result metadata and may only hold `str`, `float`, `int`, `bool`, and lists or dictionaries of those.
    - More work happens on your machine, so `run()` and `job.result()` can take longer; set the `qiskit_ibm_runtime` logger to `INFO` to follow the progress.
- **Key APIs:**
  - `from qiskit_ibm_runtime.executor_sampler import Sampler`
  - `from qiskit_ibm_runtime.executor_estimator import Estimator`
  - `Estimator.find_unique_layers(pubs)`, `NoiseLearnerV3(mode).run(layers)`, `learner_result.to_pauli_lindblad_maps()`, `options.resilience.layer_noise_model = zip(layers, maps)`
  - `options_models.SamplerOptions` and `options_models.EstimatorOptions`
  - Estimator mitigation: `options.resilience_level` (default 1, which turns on TREX; 2 adds ZNE and gate twirling), `options.resilience.measure_mitigation`, `options.resilience.measure_noise_learning.num_randomizations`, `options.resilience.zne_mitigation`, `options.resilience.zne.noise_factors`, `options.resilience.zne.extrapolator`, `options.resilience.zne.amplifier` (`"pea"` for PEA), `options.resilience.pec_mitigation`
  - `SparsePauliOp.apply_layout(isa.layout)`
- **Watch out:**
  - Keep the primitive your notebook and its grader expect. If they are built on `SamplerV2`, do not switch to the client-side class on your own; the deprecated classes keep working through the deprecation period.
  - In 0.50, `from qiskit_ibm_runtime import SamplerV2, EstimatorV2` still gives the deprecated server-side classes. Import the client-side ones from `executor_sampler` and `executor_estimator`.
  - Run locally, the client-side Estimator measures every qubit of the ISA circuit, and Aer cannot simulate a whole large device. On `FakeBoston` it fails with `ValueError: could not broadcast input array`, so test it on a small backend such as `GenericBackendV2(num_qubits=16)` (see [Local test patterns](../guides/testing-and-tips.md#local-test-patterns)), or reduce the ISA circuit to a Clifford one with `ConvertISAToClifford` as the migration guide shows. The Clifford copy has no `layout` and no parameters: lay the observable out from the original ISA circuit and drop the parameter values from the local PUB.
  - Map each observable onto the transpiled circuit with `observable.apply_layout(isa.layout)` before you build an Estimator PUB. An observable on the logical qubits is rejected because its size does not match the circuit.
  - To choose mitigation methods yourself, set `resilience_level = 0` and switch on only the options you want, for example `options.resilience.measure_mitigation = True` for TREX alone while you run ZNE from your own folded circuits. Options you set explicitly apply on top of the level's base set, so a level-0 Estimator with `zne_mitigation = True` does run ZNE.
  - Not every option works with every circuit: gate twirling (in the default group), PEA and PEC refuse fractional gates, and gate-folding ZNE cannot be combined with PEA or PEC (see the feature compatibility table in the Estimator options guide). Gate-folding ZNE (`zne_mitigation = True` with the default amplifier) is accepted on a circuit with fractional `rzz` gates but amplifies nothing: the folding touches only ECR, CX, CY, CZ and SWAP (`SUPPORTED_FOLDED_GATES`, [qiskit-mitigation](noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec)), so the noise-factor circuits are identical copies, the job costs as many times the shots as there are noise factors, and the extrapolation fits shot noise, without any warning. For a fractional-gate circuit, run TREX alone (`resilience_level = 0`, `measure_mitigation = True`) and fold the `rzz` gates yourself, one PUB per noise factor ([qiskit-mitigation](noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec), [Practical tips](../guides/testing-and-tips.md#practical-tips)), or load the backend with `use_fractional_gates=False`, so that each `rzz` becomes two CZ gates that the built-in ZNE, gate twirling, PEA and PEC can handle, at the price of a longer circuit.

**Docs:**
- [Migrate to client-side primitives](https://quantum.cloud.ibm.com/docs/en/guides/migrate-to-client-side-primitives)
- [executor_sampler.Sampler API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/executor-sampler-sampler)
- [executor_estimator.Estimator API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/executor-estimator-estimator)
- [Primitive inputs and outputs](https://quantum.cloud.ibm.com/docs/en/guides/primitive-input-output)
- [Introduction to primitive options](https://quantum.cloud.ibm.com/docs/en/guides/runtime-options-overview)
- [Sampler options](https://quantum.cloud.ibm.com/docs/en/guides/sampler-options)
- [Estimator options](https://quantum.cloud.ibm.com/docs/en/guides/estimator-options)
- [Introduction to primitives, with examples](https://quantum.cloud.ibm.com/docs/en/guides/primitives-examples)
- [Tutorial: combine error mitigation options with the Estimator](https://quantum.cloud.ibm.com/docs/en/tutorials/combine-error-mitigation-techniques)
- [Noise management with the Estimator](https://quantum.cloud.ibm.com/docs/en/guides/estimator-noise-management)
- [ConvertISAToClifford transpiler pass](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/transpiler-passes-convert-isa-to-clifford)

## Execution modes: job, batch and session

A job runs on its own. A `Batch` groups independent jobs so they are scheduled together: their classical pre-processing runs in parallel and the QPU work follows in quick succession. A `Session` reserves the QPU for a sequence of dependent jobs and uses QPU time for as long as it stays open, whether or not a job is running; sessions are not available on the Open Plan.

- **When to use:** you submit more than one job: a batch for independent jobs, a session for iterative jobs that depend on each other.
- **Tags:** family: general; step: execute; runs: qpu; status: stable
- **Key APIs:** `Batch(backend=...)`, `Session(backend=...)`, `Sampler(mode=batch)`, `Executor(mode=batch)`, `batch.usage()`, `session.usage()`
- **Watch out:**
  - Neither mode shortens the queue for the first job, so they bring nothing for a single job.
  - Batched jobs are not guaranteed to run in submission order, and other users' jobs or calibration jobs can run between them.
  - Submit every job first, then collect the results: `job.result()` blocks, so calling it inside the submission loop serializes the batch.
  - A batch or session is deactivated when no job is ready within the interactive time to live, and it ends when the maximum time to live runs out; queued jobs then fail.

**Docs:**
- [Execution modes](https://quantum.cloud.ibm.com/docs/en/guides/execution-modes)
- [Choose the right execution mode](https://quantum.cloud.ibm.com/docs/en/guides/choose-execution-mode)
- [Run jobs in a batch](https://quantum.cloud.ibm.com/docs/en/guides/run-jobs-batch)
- [Run jobs in a session](https://quantum.cloud.ibm.com/docs/en/guides/run-jobs-session)
- [Execution modes FAQ](https://quantum.cloud.ibm.com/docs/en/guides/execution-modes-faq)

## Budgeting QPU time

QPU time is the scarce resource in the QDC 2026 challenges. Most notebooks state a budget, as seconds of QPU time or as shot counts and job sizes; either way, estimate before you submit. Usage is the time the QPU is locked for your workload. A job's usage is roughly a fixed overhead (about 2 s per sub-job) plus (repetition delay + circuit duration) × number of executions, so shots, circuit count and `rep_delay` all count, and mitigation adds circuits of its own. Estimate the cost before you submit: schedule the circuit (`scheduling_method="alap"`), call `estimate_duration`, add the reset duration and the backend's default repetition delay, and multiply by the number of executions (the circuits after broadcasting, including twirled variants and calibration circuits, times the shots; the next paragraph shows how to count them); without a custom `rep_delay` or advanced mitigation, `2 + 0.00035 × executions` seconds is a quick formula. Compare the estimate with the job's reported usage afterwards. Transpiling against a live backend, reading its target and properties, and retrieving finished jobs use no QPU time.

To count the executions a client-side Sampler or Estimator will submit, run it locally with the same options on a small backend whose target has your gates ([Local test patterns](../guides/testing-and-tips.md#local-test-patterns)): `job.inputs` is the `QuantumProgram` it built, with `program.shots` and `program.items`, and the sum over the items of `prod(item.shape) × program.shots` is the number of executions (the shape counts twirling randomizations and parameter sets; each ZNE noise factor and the TREX calibration circuits are items of their own). The program is built before the simulation runs. If your account can use `dry_run=True`, compare with the dry-run job's `job.usage()` once it finishes.

- **When to use:** before every submission.
- **Tags:** family: general; step: execute; runs: local, account; status: stable; updated: 2026-09
- **Recent changes:**
  - `job.usage(partial=True)` reports the running total while a job executes (0.48); without the flag, `usage()` returns 0 until the final calculation is done.
  - `dry_run=True` (0.50) is accepted by `Executor.run`, by the primitives' `run` and by `NoiseLearnerV3.run`. It runs the job on the server against a mock twin of the backend (`ibm_<name>` becomes `mock_<name>`), not on a QPU, and returns random data with the expected shapes plus usage and timing metadata. If your account cannot see the mock backend, `run` raises `QiskitBackendNotFoundError` instead of submitting; in local mode the flag only emits a warning and has no effect. IBM notes that access may be restricted, so check that the returned job ran on the `mock_` backend.
  - The allowed `rep_delay` range on Heron grew to 20–2000 µs (February 2026); the defaults were unchanged. Ranges differ between QPUs and change over time, so read `backend.rep_delay_range` before you set one.
  - The Nighthawk r2 QPU `ibm_phoenix` defaults to a 1 µs repetition delay (September 2026), against 250 µs on most other backends.
- **Key APIs:** `service.usage()` (the active instance), `job.usage()`, `job.usage_estimation`, `job.metrics()`, `batch.usage()`, `session.usage()`, `QuantumCircuit.estimate_duration(target)`, `backend.target["reset"][(0,)].duration`, `backend.default_rep_delay`, `backend.rep_delay_range`, `options.execution.rep_delay`, `options.execution.init_qubits`, `max_execution_time`
- **Watch out:**
  - Session usage is wall-clock time while the session is active, even between jobs; batch usage is the cumulative locked time of its jobs.
  - Failed or canceled jobs still count when the cause is a user error or your own cancellation of a running job; a job canceled while it is still queued, and a job that fails from a system error, report zero usage.
  - Lowering `rep_delay` shortens a job but raises the state-preparation error rate; raising it costs usage directly.
  - The experimental `scheduler_timing` option returns the circuit time, but that is not the time used for billing.

**Docs:**
- [Workload usage (estimate and query usage)](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time)
- [Minimize job run time](https://quantum.cloud.ibm.com/docs/en/guides/minimize-time)
- [Maximum execution time](https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time)
- [Qubit initialization and repetition delay](https://quantum.cloud.ibm.com/docs/en/guides/repetition-rate-execution)
- [Circuit timing for Executor and Sampler jobs](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-runtime-circuit-timing)
- [RuntimeJobV2 (usage, metrics)](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/runtime-job-v2)
- [IBM Quantum Compute Service changelog](https://quantum.cloud.ibm.com/docs/en/guides/changelog-quantum-compute-service)

## Saving results and job IDs

Save results with `RuntimeEncoder`, and keep every job ID on disk. IBM stores the results of every job, and re-reading a finished job with `service.job(job_id)` costs no QPU time, so a kernel restart never forces a resubmission: retrieve rather than resubmit. A small job ledger keyed by the run's settings (backend, shots, options) lets a rerun with the same settings find the earlier job instead of submitting a new one. If you lost an ID, `service.jobs(created_after=...)` lists recent jobs, and job tags make them easier to find.

- **When to use:** always, and especially before long runs.
- **Tags:** family: general; step: develop; runs: local, account; status: stable; updated: 2026-09
- **Recent changes:** since 0.49, `RuntimeDecoder` warns, and suggests `pip install -U qiskit`, when a retrieved job's QPY is newer than your Qiskit; since 0.50, jobs from retired backends can be retrieved too.
- **Key APIs:** `job.job_id()`, `job.status()`, `job.cancel()`, `service.job(job_id)`, `service.jobs(created_after=..., limit=...)`, `json.dump(result, file, cls=RuntimeEncoder)`, `json.load(file, cls=RuntimeDecoder)`, `options.environment.job_tags`
- **Watch out:** result metadata from the client-side primitives only holds plain JSON types, so encode anything else as a string before you attach it to a circuit.
- **Related:** [Saving circuits with QPY](qiskit-sdk.md#saving-circuits-with-qpy)

**Docs:**
- [Retrieve and save job results](https://quantum.cloud.ibm.com/docs/en/guides/save-jobs)
- [Monitor or cancel a job](https://quantum.cloud.ibm.com/docs/en/guides/monitor-job)

## Twirling options in the primitives

The Sampler and Estimator also twirl for you through their options. Measurement twirling (`options.twirling.enable_measure`) makes readout errors symmetric and is the basis of TREX. Gate twirling (`options.twirling.enable_gates`) turns coherent two-qubit errors into stochastic ones. Left at `None`, both are off in the Sampler, while the Estimator sets them from the resilience level: measurement twirling at levels 1 and 2, gate twirling only at level 2.

- **When to use:** you use the Sampler or Estimator and want twirling without editing the circuit.
- **Tags:** family: general; step: execute; noise: suppression; runs: qpu, local; status: stable; updated: 2026-09
- **Recent changes:** the client-side primitives' `options.twirling.group` (0.50) accepts `"pauli"` (the default), `"balanced_pauli"`, `"local_c1"` and `"local_pauli"`. With `"local_pauli"` they can gate-twirl circuits that contain fractional `RZZ` gates, which the default group refuses; see the fractional-gates caveat in [Fractional gates](#fractional-gates).
- **Key APIs:** `options.twirling.enable_gates`, `options.twirling.enable_measure`, `options.twirling.group`, `options.twirling.num_randomizations`, `options.twirling.shots_per_randomization`, `options.twirling.strategy` (`"active"`, `"active-circuit"`, `"active-accum"` by default, or `"all"`)
- **Watch out:**
  - Leave `num_randomizations` and `shots_per_randomization` at `"auto"` unless you have checked them.
  - With the legacy `SamplerV2`, a job fails when their product is smaller than the shots of a PUB.
  - The client-side Sampler and Estimator instead run exactly `num_randomizations × shots_per_randomization` shots when all three values are set, so you silently get fewer or more shots than you asked for; omit the two twirling counts when `shots` should win.
  - With `"auto"`, the total shot count may be rounded up (1,000 requested shots return 1,024).
  - Measurement twirling only applies to terminal measurements outside conditional blocks. Gate twirling does not work with non-Clifford entanglers, with stretches or, in the default group, with fractional gates.
- **Related:** [Dynamical decoupling options](#dynamical-decoupling-options), the other suppression option of the primitives.

**Docs:**
- [Noise management with the Sampler](https://quantum.cloud.ibm.com/docs/en/guides/sampler-noise-management)
- [TwirlingOptions (client-side)](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-models-twirling-options)
- [Estimator options: feature compatibility](https://quantum.cloud.ibm.com/docs/en/guides/estimator-options#options-compatibility-table)

## NoiseLearnerV3

`NoiseLearnerV3` learns sparse Pauli–Lindblad noise models for the boxed layers you pass in; you choose the layering, for example with Samplomatic's boxing pass manager, and the learner makes no layer-finding decisions of its own. Its inputs are twirl-annotated `BoxOp` instructions of ISA operations on physical qubits. It runs two protocols: Lindblad learning for boxes whose content casts to a `Clifford` with a single layer of (up to) two-qubit gates, and TREX-style readout learning for boxes that contain exactly one measurement per qubit. The result is a `NoiseLearnerV3Results` sequence with one `NoiseLearnerV3Result` per input instruction, each convertible to a `PauliLindbladMap`. The noise models feed PEC, PEA (noise amplification for ZNE), PNA and shaded light cones, both in `Executor` workflows and with the client-side Estimator.

- **When to use:** you need learned noise models for PEC, PEA, PNA or shaded light cones.
- **Tags:** family: trotter, floquet, echoes, verifiable; step: execute; noise: characterization; runs: qpu; status: beta; updated: 2026-09
- **Recent changes:** part of `qiskit-ibm-runtime` since 0.47 (May 2026; beta, like the rest of the directed execution model). It gained `bit_flip_checks` options (0.49) and `NoiseLearnerV3Results.to_pauli_lindblad_maps()` (0.50). The client-side Estimator of 0.50 consumes those maps: `layers = estimator.find_unique_layers(pubs)`, then `estimator.options.resilience.layer_noise_model = zip(layers, maps)`. The older `NoiseLearner` works only with the legacy server-side Estimator, which 0.50 deprecates.
- **Key APIs:** `NoiseLearnerV3(mode=backend, options=...)`, `.run(box_instructions, dry_run=False)`, `NoiseLearnerV3Results.to_pauli_lindblad_maps()`, `NoiseLearnerV3Results.to_dict(instructions, require_refs=True)` (keyed by each box's `InjectNoise.ref`; `require_refs=False` skips boxes without one), `NoiseLearnerV3Result.to_pauli_lindblad_map()`, `NoiseLearnerV3Options` (`num_randomizations`, default 32; `shots_per_randomization`, default 128; `layer_pair_depths`, default `[0, 1, 2, 4, 16, 32]`, counted in layer pairs and ignored by TREX; `bit_flip_checks` (0.49; `post_selection` is deprecated since 0.49)), `samplomatic.transpiler.generate_boxing_pass_manager`, `samplomatic.utils.find_unique_box_instructions`
- **Watch out:**
  - It needs a real backend; there is no local mode. `run(..., dry_run=True)` validates the job and estimates usage on the server side, but its results are randomized mock data.
  - It learns layers that can be Pauli-twirled, so layers of non-Clifford fractional `RZZ` cannot be learned. `run` raises `IBMInputValueError` for boxes without a twirl annotation, with non-ISA gates, or on qubits outside the backend's physical register.
  - Learn close in time to the run you mitigate: device drift between learning and PEC makes PEC over-correct.
  - Learn with the same boxing as the circuit you mitigate, and call `find_unique_box_instructions` first so each unique layer is learned once. `to_dict` raises if the number of instructions differs from the number of results or if two instructions share a `ref`.
  - Make the deepest `layer_pair_depths` value match your circuit: the shaded-light-cone tutorial sets it to half the circuit's two-qubit-layer depth, since depths count layer pairs.
  - Budget the cost: with six depths, `num_randomizations=32` means 32 × 9 × 6 learning circuits (9 measurement-basis settings per layer) at `shots_per_randomization` shots each.
  - The client-side Estimator accepts only Pauli–Lindblad models of two-qubit layers, not the TREX results.
- **Related:** [qiskit-mitigation: TREX, ZNE, PEA and PEC](noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) consumes the learned maps; [Pauli twirling with Samplomatic](noise-management.md#pauli-twirling-with-samplomatic) builds the boxed layers it learns; [Custom noise learning](noise-management.md#custom-noise-learning-qiskit-noise-learning) covers learning experiments of your own.

**Docs:**
- [Migrate to NoiseLearnerV3](https://quantum.cloud.ibm.com/docs/en/guides/migrate-to-noise-learner-v3)
- [Noise learning helper](https://quantum.cloud.ibm.com/docs/en/guides/noise-learning)
- [NoiseLearnerV3 API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/noise-learner-v3-noise-learner-v3)
- [NoiseLearnerV3Options API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-models-noise-learner-v3-options)
- [NoiseLearnerV3Results API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/results-noise-learner-v3-results)

## Fake backends and local testing mode

`qiskit_ibm_runtime.fake_provider` ships offline snapshots of IBM QPUs: coupling map, basis gates and calibration-derived noise. Pass a fake backend or an `AerSimulator` as the `mode` of a primitive or of `Executor`, and the job runs on your machine without spending QPU time. Install `qiskit-aer` as well: the fake backends simulate with Aer when it is available, to take advantage of its performance.

- **When to use:** before any hardware run, to test the whole pipeline, including the client-side primitives' error-suppression and mitigation options, offline and without QPU time.
- **Tags:** family: general; step: develop; runs: local; status: stable; updated: 2026-09
- **Recent changes:**
  - `FakeNighthawk` (0.44, December 2025) has the Nighthawk connectivity and basis gates, but its properties are not meant to represent typical Nighthawk error rates.
  - `FakeKingston`, `FakeBoston`, `FakePittsburgh`, `FakeAachen`, `FakeMiami`, `FakeBerlin`, `FakeBrussels` and `FakeStrasbourg` were added in 0.47 (May 2026). Since 0.49 they also appear in `FakeProviderForBackendV2().backends()`, and `refresh()` accepts `persist=False` for read-only installs.
  - Version 0.50 (September 2026) adds local mode for `Executor` (simulator behavior is set through the `simulator` options attribute), and the client-side `Sampler` and `Estimator` apply their error-suppression and mitigation options locally. The legacy server-side primitives ignored every option except `shots` on a local simulator.
- **Key APIs:** `qiskit_ibm_runtime.fake_provider.FakeBoston()` (and the other fake backends), `FakeBackendV2.refresh(service, use_fractional_gates=True, persist=True)` (the class lives in `qiskit_ibm_runtime.fake_provider.fake_backend`), `qiskit_ibm_runtime.executor_sampler.Sampler(mode=fake_backend, options={"simulator": {"seed_simulator": 42}})` for reproducible local runs, `Executor(mode=fake_backend)`, `qiskit.providers.fake_provider.GenericBackendV2`, `qiskit_ibm_runtime.transpiler.passes.ConvertISAToClifford`
- **Watch out:**
  - Fake backends simulate with a noisy statevector, so keep local tests small or Cliffordize the circuit (`ConvertISAToClifford` handles fractional gates since 0.43). Noisy simulation cost grows exponentially with the qubit count, most laptops stop at about 32 qubits, and the locally applied mitigation adds memory and time on top; use a small fake backend, a noiseless simulator or Clifford simulation.
  - Snapshots leave out fractional gates by default (`FakeBoston` has no `rzz` or `rx`). `fake.refresh(service, use_fractional_gates=True)` adds them from the real backend; it needs an account with access to that QPU, but no QPU time, and by default it overwrites the snapshot files bundled with the installed package. Some special instructions, such as `xslow`, are missing either way, and Aer cannot simulate the specially calibrated ones even where the target lists them ([Specially calibrated instructions and calibration sets](#specially-calibrated-instructions-and-calibration-sets)).
  - To test a fractional-gate circuit without an account, build a small backend whose target has the gates: `GenericBackendV2(num_qubits=16, basis_gates=["rz", "sx", "x", "cz", "rzz", "rx"], coupling_map=<list of edges>, seed=1)`. Its default basis is `cx`, not Heron's `cz`, and its default coupling map is all-to-all, so pass both arguments when the test should resemble a Heron circuit. The client-side primitives accept it as `mode`, and it exercises the twirling-group and folding checks ([Local test patterns](../guides/testing-and-tips.md#local-test-patterns)). Its target has no angle bounds, so a local run accepts `rzz` angles outside (0, π/2] that the QPU rejects; check the bound angles before you submit.
  - The client-side Estimator measures every qubit, so run it locally on a small backend (see [Sampler and Estimator](#sampler-and-estimator-client-side-primitives)).
  - `FakeNighthawk`, `FakeMiami` and `FakeBerlin` model Nighthawk r1, not the r2 `ibm_phoenix`.
  - Snapshot noise models can be years old and never match the current device; `Session` syntax is accepted but ignored in local mode.
- **Related:** [Qiskit Aer](simulators.md#qiskit-aer) for custom noise models and larger or Clifford simulations.

**Docs:**
- [Local testing mode](https://quantum.cloud.ibm.com/docs/en/guides/local-testing-mode)
- [Fake provider API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider)
- [IBM Quantum Compute client release notes](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/release-notes)
- [Debug jobs with NEAT (noise impact on an Estimator workload)](https://quantum.cloud.ibm.com/docs/en/guides/debug-qiskit-runtime-jobs)

## NEAT: the noisy Estimator analyzer

`Neat` (the Noisy Estimator Analyzer Tool, in `qiskit_ibm_runtime.debug_tools`) predicts how an Estimator workload will fare before you spend QPU time. `to_clifford` rounds every `rz`, `rx` and `rzz` angle of an ISA circuit to the nearest multiple of π/2 (the `ConvertISAToClifford` pass, see [Calibrated instructions](#specially-calibrated-instructions-and-calibration-sets)) and leaves the two-qubit gates, qubit count and depth unchanged, so Aer's stabilizer method simulates it at any size. `ideal_sim` and `noisy_sim` return expectation values without and with the backend's Aer noise model. A `NeatPubResult` supports arithmetic with `PubResult`s, so `abs(noisy[0] - measured[0])` is a valid figure of merit.

- **When to use:** before a large Estimator run, to see how much noise hurts circuits of your depth, to compare noise models, or to pick mitigation settings from the Clifford version run on the QPU.
- **Tags:** family: general; step: develop; noise: characterization; runs: local; status: stable; updated: 2026-09
- **Recent changes:** added in runtime 0.31 (October 2024); the pass handles fractional `rzz` and `rx` since 0.43. The 0.50 guide pairs it with the client-side Estimator ([Sampler and Estimator](#sampler-and-estimator-client-side-primitives)); its results also combine with the legacy `EstimatorV2`'s (both checked locally on 0.50). It only runs Aer simulations itself, never a QPU job.
- **Key APIs:** `Neat(backend, noise_model=None)`, `.to_clifford(pubs)`, `.ideal_sim(pubs, cliffordize=False, seed_simulator=None, precision=0)`, `.noisy_sim(...)`, `.noise_model` (settable), `NeatResult`, `NeatPubResult.vals`
- **Watch out:**
  - Needs `qiskit-aer`. Non-Clifford PUBs raise `ValueError` unless you Cliffordize them (`to_clifford`, or `cliffordize=True`). The pass accepts only ISA gates, barriers and measurements; delays, resets, boxes and `if_else` raise `Operation not supported` (checked on 0.50), so Cliffordize before adding dynamical decoupling or boxing.
  - Aer's noise model from the backend covers only some error sources and is likely milder than the real QPU, so treat the noisy values as optimistic. IBM's guide reads the mean absolute difference between ideal and noisy results as a rough guide to mitigation: above 90% mitigation will likely not help, below 90% PEA likely helps, below 80% ZNE by gate folding as well.
  - Map observables onto the ISA layout with `apply_layout` ([Sampler and Estimator](#sampler-and-estimator-client-side-primitives)); see [NoiseLearnerV3](#noiselearnerv3) and [qiskit-mitigation](noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) for noise learning and mitigation.

**Docs:**
- [Debug jobs with NEAT](https://quantum.cloud.ibm.com/docs/en/guides/debug-qiskit-runtime-jobs)
- [Neat API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/debug-tools-neat)
- [debug_tools API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/debug-tools)
- [ConvertISAToClifford](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/transpiler-passes-convert-isa-to-clifford)
- [Debugging tools introduction](https://quantum.cloud.ibm.com/docs/en/guides/debugging-tools)
