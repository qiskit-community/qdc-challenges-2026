---
title: Simulators and classical references
short_title: Simulators
tool_type: simulators
owner: Qiskit Aer team
last_verified: 2026-10-20
---

# Simulators and classical references

Simulating circuits on your machine with Qiskit Aer, and the exact and tensor-network methods that give reference values. Fake backends and local testing mode belong to the [IBM Quantum Compute client](ibm-quantum-compute-client.md#fake-backends-and-local-testing-mode).

Package versions this document was checked against:

| Package | Version checked | Notes |
|---|---|---|
| `qiskit-aer` | 0.17.2 | 0.18.0 (20 October 2026) is now the newest release and requires Python 3.11 or later |
| `quimb` | 1.15.0 | Requires Python 3.11 or later |

## Qiskit Aer

Qiskit's high-performance local simulators. `AerSimulator` offers the `statevector`, `density_matrix`, `stabilizer`, `extended_stabilizer`, `matrix_product_state`, `unitary` and `superop` methods (`tensor_network` needs a GPU), and `automatic` picks one from the circuit and the noise model. `NoiseModel` builds device-like noise, either by hand or with `NoiseModel.from_backend` from a real or fake backend's calibration data. Aer also ships `SamplerV2` and `EstimatorV2` primitives and save instructions such as `save_probabilities_dict`.

- **When to use:** you need fast exact, noisy, stabilizer or matrix-product-state simulation on your machine, or a custom noise model that the fake backends do not offer.
- **Tags:** family: general, thermal, detection, verifiable; step: develop; runs: local; status: stable; updated: 2026-10
- **Current version:** 0.18.0 (20 October 2026), which requires Python 3.11 or later. This cheat sheet's import lines and snippets were run with 0.17.2 (September 2025).
- **Key APIs:** `AerSimulator(method="stabilizer")` (and the other methods), `AerSimulator.from_backend(backend)`, `qiskit_aer.noise.NoiseModel`, `NoiseModel.from_backend`, `depolarizing_error`, `pauli_error`, `thermal_relaxation_error`, `ReadoutError`, `qiskit_aer.primitives.SamplerV2` and `EstimatorV2` (noise goes in through `options=dict(backend_options=dict(noise_model=...))`), `qiskit_aer.library.save_probabilities_dict`
- **Watch out:**
  - Noisy statevector sampling has shown biased, under-dispersed samples in Aer 0.17.2; check against the density-matrix method on small cases. The statevector method draws one randomly sampled noisy circuit per shot, while the density-matrix method evolves the mixed state and samples outcomes from it.
  - The Aer primitives do not take the IBM primitive options (no resilience level, for example), and a noise model built from a backend covers only some of the real device's error sources, so it is usually milder than the hardware.
  - The `stabilizer` method simulates noisy Clifford circuits only when every error in the noise model is a Clifford error; `extended_stabilizer` grows with the number of non-Clifford gates.
- **Related:** [Fake backends and local testing mode](ibm-quantum-compute-client.md#fake-backends-and-local-testing-mode) to run the IBM primitives on an `AerSimulator`; [Exact and tensor-network references](#exact-and-tensor-network-references) for exact and Clifford references.

**Docs:**
- [Exact and noisy simulation with Qiskit Aer](https://quantum.cloud.ibm.com/docs/en/guides/simulate-with-qiskit-aer)
- [Efficient simulation of stabilizer circuits with Qiskit Aer primitives](https://quantum.cloud.ibm.com/docs/en/guides/simulate-stabilizer-circuits)
- [Build noise models](https://quantum.cloud.ibm.com/docs/en/guides/build-noise-models)
- [Aer documentation](https://qiskit.github.io/qiskit-aer/)
- [Matrix product state method](https://qiskit.github.io/qiskit-aer/tutorials/7_matrix_product_state_method.html)

## Exact and tensor-network references

For small instances and for checks, Qiskit's reference primitives (`StatevectorSampler`, `StatevectorEstimator`) and the `quantum_info` classes (`Statevector`, `Clifford`, `StabilizerState`) give exact answers. Clifford circuits stay cheap at any size, so a Clifford version of your circuit (rotation angles restricted to multiples of π/2) is a good way to test acquisition code and to compare hardware output against a simulable circuit of the same structure. For larger, weakly entangled circuits, tensor-network simulators such as quimb's `CircuitMPS` give approximate references.

- **When to use:** you need exact answers for small instances or for Clifford versions of your circuit, or approximate references for larger, weakly entangled circuits.
- **Tags:** family: trotter, floquet, echoes, thermal, detection, verifiable; step: develop; runs: local; status: stable; updated: 2026-08
- **Recent changes:** quimb 1.15 (August 2026) restructured its circuit simulators (`CircuitMPS` now derives from `CircuitBase` and lost some exact-only methods) and added OpenQASM 3 loaders (`from_openqasm3_file`, `from_openqasm3_str`, `from_openqasm3_url`) that read custom gate definitions, so tracker files load directly.
- **Key APIs:** `qiskit.primitives.StatevectorSampler`, `StatevectorEstimator` (exact by default; pass `precision` in `run()` to add shot noise), `qiskit.quantum_info.Statevector`, `Clifford`, `StabilizerState`, `DensityMatrix`, `quimb.tensor.CircuitMPS` (with `CircuitMPS.from_openqasm3_file(path)`)
- **Watch out:**
  - The reference primitives accept abstract (non-ISA) circuits, but a circuit passed to an Estimator must not contain measurements.
  - Resets are not exact here. `Statevector`, `StatevectorEstimator` and `StatevectorSampler` follow one random outcome per `reset`, so a circuit that resets an entangled qubit gives a different value on each call; `StatevectorSampler` draws every shot of a call from that one outcome, so its counts look clean but are wrong. `StatevectorSampler` also raises on mid-circuit measurements and control flow. Use `DensityMatrix(circuit)` for circuits with resets but no measurements, or Aer's `density_matrix` method, which also runs mid-circuit measurements and feed-forward.
  - For a reset-based (dissipative) protocol, evolve one round at a time with `DensityMatrix.evolve(round_circuit)` until the observable stops changing; that is the noiseless steady state the hardware should approach. For a Gibbs target, compute tr(e^(−βH)O)/tr(e^(−βH)) directly with `scipy.linalg.expm(-beta * H.to_matrix())`, which is practical up to about 12 qubits.
  - An MPS value is a reference only once it has converged: rerun with a larger bond dimension until the value stops changing.
  - A noisy Clifford reference needs Aer's `stabilizer` method with a Clifford noise model (see [Qiskit Aer](#qiskit-aer)); the reference primitives model no noise.
- **Related:** [Qiskit Aer](#qiskit-aer) for its `matrix_product_state` and `density_matrix` methods; [Propagated noise absorption and Pauli propagation](qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) to estimate Pauli expectation values classically.

**Docs:**
- [Exact simulation with Qiskit SDK primitives](https://quantum.cloud.ibm.com/docs/en/guides/simulate-with-qiskit-sdk-primitives)
- [Efficient simulation of stabilizer circuits with Qiskit Aer primitives](https://quantum.cloud.ibm.com/docs/en/guides/simulate-stabilizer-circuits)
- [quimb documentation](https://quimb.readthedocs.io/)
