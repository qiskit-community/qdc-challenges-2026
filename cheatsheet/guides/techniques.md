---
title: Techniques
short_title: Techniques
tool_type: guide
owner: QDC organizers
last_verified: 2026-10-20
---

# Techniques

Methods that combine several tools and have no package of their own.

## Choosing error detection and mitigation

Error detection adds redundancy that flags runs hit by an error. You then discard (post-select away) the flagged shots. You pay with extra qubits, extra gates and lost shots, and get results that are closer to the noiseless ones. The IBM Quantum Learning course *Error correction I* covers both approaches: end-of-circuit detection with a small stabilizer code, the [Iceberg code](#the-iceberg-code), and mid-circuit spacetime checks with [Paulice](../reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice). The [non-Markovian error checks](../reference/noise-management.md#post-selection-on-non-markovian-error-checks) are different in kind: they test the qubits, not the computation, and filter out shots hit by non-Markovian noise.

Choose a mitigation method by what your result is and what you can afford. Readout mitigation is cheap. Gate-error mitigation costs shots or QPU time, and each method trades bias against variance differently. Managed alternatives such as QESEM and TEM are [circuit functions](../reference/ibm-quantum-functions-catalog.md#circuit-functions).

| Method | Corrects | Bias | Extra cost | Needs learned noise | Fractional `RZZ` | Test locally |
|---|---|---|---|---|---|---|
| TREX ([qiskit-mitigation](../reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec)) | Readout errors in expectation values | None under its readout model | Readout-calibration circuits | No | Yes | Yes, with the client-side Estimator on a small backend ([Local test patterns](testing-and-tips.md#local-test-patterns)) |
| [M3](../reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree) | Readout errors in counts | Small; it corrects within the observed bitstrings | One calibration job | No | Yes | Yes |
| ZNE by gate folding ([qiskit-mitigation](../reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec)) | Gate errors | Depends on the fit model | Runs at several noise factors, each deeper than the original; extrapolation amplifies shot noise | No | Only with hand-built folding: the built-in option runs but folds nothing (`GateFolding` folds only CX, CY, CZ, ECR and SWAP); map the negative angles that folding creates back into range | Yes |
| PEA ([qiskit-mitigation](../reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec)) | Gate errors | Depends on the fit model | Noise learning, then runs at several noise levels | Yes | No | Only with a noise model you supply |
| PEC ([qiskit-mitigation](../reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec)) | Gate errors | None if the learned model matches the device | Shots grow with the sampling overhead γ², which grows exponentially with circuit size | Yes | No | Only with a noise model you supply |
| PEC with [shaded light cones](../reference/qiskit-addons.md#shaded-light-cones-qiskit-addon-slc) | Gate errors that reach your observable | Bounded by your `bias_tolerance` | A much smaller γ² than full PEC, plus classical bound computations | Yes | No | The bounds, yes |
| [PNA](../reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) | Gate errors, moved into the observable | From truncating the propagated observable | Classical propagation, and more Pauli terms to measure | Yes | No | The classical part, yes |

"Needs learned noise" means [`NoiseLearnerV3`](../reference/ibm-quantum-compute-client.md#noiselearnerv3), which runs only on hardware.

- **When to use:** you plan how to handle noise in a hardware run: whether to detect and discard errors, which mitigation method fits your result, and what each one costs.
- **Tags:** family: general; step: post-process; noise: detection, mitigation-expval, mitigation-sampling; runs: local; status: stable

**Docs:**
- [Overview of noise management techniques](https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-overview)

## The Iceberg code

The Iceberg code (Self, Benedetti and Amaro, 2024) is a family of [[k+2, k, 2]] error-detecting codes: k logical qubits in k+2 data qubits with two stabilizers, an all-Z and an all-X parity check, plus ancillas for fault-tolerant syndrome extraction. Every weight-1 Pauli error on a data qubit anti-commutes with at least one stabilizer, so the code detects any single error. With distance 2 the two-bit syndrome tells you the error type but not the qubit, so the code is used for detection and post-selection, not correction. The smallest member, the [[4,2,2]] code, encodes two logical qubits in four data qubits. Its logical |00⟩ is the 4-qubit GHZ state (|0000⟩ + |1111⟩)/√2, and each logical basis state pairs two bit-complementary strings.

To use it, prepare the encoded state, run the circuit on the logical qubits and extract the syndromes through the ancillas. CNOTs from every data qubit into one ancilla read the Z stabilizer, and the same pattern conjugated by Hadamards on the data qubits reads the X stabilizer onto a second ancilla. Because only the ancillas are measured, the extraction can in principle be applied without collapsing the encoded state. Parity post-selection keeps the shots whose syndromes are all zero. Unlike [spacetime Pauli checks](../reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice), which need a Clifford or near-Clifford payload, the code encodes the qubits themselves, so it also protects non-Clifford circuits.

- **When to use:** you want error detection on many logical qubits with only two extra data qubits per code block, and can afford to discard flagged shots.
- **Tags:** family: detection; step: map, post-process; noise: detection; runs: local, qpu; status: stable
- **Key APIs:** plain Qiskit: `QuantumCircuit` with separate `QuantumRegister`s for data and ancilla qubits, `h`, `cx` and `measure` into `ClassicalRegister`s of their own for the syndrome bits; `qiskit.primitives.BitArray` (`get_bitstrings`, `from_samples`, `slice_bits`, `postselect(indices, selection)`, `expectation_values`); `SamplerPubResult.join_data(names=[...])` to join the syndrome and data registers in a chosen order
- **Watch out:**
  - Distance 2 means some weight-2 errors pass undetected, including an even number of bit flips, so post-selected results are closer to noiseless, not noiseless.
  - Name the syndrome registers so that post-selection code can find them, and keep them apart from the data registers.
  - `BitArray.postselect` discards a shot unless every listed bit has the requested value, and it flattens any shape axes (for example a parameter sweep) into one list of shots before it filters.

**Docs:**
- [Paper: Protecting expressive circuits with a quantum error detection code (Self, Benedetti, Amaro; Nature Physics 20, 219–224, 2024)](https://arxiv.org/abs/2211.06703)
- [IBM Quantum Learning: error detection and post-selection (Error correction I)](https://quantum.cloud.ibm.com/learning/en/courses/tools-for-quantum-advantage/error-correction-i)
- [BitArray API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.BitArray)
- [SamplerPubResult API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.SamplerPubResult)

## CVaR of sampled results

For optimization problems, the conditional value at risk CVaR_α of a set of shots is the average objective over the best α fraction of them. α = 1 gives the ordinary expected value, α → 0 the best single shot, and values in between trade a focus on the good shots against a smoother optimization landscape. Formally, CVaR_α(X) = E[X | X ≤ F_X^{-1}(α)] is the mean of the lower α-tail of X, so for a cost you minimize the "best" shots are the lowest-cost ones. CVaR also works as an error mitigation technique for the objective value: the QAOA tutorial below derives α from the two-qubit gate count and the backend's layer fidelity and raises the shot count by the same factor.

- **When to use:** you score samples from an optimization circuit by their best fraction, as the objective for a variational optimizer or as a mitigated estimate of the objective.
- **Tags:** family: qaoa, lcu; step: post-process; noise: mitigation-sampling; runs: local; status: stable
- **Key APIs:** numpy over the sampled distribution: take `BitArray.get_int_counts()` or `get_counts()` from the Sampler result, evaluate the objective once per distinct bitstring, sort by objective and accumulate probability mass until α is reached (the tutorial's `_get_cvar_aggregation(alpha)` does this in pure Python; `numpy.argsort` and `numpy.cumsum` give a vectorized version). `qiskit-addon-opt-mapper` 0.1.0 has no CVaR helper; it formulates the problem (`OptimizationProblem`, the `applications` classes and the `OptimizationProblemToQubo` converter), and the per-bitstring objective evaluation is yours.
- **Watch out:**
  - The α cutoff usually splits one bitstring's probability mass; weight the last bitstring by `min(probability, alpha - accumulated)` and divide the sum by α.
  - Evaluate the objective with the same bit order as the observable; the tutorial packs the Z mask with `bitorder="little"`.
  - Readout-corrected distributions from [M3](../reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree) are quasi-probabilities that can have small negative entries; call `nearest_probability_distribution()` on the `QuasiDistribution` before you accumulate the α tail.
  - This CVaR averages the best shots of one circuit. A CVaR limit in a risk-constrained problem is a property of each candidate solution, computed from that solution's own loss distribution; evaluate it per bitstring, not over the shots.

**Docs:**
- [Tutorial: advanced techniques for QAOA (CVaR cost function)](https://quantum.cloud.ibm.com/docs/en/tutorials/advanced-techniques-for-qaoa)
- [Barkoutsos et al., Improving variational quantum optimization using CVaR](https://arxiv.org/abs/1907.04769)
- [Barron et al., Provable bounds for noise-free expectation values computed from noisy samples](https://arxiv.org/abs/2312.00733)

## Sampling a linear combination of unitaries

When the costliest block of a circuit is a weighted sum of cheaper unitaries, U = Σ_j c_j V_j, a sampling task can do without the ancilla and controlled gates of a coherent linear combination of unitaries (LCU). Run the circuit with V_j in place of U with probability |c_j| / Σ_k |c_k|, and combine all the shots. By the Cauchy-Schwarz inequality, every bitstring then keeps at least 1/Γ of the probability it has in the circuit with U, where Γ = (Σ_j |c_j|)² is the sampling overhead: the ensemble needs none of the block's two-qubit gates but up to Γ times the shots. The bound holds bitstring by bitstring, so it protects the best samples, not the mean. Because the pooled distribution contains the coherent one scaled by 1/Γ, the CVaR at level 1/Γ of the pooled samples ([CVaR of sampled results](#cvar-of-sampled-results)) is at least as good as the expected value of the circuit with U, while the pooled mean can be much worse.

- **When to use:** you sample from a circuit whose costliest block is a weighted sum of cheap unitaries, and extra shots cost you less than that block's two-qubit gates.
- **Tags:** family: lcu; step: map; runs: local, qpu; status: stable
- **Related:** [Executor and QuantumProgram](../reference/ibm-quantum-compute-client.md#executor-and-quantumprogram) and [Execution modes](../reference/ibm-quantum-compute-client.md#execution-modes-job-batch-and-session) run many circuits or parameter sets with few jobs; [Circuit cutting](../reference/qiskit-addons.md#circuit-cutting-qiskit-addon-cutting) and PEC make the same trade for expectation values.

**Docs:**
- [Childs and Wiebe, Hamiltonian simulation using linear combinations of unitary operations (arXiv:1202.5822)](https://arxiv.org/abs/1202.5822)

## Fidelity of a Clifford circuit from sampled stabilizers

A Clifford circuit prepares a stabilizer state |ψ⟩, and the fidelity of the state ρ that the hardware prepares is F = ⟨ψ|ρ|ψ⟩ = 2^(−n) Σ_S tr(ρS), a sum over the 2^n elements S of its stabilizer group. Direct fidelity estimation (DFE) averages the measured expectation values of K stabilizers drawn uniformly from the group; for a stabilizer state its statistical error shrinks with K and does not grow with the number of qubits. Each stabilizer needs only a basis change before the final measurement.

- **When to use:** your circuit is Clifford and you want the fidelity of the state it prepares on hardware.
- **Tags:** family: verifiable, detection; step: post-process; runs: local, qpu; status: stable
- **Key APIs:** `qiskit.quantum_info.Clifford(circuit).to_labels(mode="S")` returns the signed stabilizer generators; the product of a uniformly random subset of them is a uniformly random stabilizer.
- **Watch out:** keep each stabilizer's sign in its observable. A value near −1 means a dropped sign, not a bad state.
- **Related:** [Executor and QuantumProgram](../reference/ibm-quantum-compute-client.md#executor-and-quantumprogram) measures many basis settings in one program, [Expectation values from shot data](../reference/qiskit-addons.md#expectation-values-from-shot-data) turns the bits into one value per stabilizer, [qiskit-mitigation](../reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) corrects readout with TREX, and [Spacetime Pauli checks](../reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice) post-select the shots.

**Docs:**
- [Flammia and Liu, Direct fidelity estimation from few Pauli measurements (arXiv:1104.4695)](https://arxiv.org/abs/1104.4695)
