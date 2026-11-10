---
title: Ecosystem and third-party packages
short_title: Ecosystem
tool_type: ecosystem
owner: QDC organizers
last_verified: 2026-10-20
---

# Ecosystem and third-party packages

Packages outside the Qiskit SDK, the Compute client and the addons: IBM research and Qiskit community packages for optimization, layout and benchmarking; the Quantum Advantage Tracker's reference circuits; and third-party scientific packages for chemistry and classical optimization. Several of them install only from GitHub.

Package versions this document was checked against:

| Package | Version checked | Notes |
|---|---|---|
| `mapomatic` | 0.14.0 |  |
| `pyscf` / `ffsim` | 2.14.0 / 0.0.84 | PySCF has no Windows support. ffsim 0.0.84 requires Python 3.11 or later and `pyscf>=2.12`; `generate_lucj_pass_manager` arrived in ffsim 0.0.75 ([Chemistry: PySCF and ffsim](#chemistry-pyscf-and-ffsim)) |
| `docplex` / `cplex` / `moocore` | 2.32.264 / 22.2.0.1 / 0.3.2 | `cplex` 22.2.0.1 has wheels for Python 3.10–3.14 on Linux, Windows and Apple-silicon macOS, but none for Intel Macs |

## Quantum optimization best practices (qopt-best-practices)

A community repository of reference code for running QAOA at scale. It includes swap-strategy routing for commuting cost layers, an annotated QAOA ansatz with a preset QAOA pass manager, qubit selection (`BackendEvaluator`), and SAT and simulated-annealing qubit mappers. It is not on PyPI, so install it from GitHub; some notebooks ship a copy of the modules they need.

- **When to use:** you run QAOA on many qubits and want ready-made swap-strategy routing, a QAOA pass manager or qubit selection.
- **Tags:** family: qaoa, multi-objective; step: optimize; runs: local; status: stable; updated: 2026-06
- **Recent changes:** compatibility with Qiskit 2.3 (February 2026) and 2.4 (May 2026), SAT mapping for parametric Hamiltonians (May 2026) and a simulated-annealing mapper (30 June 2026).
- **Key APIs:** `transpilation.cost_layer.get_cost_layer`, `transpilation.prepare_cost_layer.PrepareCostLayer`, `transpilation.swap_cancellation_pass.SwapToFinalMapping`, `transpilation.generate_preset_qaoa_pass_manager`, `qubit_selection.BackendEvaluator`
- **Related:** [Swap strategies for commuting gates](qiskit-sdk.md#swap-strategies-for-commuting-gates), [Qubit selection and layout](qiskit-sdk.md#qubit-selection-and-layout)

**Docs:**
- [GitHub repository and quick start](https://github.com/qiskit-community/qopt-best-practices#quick-start)
- [Tutorial: advanced techniques for QAOA](https://quantum.cloud.ibm.com/docs/en/tutorials/advanced-techniques-for-qaoa)

## QAOA training pipeline

Community code for training QAOA angles classically before you spend QPU time. It has trainers (depth-one scans, SciPy, LR-QAOA, transfer, recursion) and evaluators (efficient depth-one, light cone, Pauli propagation, MPS). Install it from GitHub; the repository name uses underscores.

- **When to use:** you want good QAOA angles from classical simulation before you spend QPU time.
- **Tags:** family: qaoa, multi-objective; step: map; runs: local, qpu; status: stable; updated: 2026-10
- **Recent changes:** an LR-QAOA trainer class, MPS-sample and QPU-sample evaluators (18 August 2026), and a `pyproject.toml` install with optional extras; since 1 October 2026 the tensor-network extra `tns` is optional and Qiskit Aer is a core dependency.
- **Key APIs:** `training.DepthOneScanTrainer`, `training.LRQAOATrainer`, `evaluation.EfficientDepthOneEvaluator`, `utils.graph_utils.graph_to_operator`

**Docs:**
- [GitHub repository](https://github.com/qiskit-community/qaoa_training_pipeline)

## Classical solvers and multi-objective tools

Three classical tools provide references, metrics and schedules:
* **docplex and CPLEX** model and solve classical reference problems, from single-objective QUBOs to the scalarized sub-problems of a multi-objective model. The PyPI `cplex` wheel is the Community Edition, limited to 1,000 variables and 1,000 constraints. It is built per Python version, with no wheels for Intel Macs.
* **moocore** computes hypervolumes, Pareto ranks and nondominated sets, the standard metrics for comparing a set of candidate solutions with a reference front.
* **OR-Tools CP-SAT** solves scheduling and assignment problems with hard constraints, as a classical baseline or as a helper for ordering and layout sub-problems.

- **When to use:** you need classical reference solutions, Pareto-front metrics such as the hypervolume, or a constraint solver for scheduling.
- **Tags:** family: multi-objective, qaoa; step: post-process; runs: local; status: stable
- **Key APIs:** `docplex.mp.model.Model`, `moocore.hypervolume`, `moocore.is_nondominated`, `ortools.sat.python.cp_model.CpModel`, `cp_model.CpSolver`
- **Related:** [Optimization mapper (qiskit-addon-opt-mapper)](qiskit-addons.md#optimization-mapper-qiskit-addon-opt-mapper) imports docplex models.

**Docs:**
- [moocore Python documentation](https://multi-objective.github.io/moocore/python/)
- [moocore.hypervolume](https://multi-objective.github.io/moocore/python/reference/generated/moocore.hypervolume.html)
- [OR-Tools CP-SAT solver](https://developers.google.com/optimization/cp/cp_solver)

## Chemistry: PySCF and ffsim

The quantum-centric chemistry workflow chains three tools. PySCF builds the molecular Hamiltonian (integrals, Hartree-Fock and CCSD). ffsim prepares a local unitary cluster Jastrow (LUCJ) circuit seeded by the CCSD amplitudes, and the sampled bitstrings are post-processed with [SQD](qiskit-addons.md#sample-based-quantum-diagonalization-qiskit-addon-sqd). The SQD tutorial walks through all three stages on a single molecule. Embedding packages such as [Vayesta](https://github.com/BoothGroup/Vayesta) (embedded wave function, EWF) can split a large molecule into fragments that each go through the same pipeline.

- **When to use:** you build molecular Hamiltonians or prepare LUCJ circuits for SQD.
- **Tags:** family: sqd; step: map; runs: local; status: stable; updated: 2026-04
- **Recent changes:** `ffsim.qiskit.generate_lucj_pass_manager` (ffsim 0.0.75, April 2026) picks a hardware-aware layout for LUCJ circuits on heavy-hex or square topologies and returns the alpha-beta interaction pairs the device can host.
- **Key APIs:** `ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, n_reps=..., interaction_pairs=...)`, `ffsim.qiskit.PrepareHartreeFockJW`, `ffsim.qiskit.UCJOpSpinBalancedJW`, `ffsim.qiskit.generate_lucj_pass_manager`, `pyscf.cc.CCSD`
- **Watch out:**
  - `generate_lucj_pass_manager` drops the alpha-beta interactions the device cannot host, from the end of the requested `pairs_ab` list. Check the warning and the pairs it returns, list the pairs in descending order of priority, and feed the returned pairs back into `from_t_amplitudes(interaction_pairs=...)` so that the ansatz matches the device and the router inserts few swaps.
  - `generate_lucj_pass_manager` removes coupling-graph edges whose two-qubit error is at or above `two_qubit_error_threshold` (default 1.0, so only completely faulty edges) and qubits whose readout error is at or above `readout_error_threshold` (default 0.1).
- **Related:** [Qiskit Fermions](qiskit-addons.md#qiskit-fermions-qiskit-fermions) wraps ffsim operators as fermionic gates.

**Docs:**
- [ffsim documentation](https://qiskit-community.github.io/ffsim/)
- [The LUCJ ansatz](https://qiskit-community.github.io/ffsim/explanations/lucj.html)
- [Tutorial: sample-based quantum diagonalization of a chemistry Hamiltonian](https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-quantum-diagonalization)
- [PySCF](https://pyscf.org/)
- [Vayesta](https://github.com/BoothGroup/Vayesta)

## Reference circuits: the Quantum Advantage Tracker

The Quantum Advantage Tracker collects candidate quantum-advantage experiments, with the circuits and data needed to check them. The circuits are `.qasm` files under `data/` in its GitHub repository. Most are OpenQASM 3 (33 of the 43 files on 20 October 2026 start with `OPENQASM 3.0; include "stdgates.inc";`); the rest are OpenQASM 2 (`OPENQASM 2.0; include "qelib1.inc";`). Read the first line before you pick a loader.

- **When to use:** you need a published benchmark circuit, or want to compare a result with a candidate quantum-advantage experiment.
- **Tags:** family: verifiable, echoes, floquet, sqd, trotter; step: map; runs: local; status: stable
- **Key APIs:** `qiskit.qasm3.load(path)` for the OpenQASM 3 files (needs the `qiskit-qasm3-import` package, which `pip install "qiskit[all]"` includes); `qiskit.qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)` for the OpenQASM 2 files
- **Watch out:**
  - `qasm2.load` on an OpenQASM 3 file stops with `QASM2ParseError: can only handle OpenQASM 2.0, but given 3.0`, with or without custom instructions, and `qasm3.load` on an OpenQASM 2 file stops with `QASM3ImporterError: non-stdgates imports not currently supported`. On the OpenQASM 2 files, plain `qasm2.load` stops with `'sx'`, `'swap'` or `'u'` is not defined in this scope, because Qiskit's `qelib1.inc` has none of them; pass the legacy custom instructions.
  - Some files define gates that `stdgates.inc` lacks, such as `gate rzz(p0)` or `gate sxdg`. `qasm3.load` turns them into custom gates that keep those names, and on a backend loaded with `use_fractional_gates=True` the transpiler leaves a custom `rzz` in place instead of the native `RZZGate`. Swap in the library gate before you transpile.
  - Some files are already laid out on a 156-qubit device register (`qubit[156] q;`), so build observables on those physical indices.

**Docs:**
- [Quantum Advantage Tracker](https://quantum-advantage-tracker.github.io/)
- [Tracker circuits and data on GitHub](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io)
- [OpenQASM 2 and the Qiskit SDK](https://quantum.cloud.ibm.com/docs/en/guides/interoperate-qiskit-qasm2)
- [OpenQASM 3 and the Qiskit SDK](https://quantum.cloud.ibm.com/docs/en/guides/interoperate-qiskit-qasm3)

## Layout ranking with mapomatic

`mapomatic` finds every placement of a transpiled circuit on one or more backends and ranks the placements by calibrated error.

- **When to use:** you have a transpiled circuit and want its lowest-error placement, on one backend or across several.
- **Tags:** family: general, floquet, verifiable; step: optimize; runs: local, account; status: stable
- **Key APIs:** `mapomatic.matching_layouts`, `mapomatic.evaluate_layouts`
- **Related:** [Qubit selection and layout](qiskit-sdk.md#qubit-selection-and-layout)

**Docs:**
- [mapomatic README](https://github.com/qiskit-community/mapomatic/blob/main/README.md)

## Device benchmarking (qiskit-device-benchmarking)

A community GitHub repository, not on PyPI (`pip install git+https://github.com/qiskit-community/qiskit-device-benchmarking.git`), with example code for reproducing the device metrics IBM reports. Layer fidelity (LF) runs direct randomized benchmarking simultaneously on disjoint layers of two-qubit gates along a qubit chain; the reported "2Q error (layered)", EPLG, is derived from the layer fidelity of 100-qubit chains (the best of six pre-selected chains).

Its example notebooks cover layer fidelity on a reported or chosen chain, LF-based error maps for qubit placement, full-device two-qubit RB, mid-circuit-measurement and dynamic-circuit RB, Clifford and XEB benchmarks, and coherence characterization. `BackendCharacterization` re-measures readout, single-qubit RB, layered two-qubit RB, T1 and T2 in about 70 seconds of QPU time for the full suite on a Heron r2, and writes the results into a copy of the backend; nothing changes on the device itself. CLOPS and MCPS throughput benchmarks and a command-line mirror-circuit validation suite complete the set.

- **When to use:** the calibration data is hours old and you want fresh layer-fidelity or coherence numbers for your qubits, or you want to reproduce a reported benchmark.
- **Tags:** family: general, verifiable, echoes; step: optimize; noise: characterization; runs: qpu; status: experimental; updated: 2026-08
- **Current version:** 0.1.0 on `main` (last commit 24 August 2026, which added MCPS). It needs `qiskit-experiments>=0.12` (0.14.2, August 2026) and `qiskit-ibm-runtime>=0.28`; the README says the code base is not guaranteed to be stable and may have breaking changes.
- **Key APIs:** `qiskit_device_benchmarking.utilities.characterization_utils.BackendCharacterization(backend)` (`run_and_update`, `run_experiments`, `analyze_results`, `update_backend`), `plot_characterization_comparison`, `utilities.layer_fidelity_utils` (`run_lf_chain`, `get_lf_chain`, `best_chain`), `qiskit_experiments.library.randomized_benchmarking.LayerFidelity`, `mcps.mcps_benchmark.MCPSBenchmark`
- **Watch out:**
  - Every benchmark costs QPU time, and the refreshed numbers live only in the backend copy you get back, so re-measure close to the run that uses them.
  - Layered two-qubit errors come out above the reported isolated-gate errors because they include crosstalk, and parallel single-echo T2 values come out well below the reported ones. A gap is not drift until you have compared methods.
  - The single-chain layer-fidelity example and the CLOPS code default to the layer-fidelity chain the backend reports (`backend.properties().general_qlists`, see [Calibration data](hardware.md#calibration-data-and-layer-fidelity)).

**Docs:**
- [Repository README](https://github.com/qiskit-community/qiskit-device-benchmarking)
- [Notebooks index](https://github.com/qiskit-community/qiskit-device-benchmarking/tree/main/notebooks)
- [Tutorial: refresh backend properties with real-time benchmarking](https://quantum.cloud.ibm.com/docs/en/tutorials/refresh-backend-properties-with-real-time-benchmarking)
- [Tutorial: real-time benchmarking for qubit selection](https://quantum.cloud.ibm.com/docs/en/tutorials/real-time-benchmarking-for-qubit-selection)
- [View backend details (layer fidelity and EPLG)](https://quantum.cloud.ibm.com/docs/en/guides/qpu-information)
- [Monitor, calibrate, and benchmark](https://quantum.cloud.ibm.com/docs/en/guides/calibration-jobs)
- [LayerFidelity in Qiskit Experiments](https://qiskit-community.github.io/qiskit-experiments/stubs/qiskit_experiments.library.randomized_benchmarking.LayerFidelity.html)
- [McKay et al., Benchmarking quantum processor performance at scale](https://arxiv.org/abs/2311.05933)
