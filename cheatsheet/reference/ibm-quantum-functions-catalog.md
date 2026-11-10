---
title: IBM Quantum Functions Catalog
short_title: IBM Quantum Functions Catalog
tool_type: functions
owner: Qiskit Functions team
last_verified: 2026-11-02
---

# IBM Quantum Functions Catalog

The IBM Quantum Functions Catalog lists ready-made functions (provider functions), mostly from IBM partners, that you load and run on IBM QPUs without deploying anything yourself. To deploy a workflow of your own on the same platform, see [Managed Workflows](managed-workflows.md). For additional information on any function (its guide, tutorials and full API reference), look at the [IBM Quantum Functions Catalog](https://quantum.cloud.ibm.com/functions).

* **Kinds:** circuit functions take PUBs and return results after their own error suppression and mitigation. Application functions take a problem description (a molecule, a graph, a cost function, a PDE) and return a solution.
* **Status and access:** all functions in the IBM Quantum Functions Catalog are in beta, for Premium, Flex and On-Prem plan users; the Open Plan cannot use them. Access it through the instance provided for the QDC challenge.
* **QPU time:** functions submit jobs from your instance, so their QPU time counts against your budget.

## Using the IBM Quantum Functions Catalog

`FunctionsCatalog` from `qiskit-ibm-catalog` (version 0.22.0, which requires `qiskit-serverless` 0.38) is the client for the IBM Quantum Functions Catalog. It authenticates with the same channel, instance CRN and API key as `qiskit-ibm-runtime`, lists the functions your instance can run, loads one as `"<provider>/<name>"` and runs it as an asynchronous job whose status, logs and result you poll. A function job may submit QPU jobs of its own, and `job.runtime_jobs()` returns their IDs. Besides `job.status()`, you can check the running status of your function jobs in your [workloads table](https://quantum.cloud.ibm.com/workloads). Every argument of `run()` is specific to the function. The current method names are `catalog.job()`, `job.cancel()` and `function.jobs()`; the older `get_job_by_id()`, `job.stop()` and `get_jobs()` still work but raise a `DeprecationWarning`.

- **When to use:** a function already does what you want to compare against, such as a partner's mitigation or optimization pipeline, and your instance has access to it.
- **Tags:** family: general; step: scale; runs: premium, qpu; status: beta
- **Related:** [Managed Workflows](managed-workflows.md) for deploying a workflow of your own; [QPUs at a glance](hardware.md#qpus-at-a-glance) for which QPUs an instance can reach.

```python
from qiskit_ibm_catalog import FunctionsCatalog

catalog = FunctionsCatalog(channel="ibm_quantum_platform", instance="<instance CRN>", token="<API key>")
print(catalog.list())                    # the functions your instance can run
qesem = catalog.load("qedma/qesem")      # "<provider>/<function>"
job = qesem.run(
    pubs=[(isa_circuit, observable)],      # analytical estimates need a single Pauli or Paulis on disjoint qubits
    backend_name="<QPU name>",
    options={"estimate_time_only": "analytical"},  # QESEM: an upper bound on QPU time, using none
)
print(job.status())                      # QUEUED, INITIALIZING, RUNNING[: <stage>], DONE, ERROR or CANCELED
estimate = job.result()                  # estimate[0].metadata["time_estimation_sec"] holds the estimate
```

## Circuit functions

| Function (provider) | Load with | What it does | Good to know | Suits |
|---|---|---|---|---|
| QESEM (Qedma) | `"qedma/qesem"` | Unbiased error-mitigated expectation values with error bars, from noise-aware transpilation, error suppression and quasi-probabilistic mitigation. QPU time grows with the circuit's active volume, not with its qubit count | Estimate QPU time before a real run: the analytical estimate is free but pessimistic (30-minute steps) and works only for a single Pauli or Paulis on disjoint qubits; the empirical one is more accurate (5-minute steps) and uses under 10 minutes of QPU time. Estimates are per backend, so run on the backend you estimated on. Multi-PUB and parameterized jobs need circuits already in the backend's basis gates. Supports fractional `RZZ` gates and high-weight observables. Estimator PUBs only | Error-mitigated expectation values with error bars: time-evolution (Trotter) circuits, VQE, QAOA and amplitude-estimation circuits; many-term or high-weight observables; fractional-gate circuits |
| Tensor-network error mitigation (TEM) (Algorithmiq) | `"algorithmiq/tem"` | Mitigated expectation values: informationally complete measurements, then tensor-network inversion of a noise model learned with the runtime noise learner, all in post-processing | Estimator PUBs with bound parameters only, and no loops, resets, mid-circuit measurements or control flow. Best for low-weight observables. A non-ISA circuit is transpiled for you; an ISA circuit is not, and then the observable must be defined on the whole QPU. Reusing an earlier noise-learner result saves QPU time | Time-evolution expectation values and other loop-free circuits with low-weight observables; several observables from one measurement data set |
| Performance Management (Q-CTRL Fire Opal) | `"q-ctrl/performance-management"` | A drop-in Sampler or Estimator with automatic error suppression and no sampling overhead | Pass untranspiled circuits: it transpiles them itself, and pre-transpiled circuits give worse results. Not every backend is supported | Error-suppressed Sampler or Estimator baselines: parametric or deep circuits, many circuit executions, QAOA sampling, time-evolution expectation values, observables of any weight |
| Orbit (Quantum Elements) | `"quantum-elements/orbit"` | Optimized transpilation, dynamical decoupling in scheduled idle windows and M3 readout mitigation, with a raw baseline in the same job | Supports dynamic circuits (mid-circuit measurement and feedforward); the guide does not say whether mid-circuit reset is supported. Can compare raw, default and custom DD configurations in one job. Preview mode shows the DD plan without using the QPU. Runtime DD and resilience are off unless you turn them on. M3 is Sampler-only | Circuits with idle periods: error-detected and error-corrected circuits, dynamic circuits with mid-circuit measurement, Bernstein-Vazirani, QFT, VQE and time-evolution circuits; DD-sequence comparisons against a raw baseline |
- **When to use:** you want a managed, error-suppressed or error-mitigated Sampler or Estimator baseline from a partner function.
- **Tags:** family: trotter, floquet, echoes, qaoa, detection; step: scale; noise: suppression, mitigation-expval, mitigation-sampling; runs: premium, qpu; status: beta

## Application functions

| Function (provider) | Load with | What it does | Good to know | Suits |
|---|---|---|---|---|
| Parity Twine Optimizer (ParityQC) | `"parityqc/parity-twine-optimizer"` | Compiles a QUBO into SWAP-free Parity Twine circuits of minimal two-qubit depth, runs one QAOA layer with grid-searched angles and post-processes greedily | Version 1.0.2 is the first stable release. QUBO only (higher-order compilation is planned). Has MIS-specific post-processing | QAOA sampling for QUBOs with all-to-all or otherwise non-local couplings (Sherrington-Kirkpatrick, market split, maximum independent set) |
| Constrained Quantum Optimizer (Aqarios) | `"aqarios/constrained-quantum-optimizer"` | Warm-started, fixed-angle (linear-ramp) QAOA, with XY mixers for one-hot and set-packing constraints and penalties for the rest | Binary or bounded-integer variables only; continuous variables fail with error code `4710`. One objective per run. More parallel runs, shots and QAOA layers improve quality at the cost of runtime | Constrained binary optimization, one objective per run (maximum independent set, Max-k-Cut, graph coloring) |
| Iskay Quantum Optimizer (Kipu Quantum) | `"kipu-quantum/iskay-quantum-optimizer"` | Unconstrained QUBO and higher-order binary problems with bf-DCQO, a non-variational counterdiabatic algorithm, plus local search | Up to 156 variables, one qubit each. Constraints must go in as penalties. Applies no error mitigation. Opens a session by default, which keeps the QPU reserved while it runs | Unconstrained QUBO and HUBO baselines (MaxCut, market split, scheduling, portfolio QUBOs) |
| Optimization Solver (Q-CTRL Fire Opal) | `"q-ctrl/optimization-solver"` | A noise-aware, QAOA-inspired solver for binary problems of any order, built on Fire Opal's error suppression | Runs at full device scale. Supports hard Hamming-weight-1 constraints; a size-k constraint still needs a penalty. Creates a new session unless you pass one | Binary optimization baselines of any order at full device scale (MaxCut, spin glasses, graph partitioning) |
| SQOA Solver (JIJ) | `"jij/sqoa-solver"` | Solves QUBO problems with more binary variables than the QPU has qubits: quantum relaxation (QRAO, up to three variables per qubit), QPU sampling and sample-based diagonalization (SQD), then rounding and a classical local search | QUBO only: constrained problems need penalties and higher-order ones need quadratizing before submission. Takes a coefficient dictionary or a padded-Base64 OMMX instance (`variable_type="binary"` or `"spin"`; OMMX is binary only); dictionaries are always minimized, so a maximization returns the negated objective. How many qubits it saves depends on how sparse the problem is. Picks the least busy QPU when `backend_name` is omitted. `shots` is a total budget across its circuits, not a per-circuit count | QUBO baselines larger than the device (maximum independent set, MaxCut, penalized constrained problems) |
| Quantum Portfolio Optimizer (Global Data Quantum) | `"global-data-quantum/quantum-portfolio-optimizer"` | Dynamic portfolio rebalancing: a QUBO solved with VQE, then SQD-based post-processing | One objective only. Qubit count grows with time steps × resolution qubits × assets; the guide's example needs 112 qubits | Single-objective dynamic portfolio optimization and strategy back-testing |
| HI-VQE Chemistry (Qunova Computing) | `"qunova/hivqe-chemistry"` | Molecular ground states by iterative subspace diagonalization, an extension of SQD | Free up to 20 qubits; licensed use above that goes through the `hivqe-qiskit-function-utils` wrapper | Chemistry: ground-state energies and potential energy surfaces of small molecules |
| Dynamics Simulator (Q-CTRL Fire Opal) | `"q-ctrl/dynamics-simulator"` | Time traces of observables for the one-dimensional Fermi-Hubbard model with open boundaries: you give the model, an initial Fock state, a Trotter schedule and occupation observables, and it builds, maps and error-suppresses the circuits itself | No circuits in or out, and nothing to transpile. Only `"trotter"` is supported. An `L`-site chain uses `2L` qubits; circuit depth grows with the number of Trotter steps, not with `L`. Larger time steps reach later times at the same depth but add Trotter error, which grows with the on-site interaction. Observables are sums of products of `Nup`/`Ndn` site occupations, so charge, spin and spin-spin correlators come from the same shots. `shots` is per circuit, and small signals need many shots (the tutorial uses 20,000) | Fermi-Hubbard dynamics on long chains (spin-charge separation, light cones, quench relaxation), past the reach of exact diagonalization |
| QUICK-PDE (ColibriTD) | `"colibritd/quick-pde"` | Solves partial differential equations with a variational Chebyshev-basis solver | Fixed use cases only, no arbitrary PDEs. Solutions come back on fixed grids | PDEs: fluid dynamics (inviscid Burgers, pressureless Euler) and material deformation (one-dimensional tensile test) |
| Singularity Machine Learning – Classification (Multiverse Computing) | `"multiverse/singularity"` | A binary classifier: a classical ensemble pruned with QAOA | Binary classification only. Large datasets go in as files rather than as arguments | Binary classification of labeled feature data |
- **When to use:** you want a managed baseline for an optimization, chemistry, many-body dynamics, PDE or classification problem from a partner function.
- **Tags:** family: qaoa, multi-objective, sqd, trotter, other; step: scale; runs: premium, qpu; status: beta
