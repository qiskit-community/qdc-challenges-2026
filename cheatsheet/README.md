# QDC 2026 Challenges Cheat Sheet

## Start here!
- **Participants:** set up with [Getting started](guides/getting-started.md). Then start from [I want to…](#i-want-to) for a specific goal, or from [By algorithm family](#by-algorithm-family) for the tools your challenge's method uses. Your challenge notebook is the authority on what is required and graded.
- **AI agents:** read [How to use this cheat sheet](guides/getting-started.md#how-to-use-this-cheat-sheet) before you act; it has rules on QPU time, credentials and graded code. This page is long, so read only the view you need: [I want to…](#i-want-to) maps tasks to entries, and [All entries](#all-entries) lists every entry with its tags, which [taxonomy.toml](tools/taxonomy.toml) defines. Every link names a document and a heading, so load only those sections.

**Browse by** [task](#i-want-to) · [algorithm family](#by-algorithm-family) · [all entries](#all-entries) · [workflow step](#by-workflow-step) · [noise management](#by-noise-management-layer) · [what's new](#new-and-changing) · [tool type](#by-tool-type)

**Guides:** [Getting started](guides/getting-started.md) · [Techniques](guides/techniques.md) · [Testing and tips](guides/testing-and-tips.md) · [Troubleshooting](guides/troubleshooting.md)

<!-- BEGIN GENERATED: views -->
## I want to…

Tasks grouped by what you are doing. Each link names the entry or section that answers it.

### Set up and get help

| Task | Go to |
|---|---|
| Install the package versions my notebook needs | [Versions and environment](guides/getting-started.md#versions-and-environment) |
| Save my IBM Quantum API key and choose an instance | [IBM Quantum account](guides/getting-started.md#ibm-quantum-account) |
| Find the import line for a class or function | [API quick reference](guides/testing-and-tips.md#api-quick-reference) |
| Fix an error message or an unexpected result | [Troubleshooting](guides/troubleshooting.md) |
| Give my AI assistant Qiskit tools and documentation | [Setting up an AI assistant](guides/getting-started.md#setting-up-an-ai-assistant) · [Qiskit MCP servers](reference/ai-tooling.md#qiskit-mcp-servers) · [Documentation for agents](reference/ai-tooling.md#documentation-for-agents) |
| Make my AI assistant ask before it submits or cancels jobs | [Qiskit MCP servers](reference/ai-tooling.md#qiskit-mcp-servers) · [How to use this cheat sheet](guides/getting-started.md#how-to-use-this-cheat-sheet) |
| Move my own older code to the client-side primitives | [Sampler and Estimator: client-side primitives](reference/ibm-quantum-compute-client.md#sampler-and-estimator-client-side-primitives) · [Qiskit AI skills](reference/ai-tooling.md#qiskit-ai-skills) |
| Submit answers for grading | [Grading with qc-grader](guides/getting-started.md#grading-with-qc-grader) |

### Build circuits

| Task | Go to |
|---|---|
| Build Trotter or Floquet circuits for a spin model | [Circuit library](reference/qiskit-sdk.md#circuit-library) · [Swap strategies for commuting gates](reference/qiskit-sdk.md#swap-strategies-for-commuting-gates) · [Fractional gates](reference/ibm-quantum-compute-client.md#fractional-gates) |
| Solve an optimization problem with QAOA | [Circuit library](reference/qiskit-sdk.md#circuit-library) · [QAOA training pipeline](reference/ecosystem.md#qaoa-training-pipeline) · [Quantum optimization best practices](reference/ecosystem.md#quantum-optimization-best-practices-qopt-best-practices) · [Optimization mapper](reference/qiskit-addons.md#optimization-mapper-qiskit-addon-opt-mapper) · [Swap strategies for commuting gates](reference/qiskit-sdk.md#swap-strategies-for-commuting-gates) · [Qubit selection and layout](reference/qiskit-sdk.md#qubit-selection-and-layout) · [CVaR of sampled results](guides/techniques.md#cvar-of-sampled-results) |
| Replace a costly circuit block with a sampled ensemble of cheaper circuits | [Sampling a linear combination of unitaries](guides/techniques.md#sampling-a-linear-combination-of-unitaries) · [Circuit cutting](reference/qiskit-addons.md#circuit-cutting-qiskit-addon-cutting) |
| Estimate a ground-state energy from samples | [Sample-based quantum diagonalization](reference/qiskit-addons.md#sample-based-quantum-diagonalization-qiskit-addon-sqd) · [Chemistry: PySCF and ffsim](reference/ecosystem.md#chemistry-pyscf-and-ffsim) · [Qiskit Fermions](reference/qiskit-addons.md#qiskit-fermions-qiskit-fermions) |
| Load a published benchmark circuit | [Reference circuits: the Quantum Advantage Tracker](reference/ecosystem.md#reference-circuits-the-quantum-advantage-tracker) |
| Measure or reset qubits in the middle of a circuit | [Dynamic circuits and resets](reference/ibm-quantum-compute-client.md#dynamic-circuits-and-resets) · [Specially calibrated instructions and calibration sets](reference/ibm-quantum-compute-client.md#specially-calibrated-instructions-and-calibration-sets) |

### Fit the hardware

| Task | Go to |
|---|---|
| Choose a QPU | [QPUs at a glance](reference/hardware.md#qpus-at-a-glance) · [Layout ranking with mapomatic](reference/ecosystem.md#layout-ranking-with-mapomatic) |
| Pick the best physical qubits for my circuit | [Qubit selection and layout](reference/qiskit-sdk.md#qubit-selection-and-layout) · [Calibration data and layer fidelity](reference/hardware.md#calibration-data-and-layer-fidelity) · [Layout ranking with mapomatic](reference/ecosystem.md#layout-ranking-with-mapomatic) · [Device benchmarking](reference/ecosystem.md#device-benchmarking-qiskit-device-benchmarking) |
| Cut the depth or two-qubit gate count of a circuit | [Transpiler: preset and custom pass managers](reference/qiskit-sdk.md#transpiler-preset-and-custom-pass-managers) · [Swap strategies for commuting gates](reference/qiskit-sdk.md#swap-strategies-for-commuting-gates) · [AI-powered transpiler passes](reference/qiskit-sdk.md#ai-powered-transpiler-passes-qiskit-ibm-transpiler) |
| Run RZZ and RX rotations as single native gates | [Fractional gates](reference/ibm-quantum-compute-client.md#fractional-gates) |
| Shrink a circuit that is too deep or too wide for the hardware | [Operator backpropagation](reference/qiskit-addons.md#operator-backpropagation-qiskit-addon-obp) · [AQC-Tensor](reference/qiskit-addons.md#aqc-tensor-qiskit-addon-aqc-tensor) · [Circuit cutting](reference/qiskit-addons.md#circuit-cutting-qiskit-addon-cutting) |
| Reduce Trotter error without deeper circuits | [Multi-product formulas](reference/qiskit-addons.md#multi-product-formulas-qiskit-addon-mpf) |

### Test and budget

| Task | Go to |
|---|---|
| Test my workflow without an account or QPU time | [Fake backends and local testing mode](reference/ibm-quantum-compute-client.md#fake-backends-and-local-testing-mode) · [Qiskit Aer](reference/simulators.md#qiskit-aer) · [Local test patterns](guides/testing-and-tips.md#local-test-patterns) |
| Simulate a circuit with mid-circuit resets, measurements or feed-forward | [Qiskit Aer](reference/simulators.md#qiskit-aer) · [Exact and tensor-network references](reference/simulators.md#exact-and-tensor-network-references) |
| Get a reference value for a circuit too large to simulate exactly | [Exact and tensor-network references](reference/simulators.md#exact-and-tensor-network-references) · [Qiskit Aer](reference/simulators.md#qiskit-aer) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) |
| Check how much noise will hurt my circuit before a large run | [NEAT: the noisy Estimator analyzer](reference/ibm-quantum-compute-client.md#neat-the-noisy-estimator-analyzer) |
| Estimate QPU time before I submit | [Budgeting QPU time](reference/ibm-quantum-compute-client.md#budgeting-qpu-time) · [Count the executions a client-side Estimator job will submit](guides/testing-and-tips.md#count-the-executions-a-client-side-estimator-job-will-submit) |
| See how much QPU time I have used or have left | [Budgeting QPU time](reference/ibm-quantum-compute-client.md#budgeting-qpu-time) · [Before you spend QPU time](guides/testing-and-tips.md#before-you-spend-qpu-time) |
| Stop a duplicate job, or recover after a job cell ran twice | [Before you spend QPU time](guides/testing-and-tips.md#before-you-spend-qpu-time) · [Saving results and job IDs](reference/ibm-quantum-compute-client.md#saving-results-and-job-ids) |

### Run jobs

| Task | Go to |
|---|---|
| Run many circuit variants or parameter sets with few jobs | [Sampler and Estimator: client-side primitives](reference/ibm-quantum-compute-client.md#sampler-and-estimator-client-side-primitives) · [Executor and QuantumProgram](reference/ibm-quantum-compute-client.md#executor-and-quantumprogram) · [Execution modes: job, batch and session](reference/ibm-quantum-compute-client.md#execution-modes-job-batch-and-session) |
| Recover a job instead of resubmitting it | [Saving results and job IDs](reference/ibm-quantum-compute-client.md#saving-results-and-job-ids) · [Retrieve a finished job instead of resubmitting it](guides/testing-and-tips.md#retrieve-a-finished-job-instead-of-resubmitting-it) |
| Share circuits between Qiskit versions | [Saving circuits with QPY](reference/qiskit-sdk.md#saving-circuits-with-qpy) |
| Run long classical steps next to the QPU | [Write, upload and run a workflow](reference/managed-workflows.md#write-upload-and-run-a-workflow) (Premium plans) · [Fleets runner and compute profiles](reference/managed-workflows.md#fleets-runner-and-compute-profiles) (Premium plans) |
| Try a ready-made solver or error-mitigation service | [Using the IBM Quantum Functions Catalog](reference/ibm-quantum-functions-catalog.md#using-the-ibm-quantum-functions-catalog) (Premium plans) · [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions) (Premium plans) · [Application functions](reference/ibm-quantum-functions-catalog.md#application-functions) (Premium plans) |

### Reduce noise

| Task | Go to |
|---|---|
| Choose an error-mitigation method | [Choosing error detection and mitigation](guides/techniques.md#choosing-error-detection-and-mitigation) |
| Suppress decoherence, crosstalk and coherent errors | [Dynamical decoupling options](reference/ibm-quantum-compute-client.md#dynamical-decoupling-options) · [Dynamical decoupling passes](reference/qiskit-sdk.md#dynamical-decoupling-passes) · [Twirling options in the primitives](reference/ibm-quantum-compute-client.md#twirling-options-in-the-primitives) · [Pauli twirling with Samplomatic](reference/noise-management.md#pauli-twirling-with-samplomatic) |
| Mitigate readout errors in counts or sampled distributions | [Measurement mitigation with M3](reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree) |
| Mitigate readout errors in expectation values | [Sampler and Estimator: client-side primitives](reference/ibm-quantum-compute-client.md#sampler-and-estimator-client-side-primitives) · [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) |
| Mitigate gate errors in expectation values | [Sampler and Estimator: client-side primitives](reference/ibm-quantum-compute-client.md#sampler-and-estimator-client-side-primitives) · [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) · [Shaded light cones](reference/qiskit-addons.md#shaded-light-cones-qiskit-addon-slc) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) · [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions) (Premium plans) |
| Discard shots hit by an error | [Post-selection on non-Markovian error checks](reference/noise-management.md#post-selection-on-non-markovian-error-checks) · [Spacetime Pauli checks](reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice) · [The Iceberg code](guides/techniques.md#the-iceberg-code) |
| Learn a noise model for the layers of my circuit | [NoiseLearnerV3](reference/ibm-quantum-compute-client.md#noiselearnerv3) · [Custom noise learning](reference/noise-management.md#custom-noise-learning-qiskit-noise-learning) |

### Analyze results

| Task | Go to |
|---|---|
| Turn shot data into expectation values | [Expectation values from shot data](reference/qiskit-addons.md#expectation-values-from-shot-data) |
| Estimate the fidelity of a Clifford circuit on hardware | [Fidelity of a Clifford circuit from sampled stabilizers](guides/techniques.md#fidelity-of-a-clifford-circuit-from-sampled-stabilizers) · [Expectation values from shot data](reference/qiskit-addons.md#expectation-values-from-shot-data) · [Spacetime Pauli checks](reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice) |
| Score samples by their best fraction | [CVaR of sampled results](guides/techniques.md#cvar-of-sampled-results) |
| Compare with a classical optimum or a Pareto front | [Classical solvers and multi-objective tools](reference/ecosystem.md#classical-solvers-and-multi-objective-tools) |

## By algorithm family

Each family lists the tools specific to it, grouped by workflow step. Tools that every family uses, such as the transpiler, the primitives and QPU budgeting, are listed under [By workflow step](#by-workflow-step).

Families: [QAOA](#qaoa) · [Multi-objective and constrained optimization](#multi-objective-and-constrained-optimization) · [LCU and circuit ensembles](#lcu-and-circuit-ensembles) · [Trotterized time evolution](#trotterized-time-evolution) · [Floquet dynamics](#floquet-dynamics) · [Loschmidt echoes and scrambling](#loschmidt-echoes-and-scrambling) · [Thermal and dissipative states](#thermal-and-dissipative-states) · [Sample-based diagonalization and chemistry](#sample-based-diagonalization-and-chemistry) · [Verifiable circuit sampling](#verifiable-circuit-sampling) · [Error detection and encoded circuits](#error-detection-and-encoded-circuits) · [Other application domains](#other-application-domains)

### QAOA

Quantum approximate optimization and its variants: linear-ramp and warm-start QAOA, constraint-preserving mixers, CVaR objectives, and angles trained classically or transferred between instances.

- **Map:** [Circuit library](reference/qiskit-sdk.md#circuit-library) · [Optimization mapper](reference/qiskit-addons.md#optimization-mapper-qiskit-addon-opt-mapper) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) · [QAOA training pipeline](reference/ecosystem.md#qaoa-training-pipeline)
- **Optimize:** [Swap strategies for commuting gates](reference/qiskit-sdk.md#swap-strategies-for-commuting-gates) · [Qubit selection and layout](reference/qiskit-sdk.md#qubit-selection-and-layout) · [Fractional gates](reference/ibm-quantum-compute-client.md#fractional-gates) · [Quantum optimization best practices](reference/ecosystem.md#quantum-optimization-best-practices-qopt-best-practices) · [Calibration data and layer fidelity](reference/hardware.md#calibration-data-and-layer-fidelity)
- **Post-process:** [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) · [Measurement mitigation with M3](reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree) · [Classical solvers and multi-objective tools](reference/ecosystem.md#classical-solvers-and-multi-objective-tools) · [CVaR of sampled results](guides/techniques.md#cvar-of-sampled-results)
- **Scale:** [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions) · [Application functions](reference/ibm-quantum-functions-catalog.md#application-functions)

### Multi-objective and constrained optimization

Problems with several objectives or hard constraints: Pareto fronts and their hypervolume, scalarized sub-problems, and constraints evaluated on each candidate solution.

- **Map:** [Optimization mapper](reference/qiskit-addons.md#optimization-mapper-qiskit-addon-opt-mapper) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) · [QAOA training pipeline](reference/ecosystem.md#qaoa-training-pipeline)
- **Optimize:** [Swap strategies for commuting gates](reference/qiskit-sdk.md#swap-strategies-for-commuting-gates) · [Qubit selection and layout](reference/qiskit-sdk.md#qubit-selection-and-layout) · [Quantum optimization best practices](reference/ecosystem.md#quantum-optimization-best-practices-qopt-best-practices) · [Calibration data and layer fidelity](reference/hardware.md#calibration-data-and-layer-fidelity)
- **Post-process:** [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) · [Measurement mitigation with M3](reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree) · [Classical solvers and multi-objective tools](reference/ecosystem.md#classical-solvers-and-multi-objective-tools)
- **Scale:** [Application functions](reference/ibm-quantum-functions-catalog.md#application-functions)

### LCU and circuit ensembles

Methods that replace one deep circuit by a weighted ensemble of shallower circuits and pay in shots: linear combinations of unitaries sampled term by term, multi-product formulas, quasi-probability decompositions and circuit cutting.

- **Map:** [Sampling a linear combination of unitaries](guides/techniques.md#sampling-a-linear-combination-of-unitaries)
- **Optimize:** [Dynamical decoupling passes](reference/qiskit-sdk.md#dynamical-decoupling-passes) · [Fractional gates](reference/ibm-quantum-compute-client.md#fractional-gates) · [Circuit cutting](reference/qiskit-addons.md#circuit-cutting-qiskit-addon-cutting)
- **Execute:** [Executor and QuantumProgram](reference/ibm-quantum-compute-client.md#executor-and-quantumprogram)
- **Post-process:** [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) · [Expectation values from shot data](reference/qiskit-addons.md#expectation-values-from-shot-data) · [Measurement mitigation with M3](reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree) · [Multi-product formulas](reference/qiskit-addons.md#multi-product-formulas-qiskit-addon-mpf) · [CVaR of sampled results](guides/techniques.md#cvar-of-sampled-results)

### Trotterized time evolution

Hamiltonian simulation of spin and lattice models with Trotter circuits, and the methods that compress, extrapolate or mitigate them.

- **Map:** [Circuit library](reference/qiskit-sdk.md#circuit-library) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) · [Reference circuits: the Quantum Advantage Tracker](reference/ecosystem.md#reference-circuits-the-quantum-advantage-tracker)
- **Optimize:** [Swap strategies for commuting gates](reference/qiskit-sdk.md#swap-strategies-for-commuting-gates) · [Fractional gates](reference/ibm-quantum-compute-client.md#fractional-gates) · [Pauli twirling with Samplomatic](reference/noise-management.md#pauli-twirling-with-samplomatic) · [Operator backpropagation](reference/qiskit-addons.md#operator-backpropagation-qiskit-addon-obp) · [AQC-Tensor](reference/qiskit-addons.md#aqc-tensor-qiskit-addon-aqc-tensor) · [Circuit cutting](reference/qiskit-addons.md#circuit-cutting-qiskit-addon-cutting)
- **Execute:** [Executor and QuantumProgram](reference/ibm-quantum-compute-client.md#executor-and-quantumprogram) · [NoiseLearnerV3](reference/ibm-quantum-compute-client.md#noiselearnerv3) · [Custom noise learning](reference/noise-management.md#custom-noise-learning-qiskit-noise-learning)
- **Post-process:** [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) · [Shaded light cones](reference/qiskit-addons.md#shaded-light-cones-qiskit-addon-slc) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) · [Multi-product formulas](reference/qiskit-addons.md#multi-product-formulas-qiskit-addon-mpf)
- **Scale:** [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions) · [Application functions](reference/ibm-quantum-functions-catalog.md#application-functions) · [Qiskit Function templates](reference/managed-workflows.md#qiskit-function-templates)
- **Develop:** [Exact and tensor-network references](reference/simulators.md#exact-and-tensor-network-references)

### Floquet dynamics

Periodically driven circuits, which repeat one layer many times, and their long-time, prethermal behavior.

- **Map:** [Circuit library](reference/qiskit-sdk.md#circuit-library) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) · [Reference circuits: the Quantum Advantage Tracker](reference/ecosystem.md#reference-circuits-the-quantum-advantage-tracker)
- **Optimize:** [Swap strategies for commuting gates](reference/qiskit-sdk.md#swap-strategies-for-commuting-gates) · [Qubit selection and layout](reference/qiskit-sdk.md#qubit-selection-and-layout) · [Fractional gates](reference/ibm-quantum-compute-client.md#fractional-gates) · [Pauli twirling with Samplomatic](reference/noise-management.md#pauli-twirling-with-samplomatic) · [Operator backpropagation](reference/qiskit-addons.md#operator-backpropagation-qiskit-addon-obp) · [AQC-Tensor](reference/qiskit-addons.md#aqc-tensor-qiskit-addon-aqc-tensor) · [Layout ranking with mapomatic](reference/ecosystem.md#layout-ranking-with-mapomatic) · [Calibration data and layer fidelity](reference/hardware.md#calibration-data-and-layer-fidelity)
- **Execute:** [Executor and QuantumProgram](reference/ibm-quantum-compute-client.md#executor-and-quantumprogram) · [NoiseLearnerV3](reference/ibm-quantum-compute-client.md#noiselearnerv3) · [Custom noise learning](reference/noise-management.md#custom-noise-learning-qiskit-noise-learning)
- **Post-process:** [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) · [Shaded light cones](reference/qiskit-addons.md#shaded-light-cones-qiskit-addon-slc) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) · [Measurement mitigation with M3](reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree)
- **Scale:** [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions)
- **Develop:** [Exact and tensor-network references](reference/simulators.md#exact-and-tensor-network-references)

### Loschmidt echoes and scrambling

Circuits that evolve forward, apply a small perturbation and evolve backward, such as operator Loschmidt echoes and out-of-time-order correlators, which measure how sensitive an observable is to the perturbation and how fast information spreads.

- **Map:** [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) · [Reference circuits: the Quantum Advantage Tracker](reference/ecosystem.md#reference-circuits-the-quantum-advantage-tracker)
- **Optimize:** [Qubit selection and layout](reference/qiskit-sdk.md#qubit-selection-and-layout) · [Specially calibrated instructions and calibration sets](reference/ibm-quantum-compute-client.md#specially-calibrated-instructions-and-calibration-sets) · [Pauli twirling with Samplomatic](reference/noise-management.md#pauli-twirling-with-samplomatic) · [Operator backpropagation](reference/qiskit-addons.md#operator-backpropagation-qiskit-addon-obp) · [Device benchmarking](reference/ecosystem.md#device-benchmarking-qiskit-device-benchmarking) · [Calibration data and layer fidelity](reference/hardware.md#calibration-data-and-layer-fidelity)
- **Execute:** [Executor and QuantumProgram](reference/ibm-quantum-compute-client.md#executor-and-quantumprogram) · [NoiseLearnerV3](reference/ibm-quantum-compute-client.md#noiselearnerv3) · [Custom noise learning](reference/noise-management.md#custom-noise-learning-qiskit-noise-learning)
- **Post-process:** [Post-selection on non-Markovian error checks](reference/noise-management.md#post-selection-on-non-markovian-error-checks) · [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) · [Expectation values from shot data](reference/qiskit-addons.md#expectation-values-from-shot-data) · [Shaded light cones](reference/qiskit-addons.md#shaded-light-cones-qiskit-addon-slc) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation)
- **Scale:** [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions)
- **Develop:** [Exact and tensor-network references](reference/simulators.md#exact-and-tensor-network-references)

### Thermal and dissipative states

Preparing Gibbs-like or steady states, for example by engineered dissipation with mid-circuit resets, and measuring their observables.

- **Map:** [Dynamic circuits and resets](reference/ibm-quantum-compute-client.md#dynamic-circuits-and-resets)
- **Optimize:** [Dynamical decoupling passes](reference/qiskit-sdk.md#dynamical-decoupling-passes) · [Fractional gates](reference/ibm-quantum-compute-client.md#fractional-gates) · [Specially calibrated instructions and calibration sets](reference/ibm-quantum-compute-client.md#specially-calibrated-instructions-and-calibration-sets) · [Pauli twirling with Samplomatic](reference/noise-management.md#pauli-twirling-with-samplomatic)
- **Execute:** [Executor and QuantumProgram](reference/ibm-quantum-compute-client.md#executor-and-quantumprogram)
- **Post-process:** [Post-selection on non-Markovian error checks](reference/noise-management.md#post-selection-on-non-markovian-error-checks) · [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) · [Expectation values from shot data](reference/qiskit-addons.md#expectation-values-from-shot-data)
- **Develop:** [Qiskit Aer](reference/simulators.md#qiskit-aer) · [Exact and tensor-network references](reference/simulators.md#exact-and-tensor-network-references)

### Sample-based diagonalization and chemistry

Ground-state energies from sampled configurations (sample-based quantum diagonalization and its Krylov variant), LUCJ circuits, and embedding methods that split large molecules into fragments.

- **Map:** [Qiskit Fermions](reference/qiskit-addons.md#qiskit-fermions-qiskit-fermions) · [Chemistry: PySCF and ffsim](reference/ecosystem.md#chemistry-pyscf-and-ffsim) · [Reference circuits: the Quantum Advantage Tracker](reference/ecosystem.md#reference-circuits-the-quantum-advantage-tracker)
- **Post-process:** [Measurement mitigation with M3](reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree) · [Sample-based quantum diagonalization](reference/qiskit-addons.md#sample-based-quantum-diagonalization-qiskit-addon-sqd)
- **Scale:** [Application functions](reference/ibm-quantum-functions-catalog.md#application-functions) · [Write, upload and run a workflow](reference/managed-workflows.md#write-upload-and-run-a-workflow) · [Fleets runner and compute profiles](reference/managed-workflows.md#fleets-runner-and-compute-profiles) · [Qiskit Function templates](reference/managed-workflows.md#qiskit-function-templates)

### Verifiable circuit sampling

Sampling circuits that are hard to simulate and certifying their fidelity, for example by doping checked Clifford circuits with a few non-Clifford gates, estimating fidelity from sampled stabilizers and post-selecting on spacetime checks.

- **Map:** [Spacetime Pauli checks](reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice) · [Reference circuits: the Quantum Advantage Tracker](reference/ecosystem.md#reference-circuits-the-quantum-advantage-tracker)
- **Optimize:** [Specially calibrated instructions and calibration sets](reference/ibm-quantum-compute-client.md#specially-calibrated-instructions-and-calibration-sets) · [Pauli twirling with Samplomatic](reference/noise-management.md#pauli-twirling-with-samplomatic) · [Layout ranking with mapomatic](reference/ecosystem.md#layout-ranking-with-mapomatic) · [Device benchmarking](reference/ecosystem.md#device-benchmarking-qiskit-device-benchmarking)
- **Execute:** [Executor and QuantumProgram](reference/ibm-quantum-compute-client.md#executor-and-quantumprogram) · [NoiseLearnerV3](reference/ibm-quantum-compute-client.md#noiselearnerv3) · [Custom noise learning](reference/noise-management.md#custom-noise-learning-qiskit-noise-learning)
- **Post-process:** [Post-selection on non-Markovian error checks](reference/noise-management.md#post-selection-on-non-markovian-error-checks) · [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) · [Spacetime Pauli checks](reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice) · [Expectation values from shot data](reference/qiskit-addons.md#expectation-values-from-shot-data) · [Measurement mitigation with M3](reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree) · [Fidelity of a Clifford circuit from sampled stabilizers](guides/techniques.md#fidelity-of-a-clifford-circuit-from-sampled-stabilizers)
- **Scale:** [Write, upload and run a workflow](reference/managed-workflows.md#write-upload-and-run-a-workflow)
- **Develop:** [Qiskit Aer](reference/simulators.md#qiskit-aer) · [Exact and tensor-network references](reference/simulators.md#exact-and-tensor-network-references)

### Error detection and encoded circuits

Small error-detecting codes and checks that flag errors so that shots can be post-selected, an early step toward fault tolerance.

- **Map:** [Dynamic circuits and resets](reference/ibm-quantum-compute-client.md#dynamic-circuits-and-resets) · [Spacetime Pauli checks](reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice) · [The Iceberg code](guides/techniques.md#the-iceberg-code)
- **Optimize:** [Dynamical decoupling passes](reference/qiskit-sdk.md#dynamical-decoupling-passes) · [Specially calibrated instructions and calibration sets](reference/ibm-quantum-compute-client.md#specially-calibrated-instructions-and-calibration-sets)
- **Post-process:** [Post-selection on non-Markovian error checks](reference/noise-management.md#post-selection-on-non-markovian-error-checks) · [Spacetime Pauli checks](reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice) · [Expectation values from shot data](reference/qiskit-addons.md#expectation-values-from-shot-data) · [The Iceberg code](guides/techniques.md#the-iceberg-code) · [Fidelity of a Clifford circuit from sampled stabilizers](guides/techniques.md#fidelity-of-a-clifford-circuit-from-sampled-stabilizers)
- **Scale:** [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions)
- **Develop:** [Qiskit Aer](reference/simulators.md#qiskit-aer) · [Exact and tensor-network references](reference/simulators.md#exact-and-tensor-network-references)

### Other application domains

Partial differential equations, classification and finance, reached through application functions.

- **Scale:** [Application functions](reference/ibm-quantum-functions-catalog.md#application-functions)

## All entries

Every entry with its tags. Tag values are defined in [taxonomy.toml](tools/taxonomy.toml).

Runs on: **local** runs on your machine without an account; **account** needs an IBM Quantum account but no QPU time; **qpu** uses QPU time; **serverless** runs as a Managed Workflow (Qiskit Serverless), which needs a Premium, Flex or On-Prem plan; **premium** needs a Premium, Flex or On-Prem plan.

| Entry | Document | Families | Step | Noise | Runs on | Status | Updated |
|---|---|---|---|---|---|---|---|
| [Circuit library](reference/qiskit-sdk.md#circuit-library) | Qiskit SDK | General, QAOA, Trotter, Floquet | Map |  | local | stable |  |
| [Transpiler: preset and custom pass managers](reference/qiskit-sdk.md#transpiler-preset-and-custom-pass-managers) | Qiskit SDK | General | Optimize |  | local | stable | 2026-07 |
| [AI-powered transpiler passes (qiskit-ibm-transpiler)](reference/qiskit-sdk.md#ai-powered-transpiler-passes-qiskit-ibm-transpiler) | Qiskit SDK | General | Optimize |  | local, account | beta | 2026-03 |
| [Swap strategies for commuting gates](reference/qiskit-sdk.md#swap-strategies-for-commuting-gates) | Qiskit SDK | QAOA, Multi-objective, Trotter, Floquet | Optimize |  | local | stable | 2026-07 |
| [Qubit selection and layout](reference/qiskit-sdk.md#qubit-selection-and-layout) | Qiskit SDK | General, QAOA, Multi-objective, Floquet, Echoes | Optimize |  | local, account | stable | 2026-06 |
| [Dynamical decoupling passes](reference/qiskit-sdk.md#dynamical-decoupling-passes) | Qiskit SDK | General, LCU, Thermal, Error detection | Optimize | Suppression | local | stable |  |
| [Saving circuits with QPY](reference/qiskit-sdk.md#saving-circuits-with-qpy) | Qiskit SDK | General | Develop |  | local | stable | 2026-10 |
| [Fractional gates](reference/ibm-quantum-compute-client.md#fractional-gates) | Compute client | QAOA, LCU, Trotter, Floquet, Thermal | Optimize |  | qpu, local | experimental | 2026-07 |
| [Specially calibrated instructions and calibration sets](reference/ibm-quantum-compute-client.md#specially-calibrated-instructions-and-calibration-sets) | Compute client | Thermal, Error detection, Verifiable sampling, Echoes | Optimize |  | qpu | stable | 2026-09 |
| [Dynamic circuits and resets](reference/ibm-quantum-compute-client.md#dynamic-circuits-and-resets) | Compute client | Thermal, Error detection | Map |  | qpu, local | stable | 2026-09 |
| [Dynamical decoupling options](reference/ibm-quantum-compute-client.md#dynamical-decoupling-options) | Compute client | General | Execute | Suppression | qpu, local | stable | 2026-09 |
| [Executor and QuantumProgram](reference/ibm-quantum-compute-client.md#executor-and-quantumprogram) | Compute client | General, LCU, Trotter, Floquet, Echoes, Thermal, Verifiable sampling | Execute |  | qpu, local | beta | 2026-09 |
| [Sampler and Estimator: client-side primitives](reference/ibm-quantum-compute-client.md#sampler-and-estimator-client-side-primitives) | Compute client | General | Execute | Mitigation (expectation values) | qpu, local | stable | 2026-09 |
| [Execution modes: job, batch and session](reference/ibm-quantum-compute-client.md#execution-modes-job-batch-and-session) | Compute client | General | Execute |  | qpu | stable |  |
| [Budgeting QPU time](reference/ibm-quantum-compute-client.md#budgeting-qpu-time) | Compute client | General | Execute |  | local, account | stable | 2026-09 |
| [Saving results and job IDs](reference/ibm-quantum-compute-client.md#saving-results-and-job-ids) | Compute client | General | Develop |  | local, account | stable | 2026-09 |
| [Twirling options in the primitives](reference/ibm-quantum-compute-client.md#twirling-options-in-the-primitives) | Compute client | General | Execute | Suppression | qpu, local | stable | 2026-09 |
| [NoiseLearnerV3](reference/ibm-quantum-compute-client.md#noiselearnerv3) | Compute client | Trotter, Floquet, Echoes, Verifiable sampling | Execute | Characterization | qpu | beta | 2026-09 |
| [Fake backends and local testing mode](reference/ibm-quantum-compute-client.md#fake-backends-and-local-testing-mode) | Compute client | General | Develop |  | local | stable | 2026-09 |
| [NEAT: the noisy Estimator analyzer](reference/ibm-quantum-compute-client.md#neat-the-noisy-estimator-analyzer) | Compute client | General | Develop | Characterization | local | stable | 2026-09 |
| [Pauli twirling with Samplomatic](reference/noise-management.md#pauli-twirling-with-samplomatic) | Samplomatic and mitigation | General, Trotter, Floquet, Echoes, Thermal, Verifiable sampling | Optimize | Suppression | local, qpu | beta | 2026-08 |
| [Post-selection on non-Markovian error checks](reference/noise-management.md#post-selection-on-non-markovian-error-checks) | Samplomatic and mitigation | Error detection, Verifiable sampling, Thermal, Echoes | Post-process | Post-selection | local, qpu | beta | 2026-09 |
| [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) | Samplomatic and mitigation | General, LCU, Trotter, Floquet, Echoes, Verifiable sampling, Thermal | Post-process | Mitigation (expectation values) | local, qpu | beta | 2026-09 |
| [Custom noise learning (qiskit-noise-learning)](reference/noise-management.md#custom-noise-learning-qiskit-noise-learning) | Samplomatic and mitigation | Trotter, Floquet, Echoes, Verifiable sampling | Execute | Characterization | local, qpu | alpha | 2026-10 |
| [Optimization mapper (qiskit-addon-opt-mapper)](reference/qiskit-addons.md#optimization-mapper-qiskit-addon-opt-mapper) | Qiskit addons | QAOA, Multi-objective | Map |  | local | stable | 2025-11 |
| [Qiskit Fermions (qiskit-fermions)](reference/qiskit-addons.md#qiskit-fermions-qiskit-fermions) | Qiskit addons | SQD | Map |  | local | beta | 2026-10 |
| [Operator backpropagation (qiskit-addon-obp)](reference/qiskit-addons.md#operator-backpropagation-qiskit-addon-obp) | Qiskit addons | Trotter, Floquet, Echoes | Optimize | Mitigation (expectation values) | local | stable | 2025-05 |
| [AQC-Tensor (qiskit-addon-aqc-tensor)](reference/qiskit-addons.md#aqc-tensor-qiskit-addon-aqc-tensor) | Qiskit addons | Trotter, Floquet | Optimize | Suppression | local, serverless | stable | 2026-07 |
| [Circuit cutting (qiskit-addon-cutting)](reference/qiskit-addons.md#circuit-cutting-qiskit-addon-cutting) | Qiskit addons | LCU, Trotter | Optimize |  | local | stable | 2025-03 |
| [Spacetime Pauli checks (qiskit-paulice)](reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice) | Qiskit addons | Verifiable sampling, Error detection | Map, Post-process | Post-selection | local | stable | 2026-10 |
| [Expectation values from shot data](reference/qiskit-addons.md#expectation-values-from-shot-data) | Qiskit addons | General, LCU, Thermal, Verifiable sampling, Error detection, Echoes | Post-process |  | local | stable | 2026-09 |
| [Shaded light cones (qiskit-addon-slc)](reference/qiskit-addons.md#shaded-light-cones-qiskit-addon-slc) | Qiskit addons | Trotter, Floquet, Echoes | Post-process | Mitigation (expectation values) | local, qpu | stable | 2026-10 |
| [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) | Qiskit addons | Trotter, Floquet, Echoes, QAOA, Multi-objective | Map, Post-process | Mitigation (expectation values) | local | stable | 2026-08 |
| [Measurement mitigation with M3 (mthree)](reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree) | Qiskit addons | QAOA, Multi-objective, LCU, SQD, Verifiable sampling, Floquet | Post-process | Mitigation (sampling) | local, qpu | stable | 2025-03 |
| [Sample-based quantum diagonalization (qiskit-addon-sqd)](reference/qiskit-addons.md#sample-based-quantum-diagonalization-qiskit-addon-sqd) | Qiskit addons | SQD | Post-process | Mitigation (sampling) | local, serverless | stable | 2026-10 |
| [Multi-product formulas (qiskit-addon-mpf)](reference/qiskit-addons.md#multi-product-formulas-qiskit-addon-mpf) | Qiskit addons | Trotter, LCU | Post-process |  | local | stable | 2025-03 |
| [Using the IBM Quantum Functions Catalog](reference/ibm-quantum-functions-catalog.md#using-the-ibm-quantum-functions-catalog) | IBM Quantum Functions Catalog | General | Scale |  | premium, qpu | beta |  |
| [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions) | IBM Quantum Functions Catalog | Trotter, Floquet, Echoes, QAOA, Error detection | Scale | Suppression, Mitigation (expectation values), Mitigation (sampling) | premium, qpu | beta |  |
| [Application functions](reference/ibm-quantum-functions-catalog.md#application-functions) | IBM Quantum Functions Catalog | QAOA, Multi-objective, SQD, Trotter, Other | Scale |  | premium, qpu | beta |  |
| [Write, upload and run a workflow](reference/managed-workflows.md#write-upload-and-run-a-workflow) | Managed Workflows | General, SQD, Verifiable sampling | Scale |  | serverless, premium | experimental |  |
| [Fleets runner and compute profiles](reference/managed-workflows.md#fleets-runner-and-compute-profiles) | Managed Workflows | General, SQD | Scale |  | serverless, premium | experimental |  |
| [Data and files](reference/managed-workflows.md#data-and-files) | Managed Workflows | General | Scale |  | serverless | experimental |  |
| [External dependencies](reference/managed-workflows.md#external-dependencies) | Managed Workflows | General | Scale |  | serverless | experimental |  |
| [Bring your own wheels](reference/managed-workflows.md#bring-your-own-wheels) | Managed Workflows | General | Scale |  | serverless | experimental |  |
| [Qiskit patterns in a workflow](reference/managed-workflows.md#qiskit-patterns-in-a-workflow) | Managed Workflows | General | Map, Optimize, Execute, Post-process |  | serverless | stable |  |
| [Qiskit Function templates](reference/managed-workflows.md#qiskit-function-templates) | Managed Workflows | SQD, Trotter | Scale |  | serverless, premium | stable |  |
| [Qiskit Aer](reference/simulators.md#qiskit-aer) | Simulators | General, Thermal, Error detection, Verifiable sampling | Develop |  | local | stable | 2026-10 |
| [Exact and tensor-network references](reference/simulators.md#exact-and-tensor-network-references) | Simulators | Trotter, Floquet, Echoes, Thermal, Error detection, Verifiable sampling | Develop |  | local | stable | 2026-08 |
| [Quantum optimization best practices (qopt-best-practices)](reference/ecosystem.md#quantum-optimization-best-practices-qopt-best-practices) | Ecosystem | QAOA, Multi-objective | Optimize |  | local | stable | 2026-06 |
| [QAOA training pipeline](reference/ecosystem.md#qaoa-training-pipeline) | Ecosystem | QAOA, Multi-objective | Map |  | local, qpu | stable | 2026-10 |
| [Classical solvers and multi-objective tools](reference/ecosystem.md#classical-solvers-and-multi-objective-tools) | Ecosystem | Multi-objective, QAOA | Post-process |  | local | stable |  |
| [Chemistry: PySCF and ffsim](reference/ecosystem.md#chemistry-pyscf-and-ffsim) | Ecosystem | SQD | Map |  | local | stable | 2026-04 |
| [Reference circuits: the Quantum Advantage Tracker](reference/ecosystem.md#reference-circuits-the-quantum-advantage-tracker) | Ecosystem | Verifiable sampling, Echoes, Floquet, SQD, Trotter | Map |  | local | stable |  |
| [Layout ranking with mapomatic](reference/ecosystem.md#layout-ranking-with-mapomatic) | Ecosystem | General, Floquet, Verifiable sampling | Optimize |  | local, account | stable |  |
| [Device benchmarking (qiskit-device-benchmarking)](reference/ecosystem.md#device-benchmarking-qiskit-device-benchmarking) | Ecosystem | General, Verifiable sampling, Echoes | Optimize | Characterization | qpu | experimental | 2026-08 |
| [Qiskit MCP servers](reference/ai-tooling.md#qiskit-mcp-servers) | AI tooling | General | Develop |  | local, account, qpu | alpha | 2026-10 |
| [Qiskit AI skills](reference/ai-tooling.md#qiskit-ai-skills) | AI tooling | General | Develop |  | local | stable | 2026-09 |
| [Qiskit Code Assistant models](reference/ai-tooling.md#qiskit-code-assistant-models) | AI tooling | General | Develop |  | local | stable | 2026-05 |
| [Documentation for agents](reference/ai-tooling.md#documentation-for-agents) | AI tooling | General | Develop |  | local | stable |  |
| [QPUs at a glance](reference/hardware.md#qpus-at-a-glance) | IBM QPUs | General | Execute |  | account | stable | 2026-09 |
| [Calibration data and layer fidelity](reference/hardware.md#calibration-data-and-layer-fidelity) | IBM QPUs | General, QAOA, Multi-objective, Floquet, Echoes | Optimize |  | account | stable |  |
| [Grading with qc-grader](guides/getting-started.md#grading-with-qc-grader) | Getting started | General | Develop |  | account | stable | 2026-09 |
| [Choosing error detection and mitigation](guides/techniques.md#choosing-error-detection-and-mitigation) | Techniques | General | Post-process | Post-selection, Mitigation (expectation values), Mitigation (sampling) | local | stable |  |
| [The Iceberg code](guides/techniques.md#the-iceberg-code) | Techniques | Error detection | Map, Post-process | Post-selection | local, qpu | stable |  |
| [CVaR of sampled results](guides/techniques.md#cvar-of-sampled-results) | Techniques | QAOA, LCU | Post-process | Mitigation (sampling) | local | stable |  |
| [Sampling a linear combination of unitaries](guides/techniques.md#sampling-a-linear-combination-of-unitaries) | Techniques | LCU | Map |  | local, qpu | stable |  |
| [Fidelity of a Clifford circuit from sampled stabilizers](guides/techniques.md#fidelity-of-a-clifford-circuit-from-sampled-stabilizers) | Techniques | Verifiable sampling, Error detection | Post-process |  | local, qpu | stable |  |

## By workflow step

The four steps of a Qiskit pattern (map, optimize, execute, post-process), plus scaling out, and developing and debugging.

### Map the problem

Turn the problem into circuits and operators: ansatzes, encodings, codes and benchmark circuits.

- **Qiskit SDK:** [Circuit library](reference/qiskit-sdk.md#circuit-library)
- **Compute client:** [Dynamic circuits and resets](reference/ibm-quantum-compute-client.md#dynamic-circuits-and-resets)
- **Qiskit addons:** [Optimization mapper](reference/qiskit-addons.md#optimization-mapper-qiskit-addon-opt-mapper) · [Qiskit Fermions](reference/qiskit-addons.md#qiskit-fermions-qiskit-fermions) · [Spacetime Pauli checks](reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation)
- **Managed Workflows:** [Qiskit patterns in a workflow](reference/managed-workflows.md#qiskit-patterns-in-a-workflow)
- **Ecosystem:** [QAOA training pipeline](reference/ecosystem.md#qaoa-training-pipeline) · [Chemistry: PySCF and ffsim](reference/ecosystem.md#chemistry-pyscf-and-ffsim) · [Reference circuits: the Quantum Advantage Tracker](reference/ecosystem.md#reference-circuits-the-quantum-advantage-tracker)
- **Techniques:** [The Iceberg code](guides/techniques.md#the-iceberg-code) · [Sampling a linear combination of unitaries](guides/techniques.md#sampling-a-linear-combination-of-unitaries)

### Optimize for the hardware

Transpile, choose qubits, use native gates, shrink circuits and add error suppression.

- **Qiskit SDK:** [Transpiler: preset and custom pass managers](reference/qiskit-sdk.md#transpiler-preset-and-custom-pass-managers) · [AI-powered transpiler passes](reference/qiskit-sdk.md#ai-powered-transpiler-passes-qiskit-ibm-transpiler) · [Swap strategies for commuting gates](reference/qiskit-sdk.md#swap-strategies-for-commuting-gates) · [Qubit selection and layout](reference/qiskit-sdk.md#qubit-selection-and-layout) · [Dynamical decoupling passes](reference/qiskit-sdk.md#dynamical-decoupling-passes)
- **Compute client:** [Fractional gates](reference/ibm-quantum-compute-client.md#fractional-gates) · [Specially calibrated instructions and calibration sets](reference/ibm-quantum-compute-client.md#specially-calibrated-instructions-and-calibration-sets)
- **Samplomatic and mitigation:** [Pauli twirling with Samplomatic](reference/noise-management.md#pauli-twirling-with-samplomatic)
- **Qiskit addons:** [Operator backpropagation](reference/qiskit-addons.md#operator-backpropagation-qiskit-addon-obp) · [AQC-Tensor](reference/qiskit-addons.md#aqc-tensor-qiskit-addon-aqc-tensor) · [Circuit cutting](reference/qiskit-addons.md#circuit-cutting-qiskit-addon-cutting)
- **Managed Workflows:** [Qiskit patterns in a workflow](reference/managed-workflows.md#qiskit-patterns-in-a-workflow)
- **Ecosystem:** [Quantum optimization best practices](reference/ecosystem.md#quantum-optimization-best-practices-qopt-best-practices) · [Layout ranking with mapomatic](reference/ecosystem.md#layout-ranking-with-mapomatic) · [Device benchmarking](reference/ecosystem.md#device-benchmarking-qiskit-device-benchmarking)
- **IBM QPUs:** [Calibration data and layer fidelity](reference/hardware.md#calibration-data-and-layer-fidelity)

### Execute

Run on hardware or simulators: primitives, execution modes, noise learning, QPU-time budgeting and the QPUs themselves.

- **Compute client:** [Dynamical decoupling options](reference/ibm-quantum-compute-client.md#dynamical-decoupling-options) · [Executor and QuantumProgram](reference/ibm-quantum-compute-client.md#executor-and-quantumprogram) · [Sampler and Estimator: client-side primitives](reference/ibm-quantum-compute-client.md#sampler-and-estimator-client-side-primitives) · [Execution modes: job, batch and session](reference/ibm-quantum-compute-client.md#execution-modes-job-batch-and-session) · [Budgeting QPU time](reference/ibm-quantum-compute-client.md#budgeting-qpu-time) · [Twirling options in the primitives](reference/ibm-quantum-compute-client.md#twirling-options-in-the-primitives) · [NoiseLearnerV3](reference/ibm-quantum-compute-client.md#noiselearnerv3)
- **Samplomatic and mitigation:** [Custom noise learning](reference/noise-management.md#custom-noise-learning-qiskit-noise-learning)
- **Managed Workflows:** [Qiskit patterns in a workflow](reference/managed-workflows.md#qiskit-patterns-in-a-workflow)
- **IBM QPUs:** [QPUs at a glance](reference/hardware.md#qpus-at-a-glance)

### Post-process results

Turn shots into answers: expectation values, mitigation, post-selection, diagonalization and extrapolation.

- **Samplomatic and mitigation:** [Post-selection on non-Markovian error checks](reference/noise-management.md#post-selection-on-non-markovian-error-checks) · [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec)
- **Qiskit addons:** [Spacetime Pauli checks](reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice) · [Expectation values from shot data](reference/qiskit-addons.md#expectation-values-from-shot-data) · [Shaded light cones](reference/qiskit-addons.md#shaded-light-cones-qiskit-addon-slc) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) · [Measurement mitigation with M3](reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree) · [Sample-based quantum diagonalization](reference/qiskit-addons.md#sample-based-quantum-diagonalization-qiskit-addon-sqd) · [Multi-product formulas](reference/qiskit-addons.md#multi-product-formulas-qiskit-addon-mpf)
- **Managed Workflows:** [Qiskit patterns in a workflow](reference/managed-workflows.md#qiskit-patterns-in-a-workflow)
- **Ecosystem:** [Classical solvers and multi-objective tools](reference/ecosystem.md#classical-solvers-and-multi-objective-tools)
- **Techniques:** [Choosing error detection and mitigation](guides/techniques.md#choosing-error-detection-and-mitigation) · [The Iceberg code](guides/techniques.md#the-iceberg-code) · [CVaR of sampled results](guides/techniques.md#cvar-of-sampled-results) · [Fidelity of a Clifford circuit from sampled stabilizers](guides/techniques.md#fidelity-of-a-clifford-circuit-from-sampled-stabilizers)

### Scale out

Run classical work next to the QPUs with Managed Workflows, and use ready-made functions from the IBM Quantum Functions Catalog.

- **IBM Quantum Functions Catalog:** [Using the IBM Quantum Functions Catalog](reference/ibm-quantum-functions-catalog.md#using-the-ibm-quantum-functions-catalog) · [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions) · [Application functions](reference/ibm-quantum-functions-catalog.md#application-functions)
- **Managed Workflows:** [Write, upload and run a workflow](reference/managed-workflows.md#write-upload-and-run-a-workflow) · [Fleets runner and compute profiles](reference/managed-workflows.md#fleets-runner-and-compute-profiles) · [Data and files](reference/managed-workflows.md#data-and-files) · [External dependencies](reference/managed-workflows.md#external-dependencies) · [Bring your own wheels](reference/managed-workflows.md#bring-your-own-wheels) · [Qiskit Function templates](reference/managed-workflows.md#qiskit-function-templates)

### Develop and debug

Test locally, simulate, save your work, grade answers and work with AI assistants.

- **Qiskit SDK:** [Saving circuits with QPY](reference/qiskit-sdk.md#saving-circuits-with-qpy)
- **Compute client:** [Saving results and job IDs](reference/ibm-quantum-compute-client.md#saving-results-and-job-ids) · [Fake backends and local testing mode](reference/ibm-quantum-compute-client.md#fake-backends-and-local-testing-mode) · [NEAT: the noisy Estimator analyzer](reference/ibm-quantum-compute-client.md#neat-the-noisy-estimator-analyzer)
- **Simulators:** [Qiskit Aer](reference/simulators.md#qiskit-aer) · [Exact and tensor-network references](reference/simulators.md#exact-and-tensor-network-references)
- **AI tooling:** [Qiskit MCP servers](reference/ai-tooling.md#qiskit-mcp-servers) · [Qiskit AI skills](reference/ai-tooling.md#qiskit-ai-skills) · [Qiskit Code Assistant models](reference/ai-tooling.md#qiskit-code-assistant-models) · [Documentation for agents](reference/ai-tooling.md#documentation-for-agents)
- **Getting started:** [Grading with qc-grader](guides/getting-started.md#grading-with-qc-grader)

## By noise-management layer

Grouped as in IBM's [overview of noise management techniques](https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-overview), with error detection and noise characterization added.

### Error suppression

Reduce errors as the circuit runs: dynamical decoupling, Pauli twirling and shallower circuits.

- **Qiskit SDK:** [Dynamical decoupling passes](reference/qiskit-sdk.md#dynamical-decoupling-passes)
- **Compute client:** [Dynamical decoupling options](reference/ibm-quantum-compute-client.md#dynamical-decoupling-options) · [Twirling options in the primitives](reference/ibm-quantum-compute-client.md#twirling-options-in-the-primitives)
- **Samplomatic and mitigation:** [Pauli twirling with Samplomatic](reference/noise-management.md#pauli-twirling-with-samplomatic)
- **Qiskit addons:** [AQC-Tensor](reference/qiskit-addons.md#aqc-tensor-qiskit-addon-aqc-tensor)
- **IBM Quantum Functions Catalog:** [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions)

### Noise learning and characterization

Measure the noise: learned Pauli-Lindblad models, fresh benchmarks and noise-impact estimates.

- **Compute client:** [NoiseLearnerV3](reference/ibm-quantum-compute-client.md#noiselearnerv3) · [NEAT: the noisy Estimator analyzer](reference/ibm-quantum-compute-client.md#neat-the-noisy-estimator-analyzer)
- **Samplomatic and mitigation:** [Custom noise learning](reference/noise-management.md#custom-noise-learning-qiskit-noise-learning)
- **Ecosystem:** [Device benchmarking](reference/ecosystem.md#device-benchmarking-qiskit-device-benchmarking)

### Error detection and post-selection

Flag shots hit by an error and discard them.

- **Samplomatic and mitigation:** [Post-selection on non-Markovian error checks](reference/noise-management.md#post-selection-on-non-markovian-error-checks)
- **Qiskit addons:** [Spacetime Pauli checks](reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice)
- **Techniques:** [Choosing error detection and mitigation](guides/techniques.md#choosing-error-detection-and-mitigation) · [The Iceberg code](guides/techniques.md#the-iceberg-code)

### Mitigation of expectation values

Correct expectation values after the run: TREX, ZNE, PEA, PEC and the methods that cut their cost.

- **Compute client:** [Sampler and Estimator: client-side primitives](reference/ibm-quantum-compute-client.md#sampler-and-estimator-client-side-primitives)
- **Samplomatic and mitigation:** [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec)
- **Qiskit addons:** [Operator backpropagation](reference/qiskit-addons.md#operator-backpropagation-qiskit-addon-obp) · [Shaded light cones](reference/qiskit-addons.md#shaded-light-cones-qiskit-addon-slc) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation)
- **IBM Quantum Functions Catalog:** [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions)
- **Techniques:** [Choosing error detection and mitigation](guides/techniques.md#choosing-error-detection-and-mitigation)

### Mitigation of sampling results

Correct counts and sampled configurations: readout mitigation and sample-based diagonalization.

- **Qiskit addons:** [Measurement mitigation with M3](reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree) · [Sample-based quantum diagonalization](reference/qiskit-addons.md#sample-based-quantum-diagonalization-qiskit-addon-sqd)
- **IBM Quantum Functions Catalog:** [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions)
- **Techniques:** [Choosing error detection and mitigation](guides/techniques.md#choosing-error-detection-and-mitigation) · [CVaR of sampled results](guides/techniques.md#cvar-of-sampled-results)

## New and changing

Entries whose newest listed change is from June 2026 or later (the cheat sheet was checked in November 2026). Each entry says what changed and when.

- **October 2026**
  - [Saving circuits with QPY](reference/qiskit-sdk.md#saving-circuits-with-qpy) (Qiskit SDK)
  - [Custom noise learning](reference/noise-management.md#custom-noise-learning-qiskit-noise-learning) (Samplomatic and mitigation)
  - [Qiskit Fermions](reference/qiskit-addons.md#qiskit-fermions-qiskit-fermions) (Qiskit addons)
  - [Spacetime Pauli checks](reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice) (Qiskit addons)
  - [Shaded light cones](reference/qiskit-addons.md#shaded-light-cones-qiskit-addon-slc) (Qiskit addons)
  - [Sample-based quantum diagonalization](reference/qiskit-addons.md#sample-based-quantum-diagonalization-qiskit-addon-sqd) (Qiskit addons)
  - [Qiskit Aer](reference/simulators.md#qiskit-aer) (Simulators)
  - [QAOA training pipeline](reference/ecosystem.md#qaoa-training-pipeline) (Ecosystem)
  - [Qiskit MCP servers](reference/ai-tooling.md#qiskit-mcp-servers) (AI tooling)
- **September 2026**
  - [Specially calibrated instructions and calibration sets](reference/ibm-quantum-compute-client.md#specially-calibrated-instructions-and-calibration-sets) (Compute client)
  - [Dynamic circuits and resets](reference/ibm-quantum-compute-client.md#dynamic-circuits-and-resets) (Compute client)
  - [Dynamical decoupling options](reference/ibm-quantum-compute-client.md#dynamical-decoupling-options) (Compute client)
  - [Executor and QuantumProgram](reference/ibm-quantum-compute-client.md#executor-and-quantumprogram) (Compute client)
  - [Sampler and Estimator: client-side primitives](reference/ibm-quantum-compute-client.md#sampler-and-estimator-client-side-primitives) (Compute client)
  - [Budgeting QPU time](reference/ibm-quantum-compute-client.md#budgeting-qpu-time) (Compute client)
  - [Saving results and job IDs](reference/ibm-quantum-compute-client.md#saving-results-and-job-ids) (Compute client)
  - [Twirling options in the primitives](reference/ibm-quantum-compute-client.md#twirling-options-in-the-primitives) (Compute client)
  - [NoiseLearnerV3](reference/ibm-quantum-compute-client.md#noiselearnerv3) (Compute client)
  - [Fake backends and local testing mode](reference/ibm-quantum-compute-client.md#fake-backends-and-local-testing-mode) (Compute client)
  - [NEAT: the noisy Estimator analyzer](reference/ibm-quantum-compute-client.md#neat-the-noisy-estimator-analyzer) (Compute client)
  - [Post-selection on non-Markovian error checks](reference/noise-management.md#post-selection-on-non-markovian-error-checks) (Samplomatic and mitigation)
  - [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) (Samplomatic and mitigation)
  - [Expectation values from shot data](reference/qiskit-addons.md#expectation-values-from-shot-data) (Qiskit addons)
  - [Qiskit AI skills](reference/ai-tooling.md#qiskit-ai-skills) (AI tooling)
  - [QPUs at a glance](reference/hardware.md#qpus-at-a-glance) (IBM QPUs)
  - [Grading with qc-grader](guides/getting-started.md#grading-with-qc-grader) (Getting started)
- **August 2026**
  - [Pauli twirling with Samplomatic](reference/noise-management.md#pauli-twirling-with-samplomatic) (Samplomatic and mitigation)
  - [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) (Qiskit addons)
  - [Exact and tensor-network references](reference/simulators.md#exact-and-tensor-network-references) (Simulators)
  - [Device benchmarking](reference/ecosystem.md#device-benchmarking-qiskit-device-benchmarking) (Ecosystem)
- **July 2026**
  - [Transpiler: preset and custom pass managers](reference/qiskit-sdk.md#transpiler-preset-and-custom-pass-managers) (Qiskit SDK)
  - [Swap strategies for commuting gates](reference/qiskit-sdk.md#swap-strategies-for-commuting-gates) (Qiskit SDK)
  - [Fractional gates](reference/ibm-quantum-compute-client.md#fractional-gates) (Compute client)
  - [AQC-Tensor](reference/qiskit-addons.md#aqc-tensor-qiskit-addon-aqc-tensor) (Qiskit addons)
- **June 2026**
  - [Qubit selection and layout](reference/qiskit-sdk.md#qubit-selection-and-layout) (Qiskit SDK)
  - [Quantum optimization best practices](reference/ecosystem.md#quantum-optimization-best-practices-qopt-best-practices) (Ecosystem)

### Not yet stable

Entries whose status is not stable. Check their interfaces against the installed version before you rely on them.

- **Beta:** [AI-powered transpiler passes](reference/qiskit-sdk.md#ai-powered-transpiler-passes-qiskit-ibm-transpiler) · [Executor and QuantumProgram](reference/ibm-quantum-compute-client.md#executor-and-quantumprogram) · [NoiseLearnerV3](reference/ibm-quantum-compute-client.md#noiselearnerv3) · [Pauli twirling with Samplomatic](reference/noise-management.md#pauli-twirling-with-samplomatic) · [Post-selection on non-Markovian error checks](reference/noise-management.md#post-selection-on-non-markovian-error-checks) · [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) · [Qiskit Fermions](reference/qiskit-addons.md#qiskit-fermions-qiskit-fermions) · [Using the IBM Quantum Functions Catalog](reference/ibm-quantum-functions-catalog.md#using-the-ibm-quantum-functions-catalog) · [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions) · [Application functions](reference/ibm-quantum-functions-catalog.md#application-functions)
- **Alpha:** [Custom noise learning](reference/noise-management.md#custom-noise-learning-qiskit-noise-learning) · [Qiskit MCP servers](reference/ai-tooling.md#qiskit-mcp-servers)
- **Experimental:** [Fractional gates](reference/ibm-quantum-compute-client.md#fractional-gates) · [Write, upload and run a workflow](reference/managed-workflows.md#write-upload-and-run-a-workflow) · [Fleets runner and compute profiles](reference/managed-workflows.md#fleets-runner-and-compute-profiles) · [Data and files](reference/managed-workflows.md#data-and-files) · [External dependencies](reference/managed-workflows.md#external-dependencies) · [Bring your own wheels](reference/managed-workflows.md#bring-your-own-wheels) · [Device benchmarking](reference/ecosystem.md#device-benchmarking-qiskit-device-benchmarking)

## By tool type

One reference document per tool type, each owned by one team; the guides hold what spans tools. [Document owners](#document-owners) lists who keeps each document current.

**Reference documents**

- **[Qiskit SDK](reference/qiskit-sdk.md):** [Circuit library](reference/qiskit-sdk.md#circuit-library) · [Transpiler: preset and custom pass managers](reference/qiskit-sdk.md#transpiler-preset-and-custom-pass-managers) · [AI-powered transpiler passes](reference/qiskit-sdk.md#ai-powered-transpiler-passes-qiskit-ibm-transpiler) · [Swap strategies for commuting gates](reference/qiskit-sdk.md#swap-strategies-for-commuting-gates) · [Qubit selection and layout](reference/qiskit-sdk.md#qubit-selection-and-layout) · [Dynamical decoupling passes](reference/qiskit-sdk.md#dynamical-decoupling-passes) · [Saving circuits with QPY](reference/qiskit-sdk.md#saving-circuits-with-qpy)
- **[IBM Quantum Compute client (qiskit-ibm-runtime)](reference/ibm-quantum-compute-client.md):** [Fractional gates](reference/ibm-quantum-compute-client.md#fractional-gates) · [Specially calibrated instructions and calibration sets](reference/ibm-quantum-compute-client.md#specially-calibrated-instructions-and-calibration-sets) · [Dynamic circuits and resets](reference/ibm-quantum-compute-client.md#dynamic-circuits-and-resets) · [Dynamical decoupling options](reference/ibm-quantum-compute-client.md#dynamical-decoupling-options) · [Executor and QuantumProgram](reference/ibm-quantum-compute-client.md#executor-and-quantumprogram) · [Sampler and Estimator: client-side primitives](reference/ibm-quantum-compute-client.md#sampler-and-estimator-client-side-primitives) · [Execution modes: job, batch and session](reference/ibm-quantum-compute-client.md#execution-modes-job-batch-and-session) · [Budgeting QPU time](reference/ibm-quantum-compute-client.md#budgeting-qpu-time) · [Saving results and job IDs](reference/ibm-quantum-compute-client.md#saving-results-and-job-ids) · [Twirling options in the primitives](reference/ibm-quantum-compute-client.md#twirling-options-in-the-primitives) · [NoiseLearnerV3](reference/ibm-quantum-compute-client.md#noiselearnerv3) · [Fake backends and local testing mode](reference/ibm-quantum-compute-client.md#fake-backends-and-local-testing-mode) · [NEAT: the noisy Estimator analyzer](reference/ibm-quantum-compute-client.md#neat-the-noisy-estimator-analyzer)
- **[Samplomatic and qiskit-mitigation](reference/noise-management.md):** [Pauli twirling with Samplomatic](reference/noise-management.md#pauli-twirling-with-samplomatic) · [Post-selection on non-Markovian error checks](reference/noise-management.md#post-selection-on-non-markovian-error-checks) · [qiskit-mitigation: TREX, ZNE, PEA and PEC](reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) · [Custom noise learning](reference/noise-management.md#custom-noise-learning-qiskit-noise-learning)
- **[Qiskit addons](reference/qiskit-addons.md):** [Optimization mapper](reference/qiskit-addons.md#optimization-mapper-qiskit-addon-opt-mapper) · [Qiskit Fermions](reference/qiskit-addons.md#qiskit-fermions-qiskit-fermions) · [Operator backpropagation](reference/qiskit-addons.md#operator-backpropagation-qiskit-addon-obp) · [AQC-Tensor](reference/qiskit-addons.md#aqc-tensor-qiskit-addon-aqc-tensor) · [Circuit cutting](reference/qiskit-addons.md#circuit-cutting-qiskit-addon-cutting) · [Spacetime Pauli checks](reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice) · [Expectation values from shot data](reference/qiskit-addons.md#expectation-values-from-shot-data) · [Shaded light cones](reference/qiskit-addons.md#shaded-light-cones-qiskit-addon-slc) · [Propagated noise absorption and Pauli propagation](reference/qiskit-addons.md#propagated-noise-absorption-and-pauli-propagation) · [Measurement mitigation with M3](reference/qiskit-addons.md#measurement-mitigation-with-m3-mthree) · [Sample-based quantum diagonalization](reference/qiskit-addons.md#sample-based-quantum-diagonalization-qiskit-addon-sqd) · [Multi-product formulas](reference/qiskit-addons.md#multi-product-formulas-qiskit-addon-mpf)
- **[IBM Quantum Functions Catalog](reference/ibm-quantum-functions-catalog.md):** [Using the IBM Quantum Functions Catalog](reference/ibm-quantum-functions-catalog.md#using-the-ibm-quantum-functions-catalog) · [Circuit functions](reference/ibm-quantum-functions-catalog.md#circuit-functions) · [Application functions](reference/ibm-quantum-functions-catalog.md#application-functions)
- **[Managed Workflows](reference/managed-workflows.md):** [Write, upload and run a workflow](reference/managed-workflows.md#write-upload-and-run-a-workflow) · [Fleets runner and compute profiles](reference/managed-workflows.md#fleets-runner-and-compute-profiles) · [Data and files](reference/managed-workflows.md#data-and-files) · [External dependencies](reference/managed-workflows.md#external-dependencies) · [Bring your own wheels](reference/managed-workflows.md#bring-your-own-wheels) · [Qiskit patterns in a workflow](reference/managed-workflows.md#qiskit-patterns-in-a-workflow) · [Qiskit Function templates](reference/managed-workflows.md#qiskit-function-templates)
- **[Simulators and classical references](reference/simulators.md):** [Qiskit Aer](reference/simulators.md#qiskit-aer) · [Exact and tensor-network references](reference/simulators.md#exact-and-tensor-network-references)
- **[Ecosystem and third-party packages](reference/ecosystem.md):** [Quantum optimization best practices](reference/ecosystem.md#quantum-optimization-best-practices-qopt-best-practices) · [QAOA training pipeline](reference/ecosystem.md#qaoa-training-pipeline) · [Classical solvers and multi-objective tools](reference/ecosystem.md#classical-solvers-and-multi-objective-tools) · [Chemistry: PySCF and ffsim](reference/ecosystem.md#chemistry-pyscf-and-ffsim) · [Reference circuits: the Quantum Advantage Tracker](reference/ecosystem.md#reference-circuits-the-quantum-advantage-tracker) · [Layout ranking with mapomatic](reference/ecosystem.md#layout-ranking-with-mapomatic) · [Device benchmarking](reference/ecosystem.md#device-benchmarking-qiskit-device-benchmarking)
- **[AI-assisted development](reference/ai-tooling.md):** [Qiskit MCP servers](reference/ai-tooling.md#qiskit-mcp-servers) · [Qiskit AI skills](reference/ai-tooling.md#qiskit-ai-skills) · [Qiskit Code Assistant models](reference/ai-tooling.md#qiskit-code-assistant-models) · [Documentation for agents](reference/ai-tooling.md#documentation-for-agents)
- **[IBM QPUs](reference/hardware.md):** [QPUs at a glance](reference/hardware.md#qpus-at-a-glance) · [Calibration data and layer fidelity](reference/hardware.md#calibration-data-and-layer-fidelity)

**Guides**

- **[Getting started](guides/getting-started.md):** [How to use this cheat sheet](guides/getting-started.md#how-to-use-this-cheat-sheet) · [Versions and environment](guides/getting-started.md#versions-and-environment) · [IBM Quantum account](guides/getting-started.md#ibm-quantum-account) · [Setting up an AI assistant](guides/getting-started.md#setting-up-an-ai-assistant) · [Grading with qc-grader](guides/getting-started.md#grading-with-qc-grader)
- **[Techniques](guides/techniques.md):** [Choosing error detection and mitigation](guides/techniques.md#choosing-error-detection-and-mitigation) · [The Iceberg code](guides/techniques.md#the-iceberg-code) · [CVaR of sampled results](guides/techniques.md#cvar-of-sampled-results) · [Sampling a linear combination of unitaries](guides/techniques.md#sampling-a-linear-combination-of-unitaries) · [Fidelity of a Clifford circuit from sampled stabilizers](guides/techniques.md#fidelity-of-a-clifford-circuit-from-sampled-stabilizers)
- **[Testing and tips](guides/testing-and-tips.md):** [API quick reference](guides/testing-and-tips.md#api-quick-reference) · [Local test patterns](guides/testing-and-tips.md#local-test-patterns) · [Practical tips](guides/testing-and-tips.md#practical-tips)
- **[Troubleshooting](guides/troubleshooting.md):** [Installing and importing](guides/troubleshooting.md#installing-and-importing) · [Building and transpiling circuits](guides/troubleshooting.md#building-and-transpiling-circuits) · [Submitting and running jobs](guides/troubleshooting.md#submitting-and-running-jobs) · [Reading results](guides/troubleshooting.md#reading-results) · [Managed Workflows](guides/troubleshooting.md#managed-workflows)

## Document owners

Proposed owners until the teams confirm them. [Maintaining this cheat sheet](CONTRIBUTING.md) explains how to update a document.

| Document | Owner | Last verified |
|---|---|---|
| [Qiskit SDK](reference/qiskit-sdk.md) | Qiskit SDK team | 2026-10-20 |
| [IBM Quantum Compute client (qiskit-ibm-runtime)](reference/ibm-quantum-compute-client.md) | IBM Quantum Compute client team | 2026-10-20 |
| [Samplomatic and qiskit-mitigation](reference/noise-management.md) | Noise-management team | 2026-10-20 |
| [Qiskit addons](reference/qiskit-addons.md) | Qiskit addons team | 2026-10-20 |
| [IBM Quantum Functions Catalog](reference/ibm-quantum-functions-catalog.md) | Qiskit Functions team | 2026-11-02 |
| [Managed Workflows](reference/managed-workflows.md) | Qiskit Functions team | 2026-11-02 |
| [Simulators and classical references](reference/simulators.md) | Qiskit Aer team | 2026-10-20 |
| [Ecosystem and third-party packages](reference/ecosystem.md) | QDC organizers | 2026-10-20 |
| [AI-assisted development](reference/ai-tooling.md) | Qiskit developer experience team | 2026-10-20 |
| [IBM QPUs](reference/hardware.md) | IBM Quantum systems team | 2026-10-20 |
| [Getting started](guides/getting-started.md) | QDC organizers | 2026-10-20 |
| [Techniques](guides/techniques.md) | QDC organizers | 2026-10-20 |
| [Testing and tips](guides/testing-and-tips.md) | QDC organizers | 2026-10-20 |
| [Troubleshooting](guides/troubleshooting.md) | QDC organizers | 2026-10-20 |
<!-- END GENERATED: views -->
