---
title: Testing and tips
short_title: Testing and tips
tool_type: guide
owner: QDC organizers
last_verified: 2026-10-20
---

# Testing and tips

A checked list of imports, offline test patterns that use no QPU time, and practical advice for hardware runs.

## API quick reference

Every import below was run on Python 3.12 with one set of versions: Qiskit 2.5.2, `qiskit-ibm-runtime` 0.50.0, Samplomatic 0.21.0, `qiskit-mitigation` 0.1.1 and the other versions in [Versions and environment](getting-started.md#versions-and-environment). A notebook that pins an older Qiskit or `qiskit-ibm-runtime` release may lack some of them, so check its requirements file before you copy an import. The comment after each line names the entry that covers it. A few imports could not be checked this way; they are listed, commented out, at the end.

```python
# Qiskit SDK 2.5.2
from qiskit import QuantumCircuit, qpy, qasm2                                    # Saving circuits with QPY, Reference circuits: the Quantum Advantage Tracker (OpenQASM 2)
from qiskit.circuit.library import RZZGate, PauliEvolutionGate, qaoa_ansatz      # Circuit library
from qiskit.circuit.classical import expr                                        # Dynamic circuits and resets
from qiskit.transpiler import generate_preset_pass_manager, PassManager          # Transpiler: preset and custom pass managers
from qiskit.transpiler.basepasses import TransformationPass                      # Transpiler: preset and custom pass managers
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling  # Dynamical decoupling passes
from qiskit.transpiler.passes import VF2PostLayout                               # Qubit selection and layout
from qiskit.transpiler.passes import Commuting2qGateRouter, ElidePermutations    # Swap strategies for commuting gates
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy  # Swap strategies for commuting gates
from qiskit.quantum_info import SparsePauliOp, SparseObservable, Statevector, DensityMatrix, Clifford, StabilizerState  # Exact and tensor-network references
from qiskit.primitives import BitArray, StatevectorSampler, StatevectorEstimator  # Expectation values from shot data, Exact and tensor-network references
from qiskit.result import sampled_expectation_value                              # Expectation values from shot data
from qiskit.providers.fake_provider import GenericBackendV2                      # Fake backends and local testing mode
from qiskit.visualization import plot_gate_map, plot_error_map, plot_circuit_layout  # Qubit selection and layout

# qiskit-ibm-runtime 0.50.0 (the IBM Quantum Compute client)
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, Session              # Execution modes: job, batch and session
from qiskit_ibm_runtime import Executor, QuantumProgram                          # Executor and QuantumProgram
from qiskit_ibm_runtime.executor_sampler import Sampler                          # Sampler and Estimator: client-side primitives (client-side)
from qiskit_ibm_runtime.executor_estimator import Estimator                      # Sampler and Estimator: client-side primitives (client-side)
from qiskit_ibm_runtime import SamplerV2, EstimatorV2                            # Sampler and Estimator: client-side primitives (server-side, deprecated)
from qiskit_ibm_runtime import NoiseLearnerV3                                    # NoiseLearnerV3
from qiskit_ibm_runtime.results import NoiseLearnerV3Results                     # NoiseLearnerV3
from qiskit_ibm_runtime.options_models import SamplerOptions, EstimatorOptions, ExecutorOptions  # Executor and QuantumProgram, Sampler and Estimator: client-side primitives
from qiskit_ibm_runtime.options_models import NoiseLearnerV3Options, TwirlingOptions  # NoiseLearnerV3, Twirling options in the primitives
from qiskit_ibm_runtime.circuit.library import MidCircuitMeasure, MidCircuitReset, MeasureReset, XSlowGate  # Specially calibrated instructions and calibration sets
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle, ConvertISAToClifford  # Specially calibrated instructions and calibration sets, Fake backends and local testing mode
from qiskit_ibm_runtime.transpiler.passes import ConvertToMidCircuitMeasure, ConvertToMidCircuitResetAndMeasure  # Specially calibrated instructions and calibration sets
from qiskit_ibm_runtime.fake_provider import FakeBoston, FakeKingston, FakeMiami, FakeNighthawk  # Fake backends and local testing mode
from qiskit_ibm_runtime.debug_tools import Neat                                 # NEAT: the noisy Estimator analyzer
from qiskit_ibm_runtime import RuntimeEncoder, RuntimeDecoder                    # Saving results and job IDs

# Samplomatic 0.21.0
from samplomatic import build, Twirl, InjectNoise, Tag, ChangeBasis              # Pauli twirling with Samplomatic
from samplomatic.transpiler import generate_boxing_pass_manager                  # Pauli twirling with Samplomatic
from samplomatic.utils import find_unique_box_instructions, get_annotation       # Pauli twirling with Samplomatic

# qiskit-mitigation 0.1.1
from qiskit_mitigation import GateFolding, PEA, PEC, TREX, MitigationTask, find_combined_unique_layers  # qiskit-mitigation: TREX, ZNE, PEA and PEC
from qiskit_mitigation.utils import executor_expectation_values, load_tasks_from_result  # qiskit-mitigation: TREX, ZNE, PEA and PEC, Expectation values from shot data
from qiskit_mitigation.postselection import PostSelector                         # Post-selection on non-Markovian error checks
from qiskit_mitigation.postselection.passes import AddPostCircuitNonMarkovianErrorChecks  # Post-selection on non-Markovian error checks
from qiskit_mitigation.postselection.passes import AddSpectatorPostCircuitNonMarkovianErrorChecks  # Post-selection on non-Markovian error checks

# qiskit-addon-utils 0.4.0 (imported as modules: some names repeat those in qiskit-mitigation)
from qiskit_addon_utils import exp_vals          # exp_vals.executor_expectation_values, exp_vals.get_measurement_bases (Expectation values from shot data)
from qiskit_addon_utils.noise_management import bit_flip_checks  # bit_flip_checks.PostSelector (Post-selection on non-Markovian error checks)
from qiskit_addon_utils.slicing import slice_by_gate_types, combine_slices       # Operator backpropagation

# Qiskit Aer 0.17.2
from qiskit_aer import AerSimulator                                              # Qiskit Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError        # Qiskit Aer
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2                      # Qiskit Aer

# qiskit-paulice 0.2.1 (stabilizers=, dope() and estimate_fault_rates() need GitHub main; see Spacetime Pauli checks)
from qiskit_paulice import add_pauli_checks, CheckedCircuit                      # Spacetime Pauli checks
from qiskit_paulice.layout import get_check_qubits                               # Spacetime Pauli checks
from qiskit_paulice.noise_models import NoiseModel as PauliceNoiseModel          # Spacetime Pauli checks

# qiskit-addon-slc at GitHub commit 7156757, qiskit-addon-pna 0.2.0
from qiskit_addon_slc.bounds import LightCone, compute_bounds, compute_forward_bounds, compute_local_scales, merge_bounds  # Shaded light cones
from qiskit_addon_slc.utils import generate_noise_model_paulis                   # Shaded light cones
from qiskit_addon_pna import generate_noise_mitigating_observable                # Propagated noise absorption and Pauli propagation

# qiskit-addon-obp 0.3.0, qiskit-addon-aqc-tensor 0.3.1
from qiskit_addon_obp import backpropagate                                       # Operator backpropagation
from qiskit_addon_obp.utils.simplify import OperatorBudget                       # Operator backpropagation
from qiskit_addon_aqc_tensor.ansatz_generation import parametrize_circuit        # AQC-Tensor
from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity              # AQC-Tensor

# mthree 3.0.0, mapomatic 0.14.0
from mthree import M3Mitigation                                                  # Measurement mitigation with M3
from mthree.utils import final_measurement_mapping                               # Measurement mitigation with M3
import mapomatic as mm      # mm.matching_layouts, mm.evaluate_layouts           # Layout ranking with mapomatic

# qiskit-addon-opt-mapper 0.1.0
from qiskit_addon_opt_mapper import OptimizationProblem                          # Optimization mapper
from qiskit_addon_opt_mapper.translators import from_docplex_mp, to_ising        # Optimization mapper
from qiskit_addon_opt_mapper.converters import EqualityToPenalty, OptimizationProblemToQubo  # Optimization mapper

# qiskit-addon-sqd 0.14.0, ffsim, quimb 1.15
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch  # Sample-based quantum diagonalization
from qiskit_addon_sqd.trim import TrimPolicy                                     # Sample-based quantum diagonalization
from qiskit_addon_sqd.subsampling import partition_subsample                     # Sample-based quantum diagonalization
import ffsim   # ffsim.UCJOpSpinBalanced, ffsim.qiskit.PrepareHartreeFockJW, ffsim.qiskit.generate_lucj_pass_manager (Chemistry: PySCF and ffsim)
import quimb.tensor as qtn  # qtn.CircuitMPS                                     # Exact and tensor-network references

# qiskit-ibm-catalog 0.22.0 and qiskit-serverless 0.38.0
from qiskit_ibm_catalog import ManagedWorkflowClient, ManagedWorkflow, FunctionsCatalog  # Write, upload and run a workflow, Using the IBM Quantum Functions Catalog
from qiskit_serverless import get_arguments, get_runtime_service, save_result    # Write, upload and run a workflow (inside a managed workflow)

# Not import-checked here:
# from qc_grader.challenges.<module your notebook names> import ...  # Grading with qc-grader: the QDC 2026 graders are not in any PyPI release as of 20 Oct 2026
# Checked in a second environment with the same Qiskit 2.5.2 (these packages are not installed with the set above):
# from qiskit_addon_cutting import partition_problem, generate_cutting_experiments, reconstruct_expectation_values  # Circuit cutting
# from qiskit_addon_mpf.static import setup_static_lse  # Multi-product formulas
# qiskit-ibm-transpiler 0.18.0 does not import on Qiskit 2.5 or later; see AI-powered transpiler passes
```

## Local test patterns

None of these snippets uses QPU time, and all but the last run without an IBM Quantum account. Each was run with the versions above.

### Sample bitstrings with the client-side Sampler on a fake backend

```python
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeBoston
from qiskit_ibm_runtime.executor_sampler import Sampler

backend = FakeBoston()  # offline snapshot of ibm_boston
pm = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=7)
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
isa = pm.run(qc)

sampler = Sampler(mode=backend)
sampler.options.twirling.enable_measure = True
pub_result = sampler.run([isa], shots=1000).result()[0]
counts = pub_result.data.meas.get_counts()  # 1,024 shots: "auto" twirling rounds the count up
```

### Run the same circuit through `Executor`

This continues from the snippet above.

```python
from qiskit_ibm_runtime import Executor, QuantumProgram

program = QuantumProgram(shots=500)
program.append_circuit_item(isa)  # an ISA circuit with measurements
result = Executor(mode=backend).run(program).result()
bits = result[0]["meas"]  # NumPy bool array of shape (shots, clbits), one entry per classical register
duration_s = isa.estimate_duration(backend.target, unit="s")  # one execution, without the repetition delay
```

### Estimate an expectation value with TREX only

The client-side Estimator measures every qubit of the ISA circuit, and Aer cannot simulate a whole 156-qubit device, so test it on a small backend.

```python
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.executor_estimator import Estimator

backend = GenericBackendV2(num_qubits=16, seed=1)
pm = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=7)
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
isa = pm.run(bell)
observable = SparsePauliOp("ZZ").apply_layout(isa.layout)

estimator = Estimator(mode=backend)
estimator.options.resilience_level = 0  # switch on only what you choose
estimator.options.resilience.measure_mitigation = True  # TREX
pub_result = estimator.run([(isa, observable)]).result()[0]
print(pub_result.data.evs, pub_result.data.stds)
```

### Count the executions a client-side Estimator job will submit

The program is built on your machine before any simulation, so this also works for circuits too large to simulate; there, ignore the errors the local simulation prints, and do not call `job.result()`. The backend here has Heron's gate set (`cz` plus fractional `rzz` and `rx`) on a line; `GenericBackendV2`'s default is `cx` on an all-to-all map.

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.executor_estimator import Estimator

backend = GenericBackendV2(num_qubits=16, basis_gates=["rz", "sx", "x", "cz", "rzz", "rx"],
                           coupling_map=[[i, i + 1] for i in range(15)], seed=1)
pm = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=7)
trotter = QuantumCircuit(4)
for _ in range(2):
    trotter.rx(0.3, range(4))
    for q in range(3):
        trotter.rzz(0.4, q, q + 1)
isa = pm.run(trotter)  # keeps the fractional rzz gates
observable = SparsePauliOp("ZZII").apply_layout(isa.layout)

estimator = Estimator(mode=backend)
estimator.options.resilience_level = 0
estimator.options.resilience.measure_mitigation = True  # TREX
estimator.options.twirling.enable_gates = True    # gate twirling
estimator.options.twirling.group = "local_pauli"  # needed with fractional rzz; the default group raises BuildError
job = estimator.run([(isa, observable)])
program = job.inputs  # the QuantumProgram built on your machine
executions = sum(int(np.prod(item.shape)) * program.shots for item in program.items)
print(len(program.items), "items,", program.shots, "shots each,", executions, "executions")
# 2 items (32 twirled circuits and 32 TREX calibration circuits) of 128 shots: 8,192 executions, with or without gate twirling
```

### Retrieve a finished job instead of resubmitting it

This needs your saved account but uses no QPU time. If you did not save the ID, `service.jobs()` lists your jobs, with filters such as `backend_name`, `session_id` and `created_after`; its default `limit` is 10.

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(name="<saved account name>")
job = service.job("<job ID>")
result = job.result()
usage_seconds = job.usage()
```

## Practical tips

### Before you spend QPU time
- Get each step working locally first. Where a notebook gates hardware behind a switch, leave it off until the local work runs, and read what each switch gates before you flip it. Run job cells one at a time, never with Run All.
- Cancel a duplicate or unwanted job as soon as you notice it. Usage is the time the QPU is locked for the job, so a job canceled while it is still queued (`job.status()` reports `QUEUED`) has consumed nothing; a running job reports the time consumed up to the cancellation, including the overhead of preparing the QPU, and canceling never refunds time already used.
- If a job cell ran twice, do not run it again. Find both jobs (`service.jobs(pending=True)` lists the queued and running ones, or use the IDs the cell printed), `job.cancel()` the one that is still queued, keep the other's ID and read it back with `service.job(job_id).result()`. Then read `service.usage()` before any further submission.
- If a notebook already contains job IDs or result files from earlier runs, keep them apart from your own hardware runs so that results are not mixed.
- Estimate a job's QPU time first (see [Budgeting QPU time](../reference/ibm-quantum-compute-client.md#budgeting-qpu-time)). `job.usage()` can be 0 or pending for a while after a job finishes, so read it again before you record it; `service.usage()` returns the usage of the active instance, with a `usage_remaining_seconds` entry when the plan has a limit. A long "In progress" status is not usage: it includes classical pre-processing and waiting for the QPU, and in job and batch mode only the time the QPU is locked for your job counts. Session mode is the exception: its usage is the elapsed time while the session is active, whether or not jobs are running.
- Save every job ID and retrieve finished jobs with `service.job(job_id)` instead of resubmitting. Record the layout with the job ID: recalibration can change a layout search, and results read on the wrong qubits look plausible. `job.properties()` returns the backend properties at the time the job ran, so you can check the calibration a result was taken under.
- Every circuit you send to a Sampler or `Executor` needs measurements: a Sampler PUB circuit must include measurement instructions for each qubit to be sampled, and a circuit without them fails with error 1515 (invalid primitive inputs).
- `shots` applies to every parameter set (row) of a PUB, so a PUB with 10 parameter rows at 10,000 shots returns 100,000 bitstrings. Usage scales the same way: the number of executions is the number of circuits after broadcasting times the shots.

### Hardware and compilation
- Check `backend.target` for dead qubits and couplers: a qubit can carry error 1.0 in the target while `backend.properties()` reports no faulty qubits, so check both ([Calibration data](../reference/hardware.md#calibration-data-and-layer-fidelity)).
- Fake backends are not the live device: their noise models come from system snapshots taken in the past, so transpile and estimate costs against the live `backend.target` when you can. `FakeBackendV2.refresh(service)` replaces the snapshot with the current backend data and overwrites the local snapshot files.
- Fix the layout and choose the optimization level deliberately when your circuit has a hand-built structure (swap networks, Trotter layers, code blocks): higher levels can serialize or reroute it. Pass `seed_transpiler` for reproducible results.
- Running a pass manager on a circuit with free parameters can leave a symbolic global phase with thousands of terms that the service cannot parse. Check `isa.global_phase` before you submit. Bind the parameters before you transpile, or set the phase to a number, which changes no measured result.
- Parameter value arrays follow the order of `circuit.parameters` (for a samplex, `samplex.parameters`). That order is alphabetical, but `ParameterVector` elements stay in index order (`t[2]` before `t[10]`), so build arrays from these lists instead of sorting names yourself.
- Fold gates on the logical circuit with a `barrier` between U, U† and U, then transpile. Without the barriers, optimization levels 1–3 cancel the folded pairs and the noise is not amplified. Folding self-inverse ISA gates such as `cz` after transpilation also works. Inverting a whole ISA circuit adds gates outside the basis, such as `sxdg`.

### Mitigation and analysis
- Mitigation that returns expectation values (TREX, ZNE, PEA, PEC) does not change raw counts, so it cannot improve a result computed from the raw shots; it can still guide your choice of parameters.
- See also the Watch out notes under [Executor](../reference/ibm-quantum-compute-client.md#executor-and-quantumprogram), [qiskit-mitigation](../reference/noise-management.md#qiskit-mitigation-trex-zne-pea-and-pec) and [post-selection](../reference/noise-management.md#post-selection-on-non-markovian-error-checks).

### Environment
- `plot_gate_map`, `plot_error_map`, `plot_circuit_layout` and `plot_coupling_map` all draw through Graphviz's `dot` program. Qiskit checks for the `dot` executable (`qiskit.utils.optionals.HAS_GRAPHVIZ` runs `dot -V`), so install the Graphviz system package (for example `brew install graphviz`, `apt install graphviz` or `conda install graphviz`); the `graphviz` pip package is not what Qiskit checks for.
- Keep one Qiskit version across your notebooks, graders and shared QPY files (see [Saving circuits with QPY](../reference/qiskit-sdk.md#saving-circuits-with-qpy)).

**Docs:**
- [Monitor or cancel a job](https://quantum.cloud.ibm.com/docs/en/guides/monitor-job) (`service.jobs` filters, `service.usage`, `job.properties`, why a job stays "In progress")
- [Workload usage](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time) (what counts as usage, failed and canceled jobs, the estimation formula)
- [Sampler inputs and outputs](https://quantum.cloud.ibm.com/docs/en/guides/sampler-input-output) (measurements in Sampler PUB circuits)
- [Fake provider API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider) (snapshots and `refresh`)
- [Error code registry](https://quantum.cloud.ibm.com/docs/en/errors) (1515)
- [qiskit.utils optional dependency checkers](https://quantum.cloud.ibm.com/docs/en/api/qiskit/utils) (`HAS_GRAPHVIZ`)
