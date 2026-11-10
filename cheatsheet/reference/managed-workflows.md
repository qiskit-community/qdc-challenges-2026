---
title: Managed Workflows
short_title: Managed Workflows
tool_type: managed-workflows
owner: Qiskit Functions team
last_verified: 2026-11-02
---

# Managed Workflows

Managed Workflows run your own Python program on IBM-managed classical compute, so long classical steps such as searches, reference simulations and chemistry pre- and post-processing do not tie up your laptop and keep running after you close it. Ready-made partner functions are covered on the [IBM Quantum Functions Catalog](ibm-quantum-functions-catalog.md) page. For additional information, look at the [Managed Workflows](https://quantum.cloud.ibm.com/docs/en/managed-workflows) documentation pages.

* **Access:** Premium, Flex and On-Prem plan users; the feature is experimental. QPU jobs that a workflow submits use your instance's QPU time like any other job.
* **Packages:** `pip install qiskit-ibm-catalog qiskit-serverless` (Python 3.12 or later). Versions checked: `qiskit-ibm-catalog` 0.22.0 and `qiskit-serverless` 0.38.0. Catalog 0.22 pins `qiskit-serverless~=0.38.0`. On the client you define a workflow with `ManagedWorkflow` and upload, run and manage it with `ManagedWorkflowClient`, both from `qiskit_ibm_catalog`; inside the program, `qiskit_serverless` reads the arguments, provides a runtime service for QPU jobs and saves the result.
* **Code Engine Fleets:** a workflow runs on one of two runners, `"ray"` (the default) or `"fleets"`; this page uses Fleets. Fleets runs workflows on IBM Cloud Code Engine fleets, which start dedicated CPU or GPU workers for each job and remove them when it ends. You choose the worker's size through a compute profile (see [Fleets runner and compute profiles](#fleets-runner-and-compute-profiles)).

## Write, upload and run a workflow

A workflow is a directory of Python files with one entrypoint script. You upload it once as a `ManagedWorkflow` with `runner="fleets"`, then call it like a function from the IBM Quantum Functions Catalog: `client.load(title).run(**arguments)` returns a job whose status, logs and result you poll.

- **When to use:** classical steps are long or memory-hungry, should run next to the QPU, or must keep going after you close your laptop.
- **Tags:** family: general, sqd, verifiable; step: scale; runs: serverless, premium; status: experimental
- **Key APIs:**
  - Client: `ManagedWorkflowClient(channel="ibm_quantum_platform", instance=..., token=...)`, `client.upload(ManagedWorkflow(title=..., entrypoint=..., working_dir=..., runner="fleets", arguments_schema=..., sizes_map=..., default_size=...))`, `client.list()`, `client.load(title)`, `function.run(..., function_size=...)`, `client.jobs()`, `client.job(job_id)`; `client.dependencies_versions()` lists the preinstalled packages
  - Job: `job.job_id`, `job.status()`, `job.logs()`, `job.result()`, `job.error_message()`, `job.cancel()`, `job.runtime_jobs()`, `job.runtime_sessions()`
  - Inside the program: `qiskit_serverless.get_arguments()`, `get_runtime_service()`, `save_result()`, `update_status(Job.<STAGE>)`, `get_function_size()`, `get_compute_profile()`, `get_logger()`
- **Watch out:**
  - Pass `runner="fleets"` on every upload. `ManagedWorkflow` defaults to `runner="ray"`, which is still supported, but `sizes_map`, `default_size` and `function_size` only apply to Fleets.
  - Inside the program, get the backend from `get_runtime_service()`, not from a bare `QiskitRuntimeService()`. The wrapper links every primitive job and session to the parent workflow job, and `job.runtime_jobs()` depends on that link. In long loops, refresh the service before each QPU submission, because its token expires.
  - Arguments and results must be serializable: pass a backend name, not a backend object. `arguments_schema` (a JSON Schema) rejects bad arguments before any compute is used; Qiskit objects arrive encoded as `{"__type__": "QuantumCircuit", "__value__": ...}`, so check only their `__type__`.
  - `distribute_task`, `get` and `compute_profile=` were removed in 0.37. IBM's "Write your first program", "Manage compute and data resources" and "Port code" guides still use them, and the resource guide's "16 CPU cores and 32 GB per program" is outdated. Follow the code here. For parallel classical work, use Python's own `concurrent.futures` or `multiprocessing` inside one job, sized by its compute profile.
  - `working_dir` is limited to 50 MB. Large uploads can hit the client's 30-second request timeout; set the `REQUESTS_TIMEOUT_OVERRIDE` environment variable (in seconds) before you import `qiskit_serverless`.
- **Related:** [Sample-based quantum diagonalization (qiskit-addon-sqd)](qiskit-addons.md#sample-based-quantum-diagonalization-qiskit-addon-sqd) for SQD post-processing, a common workflow step.

```python
# ./source_files/transpile_remote.py  — runs remotely
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_serverless import get_arguments, get_runtime_service, save_result, update_status, Job

args = get_arguments()                                  # the keyword arguments of run()
backend = get_runtime_service().backend(args["backend_name"])
update_status(Job.OPTIMIZING_HARDWARE)                  # shows as "RUNNING: OPTIMIZING_FOR_HARDWARE"
pm = generate_preset_pass_manager(optimization_level=args.get("optimization_level", 3), backend=backend)
save_result({"transpiled_circuits": [pm.run(c) for c in args["circuits"]]})
```

```python
# local notebook
from qiskit_ibm_catalog import ManagedWorkflowClient, ManagedWorkflow

client = ManagedWorkflowClient(channel="ibm_quantum_platform", instance="<instance CRN>", token="<API key>")
client.upload(ManagedWorkflow(
    title="transpile_remote",
    entrypoint="transpile_remote.py",
    working_dir="./source_files/",       # the whole directory is uploaded
    runner="fleets",                     # the default runner is "ray"
    sizes_map={"S": "8x32", "M": "16x128"},
    default_size="S",
))
job = client.load("transpile_remote").run(
    circuits=circuits, backend_name="<QPU name>", optimization_level=3, function_size="M",
)
print(job.status())                      # QUEUED, INITIALIZING, RUNNING[: <stage>], DONE, ERROR or CANCELED
print(job.logs())                        # print() and logger output from the program
result = job.result()                    # waits for the end; the dict passed to save_result()
```

**Docs:**
- [Install](https://quantum.cloud.ibm.com/docs/en/managed-workflows/install-managed-workflows)
- [Deploy your first managed workflow](https://quantum.cloud.ibm.com/docs/en/managed-workflows/deploy-first-managed-workflow)
- [Define inputs and outputs](https://quantum.cloud.ibm.com/docs/en/managed-workflows/managed-workflow-inputs-and-outputs)
- [Run quantum jobs from a workflow](https://quantum.cloud.ibm.com/docs/en/managed-workflows/manage-quantum-classical-workflows)
- [Run a managed workflow](https://quantum.cloud.ibm.com/docs/en/managed-workflows/run-managed-workflow)
- [Write logs and handle errors](https://quantum.cloud.ibm.com/docs/en/managed-workflows/managed-workflow-logs-and-errors)
- [Run a parameter sweep](https://quantum.cloud.ibm.com/docs/en/managed-workflows/run-parameter-sweep)

## Fleets runner and compute profiles

Each workflow job runs on one Code Engine fleet worker, and the worker's hardware is a compute profile named `{vCPU}x{memory GB}`, or `{vCPU}x{memory GB}x{GPU count}{GPU model}` when it has GPUs (`8a100` means eight A100 cards). You never request a profile directly at run time. At upload you declare `sizes_map`, which maps your own size labels (t-shirt sizes such as `"S"`, `"M"`, `"L"`) to profiles, and `default_size`; each run then picks a label with `function_size`, and a run without one uses `default_size`. Inside the job, `get_function_size()` returns the label and `get_compute_profile()` the profile it mapped to.

`ManagedWorkflowClient.compute_profiles()` lists the compute profiles the platform supports (it prints as a table), and `compute_profile(...)` checks that a profile is valid, by ID or by exact resources, and returns it as a `ComputeProfile`. A `ComputeProfile` can go straight into `sizes_map` in place of an ID string. The table below shows the current list.

| Profile | vCPU | Memory (GB) | GPUs |
|---|---:|---:|---|
| `8x32` | 8 | 32 | — |
| `16x128` | 16 | 128 | — |
| `48x768` | 48 | 768 | — |
| `128x1500` | 128 | 1500 | — |
| `24x120x1l40s` | 24 | 120 | 1 × L40S |
| `20x320x2a100` | 20 | 320 | 2 × A100 |
| `80x1200x8a100` | 80 | 1200 | 8 × A100 |
| `160x1792x8h100` | 160 | 1792 | 8 × H100 |

- **When to use:** a step needs more memory or cores than the default worker, or a GPU (for example `gpu4pyscf` in a chemistry workflow), or you want small inputs to run on a cheaper worker than large ones.
- **Tags:** family: general, sqd; step: scale; runs: serverless, premium; status: experimental
- **Key APIs:** `client.compute_profiles()`, `client.compute_profile("<id>")`, `client.compute_profile(cpu=..., memory=..., gpu=...)` (`gpu` omitted means CPU-only); a `ComputeProfile` has `compute_profile_id`, `cpu`, `memory` and `gpu`.
- **Watch out:**
  - Write profile IDs in lowercase (`160X1792X8H100` is rejected) and without an instance-family prefix such as `gx3d-`. Size labels are matched case-insensitively.
  - `default_size` must be one of the labels in your `sizes_map`. Each upload that sends `sizes_map` replaces the stored map; an upload without it leaves the stored map unchanged.
  - A GPU comes only from the profile; there is no separate GPU flag. Code that must also run on a CPU profile should check for a device at run time and fall back to CPU, as the SKQD Fleets template does.
  - An unsupported profile makes the upload or run fail. Check it with `compute_profile()` first: it raises an error that lists the available profiles when nothing matches.
  - The CUDA packages in the Fleets image (`cupy-cuda12x`, `gpu4pyscf-cuda12x`, `cutensor-cu12`) are not reported by `dependencies_versions()`; they are importable only when your job runs on a GPU profile.

```python
print(client.compute_profiles())
# 8x32            (8 vCPU / 32 GB)
# 16x128          (16 vCPU / 128 GB)
# ...             (CPU-only profiles first, then GPU profiles)
# 24x120x1l40s    (24 vCPU / 120 GB / 1l40s)
# ...

SIZES_MAP = {
    "S": client.compute_profile(cpu=16, memory=128),                 # CPU baseline, validated
    "M": client.compute_profile(cpu=24, memory=120, gpu="1l40s"),    # + 1 × L40S
    "L": "20x320x2a100",                                             # an ID string also works
}
client.upload(ManagedWorkflow(title="skqd", entrypoint="skqd_entrypoint.py", working_dir="./source_files/",
                              runner="fleets", sizes_map=SIZES_MAP, default_size="S"))
job = client.load("skqd").run(function_size="M", krylov_dim=5)

# inside the program
from qiskit_serverless import get_function_size, get_compute_profile
print(get_function_size(), get_compute_profile())   # "M" "24x120x1l40s"
```

**Docs:**
- [Choose a compute profile](https://quantum.cloud.ibm.com/docs/en/managed-workflows/managed-workflow-compute-profiles)

## Data and files

Arguments go into `run()`; large data goes in as files. `client.file_upload(file, function)`, `client.files(function)`, `client.file_download(file, function)` and `client.file_delete(file, function)` manage a store scoped to one workflow, and the same calls work inside the program. On Fleets a workflow sees two mounted directories: `/function_user_data`, which holds that store and persists across runs, and `/job_user_data`, which is scoped to one job and holds its arguments, results and logs.

- **When to use:** inputs are too large to pass as arguments (Hamiltonians, sample sets, QPY circuit files), or a later run should reuse an earlier run's output.
- **Tags:** family: general; step: scale; runs: serverless; status: experimental
- **Watch out:**
  - `/data` and `/function_data` exist only on the Ray runner. On Fleets the program runs as a non-root user and those directories are missing, so code that hardcodes them fails; use the file API or `/function_user_data`.
  - Never write into `/job_user_data` yourself; `get_arguments()` and `save_result()` manage it.
  - The file store is flat, without subdirectories, and checks each file's type: archives (`.tar`, `.tar.gz`) are accepted, while `.h5` and `.pkl` files uploaded directly are rejected. Wrap files in a `.tar.gz` and extract it inside the program after `file_download()`.

**Docs:**
- [Manage files across runs](https://quantum.cloud.ibm.com/docs/en/managed-workflows/manage-files-across-managed-workflows)

## External dependencies

The Fleets runtime image already ships Qiskit, `qiskit-ibm-runtime`, `qiskit-serverless` and their dependencies, plus the packages in the table below, so a workflow imports any of them directly. To check what the image ships, call `dependencies_versions()`. For anything else, see [Bring your own wheels](#bring-your-own-wheels).

Preinstalled packages (`docker-images/requirements-dynamic-dependencies.txt` in the `Qiskit/qiskit-serverless` repository, `main` branch):

| Package | Version | Package | Version |
|---|---|---|---|
| `ffsim` | 0.0.70 | `qiskit-addon-obp` | 0.3.0 |
| `h5py` | 3.16.0 | `qiskit-addon-sqd` | 0.13.1 |
| `mergedeep` | 1.3.4 | `qiskit-addon-utils` | 0.2.0 |
| `mpi4py` | 4.1.2 | `qiskit-aer` | 0.17.2 |
| `mthree` | 3.0.0 | `qiskit-fermions` | 0.2.0 |
| `openmpi` | 5.0.10 | `qiskit-paulice` | 0.3 |
| `pybind11` | 3.1.0 | `qrmi[ibm]` | 0.24.4 |
| `pyscf` | 2.13.1 | `cotengrust` | 0.2.0 |
| `qiskit-addon-aqc-tensor[quimb-jax]` | 0.3.1 | | |

- **When to use:** your program imports anything beyond Qiskit and the runtime client.
- **Tags:** family: general; step: scale; runs: serverless; status: experimental
- **Watch out:**
  - Do not use `ManagedWorkflow(dependencies=[...])`: it is deprecated and installs nothing.
  - The arm64 (Apple Silicon) image for running the service locally (`requirements-dynamic-dependencies-arm64.txt`) ships fewer packages: `mergedeep`, `mthree`, `pyscf`, `qiskit-addon-obp`, `qiskit-addon-utils`, `qiskit-aer` and `qiskit-paulice` at the versions above, and `qiskit-addon-sqd` 0.12.1 instead of 0.13.1. It has no `ffsim` or `qiskit-addon-aqc-tensor` (no arm64 wheel), and no `h5py`, `mpi4py`, `openmpi`, `pybind11`, `cotengrust`, `qiskit-fermions` or `qrmi`. Code that runs locally on arm64 can still fail in the cloud, and the other way round.
  - The preinstalled versions lag PyPI. For example, `ffsim.qiskit.generate_lucj_pass_manager` (ffsim 0.0.75) and Trim SQD (`qiskit_addon_sqd.trim`, new in 0.14) are not available from the image; bring a newer version as a wheel, or run that step on your machine and upload the result, for example an ISA circuit as QPY ([Saving circuits with QPY](qiskit-sdk.md#saving-circuits-with-qpy)).
  - The preinstalled `qiskit-paulice` 0.3 includes `add_pauli_checks(..., stabilizers=...)`, `CheckedCircuit.dope()` and `estimate_fault_rates()` ([Spacetime Pauli checks](qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice)).

```python
from qiskit_ibm_catalog import ManagedWorkflowClient

client = ManagedWorkflowClient(channel="ibm_quantum_platform", instance="<instance CRN>", token="<API key>")
print(client.dependencies_versions())    # ["ffsim==0.0.70", "h5py==3.16.0", ...]: what the image ships
```

**Docs:**
- [Declare dependencies](https://quantum.cloud.ibm.com/docs/en/managed-workflows/managed-workflow-dependencies)

## Bring your own wheels

For a package that is not preinstalled, a newer version of a preinstalled one, or a private package, upload a wheel with the function and install it at the start of the entrypoint.

- **When to use:** `dependency_report()` marks a requirement `bring` (not in the image) or `clash` (the image ships a different version); `baked` means it is already there.
- **Tags:** family: general; step: scale; runs: serverless; status: experimental
- **Key APIs:** `ManagedWorkflowClient`: `dependency_report(requirements)`, `add_wheels(function, patterns)` (paths or globs, `.whl` only), `list_wheels(function)`, `remove_wheels(function, names)`. Inside the program: `qiskit_serverless.install_uploaded_wheels()`.
- **Watch out:**
  - Upload the function before you call `add_wheels()`, and call `install_uploaded_wheels()` before importing the package.
  - Wheels are installed without their dependencies: upload a wheel for every dependency the image does not ship.
  - The image runs Python 3.12 on Linux `x86_64`. Use pure-Python wheels or Linux wheels for that platform; for compiled packages, download them: `pip download <pkg>==<version> --no-deps --only-binary=:all: --python-version 3.12 --platform manylinux_2_28_x86_64 -d dist`.
  - To update a wheel, `remove_wheels()` the old one first, or both versions are installed.
  - Wheels are installed on every run, which adds start-up time.

```python
from qiskit_ibm_catalog import ManagedWorkflowClient, ManagedWorkflow

client = ManagedWorkflowClient(channel="ibm_quantum_platform", instance="<instance CRN>", token="<API key>")
print(client.dependency_report(["qiskit-addon-sqd==0.14.0", "mypkg", "pyscf"]))   # clash, bring, baked

function = client.upload(ManagedWorkflow(title="my-fn", entrypoint="main.py",
                                        working_dir="./source_files/", runner="fleets"))
client.add_wheels(function, ["dist/mypkg-1.0-*.whl"])   # built with: pip wheel . -w dist
print(client.list_wheels(function))                     # ["mypkg-1.0-py3-none-any.whl"]
```

```python
# ./source_files/main.py  — runs remotely
from qiskit_serverless import install_uploaded_wheels, save_result

install_uploaded_wheels()                # before importing the uploaded packages
import mypkg

save_result({"mypkg_version": mypkg.__version__})
```

**Docs:**
- [Declare dependencies: add your own packages](https://quantum.cloud.ibm.com/docs/en/managed-workflows/managed-workflow-dependencies#add-your-own-packages)

## Qiskit patterns in a workflow

A Qiskit pattern breaks a quantum computation into four steps: map the problem to quantum circuits and operators, optimize the circuits for the target hardware (transpile to ISA circuits), execute them with the Sampler or Estimator primitive, and post-process the results (readout mitigation, marginalizing, postselection, classical diagonalization). A managed workflow typically runs all four steps in one job, next to the QPU, and reports which step it is in with `update_status`.

| Pattern step | `update_status(...)` | Shows as |
|---|---|---|
| Map | `Job.MAPPING` | `RUNNING: MAPPING` |
| Optimize | `Job.OPTIMIZING_HARDWARE` | `RUNNING: OPTIMIZING_FOR_HARDWARE` |
| Execute | `Job.WAITING_QPU`, then `Job.EXECUTING_QPU` | `RUNNING: WAITING_FOR_QPU`, `RUNNING: EXECUTING_QPU` |
| Post-process | `Job.POST_PROCESSING` | `RUNNING: POST_PROCESSING` |

- **When to use:** you are structuring a workflow, or deciding which steps belong on the managed side; the heavy classical steps (mapping large molecules, post-processing such as SQD) gain most from running there.
- **Tags:** family: general; step: map, optimize, execute, post-process; runs: serverless; status: stable
- **Watch out:** `job.result()` metadata can report CPU, GPU and QPU seconds per stage under `resource_usage` if your program adds it to the dict it passes to `save_result()`.

**Docs:**
- [Run quantum jobs from a workflow: report progress with update_status()](https://quantum.cloud.ibm.com/docs/en/managed-workflows/manage-quantum-classical-workflows#report-progress-with-update_status)

## Qiskit Function templates

Templates are open-source workflows from [qiskit-function-templates on GitHub](https://github.com/qiskit-community/qiskit-function-templates) that you deploy as a managed workflow yourself, instead of loading a function from the IBM Quantum Functions Catalog. There are four template implementations: a chemistry template for electronic structure with an implicit solvent model (SQD IEF-PCM), built on the SQD addon; a GPU chemistry template for sample-based Krylov quantum diagonalization (SKQD) on Fleets, built on `qiskit-fermions` and the SQD addon, which shows `sizes_map` across CPU and GPU profiles; and two physics templates built on the AQC-Tensor addon, one for Hamiltonian simulation and one for AQC plus Trotter dynamics. Two base templates, a circuit function and an application function, show the interface, input and output handling, error messages, checkpoints and unit tests that a function of your own should have; the repository is a GitHub template, so you can start a function from it.

| Template | Area | Path in `qiskit-community/qiskit-function-templates` |
|---|---|---|
| SQD with implicit solvent (IEF-PCM) | Chemistry | `chemistry/sqd_pcm` |
| SKQD on Fleets (CPU and GPU) | Chemistry | `chemistry/skqd_fleets` |
| Hamiltonian simulation | Physics | `physics/hamiltonian_simulation` |
| AQC + Trotter dynamics | Physics | `physics/aqc_trotter` |
| Circuit function (base) | Base | `base_templates/circuit_function_template.py` |
| Application function (base) | Base | `base_templates/application_function_template.py` |

- **When to use:** you want a starting point for a workflow of your own rather than a ready-made function from the IBM Quantum Functions Catalog.
- **Tags:** family: sqd, trotter; step: scale; runs: serverless, premium; status: stable
- **Watch out:** check each template's upload code for `runner="fleets"` (SKQD Fleets has it) and its imports against the [preinstalled packages](#external-dependencies) before you deploy it.

**Docs:**
- [Hamiltonian simulation with AQC-Tensor](https://quantum.cloud.ibm.com/docs/en/managed-workflows/function-template-hamiltonian-simulation)
- [Chemistry with an implicit solvent model](https://quantum.cloud.ibm.com/docs/en/managed-workflows/function-template-chemistry-workflow)
- [AQC + Trotter Hamiltonian dynamics](https://quantum.cloud.ibm.com/docs/en/managed-workflows/function-template-aqc-trotter)
