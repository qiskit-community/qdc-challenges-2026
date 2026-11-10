---
title: Getting started
short_title: Getting started
tool_type: guide
owner: QDC organizers
last_verified: 2026-10-20
---

# Getting started

How to use this cheat sheet, the versions it was checked against, how to set up your environment, IBM Quantum account and AI assistant, and how grading works.

## How to use this cheat sheet

The cheat sheet is a set of documents. The [landing page](../README.md) indexes every entry in several ways; each reference document covers one tool type; the guides cover what spans tools.

**Participants.**
1. Start with your challenge notebook: it says what is required and how it is graded.
2. Set up the versions your notebook pins ([Versions and environment](#versions-and-environment)).
3. When a notebook calls something unfamiliar, find it from the [landing page](../README.md), by tool type, algorithm family, workflow step or task, or search the documents for the class or function name. The landing page's [New and changing](../README.md#new-and-changing) view lists the tools that changed recently.
4. [Troubleshooting](troubleshooting.md) maps common errors to fixes.

**AI agents helping a participant.**
- **Where to start:** the challenge notebook is the authority on what is required and graded; where it disagrees with this cheat sheet, follow the notebook and say so. Then read the entries for the tools the notebook imports: for a task, start from the landing page's [I want to…](../README.md#i-want-to) index, which links the sections that answer it. For the tools the notebook imports, the [API quick reference](testing-and-tips.md#api-quick-reference) names the entry for each import line, the landing page's [All entries](../README.md#all-entries) table lists every entry with its document and tags, and a search of the documents for a class or function name finds the rest. Load only the sections a task needs; `grep -n '^## ' <document>` lists a document's entries with their line numbers.
- **Entry format:** every entry is a second-level heading in a reference document or guide, and uses the same fields, in this order, leaving out any that do not apply: a description, **When to use**, **Tags**, **Recent changes** (or **Current version**), **Key APIs**, **Watch out**, **Related** and **Docs**. The Tags field uses the values defined in [taxonomy.toml](../tools/taxonomy.toml). Each document opens with front matter that names its owner and the date it was last verified, and most reference documents open with a table of the package versions they were checked against.
- **Versions:** check the installed versions before you rely on an API, for example `python -m pip list | grep -iE "qiskit|samplomatic|mthree"`. Keep the versions the notebook pins, even where this cheat sheet describes a newer release; [Versions and environment](#versions-and-environment) lists the releases it was checked against.
- **QPU time:** it is limited, so treat it as the participant's to spend.
  - Run everything locally first ([Fake backends and local testing mode](../reference/ibm-quantum-compute-client.md#fake-backends-and-local-testing-mode), [Local test patterns](testing-and-tips.md#local-test-patterns)), and estimate each job's cost ([Budgeting QPU time](../reference/ibm-quantum-compute-client.md#budgeting-qpu-time)).
  - Submit hardware jobs, Managed Workflow runs or IBM Quantum Functions Catalog runs only when the participant says so. Managed workflows can submit QPU jobs of their own.
  - Never execute a whole notebook (Run All, `jupyter nbconvert --execute` or papermill): a notebook may submit jobs from any cell, with or without a switch. Run cells one at a time, read what each cell does before you run it, and skip job cells until the participant approves them.
  - Deleting saved job IDs or result files makes the next run submit new jobs. Delete them only for a run the participant has approved.
  - Submit challenge jobs from the notebook's own cells, not with MCP job tools ([Qiskit MCP servers](../reference/ai-tooling.md#qiskit-mcp-servers)): a notebook's own job records and its grader do not see jobs submitted another way.
  - Save every job ID, and retrieve finished jobs instead of resubmitting them.
  - If a job cell ran twice, or a job went out by mistake, stop and tell the participant at once. With their go-ahead, cancel the copy that is still queued, which then costs nothing, and make sure the job ID the notebook records is the one you kept ([Before you spend QPU time](testing-and-tips.md#before-you-spend-qpu-time)).
- **Credentials:** never print, log or commit API keys. Keep them in a saved account (`QiskitRuntimeService.save_account`) or in a `.env` file that git ignores, never in notebooks or in MCP configuration files.
- **Graded code:** keep the code paths that graders check as the notebook sets them up, unless an exercise asks you to change them. For example, if a graded cell is built on `SamplerV2`, keep `SamplerV2` there even though runtime 0.50 deprecates it. Write variants as new functions instead of editing a function that a check has passed.
- **Beyond this cheat sheet:** search the IBM Quantum documentation, through the Qiskit Docs MCP server if it is installed. Any docs page is available as Markdown when you add `.md` to its URL ([Documentation for agents](../reference/ai-tooling.md#documentation-for-agents)).

## Versions and environment

Versions, release dates and links were checked against PyPI, release notes and the IBM Quantum documentation between 16 and 20 October 2026. Every import line in the [API quick reference](testing-and-tips.md#api-quick-reference) was run on Python 3.12 with Qiskit 2.5.2, `qiskit-ibm-runtime` 0.50.0 and the other versions listed there. A notebook that pins older versions may lack some of these APIs; keep its pins and check the table below before you rely on one.

Use the versions your challenge notebook or its requirements file pins:

* Where a notebook pins an older release, keep its pin, even if the table lists a newer one.
* Some notebooks install unpinned packages from an install cell, so a fresh install takes the newest releases. Where a notebook pins nothing, pin the versions in the table below yourself; a fresh install now resolves Qiskit 2.6.0 and Aer 0.18.0, which the table notes.
* If an install cell leaves out a package that a later cell imports, install it before you reach that cell.
* Some notebooks run their heavy computation inside a Managed Workflow and need only a few packages locally.
* Pins differ between notebooks, so give each one its own virtual environment; the [install guide](https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit) shows how to create one.

The table lists the releases this cheat sheet was checked against; the Qiskit and Aer releases of 16 and 20 October 2026 are noted where they differ. It is collected from the package tables at the top of the reference documents, whose owners keep them current, and the grading client's version is in [Grading with qc-grader](#grading-with-qc-grader).

<!-- BEGIN GENERATED: versions -->
| Package | Version checked | Notes | Document |
|---|---|---|---|
| `qiskit` | 2.5.2 | Requires Python 3.10 or later. Qiskit 2.6.0 (16 October 2026) is now the newest release; read [Saving circuits with QPY](../reference/qiskit-sdk.md#saving-circuits-with-qpy) before you mix 2.5 and 2.6 | [Qiskit SDK](../reference/qiskit-sdk.md) |
| `qiskit-ibm-transpiler` | 0.18.0 |  | [Qiskit SDK](../reference/qiskit-sdk.md) |
| `qiskit-ibm-runtime` | 0.50.0 | Requires Python 3.10 or later and Qiskit 2.3 or later; installs `samplomatic>=0.21`, `qiskit-mitigation>=0.1.1` and `ibm-quantum-schemas>=0.12.20260921` automatically. Runtime 0.49 has `SamplerV2` but not the client-side `executor_sampler.Sampler` ([Sampler and Estimator](../reference/ibm-quantum-compute-client.md#sampler-and-estimator-client-side-primitives)). Keep a notebook's 0.49 pin for that notebook. To use 0.50 in an environment built from a requirements file that pins 0.49 and an older `ibm-quantum-schemas`, do not only edit the runtime pin, because 0.50 needs the newer schemas release: run `pip install -r requirements.txt`, then `pip install qiskit-ibm-runtime==0.50.0` | [Compute client](../reference/ibm-quantum-compute-client.md) |
| `samplomatic` | 0.21.0 | Requires Qiskit 2.3 or later | [Samplomatic and mitigation](../reference/noise-management.md) |
| `qiskit-mitigation` | 0.1.1 | Its programs need runtime 0.50; requires `samplomatic>=0.21` | [Samplomatic and mitigation](../reference/noise-management.md) |
| `qiskit-noise-learning` | 0.1.0 | Requires Python 3.11 or later, Qiskit 2.5 or later and `samplomatic>=0.21` | [Samplomatic and mitigation](../reference/noise-management.md) |
| `qiskit-addon-utils` | 0.4.0 | 0.4 includes post-selection on slow-X error checks ([Post-selection on non-Markovian error checks](../reference/noise-management.md#post-selection-on-non-markovian-error-checks)) | [Qiskit addons](../reference/qiskit-addons.md) |
| `qiskit-addon-slc` | 0.1.0 (GitHub commit `7156757`) | PyPI 0.1.0 lacks some newer APIs that are only on GitHub ([Shaded light cones](../reference/qiskit-addons.md#shaded-light-cones-qiskit-addon-slc)), so install from GitHub when an import fails; needs PySCF (`pyscf>=2.10`) | [Qiskit addons](../reference/qiskit-addons.md) |
| `qiskit-addon-opt-mapper` | 0.1.0 |  | [Qiskit addons](../reference/qiskit-addons.md) |
| `qiskit-paulice` | 0.2.1 | Its public API lacks `stabilizers=`, `dope()` and `estimate_fault_rates()`, which are only on GitHub `main` ([Spacetime Pauli checks](../reference/qiskit-addons.md#spacetime-pauli-checks-qiskit-paulice)). Requires `qiskit-aer` 0.17.1 or later | [Qiskit addons](../reference/qiskit-addons.md) |
| `qiskit-addon-sqd` | 0.14.0 | Trim SQD (`TrimPolicy`) is new in 0.14. Needs PySCF (`pyscf>=2.9`) for `qiskit_addon_sqd.fermion`; pip leaves PySCF out on Windows | [Qiskit addons](../reference/qiskit-addons.md) |
| `mthree` | 3.0.0 | mthree 3.0.0 has wheels for Python 3.9–3.12 only; on 3.13 and 3.14, pip builds it from source and needs a C compiler | [Qiskit addons](../reference/qiskit-addons.md) |
| `qiskit-fermions` | 0.3.0 | Requires Python 3.11 or later | [Qiskit addons](../reference/qiskit-addons.md) |
| `qiskit-addon-obp` | 0.3.0 |  | [Qiskit addons](../reference/qiskit-addons.md) |
| `qiskit-addon-aqc-tensor` | 0.3.1 |  | [Qiskit addons](../reference/qiskit-addons.md) |
| `qiskit-addon-cutting` | 0.10.0 |  | [Qiskit addons](../reference/qiskit-addons.md) |
| `qiskit-addon-pna` / `pauli-prop` | 0.2.0 / 0.2.1 |  | [Qiskit addons](../reference/qiskit-addons.md) |
| `qiskit-addon-mpf` | 0.3.0 |  | [Qiskit addons](../reference/qiskit-addons.md) |
| `qiskit-aer` | 0.17.2 | 0.18.0 (20 October 2026) is now the newest release and requires Python 3.11 or later | [Simulators](../reference/simulators.md) |
| `quimb` | 1.15.0 | Requires Python 3.11 or later | [Simulators](../reference/simulators.md) |
| `mapomatic` | 0.14.0 |  | [Ecosystem](../reference/ecosystem.md) |
| `pyscf` / `ffsim` | 2.14.0 / 0.0.84 | PySCF has no Windows support. ffsim 0.0.84 requires Python 3.11 or later and `pyscf>=2.12`; `generate_lucj_pass_manager` arrived in ffsim 0.0.75 ([Chemistry: PySCF and ffsim](../reference/ecosystem.md#chemistry-pyscf-and-ffsim)) | [Ecosystem](../reference/ecosystem.md) |
| `docplex` / `cplex` / `moocore` | 2.32.264 / 22.2.0.1 / 0.3.2 | `cplex` 22.2.0.1 has wheels for Python 3.10–3.14 on Linux, Windows and Apple-silicon macOS, but none for Intel Macs | [Ecosystem](../reference/ecosystem.md) |
<!-- END GENERATED: versions -->

### Python

Use the version your notebook names. Where it names none, 3.12 is the safest choice: Qiskit 2.5 and `qiskit-ibm-runtime` 0.50 require 3.10 or later, ffsim 0.0.84 and quimb 1.15 require 3.11 or later, `qiskit-ibm-catalog` 0.22 and `qiskit-serverless` 0.38 require 3.12 or later, and mthree has wheels only up to 3.12. Qiskit runs only on CPython and, since 2.0, only on 64-bit platforms; the [install guide](https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit) points to the PyPI page for the Python versions the newest release supports. PySCF has no Windows support, so the packages that import it (`qiskit-addon-slc`, ffsim and the `fermion` module of `qiskit-addon-sqd`) need Linux, macOS or WSL. `qopt-best-practices` and `qaoa_training_pipeline` install from GitHub, and so do the newer `qiskit-addon-slc` and `qiskit-paulice` APIs noted in the table.

**Docs:**
- [Install Qiskit](https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit)

## IBM Quantum account

Save your API key with `QiskitRuntimeService.save_account(token="<API key>", instance="<CRN or instance name>", name="<name>", set_as_default=True)`. Run it once in a Python prompt, not in a notebook cell, and read the key with `getpass.getpass()` instead of typing it: a cell keeps the key in the notebook's source and outputs, which git then commits. The saved file is outside your project. Credentials are stored in `$HOME/.qiskit/qiskit-ibm.json`, and `QiskitRuntimeService.saved_accounts()` lists them. Notebooks open the account in a few ways:
- `QiskitRuntimeService()` opens your default saved account. If `QISKIT_IBM_TOKEN` is set in the environment, it is used instead, with `QISKIT_IBM_INSTANCE`, ahead of any saved account.
- `QiskitRuntimeService(name="<name>")` opens a named account. The name wins over `QISKIT_IBM_TOKEN` and over a `token=` passed in the same call, which is ignored.
- `QiskitRuntimeService(instance="<CRN>")` loads the default credentials but connects to that instance.
- `QiskitRuntimeService(token="<API key>", instance="<CRN>")` uses the credentials you pass and ignores saved accounts. Some notebooks read the key and the CRN from a `.env` file and pass them this way, or to `ManagedWorkflowClient(token=..., instance=...)`.
- `ManagedWorkflowClient` (and `FunctionsCatalog`) is built on `QiskitRuntimeService`, so `ManagedWorkflowClient()` with no arguments finds credentials the same way, `ManagedWorkflowClient(name="<name>")` opens a named account, and `ManagedWorkflowClient.save_account(...)` writes the same credentials file.

Save the account under the name your notebook expects, or edit the call. Each instance belongs to a region (`us-east` or `eu-de`). The same API key works in both regions, but the IBM Quantum Platform pages show only the instances of the region you are logged in to, and the region decides where jobs run and where their data is kept. Asking for a QPU your instance cannot reach raises `QiskitBackendNotFoundError: No backend matches the criteria`.

**Docs:**
- [Set up your IBM Cloud account](https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup)
- [Save your login credentials](https://quantum.cloud.ibm.com/docs/en/guides/save-credentials)
- [Initialize your account](https://quantum.cloud.ibm.com/docs/en/guides/initialize-account)
- [QiskitRuntimeService API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/qiskit-runtime-service)
- [Managed Workflows](https://quantum.cloud.ibm.com/docs/en/managed-workflows)

## Setting up an AI assistant

1. Add the Qiskit Docs server and the Qiskit server ([Qiskit MCP servers](../reference/ai-tooling.md#qiskit-mcp-servers)). Neither needs a key or can spend QPU time.
2. Add the IBM Runtime server only if you want backend data in the conversation. Save your account first ([IBM Quantum account](#ibm-quantum-account)), set `QISKIT_IBM_RUNTIME_MCP_INSTANCE`, and make its job and account tools ask for approval (the `ask` list in [Qiskit MCP servers](../reference/ai-tooling.md#qiskit-mcp-servers)).
3. Install the `migrate-qiskit-ibm-runtime` skill if you reuse older code.
4. Tell the assistant which challenge you are working on, in a `CLAUDE.md` file (Claude Code) or an `AGENTS.md` file (other assistants) in your challenge folder, and point it to the cheat sheet's [landing page](../README.md), the notebook and the versions the notebook pins.
5. Have it test everything locally (see [Local test patterns](testing-and-tips.md#local-test-patterns)) before any job reaches a QPU, and decide yourself when to submit.

An example `CLAUDE.md`:

```markdown
# Working on the <challenge name> challenge
- Read the Qiskit tools cheat sheet <path to cheat_sheet_2026>/README.md first.
- Load only the documents it points to for the task; the notebook <notebook path> is the authority.
- Use the versions in this folder's requirements file; change packages only as the notebook's setup instructions say.
- Never run the whole notebook. Run cells one at a time, and run no job cell, Managed Workflow run or MCP job tool without my approval.
- Keep graded functions as their checks expect; write variants as new functions.
- Never print or commit API keys.
```

## Grading with qc-grader

Each challenge notebook's grading cells show how it is graded and which module they import. Challenges that grade through IBM's `qc-grader` client (`pip install qc-grader`) need the release the organizers announce: as of 20 October 2026, the newest release on PyPI (2026.9.29) does not include the QDC 2026 graders. Install the version the announcement names, check that the module your notebook imports is present, then call one grading function per exercise.

- **When to use:** your challenge notebook grades through `qc-grader`.
- **Tags:** family: general; step: develop; runs: account; status: stable; updated: 2026-09
- **Current version:** `qc-grader` 2026.9.29 (29 September 2026) on PyPI, without the QDC 2026 graders.
- **Key APIs:** the grading module your notebook imports, `join_team("<team>")`, `check_progress()` (or `check_progress("<lab>")` for one lab), one `grade_*` function per exercise
- **Watch out:** (for challenges graded with `qc-grader`)
  - Call `join_team` before your first grading call; answers submitted afterwards are associated with that team.
  - Your best score per exercise counts, so a later, worse attempt does not lower it. Each grading call prints the score for that attempt, and `check_progress()` prints the exercises passed and the total score per lab.
  - The client refuses an answer whose JSON is larger than 20 MB.
  - The grading server never contacts IBM Quantum. For a hardware exercise, the grading function reads your job on your machine and sends its ID, backend and usage, plus any results the exercise grades; the notebook's grading section says which. Job IDs may be audited, so grade only your own jobs.

**Docs:**
- [qc-grader on PyPI](https://pypi.org/project/qc-grader/)
- [quantum-challenge-grader on GitHub](https://github.com/Qiskit-community/quantum-challenge-grader)
