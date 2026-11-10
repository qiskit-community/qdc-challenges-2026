---
title: AI-assisted development
short_title: AI tooling
tool_type: ai-tooling
owner: Qiskit developer experience team
last_verified: 2026-10-20
---

# AI-assisted development

Three official resources help an AI coding assistant work with Qiskit:

* **MCP servers** give it tools: documentation search, local circuit analysis and access to IBM Quantum.
* **Agent skills** teach it a procedure, such as migrating code to the current runtime.
* **Agent-readable documentation** lets it read IBM's docs as Markdown.

The hosted Qiskit Code Assistant service is discontinued; its models still run locally ([Qiskit Code Assistant models](#qiskit-code-assistant-models)).

## Qiskit MCP servers

The Model Context Protocol (MCP) connects an AI assistant to external tools. The Qiskit MCP Servers project publishes its servers as Python packages that run on your machine. Each is an early 0.x release.

| Server | Package (version) | What your assistant can do with it | API key | Caution |
|---|---|---|---|---|
| Qiskit Docs | `qiskit-docs-mcp-server` (0.3.0) | Search the IBM Quantum documentation (`search_docs_tool`), read any page as Markdown (`get_page_tool`) and look up an error code (`lookup_error_code_tool`) | No | Read-only |
| Qiskit | `qiskit-mcp-server` (0.3.2) | Load, analyze and transpile circuits, compare optimization levels, and convert between OpenQASM 2, OpenQASM 3 and QPY | No | Transpiles against generic presets (`ibm_heron`, `ibm_eagle`, `ibm_legacy`) and standard topologies, not your backend's live target. Never contacts a QPU |
| Qiskit IBM Runtime | `qiskit-ibm-runtime-mcp-server` (0.6.2) | List backends with their calibration data, find good qubit chains, check usage, and submit, monitor and cancel Sampler and Estimator jobs | Yes | See the Watch out notes below |
| Qiskit IBM Transpiler | `qiskit-ibm-transpiler-mcp-server` (0.4.1) | Run the AI-powered routing and synthesis passes (beta) for a named IBM backend | Yes | Submits no jobs. Needs Python 3.10–3.13 |
| Qiskit Gym (community) | `qiskit-gym-mcp-server` (0.4.2) | Train reinforcement-learning models that synthesize permutation, linear-function and Clifford circuits for a coupling map | No | CPU-heavy. Writes models under `~/.qiskit-gym`. Needs Python 3.10–3.13 |

**Set up in Claude Code.** These commands use `uvx`, which comes with [uv](https://docs.astral.sh/uv/) (install it first, for example with `pip install uv` or `brew install uv`) and downloads and runs the published package. Alternatively, `pip install qiskit-mcp-servers` installs the four core servers and `pip install "qiskit-mcp-servers[all]"` adds the Gym server, but only in an environment of their own: the meta-package always includes the Transpiler server, which depends on `qiskit-ibm-transpiler`, and that package pins `qiskit-serverless~=0.30` and therefore `qiskit-ibm-runtime` below 0.46 ([AI transpiler passes](qiskit-sdk.md#ai-powered-transpiler-passes-qiskit-ibm-transpiler)). Installed next to runtime 0.50 it downgrades or breaks your challenge environment. Both also need Python 3.10–3.13. In your challenge environment, or on Python 3.14, use `uvx`, which runs each server in an isolated environment of its own.

The IBM Runtime and Transpiler servers can work from the account you saved with `QiskitRuntimeService.save_account`, which your challenge notebook needs anyway. Save it first, and you can leave your API key out of every MCP configuration. If `QISKIT_IBM_TOKEN` is set in a server's environment, they use that key instead. `QISKIT_IBM_RUNTIME_MCP_INSTANCE` (an instance CRN, not a secret) fixes the instance the Runtime server uses.

```bash
claude mcp add qiskit-docs -- uvx qiskit-docs-mcp-server
claude mcp add qiskit -- uvx qiskit-mcp-server
# Optional; uses your saved account:
claude mcp add qiskit-ibm-runtime -e QISKIT_IBM_RUNTIME_MCP_INSTANCE="<your instance CRN>" -- uvx qiskit-ibm-runtime-mcp-server
# Optional; these two need Python 3.10–3.13:
claude mcp add qiskit-ibm-transpiler -- uvx --python 3.13 qiskit-ibm-transpiler-mcp-server
claude mcp add qiskit-gym -- uvx --python 3.13 qiskit-gym-mcp-server
claude mcp list    # shows each server and whether it connects
```

`claude mcp add` registers a server for the current project, for you only. Add `--scope user` to use it in all your projects, or `--scope project` to write a `.mcp.json` file that you share with your team.

**Set up in other clients.** Claude Desktop (`claude_desktop_config.json`), Cursor (`.cursor/mcp.json`) and VS Code (`.vscode/mcp.json`, with a top-level `servers` key instead of `mcpServers`) take the same `command`, `args` and `env` entries as JSON.

- **When to use:** you work with an AI assistant: the Docs server for answers that match the current APIs, the Qiskit server for quick circuit checks, and the IBM Runtime server only for live backend data.
- **Tags:** family: general; step: develop; runs: local, account, qpu; status: alpha; updated: 2026-10
- **Recent changes:** `qiskit-mcp-server` 0.3.2, `qiskit-ibm-runtime-mcp-server` 0.6.2 and `qiskit-gym-mcp-server` 0.4.2 were released on 8 October 2026.
- **Watch out:**
  - Do not use the IBM Runtime server to submit the jobs your challenge notebook is meant to submit. Its `run_sampler_tool` and `run_estimator_tool` submit real jobs with their own default options (the Estimator tool, for example, applies ZNE); those jobs consume your instance's QPU time, but they appear in none of your notebook's job records, and a grader never sees them. If you name no backend, they choose the least busy QPU. The Sampler tool does not transpile, so give it an ISA circuit.
  - The Runtime server's jobs are charged to the instance in `QISKIT_IBM_RUNTIME_MCP_INSTANCE`. If that is unset, they go to the instance of the account the server uses or, failing that, to whichever of your instances the service picks for the backend, which may not be the instance you intend to use.
  - `cancel_job_tool` cancels any job in your account, and `delete_saved_account_tool` deletes a saved account.
  - The official setup instructions pass your key as `QISKIT_IBM_TOKEN`. When that variable is set, `qiskit-ibm-runtime` uses it for every tool, ahead of your saved accounts. The `setup_ibm_quantum_account_tool` of the Runtime and Transpiler servers also saves the key with `overwrite=True` as the account named `default-ibm-quantum-platform`, dropping its instance. Code that calls `QiskitRuntimeService()` with no arguments then opens that account, unless another saved account is marked as the default. With a saved account, you need neither the variable nor the setup tool.
  - The servers do not mark which tools are read-only, so your client cannot tell which ones spend QPU time or change your accounts. In Claude Code, add those tools to the `ask` (or `deny`) list in `.claude/settings.json`. Tool names follow `mcp__<server>__<tool>`, where `<server>` is the name you gave in `claude mcp add`. The `*` in the rules below matches any server name, so the rules hold whatever you called the servers; Claude Code does not warn about MCP rules that match no tool, so a fixed prefix with a typo protects nothing:

    ```json
    {"permissions": {"ask": [
      "mcp__*__run_sampler_tool",
      "mcp__*__run_estimator_tool",
      "mcp__*__cancel_job_tool",
      "mcp__*__setup_ibm_quantum_account_tool",
      "mcp__*__delete_saved_account_tool"
    ]}}
    ```
  - These rules cover only MCP tools. Code the assistant runs through its shell (a script, `python -c`, `jupyter nbconvert --execute` or papermill) can submit jobs too, so keep such commands out of your `allow` list; Claude Code then asks before it runs them.
  - If you do put a key in a configuration file, `claude mcp add -e` writes its value into your Claude Code configuration. A project `.mcp.json` can reference an exported variable as `"${QISKIT_IBM_TOKEN}"` instead; never commit a key.
  - `get_job_results_tool` returns counts from the first PUB only and from a single classical register: `meas`, `c`, `cr` or `result` when one of those exists, otherwise the first register it finds in alphabetical order. Read multi-register and Estimator results with `qiskit-ibm-runtime`.
  - IBM's MCP page still lists a Qiskit Code Assistant server. That server and its service were discontinued.

**Docs:**
- [Qiskit MCP Servers](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-mcp-servers)
- [Qiskit MCP Servers on GitHub](https://github.com/Qiskit/mcp-servers)
- [AI-powered transpiler passes](https://quantum.cloud.ibm.com/docs/en/guides/ai-transpiler-passes)

## Qiskit AI skills

An AI skill is a reusable set of instructions that teaches an assistant how to complete a specific task. IBM publishes official skills in the [`Qiskit/skills`](https://github.com/Qiskit/skills) repository (Apache-2.0) in the open Agent Skills format. Claude Code, Codex, Cursor, IBM Bob and other assistants can use them. The first release (0.1.0, 29 September 2026) has one skill.

**`migrate-qiskit-ibm-runtime`** migrates `qiskit-ibm-runtime` code from any earlier version to 0.50. Where your runtime supports it, it continues to the client-side Sampler and Estimator (`executor_sampler`, `executor_estimator`).

* **Coverage:** it tracks 24 breaking changes, including V1 primitives, `backend.run`, the old `ibm_quantum` channel, sessions and the move to `NoiseLearnerV3`.
* **Edits:** it applies edits that do not change results directly. For edits that do, such as splitting a job with mixed shots or adding explicit noise learning for PEA and PEC, it shows a diff and asks first.
* **Checks:** it fetches the current migration guides as it runs, through the web or the Qiskit Docs MCP server. It checks its work locally on a fake backend and submits no jobs.

```text
# Claude Code, inside a session
/plugin marketplace add Qiskit/skills
/plugin install qiskit-ai-skills@qiskit
# run it on a folder or file of your own code:
/qiskit-ai-skills:migrate-qiskit-ibm-runtime my_scripts/

# Other assistants: the skills CLI asks which assistants to install for
npx skills add Qiskit/skills
```

The first command adds the repository's plugin marketplace, which is named `qiskit`. In `qiskit-ai-skills@qiskit`, the part before `@` is the plugin and the part after it is that marketplace.

- **When to use:** you adapt older code (tutorials, older notebooks, your own scripts) to `qiskit-ibm-runtime` 0.50.
- **Tags:** family: general; step: develop; runs: local; status: stable; updated: 2026-09
- **Watch out:**
  - Run it only on your own scripts, never on a challenge folder or the modules it ships with, because a grader may check exactly what the challenge code submits.
  - The skill does not write `Executor`, `QuantumProgram` or Samplomatic code. It does not choose mitigation methods or layouts either. The repository's roadmap lists skills for evaluating mitigation methods (in progress) and for problem-aware layout selection (planned), both for late 2026.
  - Its own checks compile `.py` files, so review edits to notebooks yourself.

**Docs:**
- [Qiskit AI skills](https://quantum.cloud.ibm.com/docs/en/guides/ai-skills)
- [Qiskit skills changelog](https://quantum.cloud.ibm.com/docs/en/guides/changelog-qiskit-skills)
- [Migrate to client-side primitives](https://quantum.cloud.ibm.com/docs/en/guides/migrate-to-client-side-primitives)

## Qiskit Code Assistant models

IBM discontinued the hosted Qiskit Code Assistant service, its VS Code and JupyterLab extensions and its MCP server on 29 May 2026. Its Qiskit-tuned open models remain on [Hugging Face](https://huggingface.co/Qiskit) (Apache-2.0) and run locally:

* `mistral-small-3.2-24b-qiskit`, the default;
* `Qwen2.5-Coder-14B-Qiskit`;
* `granite-3.3-8b-qiskit` and `granite-3.2-8b-qiskit`.

Each has a GGUF version for Ollama and llama.cpp, for example `ollama run hf.co/Qiskit/Qwen2.5-Coder-14B-Qiskit`, and the guide shows how to connect a local model to Continue (VS Code), Jupyter AI (JupyterLab) and OpenCode (terminal). The guide also lists the Qiskit version each model was trained on. Their training data predates parts of the 2026 stack described in this cheat sheet, so check their suggestions against the [API quick reference](../guides/testing-and-tips.md#api-quick-reference).

- **When to use:** you want a Qiskit-trained code model that runs on your own machine.
- **Tags:** family: general; step: develop; runs: local; status: stable; updated: 2026-05

**Docs:**
- [Qiskit Code Assistant](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-code-assistant)

## Documentation for agents

* [`llms.txt`](https://quantum.cloud.ibm.com/llms.txt) indexes the guides, API reference, tutorials and learning pages, each as a Markdown link.
* Add `.md` to any documentation URL to get the page as Markdown, starting with YAML front matter (`title`, `description`, `source`). Example: [qiskit-mcp-servers.md](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-mcp-servers.md).
* The [error code reference](https://quantum.cloud.ibm.com/docs/en/errors) explains every IBM Quantum error number; `lookup_error_code_tool` returns one entry.

- **When to use:** your AI assistant needs current IBM documentation, an error code or the Markdown version of a docs page.
- **Tags:** family: general; step: develop; runs: local; status: stable

**Docs:**
- [`llms.txt`, the documentation index for agents](https://quantum.cloud.ibm.com/llms.txt)
- [Error code reference](https://quantum.cloud.ibm.com/docs/en/errors)
