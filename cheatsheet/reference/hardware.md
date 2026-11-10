---
title: IBM QPUs
short_title: IBM QPUs
tool_type: hardware
owner: IBM Quantum systems team
last_verified: 2026-10-20
---

# IBM QPUs

The IBM quantum processors you can reach, their processor families and native instructions, the calibration data and layer-fidelity chains they report, and the retired QPUs whose names still appear in older code.

## QPUs at a glance

| QPU | Processor | Qubits | Notes |
|---|---|---|---|
| `ibm_boston` | Heron r3 | 156 | Opened to Premium, Flex and Pay-As-You-Go users in January 2026. Supports dynamic circuits, `measure_2`, `reset_2` and fractional gates |
| `ibm_pittsburgh` | Heron r3 | 156 | First Heron r3 (July 2025) |
| `ibm_aachen` | Heron r3 | 156 | Launched as a Heron r2 in April 2025 and upgraded to r3 in March 2026; EU region (`eu-de`) |
| `ibm_kingston` | Heron r2 | 156 | Launched April 2025; open to all Open Plan users since April 2026 |
| `ibm_fez`, `ibm_marrakesh` | Heron r2 | 156 | Open Plan since 27 October 2025 |
| `ibm_miami` | Nighthawk r1 | 120 | The first Nighthawk; square lattice with 218 couplers; early access for Premium and Flex users since January 2026 (`us-east`) |
| `ibm_berlin` | Nighthawk r1 | 120 | First Nighthawk in the EU (April 2026); faster two-qubit gates than `ibm_miami` (68 ns versus 138 ns, median) |
| `ibm_phoenix` | Nighthawk r2 | 120 | September 2026; 120 programmable qubits out of 458 physical ones. An independent fast reset gives a 1 µs default repetition delay (250 µs on most other QPUs); supports mid-circuit reset and dynamic circuits; Premium, Flex and Pay-As-You-Go plans only |

* Heron processors use a heavy-hex lattice and native CZ gates, and support fractional `RZZ` and `RX` gates. Nighthawk processors use a square lattice with more couplers per qubit (218 couplers versus Heron's 176), so many circuits need fewer SWAPs.
* Native instruction sets are per processor family: `rz`, `sx`, `x`, `id` and `delay`; `cz` on Heron and Nighthawk (`ecr` on the retired Eagles); fractional `rx` and `rzz`; `measure`, `reset` and their `measure_*` and `reset_*` variants; `if_else`; and, since May 2026, `store`. `backend.supported_instructions` lists what a backend offers, and `backend.target` only contains `rzz` when you load the backend with `use_fractional_gates=True`.
* At launch, `ibm_miami` and `ibm_berlin` temporarily ran with a 4 ms default repetition delay (`rep_delay`) and without dynamic circuits. Check the backend's current properties before you plan around them.
* Since February 2026 Heron QPUs accept `rep_delay` values from 20 to 2,000 µs, with the default unchanged, and the Nighthawk range stayed at 1,000 to 4,000 µs (announced before `ibm_phoenix`). Ranges differ per QPU and change over time, so read `backend.rep_delay_range` and `backend.default_rep_delay` before you set one. A longer `rep_delay` adds directly to execution time and usage.
* Some names in older code are retired: `ibm_torino` (1 April 2026) and the Eagle processors `ibm_kyiv` (April 2025), `ibm_sherbrooke` (July 2025), `ibm_brisbane` (November 2025), `ibm_brussels` and `ibm_strasbourg` (the last Eagles, 27 April 2026). Their fake backends still work offline, and `service.job(job_id)` still returns jobs that ran on them.
* `qiskit-ibm-runtime` 0.50.0 ships `FakeBoston`, `FakePittsburgh`, `FakeAachen`, `FakeKingston`, `FakeFez`, `FakeMarrakesh`, `FakeMiami` and `FakeBerlin`, plus the retired `FakeTorino`, `FakeBrisbane`, `FakeKyiv`, `FakeBrussels`, `FakeStrasbourg` and `FakeSherbrooke`; there is no `FakePhoenix`. The `FakeMiami` and `FakeBerlin` targets have no `reset`, `if_else`, `measure_2` or `reset_2`, and the `FakeBoston` target includes `measure_2` and `reset_2`.
* Which QPUs you can use depends on your plan, instance and region. Asking for a QPU your instance cannot reach raises `QiskitBackendNotFoundError`, sometimes after a logged 404 "device not found" warning.

- **When to use:** you choose a QPU, or check which processors, regions and native instructions your instance can use.
- **Tags:** family: general; step: execute; runs: account; status: stable; updated: 2026-09

**Docs:**
- [Processor types](https://quantum.cloud.ibm.com/docs/en/guides/processor-types)
- [View backend details](https://quantum.cloud.ibm.com/docs/en/guides/qpu-information)
- [IBM Quantum Compute Service changelog](https://quantum.cloud.ibm.com/docs/en/guides/changelog-quantum-compute-service)
- [Retired QPUs](https://quantum.cloud.ibm.com/docs/en/guides/retired-qpus)
- [Qubit initialization and repetition delay](https://quantum.cloud.ibm.com/docs/en/guides/repetition-rate-execution)
- [Fake backends (`qiskit_ibm_runtime.fake_provider`)](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/fake-provider)

## Calibration data and layer fidelity

Each backend reports calibration data for its qubits and couplers, and IBM publishes layer-fidelity chains for each QPU.

* **Calibration data.** `backend.target` holds per-instruction errors and durations (`backend.target["cz"][(q0, q1)].error`), and `backend.properties()` holds T1, T2 and readout errors (`backend.properties().t1(q)`, `readout_error(q)`); `backend.properties(datetime=...)` returns historical values. An error of exactly 1 means the benchmarking of that qubit or edge did not succeed for several days: the value is undefined, not necessarily broken, so treat it with caution.
* **Layer fidelity.** Backends publish their best layer-fidelity chains in `backend.properties().to_dict()["general_qlists"]`. Each entry has a `name` of the form `lf_<N>` for an N-qubit chain (for example `lf_100`) and a `qubits` list; the long-range-entanglement tutorial takes a prefix of one long chain for every shorter distance it runs. The reported two-qubit error (layered) is the error per layered gate (EPLG) of a 100-qubit chain.

- **When to use:** you read error rates, coherence times or layer-fidelity chains to choose qubits, or to judge whether a qubit or coupler is usable.
- **Tags:** family: general, qaoa, multi-objective, floquet, echoes; step: optimize; runs: account; status: stable
- **Key APIs:** `backend.target[gate][qargs].error`, `backend.properties(datetime=...)`, `backend.qubit_properties(q)`, `backend.properties().to_dict()["general_qlists"]`
- **Watch out:** `backend.target["rzz"]` exists only when you load the backend with `use_fractional_gates=True`, and the two guides disagree on what its error means ([Fractional gates](ibm-quantum-compute-client.md#fractional-gates)).
- **Related:** [Qubit selection and layout](qiskit-sdk.md#qubit-selection-and-layout) uses these numbers to choose qubits; [Device benchmarking](ecosystem.md#device-benchmarking-qiskit-device-benchmarking) re-measures them.

**Docs:**
- [View backend details](https://quantum.cloud.ibm.com/docs/en/guides/qpu-information)
- [Tutorial: selecting a 1D chain with the layer-fidelity string](https://quantum.cloud.ibm.com/docs/en/tutorials/long-range-entanglement#use-layer-fidelity-string-for-selecting-1d-chain)
