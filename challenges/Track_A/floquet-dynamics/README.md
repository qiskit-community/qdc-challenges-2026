# Floquet Dynamics: Resolving Structure in Prethermal Floquet Dynamics

**IBM Quantum Developer Conference 2026 challenge**

Based on E. Leviatan, T. Watad, R. Perry *et al.*, *Resolving Structure in Prethermal
Floquet Dynamics with Precision Quantum Computation*,
[arXiv:2607.24937](https://arxiv.org/abs/2607.24937) (2026).

| Notebook | Purpose |
|---|---|
| [floquet_challenge.ipynb](floquet_challenge.ipynb) | The challenge: exercises marked ✏️ |

## 1. The problem and why it matters

### The physics

Periodically driven ("Floquet") interacting quantum many-body systems can host phases
of matter with **no equilibrium analogue** — anomalous Floquet topological phases,
discrete time crystals, and other non-equilibrium order. The obstacle is that a generic
interacting system *heats*: it absorbs energy from the drive until every local observable
relaxes to its featureless infinite-temperature value.

Between the initial transient and this eventual heating there is often a long-lived
**prethermal** window. In this window local observables retain coherent, structured
time-dependence — governed by an approximately conserved effective Hamiltonian — even
while entanglement and operator complexity keep growing. Characterizing the prethermal
state requires reaching **large system sizes** and **long evolution times**, precisely
the regime where:

- **classical methods break down** — tensor networks fail to converge and sparse
  Pauli-path methods become strongly truncation-dependent; and
- **raw quantum hardware breaks down** — noise corrupts expectation values after only a
  few cycles.

### What the paper does

The authors study a **Floquet mixed-field Ising magnet on a heavy-hex lattice** and use
**error-mitigated** execution on an **IBM Heron r3** processor (`ibm_boston`) to measure
magnetization dynamics with **percent-level precision** in systems of **up to 74 qubits**.
They resolve **long-lived subharmonic prethermal oscillations** with a period ≈ 4× the
drive period, in a regime where leading classical simulations (on the Fugaku
supercomputer and NVIDIA H100 GPUs) can no longer be trusted.

Using this quantum-accessible regime, they perform **finite-size scaling** and find the
oscillation amplitude decreases unexpectedly slowly with system size — strong evidence
that the oscillatory response survives in the **thermodynamic limit** of heavy-hex
ladders.

### Why it matters

The result is a proof-of-principle that **error-mitigated quantum processors can act as
quantitative scientific instruments** — not just demonstrations — for discovering new
physics in a regime beyond the reach of state-of-the-art classical simulation. The
reliability of the findings is backed by a hierarchy of validation tests: unbiased
mitigation, agreement between independent estimators, hardware noise-model validation,
and **cross-platform corroboration** on Quantinuum H2 and Helios trapped-ion hardware.

### The model

One Floquet cycle re-applies the single-qubit drive before **each** of the three
edge-coloured $ZZ$ layers (paper Eq. 1):

$$
U_F=\prod_{r=1}^{3}\Big[e^{-i\frac{\theta_{zz}}{2}C_r}\,
e^{-i\frac{\theta_z}{2}Z_\Sigma}\,e^{-i\frac{\theta_x}{2}X_\Sigma}\Big],
\qquad C_r=\sum_{\langle i,j\rangle\in E_r} Z_iZ_j .
$$

$R_X$ does not commute with $ZZ$, so the three sub-layers cannot be merged. The
effective drive per cycle is $3\theta_x=\pi/2$. The colour classes $E_1,E_2,E_3$
are transcribed from the paper's Fig. E1/A1. Each class has disjoint bonds and
runs as one parallel layer of native fractional $R_{ZZ}$ gates.

| Item | Value |
|---|---|
| Angles | $(\theta_x,\theta_z,\theta_{zz})=(\pi/6,\pi/27,\pi/3)$ |
| Initial state | $\lvert0\rangle^{\otimes N}$ |
| Observables | $M=\frac1{N_q}\sum_q\langle Z_q\rangle$ and $M_{z=2}$ (the average over degree-2 qubits) |
| Patches | Paper-coloured heavy-hex patches with 21, 28, 35 and 51 qubits; 74 qubits in Part 5.2 |
| Cycles | $n=0,2,4,\dots,30$ |

```
One Floquet cycle (repeat N times, then measure):
  ┌ RX(θx) all ─ RZ(θz) all ─ RZZ(θzz) on colour C1 ┐
  │ RX(θx) all ─ RZ(θz) all ─ RZZ(θzz) on colour C2 │
  └ RX(θx) all ─ RZ(θz) all ─ RZZ(θzz) on colour C3 ┘
```

## 2. Resources

- **QPU:** IBM Heron r3 (`ibm_boston`), with fractional gates enabled. The whole
  notebook has a hard cap of **1,800 QPU seconds**.
- **Classical:** exact statevector references for 21/28/35 qubits are supplied in
  `reference/`. Part 1 recomputes the 21-qubit reference with **Qiskit Serverless**,
  uploading the engine in `reference/`; this uses no QPU time. The 51-qubit
  target is digitized from the paper's Fig. 3. The 74-qubit target is the paper's
  $M_{z=2}(18)\approx0.356$.

## 3. What you build

Each part improves on the mitigation of the previous one.

| Part | What you build |
|---|---|
| 1 | Place the colour-coded lattice, write `floquet_cycle`, and check it against the exact reference |
| 2 | Run unmitigated Sampler circuits and watch the signal hit the noise wall. Add `mthree` readout correction and global-fold ZNE at factors (1, 3) |
| 3 | Compile to native fractional $R_{ZZ}$, normalizing negative angles. Choose a low-error layout with `mapomatic`. Use TREX through the Runtime 0.50 client-side `Estimator`, with `M` and `M_{z=2}` as observables |
| 4 | Fold one Floquet cycle at a time at factors (1, 3, 5). Write linear, Richardson and exponential extrapolators, and select a model against the reference |
| 5.1 | Run the same workflow at 28, 35 and 51 qubits and compare with the targets |
| 5.2 | Open-ended: estimate $M_{z=2}(18)$ at 74 qubits with an error bar, by any method. You are advised to check the challenges cheat sheet for ideas about how to do this (perhaps, by using Qiskit functions!)

**Scoring.** Each ✏️ exercise is graded on the IBM challenge platform through
`qc-grader`: the notebook calls `grade_lab_floquet_<exercise>(...)` after each
exercise, and `check_progress()` prints your total. The grader runs your functions
locally on small fixed test inputs and sends only their outputs, so grading spends
no QPU time. The exercises add up to 100 points:

| Part | Graded exercises | Points |
|---|---|---:|
| 1 | lattice and colour groups 5 · `floquet_cycle` 5 | 10 |
| 2 | logical circuit and baseline magnetization 5 · first mitigation (readout mitigator, corrected magnetizations, global fold, linear ZNE) 10 | 15 |
| 3 | inverse `RZZ` 5 · layout 5 · Mapomatic 5 · TREX estimator 2.5 · observables 2.5 | 20 |
| 4 | cycle folding 5 · `measure_cycle_folded` 5 · ZNE fitter 5 · linear 2.5 · Richardson 2.5 · exponential 5 · model selection 5 | 30 |
| 5 | 5.1 multi-size extension 15 · 5.2 open-ended: $M_{z=2}(18)$ at 74 qubits with an error bar, any method (the solution demonstrates QESEM-style PEA) 10 | 25 |
| **Total** | | **100** |

Section 5.2 earns 10 points within 0.02 of the paper's value and 5 points within
0.05, provided the error bar is positive, at most 0.05, and consistent with the
distance (within three combined standard deviations). Hardware comparisons
(accuracy and trusted depth for each size) are printed as diagnostics and add
nothing to the 100 points.

**Remember that the top solutions will be carefully reviewed by the organizers.**

## 4. Getting started

In the QDC folder, run:

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.12 or later
pip install -r requirements.txt
```

Save your IBM Quantum account once with `QiskitRuntimeService.save_account(...)`,
then start Jupyter in this directory. Switches in the setup cell:

| Switch | Effect |
|---|---|
| `CONNECT_TO_QPU` | Read live backend data. If off, the offline `FakeBoston` snapshot is used |
| `RUN_QPU` | Allow new jobs to be submitted (this spends QPU time) |
| `RESUME_JOBS`, `LEDGER_TO_RESUME` | Retrieve the jobs already recorded in an earlier ledger instead of resubmitting them |

Job IDs, settings and billed seconds are recorded in `runs/walkthrough_*/`, so
rerunning a cell retrieves the earlier job.

## Directory layout

```
floquet-dynamics/
├── floquet_challenge.ipynb / .py   # challenge notebook
├── reference/        # exact statevector engine, colour-coded lattices, analysis, supplied traces
├── utils/            # helpers: references and diagnostics, plotting, ZNE...
└──doc/              # paper and figures

```

## References

- Paper: [arXiv:2607.24937](https://arxiv.org/abs/2607.24937)
- Classical reference details: [reference/README.md](reference/README.md)
