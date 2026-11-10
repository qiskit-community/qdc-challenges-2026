# Classical reference: exact statevector baseline

This folder holds the error-free ground truth for the QDC 2026 Floquet-dynamics
challenge, and the code that computes it: the magnetization of the mixed-field
Ising Floquet magnet on the paper's colour-coded heavy-hex patches, from **exact
statevector** simulation.

Two magnetizations are recorded each cycle (paper App. B):

- $M = \frac1{N_q}\sum_q\langle Z_q\rangle$: the full-lattice magnetization (Fig. 3a).
- $M_{z=2}$: the same average over only the degree-2 qubits. The paper's
  finite-size scaling (Fig. 3b/c) is built on this observable.

## Stored data

| File | Content |
|---|---|
| `traces_paper_colored_21q_v1.csv` | Exact, `complex128`, cycles 0–30 |
| `traces_paper_colored_28q_v1.csv` | Exact, `complex128`, cycles 0–30 |
| `traces_paper_colored_35q_v1.csv` | Exact, `complex64` (the only precision whose $2^{35}$ state fits one node), cycles 0–30 |
| `traces_paper_colored_*q_v1.json` | Provenance for each CSV: the model spec (ordered colour groups, angles, initial state, observables, cycles), its SHA-256 `model_signature`, and the generator and precision used |
| `paper_fig3_51q.csv` | The 51-qubit error-mitigated experiment, **digitized** from the paper's Fig. 3 (approximate, ± a few $10^{-3}$) |
| `traces.csv` | Legacy traces for an earlier automatic colouring. They define a different drive and must not be graded against the paper circuits |

The CSV columns are `n_qubits, n_z2, cycle, magnetization, magnetization_z2`.
The angles are $(\theta_x,\theta_z,\theta_{zz})=(\pi/6,\pi/27,\pi/3)$ and the
initial state is $\lvert0\rangle^{\otimes N}$.

## How the notebooks use this folder

- `utils/floquet_helpers.py` loads the `traces_paper_colored_*` files
  (`fh.REFERENCE_PATHS`) and `paper_fig3_51q.csv` as the comparison targets for
  the hardware diagnostics. The platform grader does not read them: it checks each
  exercise on small fixed probes (see `grader-floquet/EXERCISES.md`).
- `paper_colored_lattice(21 | 28 | 35 | 51 | 74)` in `lattice.py` gives each patch
  with the ordered three-colour bond layers transcribed from the paper's
  Fig. E1/A1. This ordering is what makes the circuit match the paper's drive.
- Part 1.3 uploads `lattice.py`, `simulator.py` and `backend.py` to **Qiskit
  Serverless** and recomputes the 21-qubit trace remotely.

## The model and the engine

One Floquet cycle (paper Eq. 1) is
$U_F = \prod_{r=1}^{3}\big[e^{-i\theta_{zz}/2\,C_r}\,e^{-i\theta_z/2\,Z_\Sigma}\,e^{-i\theta_x/2\,X_\Sigma}\big]$.
The product wraps the whole block, so each cycle interleaves three $R_X$/$R_Z$
layers with the three edge-coloured $ZZ$ layers $C_1, C_2, C_3$. $R_X$ does not
commute with $ZZ$, so the $ZZ$ layers cannot be merged. The effective drive per
cycle is $3\theta_x=\pi/2$.

`FloquetSimulator` (`simulator.py`) is a lean pure-NumPy engine, with an
optional CuPy GPU backend (`backend.py`):
- It stores only the state and one shared $R_Z$ diagonal, and applies each
  colour's $ZZ$ bonds as in-place phase multiplies.
- It reads both magnetizations in a single pass over the qubits.

```python
from reference import FloquetSimulator, paper_colored_lattice

result = FloquetSimulator(paper_colored_lattice(21)).run(30)
print(result.magnetization[:2], result.magnetization_z2[:2])
```

`analysis.py` fits the late-cycle damped cosine (Eq. 3) and the amplitude
scaling. `plotting.py` draws the paper-style Fig. 3 panels used by the notebooks.

## Regenerating the data (authors only)

Participants do not need to run this.

```bash
# from floquet-dynamics/
python -m reference.generate_paper_colored 21 28        # minutes on a laptop
python -m reference.generate_paper_colored 21 --verify-only   # diff against the stored CSV
sbatch reference/run_cpu_reference_paper.slurm          # 21/28/35 on a large-memory node
python -m reference.digitize_paper_51q                  # needs pdftoppm and Pillow
```

- `generate_paper_colored.py` picks the precision for each size (above), writes
  the CSV and its JSON record, and checks cycle 1 against known values.
- For long runs, `--checkpoint-dir` makes the 35-qubit job resumable after a
  wall-clock kill.
- The SLURM job uses `floquet-dynamics/.venv`; set `VENV_DIR` to use another environment.

### Memory limits

One `complex128` state vector takes $2^N\times16$ B, and the engine's peak is
about two to three state vectors. Halve every figure for `complex64`.

| Size | Precision | Peak memory | Where it runs |
|---|---|---|---|
| 21 qubits | `complex128` | negligible | anywhere |
| 28 qubits | `complex128` | ≈ 9 GiB | anywhere |
| 35 qubits | `complex64` | ≈ 620 GiB | a large-memory CPU node; one H100 (80 GB) cannot hold it |

The 51- and 74-qubit patches are beyond exact statevector simulation.

## Correctness

```bash
python -m pytest -q tests/test_reference.py     # from floquet-dynamics/
```

The tests check three things:
1. Every generated patch is a connected, degree-≤3, properly 3-edge-coloured heavy-hex lattice.
2. With $\theta_z=\theta_{zz}=0$, $M(n)=\cos(3n\theta_x)$ exactly.
3. The engine matches an independent Qiskit `Statevector` evolution to about $10^{-9}$.

The stored traces themselves are checked by `generate_paper_colored.py`
(the known cycle-1 values, and `--verify-only`).
