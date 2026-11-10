"""Authoring-only regeneration of the versioned paper-coloured references.

Produces ``reference/traces_paper_colored_{N}q_v1.csv`` and its ``.json``
signature sidecar for ``N in {21, 28, 35}`` (and, in principle, 51 -- but 51
exceeds the exact-statevector cap).  The lattices come from the *ordered*
Fig. E1 colouring :func:`reference.lattice.paper_colored_lattice`, so the three
interleaved RX/RZ drives wrap the correct RZZ colour layers.

It drives the project's own lean statevector engine
(:class:`reference.simulator.FloquetSimulator`), which is pure NumPy (no Aer
dependency on the HPC node) and is validated to reproduce a Qiskit Aer 21q trace
to < 1e-7 across every cycle.  The engine respects ``lattice.color_groups()`` order,
so the paper colouring drives the paper dynamics.

The ``.json`` ``spec`` -- and therefore the SHA-256 ``model_signature`` the
grader/notebook checks -- is a function of the *model only* (ordered colour
groups, drive angles, initial state, observables, cycle labels).  It does not
depend on which engine or dtype computed the numbers, so a sidecar written here
validates the shipped CSV regardless of precision.

Precision
---------
Without ``--dtype``, each size uses the widest precision that fits a single
large-memory node: 21q and 28q run ``complex128`` (double), and 35q runs
``complex64`` (the only dtype whose 2^35 state fits). Pass ``--dtype`` to force
one precision for every requested size (``complex64`` at 21q is only
single-precision accurate, so its published-value self-check is relaxed).

Memory (single-node, lean engine ~ 2 state-vectors peak):

    N=21  complex128  negligible
    N=28  complex128  ~8.6 GiB peak
    N=35  complex64   state 275 GiB; ~550 GiB during evolution, ~620 GiB during
                      the threaded magnetization pass (large-memory node only;
                      see reference/run_cpu_reference_paper.slurm)
    N=35  complex128  ~1.1 TiB peak (infeasible)

Checkpoint / resume
-------------------
``--checkpoint-dir DIR`` makes a long run survive a wall-clock kill (e.g. a
24h queue limit at 35q). Semantics, per requested size, in order:

* a size whose completion marker ``DIR/done_{N}q`` exists is skipped (its CSV
  was already written by an earlier run);
* otherwise, if a checkpoint ``DIR/ckpt_{N}q.npz`` exists, the statevector and
  the cycles computed so far are loaded and evolution resumes from there;
* otherwise it starts from ``|0...0>``.

Only sizes whose state is at least ``--checkpoint-min-gib`` (default 32 GiB, so
in practice only 35q) write periodic checkpoints -- every ``--checkpoint-every``
cycles (default 1), atomically (write to a temp file, then rename). A checkpoint
is the full statevector, so budget ~2x the state in scratch disk during the
rename (~550 GiB at 35q c64). On successful completion the CSV+JSON are written,
the ``done`` marker is created, and the checkpoint file is deleted.

Examples
--------
    # One job, all three sizes, 35q checkpointed and resumable:
    python -m reference.generate_paper_colored 21 28 35 \
        --device cpu --threads 96 --checkpoint-dir "$SCRATCH/floquet_ckpt"
    # ... if killed at the wall, resubmit the identical command: 21/28 are
    # skipped (done markers) and 35q resumes from its last checkpoint.

    # Re-derive a trace and diff it against the shipped CSV without writing
    python -m reference.generate_paper_colored 21 --verify-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .lattice import paper_colored_lattice
from .simulator import FloquetParams, FloquetResult, FloquetSimulator

HERE = Path(__file__).parent
N_CYCLES = 30
SOURCE = "arXiv:2607.24937v1 Fig. E1; ordered layers 1, 2, 3"

# Known cross-checks: cycle-1 (M, M_z2) at double precision. 21q is the
# independently published Aer value; 28q is this engine's double-precision
# result (regression guard). Tolerance is relaxed for complex64 (see _check).
KNOWN_CYCLE1 = {
    21: (0.38207767, 0.36568544),
    28: (0.38432551, 0.35998222),
}


def default_dtype(n_qubits: int) -> str:
    """Widest precision that fits one node: double below 35q, single at 35q."""
    return "complex64" if n_qubits >= 35 else "complex128"


def state_bytes(n_qubits: int, dtype: str) -> int:
    itemsize = 8 if dtype == "complex64" else 16
    return (1 << n_qubits) * itemsize


def build_spec(lattice, params: FloquetParams) -> dict:
    """The model spec whose SHA-256 is the versioned ``model_signature``.

    Must keep the keys ``tests`` recompute, or the signature check fails.
    """
    return {
        "source": SOURCE,
        "n_qubits": lattice.n_qubits,
        "groups": lattice.color_groups(),
        "angles": [params.theta_x, params.theta_z, params.theta_zz],
        "initial_state": "0" * lattice.n_qubits,
        "observables": ["mean single Z", "mean single Z on degree-two sites"],
        "cycles": list(range(N_CYCLES + 1)),
    }


def signature_of(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def _ckpt_file(ckpt_dir: Path, n: int) -> Path:
    return ckpt_dir / f"ckpt_{n}q.npz"


def _done_file(ckpt_dir: Path, n: int) -> Path:
    return ckpt_dir / f"done_{n}q"


def _save_checkpoint(path: Path, psi_host, completed: int, mags, mags_z2,
                     n: int, dtype: str) -> None:
    """Atomically write the full statevector + progress to ``path`` (.npz).

    Writes a temp file next to ``path`` and ``os.replace``s it in, so an
    interrupted write never corrupts the live checkpoint. The old checkpoint
    coexists with the temp until the rename, so ~2x the state size is needed in
    the directory during the write.
    """
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.savez(fh, psi=psi_host, completed=np.int64(completed),
                 mags=np.asarray(mags, dtype=float),
                 mags_z2=np.asarray(mags_z2, dtype=float),
                 n_qubits=np.int64(n), dtype=np.str_(dtype))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _load_checkpoint(path: Path, n: int, dtype: str):
    data = np.load(path, allow_pickle=False)
    if int(data["n_qubits"]) != n or str(data["dtype"]) != dtype:
        raise ValueError(
            f"checkpoint {path} is for N={int(data['n_qubits'])} "
            f"{str(data['dtype'])}, not N={n} {dtype}; remove it to restart"
        )
    return (data["psi"], int(data["completed"]),
            [float(x) for x in data["mags"]],
            [float(x) for x in data["mags_z2"]])


def evolve(sim: FloquetSimulator, n: int, n_cycles: int, dtype: str,
           ckpt_dir: Path | None, every: int, min_bytes: int,
           progress: bool = True) -> FloquetResult:
    """Run ``n_cycles`` Floquet cycles with optional checkpoint/resume.

    Uses the same per-cycle operations as :meth:`FloquetSimulator.run`
    (``_apply_cycle`` then ``_magnetizations``) so results are identical; the
    only additions are statevector checkpoints and resume.
    """
    xp = sim.backend.xp
    use_ckpt = ckpt_dir is not None and state_bytes(n, dtype) >= min_bytes
    ckpt_path = _ckpt_file(ckpt_dir, n) if ckpt_dir is not None else None

    if use_ckpt and ckpt_path.exists():
        psi_host, start, mags, mags_z2 = _load_checkpoint(ckpt_path, n, dtype)
        psi = xp.asarray(psi_host, dtype=sim.backend.cdtype).reshape(sim.dim)
        del psi_host
        print(f"[resume] N={n:>3}  from cycle {start}/{n_cycles}", flush=True)
    else:
        psi = xp.zeros(sim.dim, dtype=sim.backend.cdtype)
        psi[0] = 1.0
        m0, m0_z2 = sim._magnetizations(psi)
        mags, mags_z2 = [m0], [m0_z2]
        start = 0

    t0 = time.perf_counter()
    if progress:
        print(f"[{sim.backend.name:>3}] N={n:>3}  running {n_cycles} cycles "
              f"(dim=2^{n}, start={start}, dtype={dtype})", flush=True)
    for k in range(start + 1, n_cycles + 1):
        psi = sim._apply_cycle(psi)
        m, m_z2 = sim._magnetizations(psi)
        mags.append(m)
        mags_z2.append(m_z2)
        if progress:
            dt = time.perf_counter() - t0
            done = k - start
            print(f"[{sim.backend.name:>3}] N={n:>3}  cycle {k:>3}/{n_cycles}  "
                  f"M={m:+.4f}  Mz2={m_z2:+.4f}  {dt:7.1f}s "
                  f"({dt / done:5.1f}s/cyc)", flush=True)
        if use_ckpt and k < n_cycles and k % every == 0:
            tw = time.perf_counter()
            _save_checkpoint(ckpt_path, sim.backend.asnumpy(psi), k,
                             mags, mags_z2, n, dtype)
            if progress:
                print(f"[ckpt] N={n:>3}  saved after cycle {k} "
                      f"({time.perf_counter() - tw:.1f}s)", flush=True)

    return FloquetResult(
        n_qubits=n,
        cycles=np.arange(n_cycles + 1),
        magnetization=np.asarray(mags, dtype=float),
        magnetization_z2=np.asarray(mags_z2, dtype=float),
        n_z2=sim._n_z2,
        params=sim.params,
        device=sim.backend.name,
    )


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_outputs(res: FloquetResult, dtype: str, device: str) -> tuple[Path, str]:
    lattice = paper_colored_lattice(res.n_qubits)
    spec = build_spec(lattice, FloquetParams())
    signature = signature_of(spec)
    csv_path = HERE / f"traces_paper_colored_{res.n_qubits}q_v1.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("n_qubits", "n_z2", "cycle",
                         "magnetization", "magnetization_z2"))
        for cycle, m, mz2 in zip(res.cycles, res.magnetization,
                                 res.magnetization_z2):
            writer.writerow((res.n_qubits, res.n_z2, int(cycle),
                             f"{m:.12f}", f"{mz2:.12f}"))
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "model_signature": signature,
        "spec": spec,
        "generator": "reference/generate_paper_colored.py",
        "method": (f"reference.FloquetSimulator {dtype} statevector "
                   f"(device={device})"),
    }, indent=2) + "\n")
    return csv_path, signature


def write_json_only(n: int) -> str:
    lattice = paper_colored_lattice(n)
    spec = build_spec(lattice, FloquetParams())
    signature = signature_of(spec)
    json_path = HERE / f"traces_paper_colored_{n}q_v1.json"
    json_path.write_text(json.dumps({
        "model_signature": signature,
        "spec": spec,
        "generator": "reference/generate_paper_colored.py",
        "method": "model-only sidecar for the shipped CSV",
    }, indent=2) + "\n")
    return signature


def verify_against_existing(res: FloquetResult) -> None:
    """Diff a freshly computed trace against the shipped CSV (no writes)."""
    csv_path = HERE / f"traces_paper_colored_{res.n_qubits}q_v1.csv"
    if not csv_path.exists():
        print(f"no existing {csv_path.name} to compare against")
        return
    rows = np.genfromtxt(csv_path, delimiter=",", names=True)
    sel = rows["n_qubits"] == res.n_qubits
    order = np.argsort(rows["cycle"][sel])
    old_m = rows["magnetization"][sel][order]
    old_mz2 = rows["magnetization_z2"][sel][order]
    dm = float(np.max(np.abs(old_m - res.magnetization)))
    dmz2 = float(np.max(np.abs(old_mz2 - res.magnetization_z2)))
    print(f"max |Δ| vs shipped {csv_path.name}:  M={dm:.2e}  Mz2={dmz2:.2e}")


def _check(n: int, c1: tuple[float, float], dtype: str) -> None:
    if n not in KNOWN_CYCLE1:
        return
    atol = 1e-5 if dtype == "complex64" else 1e-7
    np.testing.assert_allclose(c1, KNOWN_CYCLE1[n], atol=atol, rtol=0)
    print(f"       cycle1 matches published {KNOWN_CYCLE1[n]} (atol={atol:g})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m reference.generate_paper_colored")
    p.add_argument("sizes", type=int, nargs="+", help="one or more of 21, 28, 35")
    p.add_argument("--dtype", choices=["complex128", "complex64"], default=None,
                   help="force one precision for every size "
                        "(default: 128 for 21/28, 64 for 35)")
    p.add_argument("--device", choices=["auto", "cpu", "gpu", "cuda"],
                   default="cpu")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="enable resume: skip finished sizes, checkpoint large "
                        "states, resume from the last checkpoint")
    p.add_argument("--checkpoint-every", type=int, default=1,
                   help="cycles between statevector checkpoints (default 1)")
    p.add_argument("--checkpoint-min-gib", type=float, default=32.0,
                   help="only checkpoint sizes whose state is at least this "
                        "many GiB (default 32 -> in practice only 35q)")
    p.add_argument("--verify-only", action="store_true",
                   help="compute and diff against the shipped CSV; write nothing")
    p.add_argument("--json-only", action="store_true",
                   help="write only the .json signature sidecar for an existing, "
                        "already-validated CSV; do not run the simulator")
    args = p.parse_args(argv)

    if args.checkpoint_every < 1:
        p.error("--checkpoint-every must be >= 1")
    min_bytes = int(args.checkpoint_min_gib * (1 << 30))
    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is not None and not (args.verify_only or args.json_only):
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for n in args.sizes:
        if n not in (21, 28, 35):
            p.error(f"paper layer transcriptions exist for 21/28/35, got {n}")
        dtype = args.dtype or default_dtype(n)

        if args.json_only:
            sig = write_json_only(n)
            print(f"N={n:>3}  wrote traces_paper_colored_{n}q_v1.json  "
                  f"signature={sig[:16]}…")
            continue

        if ckpt_dir is not None and not args.verify_only \
                and _done_file(ckpt_dir, n).exists():
            print(f"N={n:>3}  already complete (marker "
                  f"{_done_file(ckpt_dir, n).name}); skipping")
            continue

        sim = FloquetSimulator(paper_colored_lattice(n), params=FloquetParams(),
                               device=args.device, dtype=dtype,
                               threads=args.threads)
        res = evolve(sim, n, N_CYCLES, dtype, ckpt_dir,
                     args.checkpoint_every, min_bytes)
        c1 = (res.magnetization[1], res.magnetization_z2[1])
        print(f"N={n:>3}  n_z2={res.n_z2}  cycle1 M={c1[0]:.8f}  Mz2={c1[1]:.8f}")
        _check(n, c1, dtype)

        if args.verify_only:
            verify_against_existing(res)
            continue

        csv_path, signature = write_outputs(res, dtype, args.device)
        print(f"       wrote {csv_path.name} + .json  signature={signature[:16]}…")

        if ckpt_dir is not None:
            _done_file(ckpt_dir, n).write_text("")
            ckpt_path = _ckpt_file(ckpt_dir, n)
            if ckpt_path.exists():
                ckpt_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
