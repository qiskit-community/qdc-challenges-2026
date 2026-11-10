"""Exact statevector reference for the Floquet mixed-field Ising magnet.

Why statevector? It is the only method with **zero truncation knobs** -- no bond
dimension, no Pauli-weight cutoff, nothing to converge -- so participants get an
*exact* baseline without wasting time on convergence studies. It is also the
fastest route to the answer at the sizes reachable here (<= ~30 qubits on one
node) and maps cleanly onto CPU/GPU array backends.

Circuit structure
-----------------
One Floquet cycle is (Eq. 1, applied to the state right-to-left)

    U_F = prod_{r=1}^{3} [ exp(-i theta_zz/2 C_r) . exp(-i theta_z/2 Z_sum) . exp(-i theta_x/2 X_sum) ]

The product wraps the *whole* block: each cycle interleaves **three** ``X`` and
``Z`` layers between the three edge-coloured ``ZZ`` layers ``C_1, C_2, C_3``.
This matters -- the ``X`` layers do not commute with the ``ZZ`` layers, so the
``ZZ`` layers **cannot** be merged into a single diagonal. (Doing so simulates a
different unitary: one ``X`` per cycle, effective drive ``theta_x`` instead of
``3 theta_x`` -- the qubits then barely relax and the subharmonic period is
wrong. The paper's ``3 theta_x``/``3 theta_z`` phase-diagram axes are the tell.)

Memory model (lean)
-------------------
The dominant cost at the frontier (>= ~30 qubits) is memory, so the engine
stores **no** full ``2^N`` vector besides the state itself -- every diagonal
layer is applied on the fly:

* the single-qubit ``Z`` field ``exp(-i theta_z/2 Z_sum)`` -- identical in all
  three sub-layers -- commutes with every off-target ``RX`` and factorises over
  qubits, so it is **fused into the ``RX`` gate**: the fused single-qubit matrix
  ``diag(a, b) @ RX(theta_x)`` is applied in one pass, making ``RZ`` free (no
  separate traffic over the state, no materialised diagonal ``D_z``);
* each colour's ``ZZ`` layer is applied **on the fly**, bond by bond, as an
  in-place phase multiply on the two bits it couples (a strided view, no
  materialised diagonal);
* the fused ``RX``/``RZ`` layer uses one preallocated half-vector scratch.

Peak memory is therefore ~1 state-vector + one half-vector scratch (+ a
transient half-vector during the fused combine), i.e. ~2 state-vectors, versus
~2.5 when ``D_z`` was stored and ~5 for a three-stored-diagonal scheme -- the
difference between fitting and not fitting a single node at 34-35 qubits. The
only per-cycle traffic beyond the fused ``RX``/``RZ`` layer (``3N`` sub-layer
applications) is the per-bond ``ZZ`` passes, so the wall-clock cost is close to
a bare-``RX`` engine.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .backend import Backend, get_backend
from .lattice import HeavyHexLattice


@dataclass(frozen=True)
class FloquetParams:
    """Drive angles for one Floquet cycle.

    Defaults are the paper's selected point,
    ``(theta_x, theta_z, theta_zz) = (pi/6, pi/27, pi/3)``, near the Pareto front
    that balances a long-lived magnetization signal against entanglement growth.
    Each cycle applies the ``X``/``Z`` layers three times (once per colour
    sub-layer, see :class:`FloquetSimulator`), so the *effective* per-cycle single-
    qubit rotation is ``3 theta_x = pi/2`` -- hence the paper's ``3 theta_x`` axis
    and the period-~4 subharmonic response.

    Note: ``theta_z != 0`` is what *sustains* the long-lived subharmonic signal --
    at ``theta_z = 0`` the magnetization relaxes quickly (paper Fig. 2h). The
    ``RZ`` layer is diagonal (a virtual, near-free gate on hardware), so keep it
    in even though it costs nothing.
    """

    theta_x: float = math.pi / 6
    theta_z: float = math.pi / 27
    theta_zz: float = math.pi / 3


@dataclass
class FloquetResult:
    """Output of a reference run."""

    n_qubits: int
    cycles: np.ndarray             # integer cycle index 0 .. n_cycles
    magnetization: np.ndarray      # M(n) = (1/N) sum_q <Z_q>, one per cycle
    magnetization_z2: np.ndarray   # M_z2(n): average over coordination-2 qubits
    n_z2: int                      # |V_z2|, number of coordination-2 qubits
    params: FloquetParams
    device: str

    def as_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "cycles": self.cycles.tolist(),
            "magnetization": self.magnetization.tolist(),
            "magnetization_z2": self.magnetization_z2.tolist(),
            "n_z2": self.n_z2,
            "params": vars(self.params),
            "device": self.device,
        }


class FloquetSimulator:
    """Exact statevector engine for repeated applications of ``U_F``."""

    def __init__(
        self,
        lattice: HeavyHexLattice,
        params: FloquetParams | None = None,
        device: str = "auto",
        dtype: str = "complex128",
        threads: int | None = None,
    ):
        # Hard ceiling. The lean scheme stores no diagonal: resident is the state
        # + a half-vector scratch, peaking at ~2 state-vectors with the transient
        # half during the RX combine. So 35q complex128 is ~1 TiB peak (~512 GiB
        # at complex64) -- the practical frontier of a single large-memory node.
        # Above 36q even the state alone is out of reach, so refuse rather than
        # thrash.
        if lattice.n_qubits > 36:
            raise ValueError(
                f"{lattice.n_qubits} qubits exceeds the statevector limit (36). "
                "Use a smaller patch for the exact reference."
            )
        self.lattice = lattice
        self.params = params or FloquetParams()
        self.backend: Backend = get_backend(device, dtype)
        self.n = lattice.n_qubits
        self.dim = 1 << self.n

        # Fused RZ.RX single-qubit gate. Each cycle applies RX then the shared
        # RZ field to every qubit; RZ_q commutes with RX_{q'!=q}, so folding
        # RZ_q directly onto RX_q reproduces RZ_layer . RX_layer exactly. The
        # 2x2 is RZ.RX = diag(a, b) @ [[c, s], [s, c]] = [[a c, a s], [b s, b c]]
        # with c = cos(tx/2), s = -i sin(tx/2), and (a, b) = (exp(-i tz/2),
        # exp(+i tz/2)) the RZ field's per-qubit phases (a = b = 1 if tz == 0).
        # Applying this fused gate costs exactly what RX alone did, so the RZ
        # field is free: no separate layer pass and no materialised 2^N diagonal
        # D_z (which would be a whole extra state-vector -- 512 GiB at 35q c128).
        half = self.params.theta_x / 2.0
        c = self.backend.cdtype(math.cos(half))          # cos(tx/2)
        s = self.backend.cdtype(-1j * math.sin(half))    # -i sin(tx/2)
        a, b = self._build_rz_scalars()
        self._m00, self._m01 = a * c, a * s              # -> new bit=0 half
        self._m10, self._m11 = b * s, b * c              # -> new bit=1 half

        # ZZ bond phase, applied on the fly (see _apply_zz_layer): equal bits ->
        # exp(-i theta_zz/2), differing bits -> exp(+i theta_zz/2).
        zz = self.params.theta_zz / 2.0
        self._zz_eq = self.backend.cdtype(complex(math.cos(zz), -math.sin(zz)))
        self._zz_ne = self.backend.cdtype(complex(math.cos(zz), math.sin(zz)))
        self._color_groups = (self.lattice.color_groups() + [[], [], []])[:3]

        # Coordination-two qubit set V_{z=2}: the sites with exactly two
        # neighbours in the lattice. The paper's finite-size observable M_{z=2}
        # (Fig. 3b/c) averages <Z_q> over *only* these; the full-lattice M is
        # reported too. See _magnetizations.
        deg = self.lattice.degrees()
        self._z2 = tuple(q for q in range(self.n) if deg[q] == 2)
        self._n_z2 = len(self._z2)
        # One preallocated half-vector for the in-place RX combine (holds the
        # original top half while both halves are updated). dim/2 complex values.
        self._rx_scratch = self.backend.xp.empty(self.dim >> 1, dtype=self.backend.cdtype)

        # CPU worker pool for the elementwise layers. NumPy's ufuncs are
        # single-threaded and this engine is entirely elementwise (no BLAS), so
        # at the frontier a lone core is the bottleneck -- but every heavy kernel
        # releases the GIL, so splitting the state into disjoint chunks across
        # threads gives real, memory-bandwidth-bound parallelism. The GPU path is
        # already massively parallel on-device, so it stays single-call.
        self._threads = self._resolve_threads(threads, self.backend.is_gpu)
        self._pool = (
            ThreadPoolExecutor(max_workers=self._threads)
            if self._threads > 1 else None
        )

    @staticmethod
    def _resolve_threads(threads: int | None, is_gpu: bool) -> int:
        """Pick the CPU worker count: 1 on GPU, the caller's value if given, else
        the cores actually visible to the process (respects a Slurm cpuset)."""
        if is_gpu:
            return 1
        if threads is not None:
            return max(1, int(threads))
        try:
            return max(1, len(os.sched_getaffinity(0)))   # Linux: honours cpuset
        except AttributeError:                            # macOS / Windows
            return max(1, os.cpu_count() or 1)

    def _chunks(self, m: int) -> list[tuple[int, int]]:
        """Split ``range(m)`` into ``<= _threads`` contiguous ``(lo, hi)`` spans."""
        t = min(self._threads, m)
        step = -(-m // t)                     # ceil(m / t)
        return [(lo, min(lo + step, m)) for lo in range(0, m, step)]

    def _pmap(self, fn, chunks) -> None:
        """Run ``fn(lo, hi)`` on every chunk across the pool and block until done
        (the ``list`` is the barrier before the caller touches the state again)."""
        list(self._pool.map(lambda lh: fn(*lh), chunks))

    # -- circuit ingredients ------------------------------------------------- #

    def _build_rz_scalars(self):
        """The two per-qubit phases of the shared ``Z``-field
        ``exp(-i theta_z/2 sum_q z_q)``.

        The field is diagonal, *identical* in every colour sub-layer (Eq. 1
        re-applies it three times), and **factorises over qubits**:
        ``exp(-i theta_z/2 sum_q z_q) = prod_q diag(a, b)_q`` with ``a = exp(-i
        theta_z/2)`` (bit 0, ``z = +1``) and ``b = exp(+i theta_z/2)`` (bit 1,
        ``z = -1``). So there is nothing to materialise: these two scalars are
        folded straight into the ``RX`` gate (see ``__init__``), which is why the
        ``RZ`` field costs neither a stored ``2^N`` diagonal ``D_z`` (a whole
        extra state-vector -- 512 GiB at 35q complex128) nor a separate pass.

        Returns ``(1, 1)`` when ``theta_z == 0`` (the fused gate is then pure
        ``RX`` -- e.g. the decoupled closed-form check)."""
        cdtype = self.backend.cdtype
        if self.params.theta_z == 0.0:
            return cdtype(1.0), cdtype(1.0)
        coeff = self.params.theta_z / 2.0
        a = cdtype(complex(math.cos(coeff), -math.sin(coeff)))  # bit 0 -> z = +1
        b = cdtype(complex(math.cos(coeff), math.sin(coeff)))   # bit 1 -> z = -1
        return a, b

    def _apply_zz_layer(self, psi, group):
        """Apply colour-``group`` ``ZZ`` bonds ``exp(-i theta_zz/2 z_i z_j)`` in
        place, one bond at a time -- no materialised diagonal.

        For a bond ``(i, j)`` (``a = min, b = max``) reshape the state so bits
        ``a`` and ``b`` are separate axes, then scalar-multiply the four blocks:
        equal bits (``00``/``11``, ``z_i z_j = +1``) by ``exp(-i theta_zz/2)`` and
        differing bits (``01``/``10``, ``z_i z_j = -1``) by ``exp(+i theta_zz/2)``.
        Each multiply is an in-place op on a strided view -- a few % of the state
        touched per bond, no full-vector temporary.
        """
        if self.params.theta_zz == 0.0:
            return psi
        eq, ne = self._zz_eq, self._zz_ne
        for i, j in group:
            a, b = (i, j) if i < j else (j, i)
            hi = 1 << (self.n - 1 - b)   # blocks above bit b
            mid = 1 << (b - 1 - a)       # blocks between bits a and b
            lo = 1 << a                  # blocks below bit a
            v = psi.reshape(hi, 2, mid, 2, lo)
            v[:, 0, :, 0, :] *= eq       # bits (b=0, a=0) equal
            v[:, 1, :, 1, :] *= eq       # bits (b=1, a=1) equal
            v[:, 0, :, 1, :] *= ne       # differ
            v[:, 1, :, 0, :] *= ne       # differ
        return psi.reshape(self.dim)

    # -- evolution ----------------------------------------------------------- #

    @staticmethod
    def _rx_combine(v0, v1, tmp, m00, m01, m10, m11):
        """In-place fused ``RZ.RX`` mix ``[[m00, m01], [m10, m11]]`` of the paired
        half-views ``v0`` (bit = 0) and ``v1`` (bit = 1). The original ``v0`` is
        stashed in ``tmp`` so ``v1`` can use it after ``v0`` is overwritten, then
        both halves are updated with in-place AXPYs -- the only allocations are
        the two ``m * .`` products (freed on return). ``v0``, ``v1`` and ``tmp``
        are disjoint per worker, so calling this on several chunks concurrently is
        race-free."""
        tmp[...] = v0                      # stash original bit=0 half
        v0 *= m00                          # v0 <- m00*v0
        v0 += m01 * v1                     #      + m01*v1
        v1 *= m11                          # v1 <- m11*v1
        v1 += m10 * tmp                    #      + m10*(original v0)

    def _apply_rxrz_layer(self, psi):
        """Apply the fused ``RZ.RX`` gate to every qubit -- the ``RX`` rotation
        and the shared ``RZ`` field in one pass (the only off-diagonal work per
        sub-layer; ``ZZ`` follows separately).

        Per qubit, reshape to ``(left, 2, right)`` and combine the two halves via
        :meth:`_rx_combine` with the precomputed fused coefficients. Both the
        ``left`` (high bits) and ``right`` (low bits) axes index independent
        blocks, so on CPU the combine is split across the worker pool along
        whichever axis is larger -- each worker owns a disjoint block range and
        the matching disjoint slice of ``_rx_scratch``. On GPU (``_pool is None``)
        it runs as a single on-device call.
        """
        m00, m01, m10, m11 = self._m00, self._m01, self._m10, self._m11
        scratch = self._rx_scratch
        combine = self._rx_combine
        n = self.n
        for q in range(n):
            left = 1 << (n - 1 - q)        # blocks above bit q
            right = 1 << q                 # blocks below bit q
            v = psi.reshape(left, 2, right)
            tmp = scratch[: left * right].reshape(left, right)
            if self._pool is None or max(left, right) < 2 * self._threads:
                combine(v[:, 0, :], v[:, 1, :], tmp, m00, m01, m10, m11)
            elif left >= right:            # split the high-bit axis
                self._pmap(
                    lambda lo, hi: combine(
                        v[lo:hi, 0, :], v[lo:hi, 1, :], tmp[lo:hi],
                        m00, m01, m10, m11),
                    self._chunks(left),
                )
            else:                          # split the low-bit axis
                self._pmap(
                    lambda lo, hi: combine(
                        v[:, 0, lo:hi], v[:, 1, lo:hi], tmp[:, lo:hi],
                        m00, m01, m10, m11),
                    self._chunks(right),
                )
        return psi.reshape(self.dim)

    def _magnetizations(self, psi) -> tuple[float, float]:
        """Return ``(M, M_z2)`` for the current state.

        ``M = (1/N) sum_q <Z_q>`` is the full-lattice magnetization;
        ``M_z2 = (1/|V_z2|) sum_{q in V_z2} <Z_q>`` is the paper's coordination-
        two magnetization (Fig. 3b/c), averaging only over the degree-2 qubits.

        Both come from a single pass over the qubits: for each ``q`` we compute
        ``p_q = P(bit q = 1)`` as the summed probability of the ``bit==1`` half
        (a strided view), giving ``<Z_q> = 1 - 2 p_q``. This deliberately avoids
        a full-length ``|psi|^2`` (or popcount) buffer -- at 35 qubits the state
        and the ``Z`` diagonal already consume most of the node's memory, so the
        magnetization must not allocate another ``2^N`` vector. The transient is
        one ``bit==1`` half-slice squared, freed each iteration (and on CPU the
        reduction itself is threaded, see :meth:`_sumsq`).
        """
        psi = psi.reshape(self.dim)
        z2 = set(self._z2)
        z_total = 0.0
        z_sum_z2 = 0.0
        for q in range(self.n):
            left = 1 << (self.n - 1 - q)
            right = 1 << q
            hi = psi.reshape(left, 2, right)[:, 1, :]   # amplitudes with bit q = 1
            p1 = self._sumsq(hi)                         # P(bit q = 1)
            zq = 1.0 - 2.0 * p1
            z_total += zq
            if q in z2:
                z_sum_z2 += zq
        m = z_total / self.n
        m_z2 = z_sum_z2 / self._n_z2 if self._n_z2 else float("nan")
        return m, m_z2

    def _sumsq(self, arr) -> float:
        """``sum |arr|^2`` over a 2D (possibly strided) half-slice view.

        On CPU the larger axis is split into disjoint sub-reductions run across
        the pool and the partials are added in host float. Splitting changes the
        summation order, so the result differs from the serial reduction only at
        rounding level -- and chunked partials accumulated in Python float are if
        anything *more* accurate than one ``float32`` sweep at ``complex64``."""
        real_sq_sum = lambda a: float(
            self.backend.asnumpy((a.real ** 2 + a.imag ** 2).sum()))
        if self._pool is None or max(arr.shape) < 2 * self._threads:
            return real_sq_sum(arr)
        rows, cols = arr.shape
        along_rows = rows >= cols
        part = (lambda lh: real_sq_sum(arr[lh[0]:lh[1]])) if along_rows else \
               (lambda lh: real_sq_sum(arr[:, lh[0]:lh[1]]))
        return sum(self._pool.map(part, self._chunks(rows if along_rows else cols)))

    def _apply_cycle(self, psi):
        """One Floquet cycle ``U_F``: three ``(RX, RZ, colour-r ZZ)`` sub-layers
        (paper Eq. 1). The ``RX`` rotation is re-applied in each sub-layer -- it
        does not commute with the ``ZZ`` layers, so it cannot be pulled out. The
        shared ``RZ`` field is fused into the ``RX`` gate (one pass, see
        :meth:`_apply_rxrz_layer`), and the colour ``ZZ`` bonds are diagonal and
        applied on the fly per bond -- no materialised diagonal for either."""
        for group in self._color_groups:
            psi = self._apply_rxrz_layer(psi)       # exp(-i tz/2 Z).exp(-i tx/2 X)
            psi = self._apply_zz_layer(psi, group)  # exp(-i theta_zz/2 C_r)
        return psi

    def run(self, n_cycles: int, progress: bool = False) -> FloquetResult:
        """Evolve ``|0...0>`` for ``n_cycles`` Floquet cycles, recording ``M`` each
        cycle. Index 0 is the initial state (``M = 1``); index ``k`` is after ``k``
        full three-sub-layer cycles.

        ``progress=True`` prints a flushed heartbeat -- one line per cycle with the
        running ``M``/``M_z2``, elapsed time, and mean cost per cycle -- so a long
        run (e.g. 35q, where one cycle takes a while and there is otherwise no
        output until the whole size finishes) can be watched live, e.g. under
        ``srun``. The values printed are the ones already computed each cycle, so
        the heartbeat adds only the print itself."""
        xp = self.backend.xp
        psi = xp.zeros(self.dim, dtype=self.backend.cdtype)
        psi[0] = 1.0

        m0, m0_z2 = self._magnetizations(psi)
        mags = [m0]
        mags_z2 = [m0_z2]
        if progress:
            t0 = time.perf_counter()
            print(f"[{self.backend.name:>3}] N={self.n:>3}  running {n_cycles} "
                  f"cycles (dim=2^{self.n})", flush=True)
        for k in range(1, n_cycles + 1):
            psi = self._apply_cycle(psi)
            m, m_z2 = self._magnetizations(psi)
            mags.append(m)
            mags_z2.append(m_z2)
            if progress:
                dt = time.perf_counter() - t0
                print(f"[{self.backend.name:>3}] N={self.n:>3}  "
                      f"cycle {k:>3}/{n_cycles}  M={m:+.4f}  Mz2={m_z2:+.4f}  "
                      f"{dt:7.1f}s ({dt / k:5.1f}s/cyc)", flush=True)

        return FloquetResult(
            n_qubits=self.n,
            cycles=np.arange(n_cycles + 1),
            magnetization=np.asarray(mags, dtype=float),
            magnetization_z2=np.asarray(mags_z2, dtype=float),
            n_z2=self._n_z2,
            params=self.params,
            device=self.backend.name,
        )


def run_reference(
    n_qubits: int,
    n_cycles: int = 30,
    params: FloquetParams | None = None,
    device: str = "auto",
    dtype: str = "complex128",
    threads: int | None = None,
) -> FloquetResult:
    """Convenience one-shot: build an ``n_qubits`` heavy-hex patch and simulate."""
    from .lattice import heavy_hex

    lattice = heavy_hex(n_qubits)
    sim = FloquetSimulator(
        lattice, params=params, device=device, dtype=dtype, threads=threads)
    return sim.run(n_cycles)
