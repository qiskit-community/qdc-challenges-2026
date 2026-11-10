"""Array-backend selection for CPU (NumPy) / CUDA GPU (CuPy).

The reference simulator is written against a NumPy-compatible array module so the
*same* code runs on:

* AArch64 macOS / Linux and x86 Windows / Linux  -> NumPy (CPU)
* any CUDA machine (x86 or AArch64, e.g. an H100 node) -> CuPy (GPU)

GPU is CUDA-only (per the challenge spec); there is no macOS GPU path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Public dtype aliases: participants pick precision vs. reach/speed.
DTYPES = {
    "complex128": (np.complex128, np.float64),
    "complex64": (np.complex64, np.float32),
    "c128": (np.complex128, np.float64),
    "c64": (np.complex64, np.float32),
}


@dataclass(frozen=True)
class Backend:
    """A resolved compute backend: an array module plus dtypes."""

    xp: Any            # numpy or cupy module
    name: str          # "cpu" or "gpu"
    cdtype: Any        # complex dtype for the statevector
    rdtype: Any        # matching real dtype for phase accumulators

    @property
    def is_gpu(self) -> bool:
        return self.name == "gpu"

    def asnumpy(self, a):
        """Return a host NumPy array regardless of backend."""
        if self.is_gpu:
            import cupy as cp  # local import; only needed on GPU

            return cp.asnumpy(a)
        return np.asarray(a)


def get_backend(device: str = "auto", dtype: str = "complex128") -> Backend:
    """Resolve a compute backend.

    Parameters
    ----------
    device : {"auto", "cpu", "gpu", "cuda"}
        ``"auto"`` uses the GPU if CuPy + a CUDA device are available, else CPU.
    dtype : {"complex128", "complex64", "c128", "c64"}
        Statevector precision. ``complex64`` roughly doubles the reachable qubit
        count and speeds up the GPU, at reduced precision.
    """
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    cdtype, rdtype = DTYPES[dtype]

    device = device.lower()
    if device in ("gpu", "cuda"):
        xp = _import_cupy(required=True)
        return Backend(xp=xp, name="gpu", cdtype=cdtype, rdtype=rdtype)
    if device == "cpu":
        return Backend(xp=np, name="cpu", cdtype=cdtype, rdtype=rdtype)
    if device == "auto":
        xp = _import_cupy(required=False)
        if xp is not None:
            return Backend(xp=xp, name="gpu", cdtype=cdtype, rdtype=rdtype)
        return Backend(xp=np, name="cpu", cdtype=cdtype, rdtype=rdtype)
    raise ValueError(f"device must be auto|cpu|gpu, got {device!r}")


def _import_cupy(required: bool):
    try:
        import cupy as cp

        # Fail fast if no usable CUDA device is present.
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy imported but no CUDA device found")
        return cp
    except Exception as exc:  # ImportError or CUDA runtime error
        if required:
            raise RuntimeError(
                "GPU backend requested but CuPy/CUDA is unavailable "
                f"({type(exc).__name__}: {exc}). "
                "Common causes: the Slurm job has no GPU allocated "
                "(missing/wrong --gres), or the installed CuPy wheel does not "
                "match the node's CUDA driver. Install a matching CuPy wheel "
                "(e.g. `pip install cupy-cuda12x`) on a CUDA machine, or use "
                "device='cpu'."
            ) from exc
        return None
