"""
classical_fragment_runner — Independent single-fragment classical solver execution.

Executes one FCI or CCSD fragment task in isolation, persists the full SolverResult
via ``fragment_result_store``, and optionally integrates with the :class:`FragmentTaskRegistry`
to claim a task, record results, and transition state.

This module replaces the classical execution path that was previously embedded inside
the monolithic ``step3_solve.run_solve``.  The scientific logic (FCI/CCSD solver calls,
Hamiltonian extraction from HDF5, RDM computation) is **unchanged** — only the
orchestration layer (multi-fragment loop, quantum routing, pkl-based transport) is
removed.

Design constraints
------------------
* Reads fragment Hamiltonians from the Stage 2 HDF5 dumpfile (``ewf_dumpfile.h5``).
* Writes results via ``fragment_result_store.save_fragment_result`` (frs/1 schema).
* Never modifies ``solver_results.pkl``, the quantum path, or EWF reconstruction internals.
* Never touches the QPU sampling ledger.
* Never imports SQD / TrimSQD / Vayesta at module level.
* Quantum fragments (SQD, TrimSQD) are explicitly rejected — use the existing
  ``step3_solve.run_solve`` path for those.

Integration with TaskRegistry (Task 5.3)
-----------------------------------------
``run_classical_fragment`` can be called in standalone mode (no registry) or with an
explicit ``task_id`` and ``registry`` for durable orchestration.  When a registry is
supplied, it transitions the task through READY → SUBMITTING → RUNNING → COMPLETE and
persists the artifact reference alongside the task.

Backward compatibility
-----------------------
``step3_solve.run_solve`` is unchanged.  Classical results produced by
``run_classical_fragment`` are stored in frs/1 HDF5 format and are compatible with
``run_reconstruct_from_refs`` (Task 5.1).  The legacy ``solver_results.pkl`` transport
is not produced by this runner.

Deprecation note
-----------------
The multi-fragment classical orchestration loop in ``step3_solve.run_solve`` (lines
583–798 for the FCI/CCSD path) remains in place because the quantum path still
depends on the full ``run_solve`` function.  Once Tasks 5.5–5.7 replace the quantum
submission path, the classical-only parts of ``run_solve`` can be removed.

Public API
----------
``run_classical_fragment(
    ewf_dumpfile_path, fragment_id, solver, result_h5_path, run_id, *,
    registry=None, task_id=None, lease_token=None, worker_id=None,
    compute_rdms=True, extra=None,
) -> ClassicalFragmentResult``
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "ClassicalFragmentResult",
    "ClassicalFragmentError",
    "run_classical_fragment",
    "load_fragment_hamiltonian",
    "validate_solver_config",
    "CLASSICAL_SOLVERS",
]

CLASSICAL_SOLVERS: frozenset[str] = frozenset({"FCI", "CCSD"})

# Allow-listed solver-constructor overrides (empty config = inherited defaults).
_SOLVER_CONFIG_KEYS: dict[str, dict[str, type]] = {
    "FCI":  {"conv_tol": float, "max_cycle": int},
    "CCSD": {"conv_tol": float, "max_cycle": int, "diis_space": int},
}


def validate_solver_config(solver: str, config: Optional[dict]) -> dict:
    """Return a validated copy of a solver configuration (allow-list; fail closed)."""
    if solver not in _SOLVER_CONFIG_KEYS:
        raise ClassicalFragmentError(f"no solver configuration schema for {solver!r}")
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ClassicalFragmentError("solver_config must be a dictionary")
    allowed = _SOLVER_CONFIG_KEYS[solver]
    unknown = sorted(set(config) - set(allowed))
    if unknown:
        raise ClassicalFragmentError(f"unsupported {solver} solver_config keys: {unknown}")
    out: dict = {}
    for key, value in config.items():
        kind = allowed[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ClassicalFragmentError(f"solver_config[{key!r}] must be numeric")
        if kind is int:
            if int(value) != value or int(value) < 1:
                raise ClassicalFragmentError(f"solver_config[{key!r}] must be a positive integer")
            out[key] = int(value)
        else:
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ClassicalFragmentError(f"solver_config[{key!r}] must be a positive finite number")
            out[key] = float(value)
    return out


class ClassicalFragmentError(RuntimeError):
    """A classical fragment solver failed or was given an invalid request."""


# ─────────────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassicalFragmentResult:
    """Outcome of a single classical fragment execution.

    Attributes
    ----------
    fragment_id         : fragment index (0-based)
    run_id              : run scoping namespace
    solver              : "FCI" or "CCSD"
    hamiltonian_sha256  : sha256(h1e.tobytes() + h2e.tobytes())
    energy              : total fragment energy (Ha)
    e_corr              : correlation energy contribution (Ha)
    converged           : solver convergence flag
    n_orb               : number of spatial orbitals
    n_occ               : number of occupied orbitals
    result_h5_path      : path to the persisted frs/1 HDF5 artifact
    fragment_ref        : FragmentResultReference from fragment_result_store
    elapsed_seconds     : wall-clock time for the solve
    software_versions   : {"pyscf": "...", "numpy": "...", "scipy": "..."}
    """
    fragment_id:        int
    run_id:             str
    solver:             str
    hamiltonian_sha256: str
    energy:             float
    e_corr:             float
    converged:          bool
    n_orb:              int
    n_occ:              int
    result_h5_path:     Path
    fragment_ref:       Any          # FragmentResultReference
    elapsed_seconds:    float
    software_versions:  dict

    def as_dict(self) -> dict:
        return {
            "fragment_id":        self.fragment_id,
            "run_id":             self.run_id,
            "solver":             self.solver,
            "hamiltonian_sha256": self.hamiltonian_sha256,
            "energy_ha":          self.energy,
            "e_corr_ha":          self.e_corr,
            "converged":          self.converged,
            "n_orb":              self.n_orb,
            "n_occ":              self.n_occ,
            "result_h5_path":     str(self.result_h5_path),
            "payload_digest":     self.fragment_ref.payload_digest,
            "file_sha256":        self.fragment_ref.file_sha256,
            "elapsed_seconds":    self.elapsed_seconds,
            "software":           self.software_versions,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Hamiltonian extraction (reuses the same HDF5 convention as step3_solve)
# ─────────────────────────────────────────────────────────────────────────────

def load_fragment_hamiltonian(
    ewf_dumpfile_path: str | Path,
    fragment_id: int,
) -> dict:
    """Extract the fragment Hamiltonian integrals from a Stage 2 HDF5 dumpfile.

    Returns a dict with keys:
        h1e, h2e  — numpy arrays (Hamiltonian integrals)
        norb      — number of spatial orbitals (int)
        nocc      — number of occupied orbitals (int)
        hamiltonian_sha256 — sha256(h1e.tobytes() + h2e.tobytes())

    Raises
    ------
    ClassicalFragmentError
        If the dumpfile is missing, the fragment group is absent, or the arrays
        cannot be read.
    """
    import h5py
    import numpy as np

    path = Path(ewf_dumpfile_path)
    if not path.is_file():
        raise ClassicalFragmentError(f"EWF dumpfile not found: {path}")

    frag_key = f"fragment_{int(fragment_id)}"
    try:
        with h5py.File(path, "r") as hf:
            if frag_key not in hf:
                raise ClassicalFragmentError(
                    f"Fragment {fragment_id} not found in {path} "
                    f"(available: {sorted(hf.keys())})"
                )
            g    = hf[frag_key]
            h1e  = g["heff"][:]
            h2e  = g["eris"][:]
            norb = int(g.attrs["norb"])
            nocc = int(g.attrs["nocc"])
    except ClassicalFragmentError:
        raise
    except Exception as exc:
        raise ClassicalFragmentError(
            f"Failed to read fragment {fragment_id} from {path}: {exc}"
        ) from exc

    ham_sha = hashlib.sha256(h1e.tobytes() + h2e.tobytes()).hexdigest()
    return {"h1e": h1e, "h2e": h2e, "norb": norb, "nocc": nocc,
            "hamiltonian_sha256": ham_sha}


# ─────────────────────────────────────────────────────────────────────────────
#  Software version helper (identical to step3_solve._software_versions)
# ─────────────────────────────────────────────────────────────────────────────

def _software_versions() -> dict:
    out = {}
    for name in ("pyscf", "numpy", "scipy"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "unavailable"
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Core runner
# ─────────────────────────────────────────────────────────────────────────────

def run_classical_fragment(
    ewf_dumpfile_path: str | Path,
    fragment_id: int,
    solver: str,
    result_h5_path: str | Path,
    run_id: str,
    *,
    compute_rdms: bool = True,
    solver_config: dict | None = None,
    registry=None,
    task_id: str | None = None,
    lease_token: str | None = None,
    worker_id: str | None = None,
    extra: dict | None = None,
) -> ClassicalFragmentResult:
    """Execute one classical (FCI or CCSD) fragment and persist the result.

    This is the authoritative entry point for independent classical fragment
    execution.  It replaces the FCI/CCSD branch of the multi-fragment loop
    inside ``step3_solve.run_solve`` for registry-coordinated orchestration.

    The existing ``step3_solve.run_solve`` is unchanged and continues to be
    used for: (a) legacy monolithic runs, (b) the quantum path (Tasks 5.5–5.7).

    Parameters
    ----------
    ewf_dumpfile_path   : path to the Stage 2 ``ewf_dumpfile.h5``
    fragment_id         : which fragment to solve (integer index)
    solver              : ``"FCI"`` or ``"CCSD"``
    result_h5_path      : where to write the frs/1 HDF5 result artifact
    run_id              : run scoping namespace (must match the registry task)
    compute_rdms        : compute and persist 1- and 2-RDMs (required for EWF
                          reconstruction; default True)
    registry            : optional :class:`FragmentTaskRegistry` instance
    task_id             : task ID in the registry (required when registry is given)
    lease_token         : fencing token from ``claim_task`` (required when using
                          the registry)
    worker_id           : identifier of the calling worker (for registry audit)
    extra               : optional dict stored in the result reference

    Returns
    -------
    :class:`ClassicalFragmentResult`

    Raises
    ------
    ClassicalFragmentError
        - solver is not FCI or CCSD
        - EWF dumpfile missing or fragment not found
        - Solver raises an exception
        - frs/1 persistence fails
        - registry transition fails (if registry is given)
    """
    # Reject quantum solvers immediately
    if solver not in CLASSICAL_SOLVERS:
        raise ClassicalFragmentError(
            f"run_classical_fragment only supports {sorted(CLASSICAL_SOLVERS)}, "
            f"got {solver!r}.  Use step3_solve.run_solve for SQD/TrimSQD."
        )
    if not run_id:
        raise ClassicalFragmentError("run_id must be a non-empty string")

    result_path = Path(result_h5_path)
    if result_path.suffix.lower() != ".h5":
        raise ClassicalFragmentError(
            f"result_h5_path must end in .h5, got {result_path}"
        )
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # ── frs/1 requires RDMs for persistence and reconstruction ────────────────
    # fragment_result_store (frs/1) requires rdm1/rdm2.  CCSD with
    # compute_rdms=False returns None.  Force True so the artifact is always
    # reconstruction-ready when using this runner.
    if not compute_rdms:
        import warnings
        warnings.warn(
            "run_classical_fragment: compute_rdms=False is not compatible with "
            "frs/1 persistence (RDMs are required).  Setting compute_rdms=True.",
            stacklevel=2,
        )
        compute_rdms = True

    # ── Optional registry: transition READY → SUBMITTING → RUNNING ───────────
    if registry is not None:
        if task_id is None:
            raise ClassicalFragmentError(
                "task_id is required when a registry is provided"
            )
        _transition(registry, task_id, "READY", "SUBMITTING",
                    worker_id=worker_id, lease_token=lease_token,
                    reason="classical solver starting")
        _transition(registry, task_id, "SUBMITTING", "RUNNING",
                    worker_id=worker_id, lease_token=lease_token,
                    reason="solver running")

    # ── Load Hamiltonian ──────────────────────────────────────────────────────
    ham = load_fragment_hamiltonian(ewf_dumpfile_path, fragment_id)
    h1e            = ham["h1e"]
    h2e            = ham["h2e"]
    norb           = ham["norb"]
    nocc           = ham["nocc"]
    ham_sha        = ham["hamiltonian_sha256"]

    # ── Instantiate and run the solver ────────────────────────────────────────
    config = validate_solver_config(solver, solver_config)
    t0 = time.time()
    try:
        solver_obj = _make_solver(solver, config)
        result = solver_obj.solve_from_integrals(
            h1e, h2e, norb, nocc,
            compute_rdms=compute_rdms,
        )
    except ClassicalFragmentError:
        raise
    except Exception as exc:
        err_msg = (
            f"Fragment {fragment_id} {solver} solver failed "
            f"({type(exc).__name__}: {exc})"
        )
        if registry is not None:
            try:
                registry.fail_task(task_id, reason=err_msg, lease_token=lease_token)
            except Exception:
                pass  # don't hide the original solver error
        raise ClassicalFragmentError(err_msg) from exc

    elapsed = time.time() - t0

    # ── Stamp result metadata (same convention as step3_solve) ────────────────
    result.metadata["solver_label"] = solver
    result.metadata["nocc"] = nocc
    result.metadata["hamiltonian_sha256"] = ham_sha

    e_corr    = float(result.metadata.get("e_corr", 0.0))
    converged = bool(result.metadata.get("converged", True))

    # ── Persist result via frs/1 ──────────────────────────────────────────────
    try:
        # Import lazily to avoid requiring fragment_result_store at module import time
        try:
            from fragment_result_store import save_fragment_result, FragmentStoreError
        except ImportError:
            from .fragment_result_store import save_fragment_result, FragmentStoreError  # type: ignore[no-redef]

        frag_ref = save_fragment_result(
            result,
            result_path,
            fragment_id=fragment_id,
            run_id=run_id,
            solver=solver,
            hamiltonian_sha256=ham_sha,
            extra=extra or {},
        )
    except Exception as exc:
        err_msg = (
            f"Fragment {fragment_id}: failed to persist SolverResult "
            f"({type(exc).__name__}: {exc})"
        )
        if registry is not None:
            try:
                registry.fail_task(task_id, reason=err_msg, lease_token=lease_token)
            except Exception:
                pass
        raise ClassicalFragmentError(err_msg) from exc

    # ── Optional registry: record result and transition to COMPLETE ───────────
    if registry is not None:
        try:
            registry.complete_task(
                task_id,
                artifact_uri=str(result_path),
                payload_digest=frag_ref.payload_digest,
                lease_token=lease_token,
                reason=f"{solver} fragment {fragment_id} completed",
            )
        except Exception as exc:
            raise ClassicalFragmentError(
                f"Fragment {fragment_id}: registry completion failed: {exc}"
            ) from exc

    return ClassicalFragmentResult(
        fragment_id=fragment_id,
        run_id=run_id,
        solver=solver,
        hamiltonian_sha256=ham_sha,
        energy=float(result.energy),
        e_corr=e_corr,
        converged=converged,
        n_orb=norb,
        n_occ=nocc,
        result_h5_path=result_path,
        fragment_ref=frag_ref,
        elapsed_seconds=elapsed,
        software_versions=_software_versions(),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_solver(solver: str, config: dict | None = None):
    """Instantiate the requested classical solver.

    Imports are deferred to avoid loading PySCF at module import time.
    This is identical to the import pattern in ``step3_solve.run_solve``.
    """
    try:
        from quantum_fragment_methods.application.solvers.classical_zoo.fci import FCI
        from quantum_fragment_methods.application.solvers.classical_zoo.ccsd import CCSD
    except ImportError as exc:
        raise ClassicalFragmentError(
            f"Cannot import classical solvers (is serverless_bundle on sys.path?): {exc}"
        ) from exc

    config = config or {}
    if solver == "FCI":
        return FCI(**config)
    elif solver == "CCSD":
        return CCSD(**config)
    else:
        raise ClassicalFragmentError(f"Unknown solver {solver!r}")


def _transition(registry, task_id, from_state, to_state, *, worker_id, lease_token, reason):
    """Wrap a registry transition; raises ClassicalFragmentError on failure."""
    try:
        registry.transition(
            task_id, from_state, to_state,
            worker_id=worker_id or "",
            lease_token=lease_token or "",
            reason=reason,
        )
    except Exception as exc:
        raise ClassicalFragmentError(
            f"Registry transition {from_state}→{to_state} for task {task_id!r} failed: {exc}"
        ) from exc
