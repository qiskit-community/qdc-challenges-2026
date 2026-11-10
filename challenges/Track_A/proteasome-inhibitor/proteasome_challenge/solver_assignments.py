"""
Extensible, declarative per-fragment solver allocation and contracts.

Defines:
- SolverContract: Scientific specification contract for EWF fragment solvers
  (verified HDF5 inputs, RDM conventions, required outputs, orbital basis, provenance).
- Solver registry: Extensible registration of classical and quantum solvers
  (FCI, CCSD, SQD, TrimSQD).  New solvers (e.g. DMRG) register a contract and
  an optional executable factory.
- AllocationRule & SolverAllocationPolicy: Declarative, rule-based and
  override-supported sizing policies.
- assign_solvers_by_size & generate_solver_assignments: Derivation of per-fragment
  solver maps with provenance digest.
- Solver result validation: Enforces the scientific contract before Stage 4.

Verified scientific conventions (Pass 2C)
------------------------------------------
* HDF5 datasets: ``heff`` (norb,norb), ``eris`` (norb,norb,norb,norb), ``c_cluster``,
  ``c_frag``, ``fock``; attributes ``norb``, ``nocc``, ``nvir``.
* ``eris[p,q,r,s] = (pq|rs)`` — chemist-notation Coulomb integrals, 8-fold real
  symmetry, not antisymmetrized.
* ``rdm1[p,q]`` — spatial, spin-summed; Tr(γ₁) = N_elec; same basis as HDF5.
* ``rdm2[p,q,r,s]`` — spatial, spin-summed; ``E = h1·γ₁ + 0.5·(pq|rs)·γ₂[p,q,r,s]``.
  Tr(γ₂[p,p,q,q]) = N(N-1).
* CCSD RDMs are back-transformed to the input cluster-MO basis inside
  ``solve_from_integrals`` (numerically verified; see ``test_ccsd_basis.py``).
* SQD/TrimSQD spin assembly ``dm2_aa + dm2_ab + dm2_ab.T(2,3,0,1) + dm2_bb``
  is correct (numerically verified in ``test_rdm_convention.py``).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "VALID_SOLVERS",
    "QUANTUM_SOLVERS",
    "SolverContract",
    "SolverAllocationError",
    "AllocationRule",
    "SolverAllocationPolicy",
    "register_solver",
    "register_solver_factory",
    "get_solver_contract",
    "get_solver_factory",
    "list_registered_solvers",
    "is_valid_solver",
    "is_executable_solver",
    "assign_solvers_by_size",
    "generate_solver_assignments",
    "validate_solver_result",
    "load_solver_assignments",
    "validate_assignments",
    "get_assignment",
    "assignments_to_step3_params",
]


class SolverAllocationError(ValueError):
    """Raised when solver allocation, rule evaluation, or contract validation fails."""


@dataclass(frozen=True)
class SolverContract:
    """Scientific specification contract for an EWF fragment solver.

    Parameters
    ----------
    identifier : str
        Canonical solver name (e.g. 'FCI', 'CCSD', 'TrimSQD', 'DMRG').
    kind : str
        Algorithm kind: 'classical', 'quantum', or 'hybrid'.
    required_hdf5_datasets : tuple[str, ...]
        Required HDF5 datasets in the EWF dumpfile fragment group.
        Verified names: ('heff', 'eris', 'c_cluster', 'c_frag').
    required_hdf5_attrs : tuple[str, ...]
        Required HDF5 attributes in the fragment group attrs dict.
        Verified names: ('norb', 'nocc', 'nvir').
    required_metadata : tuple[str, ...]
        Required keys in ``SolverResult.metadata`` after the solve.
        Minimum: ('solver_label', 'nocc').
    rdm1_convention : str
        Human-readable description of the 1-RDM convention.
    rdm2_convention : str
        Human-readable description of the 2-RDM convention.
    orbital_basis : str
        Orbital basis of the returned RDMs (default: 'cluster_mo_basis').
    description : str
        Short scientific description.
    """
    identifier: str
    kind: str = "classical"
    # ── HDF5 input contract ──────────────────────────────────────────────────
    required_hdf5_datasets: tuple[str, ...] = ("heff", "eris", "c_cluster", "c_frag")
    required_hdf5_attrs: tuple[str, ...] = ("norb", "nocc", "nvir")
    # ── Output contract ───────────────────────────────────────────────────────
    required_metadata: tuple[str, ...] = ("solver_label", "nocc")
    rdm1_convention: str = (
        "rdm1[p,q] = <p†q> spatial spin-summed; Tr(γ₁)=N_elec; basis=cluster_mo"
    )
    rdm2_convention: str = (
        "rdm2[p,q,r,s]: E = h1·γ₁ + 0.5·(pq|rs)·γ₂[p,q,r,s]; "
        "Tr(γ₂[p,p,q,q])=N(N-1); spatial spin-summed; basis=cluster_mo"
    )
    orbital_basis: str = "cluster_mo_basis"
    description: str = ""

    def __post_init__(self):
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise SolverAllocationError(f"solver identifier must be a non-empty string, got {self.identifier!r}")
        if self.kind not in ("classical", "quantum", "hybrid"):
            raise SolverAllocationError(f"solver kind must be 'classical', 'quantum', or 'hybrid', got {self.kind!r}")


# ─────────────────────────────────────────────────────────────────────────────
#  Solver Registry
# ─────────────────────────────────────────────────────────────────────────────

_BUILTIN_SOLVERS: dict[str, SolverContract] = {
    "FCI": SolverContract(
        identifier="FCI",
        kind="classical",
        required_hdf5_datasets=("heff", "eris", "c_cluster", "c_frag"),
        required_hdf5_attrs=("norb", "nocc", "nvir"),
        required_metadata=("solver_label", "nocc", "converged"),
        description="Exact Full Configuration Interaction within active cluster space",
    ),
    "CCSD": SolverContract(
        identifier="CCSD",
        kind="classical",
        required_hdf5_datasets=("heff", "eris", "c_cluster", "c_frag"),
        required_hdf5_attrs=("norb", "nocc", "nvir"),
        required_metadata=("solver_label", "nocc", "converged", "e_corr"),
        description="Coupled Cluster Singles and Doubles",
    ),
    "SQD": SolverContract(
        identifier="SQD",
        kind="quantum",
        required_hdf5_datasets=("heff", "eris", "c_cluster", "c_frag"),
        required_hdf5_attrs=("norb", "nocc", "nvir"),
        required_metadata=(
            "solver_label", "nocc",
            "measurement_sha256", "sqd_config", "sqd_config_hash", "sqd_version",
        ),
        description="Sample-based Quantum Diagonalization",
    ),
    "TrimSQD": SolverContract(
        identifier="TrimSQD",
        kind="quantum",
        required_hdf5_datasets=("heff", "eris", "c_cluster", "c_frag"),
        required_hdf5_attrs=("norb", "nocc", "nvir"),
        required_metadata=(
            "solver_label", "nocc",
            "measurement_sha256", "sqd_config", "sqd_config_hash", "sqd_version",
            "policy", "trim_ratio",
        ),
        description="Trimmed Subspace Quantum Diagonalization",
    ),
}

_SOLVER_REGISTRY: dict[str, SolverContract] = dict(_BUILTIN_SOLVERS)

# Solver factory registry: identifier → callable(**kwargs) → BaseSolver instance.
# Only built-in solvers have factories; third-party solvers must register their own.
# Factories are NOT populated at import time (avoids importing heavy QPU libs).
_SOLVER_FACTORIES: dict[str, Callable[..., Any]] = {}


def register_solver(contract: SolverContract, allow_override: bool = False) -> None:
    """Register a new solver contract into the active registry.

    Registration of a contract alone does NOT mean the solver is executable.
    Use :func:`register_solver_factory` to also make it dispatchable.
    """
    if not isinstance(contract, SolverContract):
        raise SolverAllocationError(f"contract must be a SolverContract instance, got {contract!r}")
    key = contract.identifier.strip()
    if key in _SOLVER_REGISTRY and not allow_override:
        raise SolverAllocationError(f"solver {key!r} is already registered; use allow_override=True to replace")
    _SOLVER_REGISTRY[key] = contract


def register_solver_factory(
    identifier: str,
    factory: Callable[..., Any],
    allow_override: bool = False,
) -> None:
    """Register an executable factory for a previously registered solver.

    The factory must accept keyword arguments and return a solver instance
    that implements ``solve_from_integrals(h1e, h2e, norb, nocc, **kwargs)``.

    Parameters
    ----------
    identifier : str
        Registered solver identifier (must already be in the registry).
    factory : callable
        ``factory(**kwargs) → solver_instance``
    allow_override : bool
        If True, replace an existing factory without raising.

    Example — registering a dummy solver for testing::

        def my_factory(**kwargs):
            return MyDummySolver(**kwargs)

        register_solver(SolverContract(identifier="DUMMY", kind="classical"))
        register_solver_factory("DUMMY", my_factory)
    """
    if not isinstance(identifier, str) or not identifier.strip():
        raise SolverAllocationError("identifier must be a non-empty string")
    key = identifier.strip()
    if key not in _SOLVER_REGISTRY:
        raise SolverAllocationError(
            f"solver {key!r} is not registered; call register_solver() first"
        )
    if key in _SOLVER_FACTORIES and not allow_override:
        raise SolverAllocationError(
            f"factory for solver {key!r} already registered; use allow_override=True to replace"
        )
    if not callable(factory):
        raise SolverAllocationError(f"factory for solver {key!r} must be callable, got {factory!r}")
    _SOLVER_FACTORIES[key] = factory


def get_solver_contract(identifier: str) -> SolverContract:
    """Retrieve the SolverContract for an identifier, or raise SolverAllocationError."""
    if not isinstance(identifier, str):
        raise SolverAllocationError(f"identifier must be a string, got {identifier!r}")
    key = identifier.strip()
    if key not in _SOLVER_REGISTRY:
        raise SolverAllocationError(
            f"unregistered solver {identifier!r}; available: {sorted(_SOLVER_REGISTRY.keys())}"
        )
    return _SOLVER_REGISTRY[key]


def get_solver_factory(identifier: str) -> Callable[..., Any]:
    """Return the executable factory for a registered solver.

    Raises
    ------
    SolverAllocationError
        If the solver is not registered at all.
    SolverAllocationError
        If the solver is registered but has no factory (i.e. it is a
        contract-only registration, not yet dispatchable).
    """
    if not isinstance(identifier, str):
        raise SolverAllocationError(f"identifier must be a string, got {identifier!r}")
    key = identifier.strip()
    if key not in _SOLVER_REGISTRY:
        raise SolverAllocationError(
            f"unregistered solver {identifier!r}; available: {sorted(_SOLVER_REGISTRY.keys())}"
        )
    if key not in _SOLVER_FACTORIES:
        raise SolverAllocationError(
            f"solver {key!r} is registered but has no executable factory. "
            f"Call register_solver_factory({key!r}, ...) to make it dispatchable."
        )
    return _SOLVER_FACTORIES[key]


def list_registered_solvers() -> list[str]:
    """List all currently registered solver identifiers."""
    return sorted(_SOLVER_REGISTRY.keys())


def is_valid_solver(identifier: str) -> bool:
    """Check if identifier is registered in the solver registry."""
    return isinstance(identifier, str) and identifier.strip() in _SOLVER_REGISTRY


def is_executable_solver(identifier: str) -> bool:
    """Return True if identifier is registered AND has an executable factory."""
    return is_valid_solver(identifier) and identifier.strip() in _SOLVER_FACTORIES


VALID_SOLVERS = frozenset({"FCI", "CCSD", "SQD", "TrimSQD"})
QUANTUM_SOLVERS = frozenset({"SQD", "TrimSQD"})


# ─────────────────────────────────────────────────────────────────────────────
#  Allocation Rules & Policy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AllocationRule:
    """Declarative sizing rule mapping orbital count intervals to a solver.

    Parameters
    ----------
    solver : str
        Registered solver identifier.
    min_norb : int
        Minimum number of orbitals (inclusive, default: 1).
    max_norb : int | None
        Maximum number of orbitals (inclusive, None for no upper bound).
    description : str
        Human-readable description of this rule.
    """
    solver: str
    min_norb: int = 1
    max_norb: Optional[int] = None
    description: str = ""

    def __post_init__(self):
        if not isinstance(self.solver, str) or not self.solver.strip():
            raise SolverAllocationError(f"solver must be a non-empty string, got {self.solver!r}")
        if not is_valid_solver(self.solver):
            raise SolverAllocationError(f"unregistered solver {self.solver!r} in rule; register it first")
        if self.min_norb < 1:
            raise SolverAllocationError(f"min_norb must be >= 1, got {self.min_norb}")
        if self.max_norb is not None and self.max_norb < self.min_norb:
            raise SolverAllocationError(f"max_norb ({self.max_norb}) cannot be less than min_norb ({self.min_norb})")

    def matches(self, n_orb: int) -> bool:
        if n_orb < self.min_norb:
            return False
        if self.max_norb is not None and n_orb > self.max_norb:
            return False
        return True


@dataclass(frozen=True)
class SolverAllocationPolicy:
    """Authoritative sizing, override, and solver allocation policy.

    Parameters
    ----------
    rules : tuple[AllocationRule, ...]
        Ordered sequence of sizing rules evaluated in order.
    overrides : Mapping[int, str]
        Explicit per-fragment overrides {fragment_id: solver_name}.
    default_quantum_solver : str
        Solver assigned to fragments designated as quantum targets (default: 'TrimSQD').
    policy_name : str
        Policy identifier for provenance tracking.
    """
    rules: tuple[AllocationRule, ...] = (
        AllocationRule(solver="FCI", min_norb=1, max_norb=10, description="Exact FCI for n_orb <= 10"),
        AllocationRule(solver="CCSD", min_norb=11, max_norb=None, description="CCSD for n_orb >= 11 (classical baseline)"),
    )
    overrides: dict[int, str] = field(default_factory=dict)
    default_quantum_solver: str = "TrimSQD"
    policy_name: str = "proteasome_challenge_preset"

    # Backwards compatibility properties / fields
    fci_max_norb: int = 10
    sqd_min_norb: int = 20
    classical_intermediate_solver: str = "CCSD"
    quantum_solver: str = "TrimSQD"

    def __post_init__(self):
        for r in self.rules:
            if not isinstance(r, AllocationRule):
                raise SolverAllocationError(f"all rules must be AllocationRule instances, got {r!r}")
        for fid, sname in self.overrides.items():
            if not is_valid_solver(sname):
                raise SolverAllocationError(f"override for fragment {fid} specifies unregistered solver {sname!r}")
        if not is_valid_solver(self.default_quantum_solver):
            raise SolverAllocationError(f"default_quantum_solver {self.default_quantum_solver!r} is not registered")

    def allocation_digest(
        self,
        norb_by_fragment: Mapping[int, int],
        quantum_fragment_ids: Sequence[int] | None = None,
    ) -> str:
        """Compute deterministic SHA-256 digest of resolved fragment solver assignments."""
        smap = assign_solvers_by_size(norb_by_fragment, policy=self, quantum_fragment_ids=quantum_fragment_ids)
        payload = {
            "policy_name": self.policy_name,
            "assignments": {str(k): smap[k] for k in sorted(smap.keys())},
            "norb_by_fragment": {str(k): int(norb_by_fragment[k]) for k in sorted(norb_by_fragment.keys())},
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def assign_solvers_by_size(
    norb_by_fragment: Mapping[int, int],
    policy: SolverAllocationPolicy = SolverAllocationPolicy(),
    quantum_fragment_ids: Sequence[int] | None = None,
) -> dict[int, str]:
    """Derive solver assignments from fragment orbital counts, policy rules, and overrides.

    Resolution order for each fragment:
    1. Explicit override in ``policy.overrides``
    2. If fragment in ``quantum_fragment_ids`` -> ``policy.default_quantum_solver``
    3. Match against ordered ``policy.rules``
    """
    q_ids = set(int(i) for i in (quantum_fragment_ids or []))
    out: dict[int, str] = {}
    for fid, norb in sorted(norb_by_fragment.items(), key=lambda kv: int(kv[0])):
        fid_int = int(fid)
        norb_int = int(norb)
        if fid_int in policy.overrides:
            solver = policy.overrides[fid_int]
        elif fid_int in q_ids:
            solver = policy.default_quantum_solver
        else:
            matched = False
            for rule in policy.rules:
                if rule.matches(norb_int):
                    solver = rule.solver
                    matched = True
                    break
            if not matched:
                raise SolverAllocationError(
                    f"Fragment {fid_int} (n_orb={norb_int}) matched no allocation rule in policy {policy.policy_name!r}"
                )
        if not is_valid_solver(solver):
            raise SolverAllocationError(f"Fragment {fid_int} resolved to unregistered solver {solver!r}")
        out[fid_int] = solver
    return out


def generate_solver_assignments(
    norb_by_fragment: Mapping[int, int],
    policy: SolverAllocationPolicy = SolverAllocationPolicy(),
    quantum_fragment_ids: Sequence[int] | None = None,
    assignment_source: str = "orbital_sizing_policy",
) -> list[dict]:
    """Generate explicit per-fragment assignment records with provenance digest."""
    smap = assign_solvers_by_size(norb_by_fragment, policy=policy, quantum_fragment_ids=quantum_fragment_ids)
    return [
        {
            "fragment_id": fid,
            "solver": solver,
            "n_orb": int(norb_by_fragment[fid]),
            "assignment_source": assignment_source,
        }
        for fid, solver in sorted(smap.items())
    ]


def validate_solver_result(
    result: Any,
    solver_identifier: Optional[str] = None,
    norb: Optional[int] = None,
    n_elec: Optional[int] = None,
) -> None:
    """Validate that a solver execution result satisfies the contract for Stage 4.

    Accepts both a ``SolverResult`` object (checked via attributes) and a
    plain ``Mapping`` (checked via keys).

    Parameters
    ----------
    result : SolverResult or Mapping
        The solver result to validate.
    solver_identifier : str, optional
        Override the solver name for contract lookup (falls back to
        ``result.metadata.get("solver_label")`` or ``result.get("solver")``).
    norb : int, optional
        Expected number of orbitals.  When given, validates RDM shapes.
    n_elec : int, optional
        Expected number of electrons.  When given, validates RDM normalization.

    Raises
    ------
    SolverAllocationError
        When any validation check fails.
    """
    import numpy as np

    # ── Normalise to attribute access ─────────────────────────────────────────
    if isinstance(result, Mapping):
        # Plain dict-like: wrap for uniform access.
        # Use explicit None checks — numpy arrays are truthy/falsy ambiguous.
        energy  = result.get("energy")
        _rdm1   = result.get("rdm1")
        rdm1    = _rdm1 if _rdm1 is not None else result.get("dm1")
        _rdm2   = result.get("rdm2")
        rdm2    = _rdm2 if _rdm2 is not None else result.get("dm2")
        meta    = result.get("metadata", result)
        solver_name = solver_identifier or result.get("solver")
    else:
        energy  = getattr(result, "energy", None)
        rdm1    = getattr(result, "rdm1", None)
        rdm2    = getattr(result, "rdm2", None)
        meta    = getattr(result, "metadata", {}) or {}
        solver_name = solver_identifier or meta.get("solver_label")

    # ── Energy ────────────────────────────────────────────────────────────────
    if energy is None:
        raise SolverAllocationError("solver result is missing 'energy'")
    try:
        e = float(energy)
    except (TypeError, ValueError):
        raise SolverAllocationError(f"solver result 'energy' must be numeric, got {energy!r}")
    if not math.isfinite(e):
        raise SolverAllocationError(f"solver result 'energy' is not finite: {e}")

    # ── RDMs ──────────────────────────────────────────────────────────────────
    if rdm1 is None:
        raise SolverAllocationError("solver result is missing 'rdm1'")
    if rdm2 is None:
        raise SolverAllocationError("solver result is missing 'rdm2'")

    rdm1 = np.asarray(rdm1)
    rdm2 = np.asarray(rdm2)

    if rdm1.ndim != 2 or rdm1.shape[0] != rdm1.shape[1]:
        raise SolverAllocationError(
            f"rdm1 must be square 2-D array; got shape {rdm1.shape}"
        )
    if rdm2.ndim != 4:
        raise SolverAllocationError(
            f"rdm2 must be 4-D array; got shape {rdm2.shape}"
        )
    if rdm2.shape != (rdm1.shape[0],) * 4:
        raise SolverAllocationError(
            f"rdm2 shape {rdm2.shape} inconsistent with rdm1 shape {rdm1.shape}"
        )

    if norb is not None:
        if rdm1.shape != (norb, norb):
            raise SolverAllocationError(
                f"rdm1 shape {rdm1.shape} != (norb={norb}, norb={norb})"
            )

    if not np.all(np.isfinite(rdm1)):
        raise SolverAllocationError("rdm1 contains non-finite values (NaN/Inf)")
    if not np.all(np.isfinite(rdm2)):
        raise SolverAllocationError("rdm2 contains non-finite values (NaN/Inf)")

    # Hermiticity: rdm1[p,q] = rdm1[q,p]* (real → symmetric)
    herm_err = float(np.max(np.abs(rdm1 - rdm1.T)))
    if herm_err > 1e-5:
        raise SolverAllocationError(
            f"rdm1 is not symmetric (max |γ₁[p,q]-γ₁[q,p]| = {herm_err:.2e}); "
            "expected Hermitian for spin-summed real spatial RDM"
        )

    # 2-RDM pair symmetry: γ₂[p,q,r,s] = γ₂[r,s,p,q]
    pair_err = float(np.max(np.abs(rdm2 - rdm2.transpose(2, 3, 0, 1))))
    if pair_err > 1e-5:
        raise SolverAllocationError(
            f"rdm2 fails pair symmetry γ₂[p,q,r,s]=γ₂[r,s,p,q] "
            f"(max error {pair_err:.2e})"
        )

    # Electron number normalization
    if n_elec is not None:
        trace1 = float(np.trace(rdm1))
        if abs(trace1 - n_elec) > 0.01:
            raise SolverAllocationError(
                f"Tr(γ₁) = {trace1:.6f} but expected N_elec = {n_elec} "
                f"(|Δ| = {abs(trace1-n_elec):.2e})"
            )
        trace2 = float(np.einsum("ppqq->", rdm2))
        expected2 = float(n_elec * (n_elec - 1))
        if abs(trace2 - expected2) > 0.1:
            raise SolverAllocationError(
                f"Tr(γ₂[p,p,q,q]) = {trace2:.4f} but expected N(N-1) = {expected2:.4f} "
                f"(|Δ| = {abs(trace2-expected2):.2e})"
            )

    # ── Required metadata ─────────────────────────────────────────────────────
    contract = (
        get_solver_contract(solver_name)
        if isinstance(solver_name, str) and is_valid_solver(solver_name)
        else None
    )
    required_meta = contract.required_metadata if contract else ("solver_label", "nocc")
    for req in required_meta:
        if not isinstance(meta, Mapping) or req not in meta:
            raise SolverAllocationError(
                f"solver result metadata missing required field {req!r} "
                f"(solver: {solver_name})"
            )
        if meta[req] is None:
            raise SolverAllocationError(
                f"solver result metadata field {req!r} cannot be None"
            )


def load_solver_assignments(json_path) -> list[dict]:
    """Load the frozen assignment map; returns the list of per-fragment records."""
    with Path(json_path).open("r") as fh:
        data = json.load(fh)
    assignments = data["assignments"] if isinstance(data, dict) else data
    return [dict(a) for a in assignments]


def validate_assignments(assignments: list[dict], expected_fragment_ids: Iterable[int]) -> None:
    """Raise ValueError if any fragment is missing, duplicated, unexpected, or has an invalid solver."""
    expected = {int(i) for i in expected_fragment_ids}
    seen: set[int] = set()
    duplicated: set[int] = set()
    for a in assignments:
        fid = int(a["fragment_id"])
        if fid in seen:
            duplicated.add(fid)
        seen.add(fid)
        solver = a.get("solver")
        if not is_valid_solver(solver):
            raise ValueError(
                f"Fragment {fid}: invalid solver {solver!r}; "
                f"must be one of {sorted(list_registered_solvers())}"
            )
    if duplicated:
        raise ValueError(f"Duplicate assignments for fragments {sorted(duplicated)}")
    missing = expected - seen
    if missing:
        raise ValueError(f"Missing assignments for fragments {sorted(missing)}")
    extra = seen - expected
    if extra:
        raise ValueError(f"Assignments for unexpected fragments {sorted(extra)}")


def get_assignment(assignments: list[dict], fragment_id: int) -> dict:
    """Return the single assignment for ``fragment_id``; raise ValueError if absent or duplicated."""
    matches = [a for a in assignments if int(a["fragment_id"]) == int(fragment_id)]
    if not matches:
        raise ValueError(f"No solver assignment for fragment {fragment_id}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate solver assignments for fragment {fragment_id}")
    return matches[0]


def assignments_to_step3_params(assignments: list[dict]) -> dict:
    """Convert the assignment map to the arguments consumed by ``step3_solve.run_solve``.

    Returns a dict with:

    ``fragment_ids``          sorted ids of every assigned fragment
    ``quantum_fragment_ids``  sorted ids assigned to SQD or TrimSQD
    ``solver_mode``           ``"sqd"`` or ``"trimsqd"`` (one quantum solver per Stage 3 job;
                              ``"sqd"`` when no fragment is quantum)
    ``solver_assignments``    ``{fragment_id: solver_name}`` for the explicit routing path
    """
    validate_assignments(assignments, [int(a["fragment_id"]) for a in assignments])
    smap = {int(a["fragment_id"]): a["solver"] for a in assignments}
    quantum_ids = sorted(f for f, s in smap.items() if s in QUANTUM_SOLVERS)
    quantum_kinds = {smap[f] for f in quantum_ids}
    if len(quantum_kinds) > 1:
        raise ValueError(
            "All quantum fragments in one Stage 3 job must share a solver "
            f"(found {sorted(quantum_kinds)})."
        )
    solver_mode = "trimsqd" if quantum_kinds == {"TrimSQD"} else "sqd"
    return {
        "fragment_ids": sorted(smap),
        "quantum_fragment_ids": quantum_ids,
        "solver_mode": solver_mode,
        "solver_assignments": dict(sorted(smap.items())),
    }
