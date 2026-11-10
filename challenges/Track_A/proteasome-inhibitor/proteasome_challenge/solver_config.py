"""
solver_config.py — Explicit, validated Stage 3 scientific solver configuration.

Provides :class:`FCIConfig`, :class:`CCSDConfig`, :class:`SQDConfig`,
:class:`TrimSQDConfig`, and :class:`Stage3Config`.

Design principles
-----------------
* Solver *allocation* (which fragment gets which solver) lives in
  :mod:`solver_assignments`.  Configuration (how each solver operates)
  lives here.  The two are independent.
* Only parameters that actually reach the solver implementation are
  exposed.  Unknown keys raise :class:`SolverConfigError`.
* Execution controls — QPU backend, credentials, Serverless resource
  families, shot-budget authorization — are *not* in solver config.
* :class:`Stage3Config` does not duplicate allocation thresholds.  Those
  belong to :class:`~solver_assignments.SolverAllocationPolicy`.

Challenge default values
------------------------
The defaults in every config class reproduce the settings from
``config_ixazomib_sto-3g.yaml`` that have been used for all graded
runs.  Changing them changes the scientific result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional

__all__ = [
    "SolverConfigError",
    "FCIConfig",
    "CCSDConfig",
    "SQDConfig",
    "TrimSQDConfig",
    "Stage3Config",
]


class SolverConfigError(ValueError):
    """A solver configuration value is invalid or unsupported."""


# ─────────────────────────────────────────────────────────────────────────────
# Per-solver configuration dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FCIConfig:
    """Scientific configuration for the FCI solver.

    Parameters
    ----------
    conv_tol : float
        Davidson convergence tolerance (default: 1e-14).
    max_cycle : int
        Maximum Davidson iterations (default: 100).
    spin : int
        Spin multiplicity 2S.  0 = singlet (default).

    Notes
    -----
    No ``e_corr`` is computed by the FCI solver.  Stage 4 uses the
    partitioned-cumulant path (RDMs) exclusively for FCI fragments.
    """
    conv_tol: float = 1e-14
    max_cycle: int = 100
    spin: int = 0

    def __post_init__(self):
        if self.conv_tol <= 0 or not _finite(self.conv_tol):
            raise SolverConfigError(f"FCIConfig.conv_tol must be positive finite, got {self.conv_tol!r}")
        if self.max_cycle < 1:
            raise SolverConfigError(f"FCIConfig.max_cycle must be >= 1, got {self.max_cycle}")
        if self.spin < 0:
            raise SolverConfigError(f"FCIConfig.spin must be >= 0, got {self.spin}")

    def to_dict(self) -> dict[str, Any]:
        return {"conv_tol": self.conv_tol, "max_cycle": self.max_cycle, "spin": self.spin}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FCIConfig":
        _check_unknown(d, {f.name for f in fields(cls)}, "FCIConfig")
        return cls(**{k: v for k, v in d.items() if v is not None})


@dataclass(frozen=True)
class CCSDConfig:
    """Scientific configuration for the CCSD solver.

    Parameters
    ----------
    conv_tol : float
        CCSD amplitude convergence tolerance (default: 1e-7).
    max_cycle : int
        Maximum CCSD iterations (default: 50).
    diis_space : int
        DIIS subspace size for convergence acceleration (default: 6).
    level_shift : float
        Orbital energy level shift for problematic convergence (default: 0.0).
        Non-zero values bias the orbital denominators; use only when needed.

    Notes
    -----
    Unconverged CCSD solves raise :class:`RuntimeError`.  The solver never
    returns a partial result.
    """
    conv_tol: float = 1e-7
    max_cycle: int = 50
    diis_space: int = 6
    level_shift: float = 0.0

    def __post_init__(self):
        if self.conv_tol <= 0 or not _finite(self.conv_tol):
            raise SolverConfigError(f"CCSDConfig.conv_tol must be positive finite, got {self.conv_tol!r}")
        if self.max_cycle < 1:
            raise SolverConfigError(f"CCSDConfig.max_cycle must be >= 1, got {self.max_cycle}")
        if self.diis_space < 1:
            raise SolverConfigError(f"CCSDConfig.diis_space must be >= 1, got {self.diis_space}")
        if not _finite(self.level_shift):
            raise SolverConfigError(f"CCSDConfig.level_shift must be finite, got {self.level_shift!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "conv_tol": self.conv_tol,
            "max_cycle": self.max_cycle,
            "diis_space": self.diis_space,
            "level_shift": self.level_shift,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CCSDConfig":
        _check_unknown(d, {f.name for f in fields(cls)}, "CCSDConfig")
        return cls(**{k: v for k, v in d.items() if v is not None})


@dataclass(frozen=True)
class SQDConfig:
    """Scientific configuration for the SQD (Sample-based Quantum Diagonalization) solver.

    Parameters
    ----------
    n_reps : int
        Number of LUCJ circuit repetition layers (default: 1).
        Affects entanglement depth and sampling quality.
    iterations : int
        Configuration-recovery (CR) iterations (default: 6).
    n_batches : int
        Disjoint batches per CR iteration (default: 5).
    samples_per_batch : int
        Bitstrings subsampled per batch per iteration (default: 3000).
    max_dim : int or None
        Maximum CI subspace dimension.  ``None`` means no cap (default: 500).
        Required for large fragments to keep solve_sci_batch tractable.
    energy_tol : float
        CR energy convergence tolerance (default: 1e-8).
    occupancies_tol : float
        Orbital occupancy convergence tolerance (default: 1e-5).
    symmetrize_spin : bool
        Merge alpha/beta CI strings before diagonalization (default: True).
    seed : int
        RNG seed for reproducibility.  Default 20260626 is the organizer
        canonical seed for graded calculations; changing it changes the
        statistical result.
    carryover_threshold : float
        Amplitude threshold for carrying CI determinants between iterations
        (default: 1e-4).  Mutually exclusive with TrimSQD ``policy=``.

    Notes
    -----
    ``classical_backend`` is an *execution* parameter (cpu/gpu/HPC selector),
    not a scientific parameter, so it is not part of this dataclass.
    It is passed separately through the execution path.
    """
    n_reps: int = 1
    iterations: int = 6
    n_batches: int = 5
    samples_per_batch: int = 3000
    max_dim: Optional[int] = 500
    energy_tol: float = 1e-8
    occupancies_tol: float = 1e-5
    symmetrize_spin: bool = True
    seed: int = 20260626
    carryover_threshold: float = 1e-4

    def __post_init__(self):
        if self.n_reps < 1:
            raise SolverConfigError(f"SQDConfig.n_reps must be >= 1, got {self.n_reps}")
        if self.iterations < 1:
            raise SolverConfigError(f"SQDConfig.iterations must be >= 1, got {self.iterations}")
        if self.n_batches < 1:
            raise SolverConfigError(f"SQDConfig.n_batches must be >= 1, got {self.n_batches}")
        if self.samples_per_batch < 1:
            raise SolverConfigError(f"SQDConfig.samples_per_batch must be >= 1, got {self.samples_per_batch}")
        if self.max_dim is not None and self.max_dim < 1:
            raise SolverConfigError(f"SQDConfig.max_dim must be None or >= 1, got {self.max_dim}")
        if self.energy_tol <= 0 or not _finite(self.energy_tol):
            raise SolverConfigError(f"SQDConfig.energy_tol must be positive finite, got {self.energy_tol!r}")
        if self.occupancies_tol <= 0 or not _finite(self.occupancies_tol):
            raise SolverConfigError(f"SQDConfig.occupancies_tol must be positive finite, got {self.occupancies_tol!r}")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise SolverConfigError(f"SQDConfig.seed must be a non-negative int, got {self.seed!r}")
        if self.carryover_threshold <= 0 or not _finite(self.carryover_threshold):
            raise SolverConfigError(f"SQDConfig.carryover_threshold must be positive finite, got {self.carryover_threshold!r}")

    def to_sqd_dict(self) -> dict[str, Any]:
        """Convert to the ``sqd_config`` dict consumed by :class:`SQDSolver`."""
        return {
            "lucj": {"n_reps": self.n_reps},
            "symmetrize_spin": self.symmetrize_spin,
            "n_batches": self.n_batches,
            "iterations": self.iterations,
            "samples_per_batch": self.samples_per_batch,
            "max_dim": self.max_dim,
            "energy_tol": self.energy_tol,
            "occupancies_tol": self.occupancies_tol,
            "carryover_threshold": self.carryover_threshold,
            "seed": self.seed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_reps": self.n_reps,
            "iterations": self.iterations,
            "n_batches": self.n_batches,
            "samples_per_batch": self.samples_per_batch,
            "max_dim": self.max_dim,
            "energy_tol": self.energy_tol,
            "occupancies_tol": self.occupancies_tol,
            "symmetrize_spin": self.symmetrize_spin,
            "seed": self.seed,
            "carryover_threshold": self.carryover_threshold,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SQDConfig":
        _check_unknown(d, {f.name for f in fields(cls)}, "SQDConfig")
        return cls(**{k: v for k, v in d.items() if v is not None})


@dataclass(frozen=True)
class TrimSQDConfig:
    """Scientific configuration for the TrimSQD solver.

    TrimSQD shares all SQD post-processing parameters and adds trimming
    controls.  ``carryover_threshold`` from :class:`SQDConfig` is unused
    (the policy controls carryover instead).

    Parameters
    ----------
    n_reps : int
        LUCJ circuit repetition layers (default: 1).
    iterations : int
        Configuration-recovery iterations (default: 6).
    n_batches : int
        Batches per CR iteration (default: 5).
    samples_per_batch : int
        Bitstrings subsampled per batch per iteration (default: 3000).
    max_dim : int or None
        Maximum CI subspace dimension (default: 500).
    energy_tol : float
        CR energy convergence tolerance (default: 1e-8).
    occupancies_tol : float
        Orbital occupancy convergence tolerance (default: 1e-5).
    symmetrize_spin : bool
        Merge alpha/beta CI strings (default: True).
    seed : int
        RNG seed (default: 20260626).
    trim_ratio : float
        Fraction of each batch's CI strings retained after trimming
        (0 < trim_ratio ≤ 1).  0.15 is the challenge default.
    max_strings_per_trim : int or None
        Hard cap on strings kept per batch before merging (default: 300).
    max_carryover : int or None
        Hard cap on merged carryover strings (default: 400).
    sector_trim : bool
        Use :class:`SectorTrimPolicy` instead of :class:`TrimPolicy`
        (default: False).

    Notes
    -----
    ``carryover_threshold`` is intentionally absent: the addon API treats
    ``policy=`` and ``carryover_threshold=`` as mutually exclusive, and
    TrimSQD always uses the policy path.
    """
    n_reps: int = 1
    iterations: int = 6
    n_batches: int = 5
    samples_per_batch: int = 3000
    max_dim: Optional[int] = 500
    energy_tol: float = 1e-8
    occupancies_tol: float = 1e-5
    symmetrize_spin: bool = True
    seed: int = 20260626
    trim_ratio: float = 0.15
    max_strings_per_trim: Optional[int] = 300
    max_carryover: Optional[int] = 400
    sector_trim: bool = False

    def __post_init__(self):
        if self.n_reps < 1:
            raise SolverConfigError(f"TrimSQDConfig.n_reps must be >= 1, got {self.n_reps}")
        if self.iterations < 1:
            raise SolverConfigError(f"TrimSQDConfig.iterations must be >= 1, got {self.iterations}")
        if self.n_batches < 1:
            raise SolverConfigError(f"TrimSQDConfig.n_batches must be >= 1, got {self.n_batches}")
        if self.samples_per_batch < 1:
            raise SolverConfigError(f"TrimSQDConfig.samples_per_batch must be >= 1, got {self.samples_per_batch}")
        if self.max_dim is not None and self.max_dim < 1:
            raise SolverConfigError(f"TrimSQDConfig.max_dim must be None or >= 1, got {self.max_dim}")
        if self.energy_tol <= 0 or not _finite(self.energy_tol):
            raise SolverConfigError(f"TrimSQDConfig.energy_tol must be positive finite, got {self.energy_tol!r}")
        if self.occupancies_tol <= 0 or not _finite(self.occupancies_tol):
            raise SolverConfigError(f"TrimSQDConfig.occupancies_tol must be positive finite, got {self.occupancies_tol!r}")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise SolverConfigError(f"TrimSQDConfig.seed must be a non-negative int, got {self.seed!r}")
        if not (0 < self.trim_ratio <= 1.0):
            raise SolverConfigError(f"TrimSQDConfig.trim_ratio must be in (0, 1], got {self.trim_ratio}")
        if self.max_strings_per_trim is not None and self.max_strings_per_trim < 1:
            raise SolverConfigError(f"TrimSQDConfig.max_strings_per_trim must be None or >= 1, got {self.max_strings_per_trim}")
        if self.max_carryover is not None and self.max_carryover < 1:
            raise SolverConfigError(f"TrimSQDConfig.max_carryover must be None or >= 1, got {self.max_carryover}")

    def to_sqd_dict(self) -> dict[str, Any]:
        """Convert to the ``sqd_config`` dict consumed by :class:`TrimSQDSolver`."""
        return {
            "lucj": {"n_reps": self.n_reps},
            "symmetrize_spin": self.symmetrize_spin,
            "n_batches": self.n_batches,
            "iterations": self.iterations,
            "samples_per_batch": self.samples_per_batch,
            "max_dim": self.max_dim,
            "energy_tol": self.energy_tol,
            "occupancies_tol": self.occupancies_tol,
            "seed": self.seed,
            # Note: carryover_threshold intentionally omitted for TrimSQD;
            # carryover is controlled by trim_ratio/max_carryover via the policy.
            "trim": {
                "trim_ratio": self.trim_ratio,
                "max_strings_per_trim": self.max_strings_per_trim,
                "max_carryover": self.max_carryover,
                "sector_trim": self.sector_trim,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_reps": self.n_reps,
            "iterations": self.iterations,
            "n_batches": self.n_batches,
            "samples_per_batch": self.samples_per_batch,
            "max_dim": self.max_dim,
            "energy_tol": self.energy_tol,
            "occupancies_tol": self.occupancies_tol,
            "symmetrize_spin": self.symmetrize_spin,
            "seed": self.seed,
            "trim_ratio": self.trim_ratio,
            "max_strings_per_trim": self.max_strings_per_trim,
            "max_carryover": self.max_carryover,
            "sector_trim": self.sector_trim,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrimSQDConfig":
        _check_unknown(d, {f.name for f in fields(cls)}, "TrimSQDConfig")
        return cls(**{k: v for k, v in d.items() if v is not None})


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 configuration container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stage3Config:
    """Stage 3 scientific solver configuration.

    Bundles per-solver scientific settings.  Solver *allocation* (FCI /
    CCSD / TrimSQD per fragment) is controlled separately by
    :class:`~solver_assignments.SolverAllocationPolicy`.

    Parameters
    ----------
    fci : FCIConfig
        FCI solver parameters.
    ccsd : CCSDConfig
        CCSD solver parameters.
    sqd : SQDConfig
        SQD solver parameters (also used as the base for TrimSQD when
        ``trimsqd`` is not explicitly provided).
    trimsqd : TrimSQDConfig
        TrimSQD solver parameters.

    Notes
    -----
    The defaults reproduce the ixazomib challenge configuration from
    ``config_ixazomib_sto-3g.yaml``.  Do not change defaults without a
    validated scientific justification.

    What does NOT belong here
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    * QPU backend name or credentials.
    * Fresh QPU sampling authorization.
    * Shot count or budget authorization.
    * Serverless resource families.
    * CPU / GPU scheduling.
    * Serverless concurrency and monitoring settings.

    Those are execution / operational concerns and must be handled by the
    caller (e.g. the notebook or the Serverless entrypoint), not by this
    configuration object.
    """
    fci: FCIConfig = field(default_factory=FCIConfig)
    ccsd: CCSDConfig = field(default_factory=CCSDConfig)
    sqd: SQDConfig = field(default_factory=SQDConfig)
    trimsqd: TrimSQDConfig = field(default_factory=TrimSQDConfig)

    def __post_init__(self):
        # Coerce plain dicts to typed configs
        def _coerce(attr, cls):
            v = getattr(self, attr)
            if isinstance(v, dict):
                object.__setattr__(self, attr, cls.from_dict(v))
            elif not isinstance(v, cls):
                raise SolverConfigError(
                    f"Stage3Config.{attr} must be a {cls.__name__} or dict, got {type(v).__name__!r}"
                )
        _coerce("fci", FCIConfig)
        _coerce("ccsd", CCSDConfig)
        _coerce("sqd", SQDConfig)
        _coerce("trimsqd", TrimSQDConfig)

    def digest(self) -> str:
        """Deterministic SHA-256 of the full scientific configuration."""
        payload = {
            "fci": self.fci.to_dict(),
            "ccsd": self.ccsd.to_dict(),
            "sqd": self.sqd.to_dict(),
            "trimsqd": self.trimsqd.to_dict(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fci": self.fci.to_dict(),
            "ccsd": self.ccsd.to_dict(),
            "sqd": self.sqd.to_dict(),
            "trimsqd": self.trimsqd.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Stage3Config":
        _check_unknown(d, {"fci", "ccsd", "sqd", "trimsqd"}, "Stage3Config")
        return cls(
            fci=FCIConfig.from_dict(d["fci"]) if "fci" in d else FCIConfig(),
            ccsd=CCSDConfig.from_dict(d["ccsd"]) if "ccsd" in d else CCSDConfig(),
            sqd=SQDConfig.from_dict(d["sqd"]) if "sqd" in d else SQDConfig(),
            trimsqd=TrimSQDConfig.from_dict(d["trimsqd"]) if "trimsqd" in d else TrimSQDConfig(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _finite(v: Any) -> bool:
    import math
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _check_unknown(d: dict, allowed: set, name: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise SolverConfigError(
            f"{name}: unknown configuration key(s) {sorted(unknown)}; "
            f"supported: {sorted(allowed)}"
        )
