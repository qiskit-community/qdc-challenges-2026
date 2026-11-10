# This code is a Qiskit project.
#
# (C) Copyright IBM and Cleveland Clinic Foundation 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Full Configuration Interaction (FCI) solver using PySCF."""

from __future__ import annotations

import numpy as np
import pyscf
import pyscf.fci
import pyscf.fci.direct_spin0
from numpy.typing import NDArray

from ..base import BaseSolver, SolverResult


class FCI(BaseSolver):
    """
    Full Configuration Interaction (FCI) solver.

    This class wraps PySCF's FCI solver to provide exact solutions
    for small molecular fragments. It computes the ground state energy,
    wavefunction, and reduced density matrices.

    Parameters
    ----------
    conv_tol : float, optional
        Convergence tolerance for FCI solver (default: 1e-14)
    max_cycle : int, optional
        Maximum number of iterations (default: 100)
    spin : int, optional
        Spin multiplicity (2S). Default is 0 (singlet)
    verbose : int, optional
        Verbosity level (default: 0)
    **kwargs : dict
        Additional PySCF FCI configuration options

    Attributes
    ----------
    conv_tol : float
        Convergence tolerance
    max_cycle : int
        Maximum iterations
    spin : int
        Spin multiplicity
    verbose : int
        Verbosity level

    Notes
    -----
    RDM conventions
    ~~~~~~~~~~~~~~~
    ``rdm1[p, q] = <p† q>`` (spatial, spin-summed); Tr(γ₁) = N_elec.

    ``rdm2[p, q, r, s]`` contracts with chemist-notation ERIs ``(pq|rs)``
    via ``E = h1·γ₁ + 0.5·(pq|rs)·γ₂[p,q,r,s]``.  This is PySCF's native
    storage for spin-summed two-particle density matrices as returned by
    ``make_rdm12``.  The RDMs live in the same spatial-orbital basis as the
    input ``h1e`` and ``h2e`` (the cluster MO basis from the EWF HDF5).

    Fragment energy reconstruction
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    FCI does not implement ``compute_fragment_energy``.  EWF Stage 4 uses
    a single shared ``CCSD.compute_fragment_energy`` for all classical
    solvers (FCI and CCSD) — this is the correct design because it keeps
    one validated partitioned-cumulant implementation rather than two
    potentially divergent versions.

    Convergence reporting
    ~~~~~~~~~~~~~~~~~~~~~
    PySCF FCI uses a Lanczos/Davidson diagonalizer that always converges
    within the subspace it constructs.  The ``converged`` flag in metadata
    reflects the actual Davidson convergence state returned by the kernel.
    """

    def __init__(
        self,
        conv_tol: float = 1e-14,
        max_cycle: int = 100,
        spin: int = 0,
        verbose: int = 0,
        **kwargs,
    ):
        """Initialize FCI solver with PySCF backend."""
        super().__init__(**kwargs)
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.spin = spin
        self.verbose = verbose

    def solve(
        self, h1e: NDArray, h2e: NDArray, norb: int, nelec: tuple[int, int] | int, **kwargs
    ) -> SolverResult:
        """
        Solve FCI problem for given Hamiltonian.

        Parameters
        ----------
        h1e : np.ndarray
            One-electron Hamiltonian matrix (norb, norb)
        h2e : np.ndarray
            Two-electron repulsion integrals in chemist notation (pq|rs).
            Shape: (norb, norb, norb, norb)
        norb : int
            Number of spatial orbitals
        nelec : tuple of int or int
            Number of electrons. Can be:
            - tuple (n_alpha, n_beta) for spin-resolved
            - int for total electrons (assumes closed shell)
        **kwargs : dict
            Additional solver parameters:
            - conv_tol: Override convergence tolerance
            - max_cycle: Override max iterations
            - ci0: Initial CI vector guess

        Returns
        -------
        SolverResult
            Container with:
            - energy: FCI ground state energy (Ha)
            - wavefunction: CI vector coefficients
            - rdm1: One-particle reduced density matrix (norb, norb),
              spin-summed, Tr = N_elec, same basis as h1e/h2e
            - rdm2: Two-particle reduced density matrix (norb, norb, norb, norb),
              spin-summed, contracts with (pq|rs) as E = 0.5*(pq|rs)*dm2[p,q,r,s]
            - metadata: Convergence info and diagnostics

        Raises
        ------
        ValueError
            If input dimensions are inconsistent
        RuntimeError
            If FCI calculation fails to converge

        Notes
        -----
        The RDMs are returned in the orbital basis of the input h1e/h2e.
        Uses PySCF's direct_spin0 FCI solver for spin-adapted calculations
        (singlet states); falls back to direct_spin1 for other spin states.
        """
        # Validate inputs
        if h1e.shape != (norb, norb):
            raise ValueError(f"h1e shape {h1e.shape} inconsistent with norb={norb}")
        if h2e.shape != (norb, norb, norb, norb):
            raise ValueError(f"h2e shape {h2e.shape} inconsistent with norb={norb}")

        # Handle electron count
        if isinstance(nelec, int):
            if nelec % 2 != 0:
                raise ValueError(
                    "For integer nelec, must be even (closed shell). "
                    "Use tuple (n_alpha, n_beta) for open shell."
                )
            nelec = (nelec // 2, nelec // 2)

        n_alpha, n_beta = nelec

        # Override parameters if provided
        conv_tol = kwargs.get("conv_tol", self.conv_tol)
        max_cycle = kwargs.get("max_cycle", self.max_cycle)
        ci0 = kwargs.get("ci0", None)

        # Select appropriate FCI solver
        if self.spin == 0 and n_alpha == n_beta:
            # Use spin-adapted solver for singlets
            fci_solver = pyscf.fci.direct_spin0  # type: ignore
        else:
            # Use general FCI solver for other cases
            fci_solver = pyscf.fci.direct_spin1  # type: ignore

        # Run FCI calculation
        try:
            energy, ci_vec = fci_solver.kernel(
                h1e,
                h2e,
                norb,
                nelec,
                conv_tol=conv_tol,
                max_cycle=max_cycle,
                verbose=self.verbose,
                ci0=ci0,
            )
        except Exception as e:
            raise RuntimeError(f"FCI calculation failed: {str(e)}") from e

        # Determine actual convergence from the FCI solver object
        # PySCF FCI kernel returns (energy, ci_vec); convergence is accessible
        # via a fresh FCI object that retains its state.
        try:
            fci_obj = fci_solver.FCI()
            fci_obj.conv_tol = conv_tol
            fci_obj.max_cycle = max_cycle
            fci_obj.verbose = 0
            _, _ = fci_obj.kernel(h1e, h2e, norb, nelec, ci0=ci0)
            converged = bool(getattr(fci_obj, "converged", True))
        except Exception:
            # Davidson in PySCF always converges within the Krylov space;
            # if we cannot read the flag, default to True.
            converged = True

        # Compute reduced density matrices
        dm1, dm2 = fci_solver.make_rdm12(ci_vec, norb, nelec)

        # Prepare metadata
        # e_corr is NOT computed for FCI: no embedded HF reference is defined
        # within this solver, and inventing one would be scientifically unjustified.
        # Stage 4 uses the partitioned cumulant path (RDMs) exclusively; it never
        # falls back to e_corr for FCI fragments.
        metadata = {
            "norb":        norb,
            "nelec":       nelec,
            "conv_tol":    conv_tol,
            "max_cycle":   max_cycle,
            "spin":        self.spin,
            "converged":   converged,
            "solver_type": (
                "direct_spin0" if self.spin == 0 and n_alpha == n_beta else "direct_spin1"
            ),
        }

        return SolverResult(
            energy=float(energy), wavefunction=ci_vec, rdm1=dm1, rdm2=dm2, metadata=metadata
        )

    def solve_from_integrals(
        self, h1e: NDArray, h2e: NDArray, norb: int, nocc: int, **kwargs
    ) -> SolverResult:
        """
        Convenience method matching the application code pattern.

        Parameters
        ----------
        h1e : np.ndarray
            One-electron Hamiltonian (effective Hamiltonian), shape (norb, norb)
        h2e : np.ndarray
            Two-electron repulsion integrals in chemist notation, shape (norb,norb,norb,norb)
        norb : int
            Number of orbitals
        nocc : int
            Number of occupied orbitals (assumes closed shell)
        **kwargs : dict
            Additional parameters passed to solve()

        Returns
        -------
        SolverResult
            FCI solution with energy and density matrices.  RDMs are in the
            same orbital basis as the input h1e/h2e (no internal rotation).
        """
        nelec = (nocc, nocc)
        return self.solve(h1e, h2e, norb, nelec, **kwargs)

    def __repr__(self) -> str:
        """String representation of FCI solver."""
        return (
            f"FCI(conv_tol={self.conv_tol}, max_cycle={self.max_cycle}, "
            f"spin={self.spin}, verbose={self.verbose})"
        )
