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

"""Coupled Cluster Singles and Doubles (CCSD) solver using PySCF."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..base import BaseSolver, SolverResult
from . import ccsd_diagnostics as _diag


class CCSD(BaseSolver):
    """
    Coupled Cluster Singles and Doubles (CCSD) solver.

    This class wraps PySCF's CCSD solver to provide accurate correlation
    energies for molecular fragments. CCSD is more efficient than FCI
    for larger systems while maintaining good accuracy.

    Parameters
    ----------
    conv_tol : float, optional
        Convergence tolerance for CCSD solver (default: 1e-7)
    max_cycle : int, optional
        Maximum number of iterations (default: 50)
    diis_space : int, optional
        DIIS space size for convergence acceleration (default: 6)
    verbose : int, optional
        Verbosity level (default: 0)
    **kwargs : dict
        Additional PySCF CCSD configuration options

    Attributes
    ----------
    conv_tol : float
        Convergence tolerance
    max_cycle : int
        Maximum iterations
    diis_space : int
        DIIS space size
    verbose : int
        Verbosity level

    Notes
    -----
    RDM conventions and orbital-basis guarantee
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``solve_from_integrals`` performs an internal SCF canonicalization so that
    CCSD amplitude denominators (ε_a − ε_i) are well-defined.  After solving,
    the RDMs returned by PySCF are in the *canonical MO basis* of that
    internal SCF.  They are back-transformed to the *input cluster-MO basis*
    (the basis of the supplied ``h1e``/``h2e``) before being placed in the
    ``SolverResult``, so that Stage 4 can contract them directly against the
    integrals stored in the EWF HDF5 file without any additional rotation.

    The back-transformation is::

        rdm1_in[p, q]       = C @ rdm1_mo @ C.T
        rdm2_in[p, q, r, s] = einsum("abcd,pa,qb,rc,sd->pqrs", rdm2_mo, C, C, C, C)

    where ``C = mf.mo_coeff`` (shape ``(norb, norb)``).

    For the ixazomib EWF HDF5 fragments the Vayesta DUMP solver already
    writes the cluster Hamiltonian in the canonical HF basis, so the Fock
    matrix is diagonal and C ≈ I (deviations < 5×10⁻⁵).  The energy
    contribution of the back-transformation is therefore < 0.1 μHa for
    the current fragment set.  The transformation is applied unconditionally
    so the implementation is correct for any embedded Hamiltonian.

    Energy formula::

        E = h1·γ₁ + 0.5·(pq|rs)·γ₂[p, q, r, s]

    where ``(pq|rs)`` is the chemist-notation Coulomb ERI (non-antisymmetrized)
    and γ₂ is the spin-summed two-particle RDM in the input basis.

    Convergence
    ~~~~~~~~~~~
    ``solve`` raises ``RuntimeError`` and propagates the PySCF failure when
    CCSD does not converge.  Unconverged results are never returned as valid.
    """

    def __init__(
        self,
        conv_tol: float = 1e-7,
        max_cycle: int = 50,
        diis_space: int = 6,
        verbose: int = 0,
        **kwargs,
    ):
        """Initialize CCSD solver with PySCF backend."""
        super().__init__(**kwargs)
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.diis_space = diis_space
        self.verbose = verbose

    def solve(self, mf, compute_rdms: bool = False, **kwargs) -> SolverResult:
        """
        Solve CCSD problem from mean-field object.

        Parameters
        ----------
        mf : pyscf.scf object
            Mean-field calculation result (RHF, UHF, etc.)
        compute_rdms : bool, optional
            Whether to compute reduced density matrices (default: False)
        **kwargs : dict
            Additional solver parameters:
            - conv_tol: Override convergence tolerance
            - max_cycle: Override max iterations
            - diis_space: Override DIIS space

        Returns
        -------
        SolverResult
            Container with:
            - energy: CCSD total energy
            - wavefunction: None (CCSD doesn't store full wavefunction)
            - rdm1: One-particle RDM (if compute_rdms=True)
            - rdm2: Two-particle RDM (if compute_rdms=True)
            - metadata: Convergence info and diagnostics

        Raises
        ------
        ImportError
            If PySCF is not installed
        TypeError
            If mf is not a valid mean-field object
        RuntimeError
            If CCSD calculation fails

        Notes
        -----
        This implementation uses PySCF's CCSD solver. For restricted
        mean-field objects (RHF), it uses RCCSD. For unrestricted (UHF),
        it uses UCCSD.
        """
        try:
            import pyscf.cc
        except ImportError as e:
            raise ImportError(
                "PySCF is required for CCSD solver. Install with: pip install pyscf"
            ) from e

        # Validate mean-field object
        if not hasattr(mf, "mo_coeff"):
            raise TypeError("Invalid mean-field object. Must have mo_coeff attribute.")

        # Override parameters if provided
        conv_tol = kwargs.get("conv_tol", self.conv_tol)
        max_cycle = kwargs.get("max_cycle", self.max_cycle)
        diis_space = kwargs.get("diis_space", self.diis_space)
        level_shift = kwargs.get("level_shift", 0.0)

        # Pop internal overrides before forwarding kwargs to PySCF
        _ao2mo_override = kwargs.pop("_ao2mo_override", None)
        _hamiltonian_hash = kwargs.pop("_hamiltonian_hash", None)

        # Create CCSD object
        try:
            ccsd = pyscf.cc.CCSD(mf)
            ccsd.conv_tol = conv_tol
            ccsd.max_cycle = max_cycle
            ccsd.diis_space = diis_space
            ccsd.level_shift = level_shift
            ccsd.verbose = self.verbose
            if _ao2mo_override is not None:
                ccsd.ao2mo = _ao2mo_override.__get__(ccsd, type(ccsd))
        except Exception as e:
            raise RuntimeError(f"Failed to create CCSD object: {str(e)}") from e

        # Run CCSD calculation
        try:
            e_corr, t1, t2 = ccsd.kernel()
            converged = ccsd.converged
        except Exception as e:
            raise RuntimeError(f"CCSD calculation failed: {str(e)}") from e

        # Reject unconverged results — amplitude denominators and RDMs are not
        # meaningful for a non-stationary solution.
        if not converged:
            raise RuntimeError(
                f"CCSD did not converge within max_cycle={max_cycle} "
                f"(conv_tol={conv_tol}).  Increase max_cycle or adjust "
                "level_shift / diis_space to improve convergence."
            )

        # Total energy
        e_tot = ccsd.e_tot

        # Compute RDMs if requested
        # Note: make_rdm1() / make_rdm2() return in the canonical MO basis of
        # the supplied mf.  solve_from_integrals() back-transforms to the input
        # cluster-MO basis (see class docstring) before returning to the caller.
        rdm1 = None
        rdm2 = None
        if compute_rdms:
            rdm1 = ccsd.make_rdm1()
            rdm2 = ccsd.make_rdm2()

        # Prepare metadata
        metadata = {
            "converged": True,   # unconverged path raises above
            "e_corr": float(e_corr),
            "e_hf": float(mf.e_tot),
            "conv_tol": conv_tol,
            "max_cycle": max_cycle,
            "diis_space": diis_space,
            "norb": mf.mol.nao,
            "nelec": mf.mol.nelectron,
        }

        # CCSD diagnostics from the amplitudes of THIS converged solve (no extra
        # correlated calculation).  A diagnostics failure must never abort the
        # energy calculation, so any problem becomes an explicit 'unavailable'
        # record rather than an exception or a zero placeholder.
        metadata["ccsd_diagnostics"] = self._extract_diagnostics(
            mf, t1, t2, bool(converged), float(e_corr), _hamiltonian_hash
        )

        return SolverResult(
            energy=float(e_tot),
            wavefunction=None,  # CCSD doesn't store full wavefunction
            rdm1=rdm1,
            rdm2=rdm2,
            metadata=metadata,
        )

    @staticmethod
    def _extract_diagnostics(mf, t1, t2, converged, e_corr, hamiltonian_hash):
        """Build the persisted CCSD diagnostic record (see ``ccsd_diagnostics``)."""
        nelec = int(mf.mol.nelectron)
        nocc = nelec // 2
        nvir = int(mf.mol.nao) - nocc
        try:
            orb_ref = _diag.orbital_reference(
                mo_energy=getattr(mf, "mo_energy", None), nocc=nocc
            )
            if nelec % 2 or isinstance(t1, (tuple, list)):
                return _diag.unavailable_record(
                    "open-shell / unrestricted CCSD amplitudes are not supported by the "
                    "restricted diagnostic definitions.",
                    converged=converged, correlation_energy=e_corr,
                    hamiltonian_hash_value=hamiltonian_hash,
                    n_correlated_electrons=nelec, nocc=nocc, nvir=nvir, orbital_ref=orb_ref,
                )
            return _diag.compute_ccsd_diagnostics(
                t1, t2, nocc=nocc, nvir=nvir, converged=converged,
                correlation_energy=e_corr, hamiltonian_hash_value=hamiltonian_hash,
                orbital_ref=orb_ref,
            )
        except Exception as exc:  # pragma: no cover - defensive; energy must survive
            return _diag.unavailable_record(
                f"diagnostic extraction failed: {type(exc).__name__}",
                converged=converged, correlation_energy=e_corr,
                hamiltonian_hash_value=hamiltonian_hash,
                n_correlated_electrons=nelec, nocc=nocc, nvir=nvir,
            )

    def solve_from_integrals(
        self, h1e: NDArray, h2e: NDArray, norb: int, nocc: int, **kwargs
    ) -> SolverResult:
        """
        Solve CCSD from Hamiltonian integrals.

        Diagonalises h1e to obtain canonical MOs, transforms h2e into
        that basis, and constructs a mean-field reference.  This avoids
        the Davidson divergence caused by using non-canonical (embedded)
        integrals directly.

        Parameters
        ----------
        h1e : np.ndarray
            One-electron Hamiltonian matrix (norb, norb)
        h2e : np.ndarray
            Two-electron repulsion integrals in chemist notation
            Shape: (norb, norb, norb, norb)
        norb : int
            Number of spatial orbitals
        nocc : int
            Number of occupied orbitals (assumes closed shell)
        **kwargs : dict
            Additional parameters passed to solve()

        Returns
        -------
        SolverResult
            CCSD solution with energy and density matrices
        """
        try:
            from pyscf import gto, scf, ao2mo as _ao2mo
            from pyscf import lib as pyscf_lib
        except ImportError as e:
            raise ImportError(
                "PySCF is required for CCSD solver. Install with: pip install pyscf"
            ) from e

        nelec = 2 * nocc
        nvir  = norb - nocc
        nvir_pair = nvir * (nvir + 1) // 2

        # ── Step 1: run SCF to get canonical MOs ─────────────────────────────
        # EWF cluster Hamiltonians are NOT diagonal in the embedding basis.
        # We must run a proper SCF (not just diagonalise h1e) because the
        # canonical Fock eigenvalues include both the one-electron h1e and the
        # two-electron Coulomb/exchange from h2e.  Using np.diag(h1e) as
        # orbital energies misses the J/K contribution, gives wrong CCSD
        # amplitude denominators ε_a − ε_i, and causes Davidson divergence
        # on bath-heavy fragments.
        mol = gto.Mole()
        mol.nelectron = nelec
        mol.nao = norb
        mol.build(dump_input=False, parse_arg=False)

        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h1e
        mf.get_ovlp  = lambda *args: np.eye(norb)
        mf._eri      = _ao2mo.restore(1, h2e, norb)
        mf.level_shift = kwargs.pop("scf_level_shift", 0.0)
        mf.kernel()   # self-consistent; sets mo_coeff, mo_energy, mo_occ

        # ── Step 2: transform h2e into canonical MO basis ────────────────────
        # PySCF's CCSD ao2mo routine internally calls ao2mo.incore.full which
        # uses pyscf.lib.einsum.  On some PySCF versions this path has a
        # tuple-unpacking bug in the contraction list.  We bypass it entirely
        # by pre-transforming h2e with numpy and injecting the result directly.
        C      = mf.mo_coeff                                    # (norb, norb), input→canonical
        h2e_mo = np.einsum("ijkl,ip,jq,kr,ls->pqrs", h2e, C, C, C, C,
                           optimize=True)

        # ── Step 3: build canonical Fock in MO basis ─────────────────────────
        fock_ao = mf.get_fock()
        fock_mo = C.T @ fock_ao @ C                             # (norb, norb)

        # ── Step 4: override CCSD.ao2mo with pre-sliced ERIs ─────────────────
        # _ChemistsERIs packed shapes (from pyscf/cc/ccsd.py pseudo-code):
        #   oooo : (nocc, nocc, nocc, nocc)
        #   ovoo : (nocc, nvir, nocc, nocc)
        #   oovv : (nocc, nocc, nvir, nvir)
        #   ovov : (nocc, nvir, nocc, nvir)
        #   ovvo : (nocc, nvir, nvir, nocc)
        #   ovvv : (nocc, nvir, nvir_pair)   pack_tril on last 2 virt indices
        #   vvvv : (nvir_pair, nvir_pair)    ao2mo.restore(4) 4-fold symmetry
        #   fock : (norb, norb)              Fock in canonical MO basis
        _fock  = fock_mo
        _h2e   = h2e_mo

        def _ao2mo_patched(cc_obj, mo_coeff=None):
            from pyscf.cc.ccsd import _ChemistsERIs
            eris = _ChemistsERIs()
            eris.nocc      = nocc
            eris.mol       = mol
            eris.mo_coeff  = C
            eris.fock      = _fock
            eris.mo_energy = _fock.diagonal().real
            o = slice(0, nocc)
            v = slice(nocc, norb)
            eris.oooo = _h2e[o, o, o, o].copy()
            eris.ovoo = _h2e[o, v, o, o].copy()
            eris.oovv = _h2e[o, o, v, v].copy()
            eris.ovov = _h2e[o, v, o, v].copy()
            eris.ovvo = _h2e[o, v, v, o].copy()
            # ovvv: pack last two virt indices; PySCF reads eris.ovvv[:,p0:p1]
            # with axis-0=nocc, axis-1=nvir, axis-2=nvir_pair
            ovvv_full = _h2e[o, v, v, v]                        # (nocc,nvir,nvir,nvir)
            eris.ovvv = pyscf_lib.pack_tril(
                ovvv_full.reshape(nocc * nvir, nvir, nvir)
            ).reshape(nocc, nvir, nvir_pair)
            # vvvv: 4-fold symmetry
            eris.vvvv = _ao2mo.restore(4, _h2e[v, v, v, v], nvir)
            return eris

        result_mo = self.solve(
            mf,
            _ao2mo_override=_ao2mo_patched,
            _hamiltonian_hash=_diag.hamiltonian_hash(h1e, h2e),
            **kwargs,
        )

        # ── Step 5: back-transform RDMs from canonical MO basis to input basis ──
        # PySCF make_rdm1/make_rdm2 return tensors in the canonical MO basis.
        # Stage 4 contracts them against the cluster-MO-basis integrals stored in
        # the EWF HDF5 file, so they must be expressed in the same basis.
        #
        # For the ixazomib fragments the embedded Fock is already diagonal
        # (Vayesta DUMP outputs canonical orbitals), so C ≈ I and this
        # transformation is numerically trivial (<0.1 μHa effect).  It is applied
        # unconditionally so the code is correct for arbitrary embedded Hamiltonians.
        #
        # Transformation:
        #   rdm1_in[p, q]       = C @ rdm1_mo @ C.T            (1-RDM: covariant rank-2)
        #   rdm2_in[p,q,r,s]    = Σ C[p,a]C[q,b]C[r,c]C[s,d] rdm2_mo[a,b,c,d]
        #
        # The energy formula E = h1·γ₁ + 0.5·(pq|rs)·γ₂[p,q,r,s] is invariant
        # under this transformation (verified numerically in test_ccsd_basis.py).
        if result_mo.rdm1 is not None or result_mo.rdm2 is not None:
            rdm1_in = (C @ result_mo.rdm1 @ C.T
                       if result_mo.rdm1 is not None else None)
            rdm2_in = (np.einsum("abcd,pa,qb,rc,sd->pqrs",
                                 result_mo.rdm2, C, C, C, C, optimize=True)
                       if result_mo.rdm2 is not None else None)
            from ..base import SolverResult as _SR
            return _SR(
                energy=result_mo.energy,
                wavefunction=result_mo.wavefunction,
                rdm1=rdm1_in,
                rdm2=rdm2_in,
                metadata=result_mo.metadata,
            )

        return result_mo

    @staticmethod
    def split_dm2(nocc: int, dm1: NDArray, dm2: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """
        Split 2-RDM into HF reference, 1-body correlation, and approximate-cumulant components.

        Implements the decomposition of Nusspickel, Ibrahim & Booth JCTC 2023 (Eq. 15–16).
        The 1-RDM input ``dm1`` is the *correlated correction* Δγ = γ − γ⁰ (HF contribution
        already removed by the caller).

        Parameters
        ----------
        nocc : int
            Number of occupied orbitals in the reference determinant.
        dm1 : np.ndarray
            Correlated correction Δγ = γ − γ⁰ (HF diagonal entries set to zero by caller).
        dm2 : np.ndarray
            Full two-particle density matrix Γ in the cluster MO basis.

        Returns
        -------
        dm2_0 : np.ndarray
            HF (disconnected) component γ⁰γ⁰ of the 2-RDM.
        dm2_1 : np.ndarray
            Mixed 1-body correlation component (γ⁰Δγ + Δγγ⁰ antisymmetrised terms).
        dm2_2 : np.ndarray
            Approximate two-body cumulant K (Eq. 16 of the paper).  This equals the
            *true* cumulant K̃ plus the (Δγ)²-disconnected terms:
            K = K̃ + ΔγΔγ − ½ΔγΔγᵀ.
            It is NOT the true cumulant K̃.  The partitioned-cumulant energy functional
            E[γˣ, Kˣ] (Eq. 18) is evaluated with K, not K̃.

        Notes
        -----
        The full decomposition is: Γ = dm2_0 + dm2_1 + dm2_2  (Eq. 15).
        Only dm2_2 (the approximate cumulant K) is projected onto the fragment space
        and contracted with the two-electron integrals in compute_fragment_energy().
        """
        dm2_2 = dm2.copy()
        dm2_1 = np.zeros_like(dm2)
        dm2_0 = np.zeros_like(dm2)

        # Build dm2_1 from dm1
        for i in range(nocc):
            dm2_1[i, i, :, :] += dm1 * 2
            dm2_1[:, :, i, i] += dm1 * 2
            dm2_1[:, i, i, :] -= dm1
            dm2_1[i, :, :, i] -= dm1.T

        # Build dm2_0 (HF part)
        for i in range(nocc):
            for j in range(nocc):
                dm2_0[i, i, j, j] += 4
                dm2_0[i, j, j, i] -= 2

        # Extract cumulant
        dm2_2 -= dm2_0 + dm2_1

        return dm2_0, dm2_1, dm2_2

    def compute_fragment_energy(
        self, result: SolverResult, fock: NDArray, h2e: NDArray, proj: NDArray, nocc: int
    ) -> tuple[float, float]:
        """
        Compute partitioned cumulant energy contributions.

        This method implements the energy decomposition used in
        fragment-based methods like EWF (Embedded Wavefunction).

        Parameters
        ----------
        result : SolverResult
            CCSD solution containing density matrices
        fock : np.ndarray
            Fock matrix in cluster basis
        h2e : np.ndarray
            Two-electron integrals in cluster basis
        proj : np.ndarray
            Fragment projector matrix
        nocc : int
            Number of occupied orbitals

        Returns
        -------
        e1_pc : float
            One-body partitioned cumulant energy
        e22_pc : float
            Two-body partitioned cumulant energy

        Raises
        ------
        ValueError
            If result doesn't contain required RDMs

        Notes
        -----
        This implements the partitioned cumulant approach where:
        - dm1_pc = dm1 - HF contribution
        - dm2 is split into different components using split_dm2()
        - Only fragment-projected contributions are included
        """
        if result.rdm1 is None:
            raise ValueError(
                "Result must contain rdm1 for fragment energy calculation. "
                "Call solve() with compute_rdms=True."
            )

        if result.rdm2 is None:
            raise ValueError(
                "Result must contain rdm2 for fragment energy calculation. "
                "Call solve() with compute_rdms=True."
            )

        dm1 = result.rdm1.copy()
        dm2 = result.rdm2.copy()

        # Remove Hartree-Fock contribution from dm1
        dm1_pc = dm1.copy()
        dm1_pc[np.diag_indices(nocc)] -= 2.0

        # Split dm2 into components
        dm2_0, dm2_1, dm2_2 = self.split_dm2(nocc, dm1_pc, dm2)

        # Project dm1 onto fragment
        dm1_pc_proj = proj @ dm1_pc

        # One-body energy contribution
        e1_pc = np.einsum("pq,pq->", fock, dm1_pc_proj)

        # Two-body contribution (only cumulant part)
        dm2_pc = np.einsum("Ijkl,iI->ijkl", dm2_2, proj)
        e22_pc = 0.5 * np.einsum("pqrs,pqrs->", h2e, dm2_pc)

        return float(e1_pc), float(e22_pc)

    def __repr__(self) -> str:
        """String representation of CCSD solver."""
        return (
            f"CCSD(conv_tol={self.conv_tol}, max_cycle={self.max_cycle}, "
            f"diis_space={self.diis_space}, verbose={self.verbose})"
        )
