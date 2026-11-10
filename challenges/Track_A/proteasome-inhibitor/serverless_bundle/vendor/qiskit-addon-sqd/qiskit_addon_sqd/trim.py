# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Trim SQD, a schedule that screens more configurations than it diagonalizes.

:func:`~qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian` diagonalizes each
subsampled batch and keeps the best result. *Trim SQD* instead treats the batches as
candidates to be screened: each is diagonalized, trimmed down to the configurations
carrying the largest weight, and the survivors of all the batches are merged and
diagonalized together. The energy of that merged diagonalization is what the iteration
reports, and its own trimmed output seeds the next one.

The point is that screening is cheaper than diagonalizing. Each batch is roughly the
size of the merged subspace, so a set of candidates several times larger than the
affordable subspace can be ranked for the cost of a few extra diagonalizations of that
size.

This is expressed as a :class:`~qiskit_addon_sqd.fermion.SubspacePolicy`, so it describes
the schedule and leaves the diagonalizations themselves to whatever ``sci_solver`` the
caller passes.
"""

from __future__ import annotations

import math

import numpy as np

from .fermion import SCIResult, SubspaceRequest, batch_to_ci_strings
from .subsampling import partition_subsample


class _TrimSchedule:
    """The parts of the trim SQD schedule that do not depend on how batches are drawn.

    Trimming, merging and carrying over are the same however the batches were built, so
    they live here. The subclasses differ only in :meth:`prepare_subspaces`, which is
    where the choice of what to make disjoint is made. This class is not part of the
    public interface; use :class:`TrimPolicy` or :class:`SectorTrimPolicy`.
    """

    def __init__(
        self,
        *,
        trim_ratio: float = 0.1,
        carryover_ratio: float | None = None,
        max_strings_per_trim: int | None = None,
        max_carryover: int | None = None,
    ) -> None:
        """Initialize the policy.

        This policy performs two kinds of trim, and they are bounded separately. The
        *screening* trims shrink each batch of the first round, once per batch; the
        *carryover* trim shrinks the merged subspace to what seeds the next iteration.

        Args:
            trim_ratio: The fraction of each batch's CI strings, per spin sector, that
                survive screening and enter the merged subspace. Ranked by summed squared
                amplitude over the other spin sector.
            carryover_ratio: The fraction of the merged subspace's CI strings that seed the
                next iteration. Defaults to ``trim_ratio``, which keeps the subspace size
                roughly stationary from one iteration to the next.
            max_strings_per_trim: Ceiling on the number of CI strings, per spin sector, that
                any one screening trim retains. Where it binds it lowers the effective
                ratio, so a trim keeps
                ``min(ceil(trim_ratio * len(strings)), max_strings_per_trim)`` strings in
                each sector. Defaults to ``None``, leaving ``trim_ratio`` to decide.

                Reach for it when a ratio would retain more than intended:
                ``symmetrize_spin`` merges the two spin sectors into a single list and so
                roughly doubles what each batch offers, and a large ``samples_per_batch``
                does the same.

                **This is not a bound on a subspace.** It applies once per batch, and the
                contributions are merged, so the merged subspace can hold up to
                ``num_batches`` times as many strings per sector -- in practice close to
                that, since the batches have little in common. The ``max_dim`` argument of
                :func:`~qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian` is what
                bounds a subspace.
            max_carryover: Ceiling on the number of CI strings, per spin sector, that seed
                the next iteration, applied the same way to ``carryover_ratio``. Unlike
                ``max_strings_per_trim`` this governs a single trim of the merged subspace,
                so it bounds the carryover as a whole.

        Note:
            Neither ceiling constrains a solver that selects its own carryover through
            :attr:`~qiskit_addon_sqd.fermion.SCIResult.carryover`, whose selection is used
            as given.

        Raises:
            ValueError: A ratio was not in ``(0, 1]``, or a ceiling was less than one.
        """
        for name, ratio in (
            ("trim_ratio", trim_ratio),
            ("carryover_ratio", carryover_ratio),
        ):
            if ratio is not None and not 0 < ratio <= 1:
                raise ValueError(f"{name} must be greater than zero and at most one. Got {ratio}.")
        for name, limit in (
            ("max_strings_per_trim", max_strings_per_trim),
            ("max_carryover", max_carryover),
        ):
            if limit is not None and limit < 1:
                raise ValueError(f"{name} must be at least one. Got {limit}.")

        self.trim_ratio = trim_ratio
        self.carryover_ratio = trim_ratio if carryover_ratio is None else carryover_ratio
        self.max_strings_per_trim = max_strings_per_trim
        self.max_carryover = max_carryover

    def refine(
        self, results: list[SCIResult], round_index: int
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """Merge what the batches retained, then stop after diagonalizing it.

        The screening round's results are trimmed to their highest-weight CI strings and
        merged into a single subspace, which the loop diagonalizes as the second round.
        """
        if round_index:
            return None
        kept_a, kept_b = [], []
        for result in results:
            strings_a, strings_b = _trim(result, self.trim_ratio, self.max_strings_per_trim)
            kept_a.append(strings_a)
            kept_b.append(strings_b)
        return [
            (
                np.unique(np.concatenate(kept_a)),
                np.unique(np.concatenate(kept_b)),
            )
        ]

    def select_result(self, results: list[SCIResult]) -> SCIResult:
        """Report the merged diagonalization, which is the only result of the last round."""
        (result,) = results
        return result

    def select_carryover(
        self,
        result: SCIResult,
        *,
        symmetrize_spin: bool = False,  # pylint: disable=unused-argument
    ) -> tuple[np.ndarray, np.ndarray]:
        """Carry over the highest-weight CI strings of the merged diagonalization.

        ``symmetrize_spin`` is unused: the strings are ranked within each spin sector, and
        merging the sectors afterwards preserves that ranking, so the loop can apply it.
        """
        return _trim(result, self.carryover_ratio, self.max_carryover)


class TrimPolicy(_TrimSchedule):
    """A :class:`~qiskit_addon_sqd.fermion.SubspacePolicy` that screens disjoint batches.

    Each iteration diagonalizes every batch, keeps the highest-weight CI strings of each,
    merges them, and performs one further diagonalization over the merged subspace. That
    diagonalization's energy is what the iteration reports, and its own trimmed CI strings
    seed the next one.

    Pass an instance as the ``policy`` argument of
    :func:`~qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`::

        result = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=300,
            norb=norb,
            nelec=nelec,
            num_batches=5,
            policy=TrimPolicy(trim_ratio=0.1),
        )

    The cost of an iteration is ``num_batches + 1`` diagonalizations rather than
    ``num_batches``, and the merged subspace is bounded by what the batches retain.

    The batches are drawn as one pool of ``samples_per_batch * num_batches`` bitstrings
    divided among them, so that no *bitstring* appears in more than one. Screening selects
    among the batches, so it is only as useful as the batches differ: batches drawn
    independently would overlap heavily, which shrinks the pool of distinct candidates and
    gives a bitstring appearing in several batches several chances to survive the trim.
    Because that follows from what this schedule does, it is not something to configure.

    What this makes disjoint is the configurations. The per-spin CI string arrays derived
    from them can still share strings between batches, since two configurations may agree
    on one of their halves. :class:`SectorTrimPolicy` makes those arrays disjoint instead.

    Note:
        A subspace here is spanned by the Cartesian product of a spin-alpha and a
        spin-beta CI string array, so trimming to a given number of *configurations*
        retains close to that many strings in each spin sector, whose product is larger.
        ``max_strings_per_trim`` and ``max_carryover`` bound the strings per sector, and the ``max_dim`` argument of
        :func:`~qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian` bounds the
        subspace that the next iteration builds from them.

    Note:
        Whether screening improves the result depends on the system and the scale. It is
        worth comparing against a run without it, which is what passing
        ``num_batches=1`` amounts to: the single batch is diagonalized, trimmed and
        diagonalized again, with no cross-batch selection. Screening has the most to
        offer when the sampled configurations greatly outnumber what can be
        diagonalized at once, and the least when a subspace of the affordable size
        already captures most of the wavefunction.
    """

    def prepare_subspaces(self, request: SubspaceRequest) -> list[tuple[np.ndarray, np.ndarray]]:
        """Divide one pool of bitstrings among the batches, so that they are disjoint."""
        batches = partition_subsample(
            request.bitstrings,
            request.probabilities,
            samples_per_batch=request.samples_per_batch,
            num_batches=request.num_batches,
            rand_seed=request.rng,
        )
        return [
            batch_to_ci_strings(
                batch,
                request.norb,
                request.carryover_strings_a,
                request.carryover_strings_b,
                symmetrize_spin=request.symmetrize_spin,
            )
            for batch in batches
        ]


class SectorTrimPolicy(_TrimSchedule):
    """A trim SQD policy that makes each spin sector's CI strings disjoint.

    This screens and merges exactly as :class:`TrimPolicy` does, and differs only in how
    the batches are built. Rather than partitioning the sampled bitstrings and deriving
    per-spin CI strings from each part, it derives the CI strings once and partitions
    *those*, one sector at a time. So a spin-alpha string appears in exactly one batch,
    which :class:`TrimPolicy` does not guarantee: two different configurations can agree
    on their alpha half and land in different batches.

    Use it as the ``policy`` argument, in place of :class:`TrimPolicy`::

        result = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=300,
            norb=norb,
            nelec=nelec,
            num_batches=5,
            policy=SectorTrimPolicy(trim_ratio=0.1),
        )

    Warning:
        As written, this policy is much worse than :class:`TrimPolicy` at the scales
        measured here: on N2 in a 6-31g basis with 16 active orbitals,
        ``samples_per_batch=300`` and ``num_batches=5``, the error was roughly 83 mHa
        against 10 mHa, across several seeds.

        The cause is the carryover, and it is worth understanding before choosing between
        the two policies. Partitioning the strings puts each carried-over string in exactly
        one batch, so with five batches every batch is anchored by only a fifth of the
        highest-weight strings from the previous iteration. The subspace *sizes* barely
        change -- the batches differ from :class:`TrimPolicy`'s by a string or two and both
        reach the ``max_dim`` cap -- so what suffers is which strings get chosen, since each
        batch ranks its candidates against a poorer reference.

        Broadcasting the carryover to every batch instead, while still partitioning the
        sampled strings, closes the gap entirely: the same measurement gives 11 to 12 mHa,
        indistinguishable from :class:`TrimPolicy`. That variant is *not* what the reference
        implementation does, which merges the carryover into the pool it divides. The
        reference runs one batch per group of processes with its sample count sized for the
        whole pool, so each of its batches keeps a large share of the carryover; dividing a
        few dozen strings five ways is a different regime, and the detail that is harmless
        there is not harmless here.

    Note:
        The configurations requested through ``include_configurations`` are unaffected by
        the division, since the loop adds those to every subspace after a policy has run.

    Note:
        The strings are dealt from the pool in descending order of the number of times
        they were sampled, so each batch holds a similar spread of weights and the
        ordering that ``max_dim`` truncates is still meaningful. The reference
        implementation of this scheme shuffles the pool instead, which is equivalent when
        no truncation binds but discards the ranking when one does.
    """

    def __init__(
        self,
        *,
        trim_ratio: float = 0.1,
        carryover_ratio: float | None = None,
        max_strings_per_trim: int | None = None,
        max_carryover: int | None = None,
        partition_carryover: bool = True,
    ) -> None:
        """Initialize the policy.

        Args:
            trim_ratio: As for :class:`TrimPolicy`.
            carryover_ratio: As for :class:`TrimPolicy`.
            max_strings_per_trim: As for :class:`TrimPolicy`.
            max_carryover: As for :class:`TrimPolicy`.
            partition_carryover: Whether the carryover strings are divided among the
                batches along with the sampled ones, which is what the reference
                implementation does and what the default reproduces. Set it to ``False`` to
                place the carryover in every batch instead, which measures far better at
                the scales tested; see the warning above.

        Raises:
            ValueError: A ratio was not in ``(0, 1]``, or a limit was less than one.
        """
        super().__init__(
            trim_ratio=trim_ratio,
            carryover_ratio=carryover_ratio,
            max_strings_per_trim=max_strings_per_trim,
            max_carryover=max_carryover,
        )
        self.partition_carryover = partition_carryover

    def prepare_subspaces(self, request: SubspaceRequest) -> list[tuple[np.ndarray, np.ndarray]]:
        """Build the CI strings from one pool, then divide each spin sector among them."""
        num_batches = request.num_batches
        # Draw a pool spanning every batch, exactly as TrimPolicy does, so that the two
        # policies screen the same number of sampled bitstrings. Taking the whole input
        # instead would ignore samples_per_batch entirely.
        (pool,) = partition_subsample(
            request.bitstrings,
            request.probabilities,
            samples_per_batch=request.samples_per_batch * num_batches,
            num_batches=1,
            rand_seed=request.rng,
        )
        empty = np.array([], dtype=np.int64)
        carryover_a = request.carryover_strings_a if self.partition_carryover else empty
        carryover_b = request.carryover_strings_b if self.partition_carryover else empty
        pool_a, pool_b = batch_to_ci_strings(
            pool,
            request.norb,
            carryover_a,
            carryover_b,
            symmetrize_spin=request.symmetrize_spin,
        )

        def batch(pool_strings: np.ndarray, carryover: np.ndarray, offset: int) -> np.ndarray:
            # Deal round-robin from the ranked pool, so every batch gets a similar spread
            # of weights, and put any un-partitioned carryover ahead of the result.
            return np.concatenate((carryover, pool_strings[offset::num_batches]))

        held_a = empty if self.partition_carryover else request.carryover_strings_a
        held_b = empty if self.partition_carryover else request.carryover_strings_b
        if request.symmetrize_spin:
            # The sectors hold the same strings, so build one array and use it for both.
            return [
                (strs, strs)
                for strs in (batch(pool_a, held_a, offset) for offset in range(num_batches))
            ]
        return [
            (batch(pool_a, held_a, offset), batch(pool_b, held_b, offset))
            for offset in range(num_batches)
        ]


def _trim(
    result: SCIResult,
    ratio: float,
    max_strings: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the highest-weight CI strings of a diagonalization, per spin sector.

    A solver that trimmed the subspace itself reports its selection in
    ``SCIResult.carryover``, which is taken as given. Otherwise the strings are ranked
    by summed squared amplitude over the other spin sector.
    """
    if result.carryover is not None:
        return result.carryover

    sci_state = result.sci_state
    if sci_state is None:
        raise ValueError(
            "TrimPolicy needs either an SCI state or a carryover from each "
            "diagonalization in order to rank the CI strings, but the solver returned "
            "neither."
        )
    amplitudes = sci_state.amplitudes
    weights_a = np.sum(np.abs(amplitudes) ** 2, axis=1)
    weights_b = np.sum(np.abs(amplitudes) ** 2, axis=0)
    return (
        _top_strings(sci_state.ci_strs_a, weights_a, ratio, max_strings),
        _top_strings(sci_state.ci_strs_b, weights_b, ratio, max_strings),
    )


def _top_strings(
    strings: np.ndarray,
    weights: np.ndarray,
    ratio: float,
    max_strings: int | None,
) -> np.ndarray:
    """Return the strings of largest weight, in descending order of weight."""
    num = max(1, math.ceil(ratio * len(strings)))
    if max_strings is not None:
        num = min(num, max_strings)
    order = np.argsort(-weights, kind="stable")[:num]
    return strings[order]
