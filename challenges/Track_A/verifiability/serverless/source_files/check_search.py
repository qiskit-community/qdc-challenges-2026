"""The spacetime Pauli check search, shared by the notebook and Qiskit Serverless.

There is nothing to edit here. The notebook imports ``add_checks`` for a local run
and ``add_checks_remote.py`` imports the same function when the search runs in the
cloud, so the two can never disagree. Every choice that makes the search good or
bad arrives as a keyword argument from the caller: the notebook's
``CHECK_OPTIONS`` is forwarded straight through to ``add_pauli_checks``.

Nothing here imports ``qiskit_serverless``, so the module loads fine on a laptop.
"""

from qiskit_paulice import add_pauli_checks


def add_checks(
    isa_circuit, target_qubits, ancilla_qubits, noise_model, seed=None, **options
):
    """Insert spacetime Pauli checks, returning one CheckedCircuit per check count.

    Parameters
    ----------
    isa_circuit
        The transpiled Clifford circuit, with no measurements.
    target_qubits, ancilla_qubits
        Paired check wiring: ``ancilla_qubits[i]`` checks ``target_qubits[i]``.
    noise_model
        A ``paulice`` noise model, scoring how much noise each check catches.
    seed
        Seed for the search, so the result is reproducible.
    options
        The search settings, forwarded to ``add_pauli_checks`` unchanged. This is
        where ``cost``, ``method`` and ``stabilizers`` arrive from the notebook.
    """
    return add_pauli_checks(
        isa_circuit,
        target_qubits,
        noise_model,
        ancilla_qubits=ancilla_qubits,
        seed=seed,
        **options,
    )
