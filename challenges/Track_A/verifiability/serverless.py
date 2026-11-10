"""Optional: run the Step 3 and Step 4 check search on Qiskit Serverless.

The search needs no QPU, only CPU time, so you can send it to IBM's cloud and
get the same ``CheckedCircuit`` objects back. Running locally is always valid:
the grader scores the checks you find, not the machine that found them.

How to use it from the notebook:

1. Install ``qiskit-ibm-catalog`` (it is in ``requirements.txt``) and have your
   IBM Quantum account saved, as for the QPU steps.
2. Upload the search once. This sends ``serverless/source_files/`` as a Qiskit
   Function; repeat it only if you edit those files::

       import serverless

       serverless.upload()

3. In the Step 3 exercise cell, write ``add_checks`` as usual, but call
   ``serverless.add_checks`` where you would call ``add_pauli_checks``. It takes
   the same keyword arguments::

       def add_checks(isa_circuit, target_qubits, ancilla_qubits, noise_model, **options):
           return serverless.add_checks(
               isa_circuit,
               target_qubits,
               ancilla_qubits,
               noise_model,
               seed=seed,
               cost="gamma",  # ...and any other add_pauli_checks option
               **options,
           )

   Step 4 calls the same ``add_checks``, so it runs in the cloud too.
4. Run the cell. It waits for the remote job (about 30 minutes for Step 3 with
   ``windowed_genetic``), prints the job's logs, and returns the usual list, so
   the pickling and grading cells need no change.

The backend stays on your side: the remote function only gets the noise model
you built in Step 2, which must be uniform
(``NoiseModel.from_backend(..., uniform_gate_noise=True)``, as in the notebook).
If your account is saved under a name, pass it as ``name=`` to both calls.
"""

import time
from pathlib import Path

FUNCTION_TITLE = "add_checks_remote"
SOURCE_DIR = Path(__file__).parent / "serverless" / "source_files"
# The paulice installed remotely. Bump it to the release that matches your local
# version once it ships; until then remote and local searches can differ.
PAULICE = "qiskit-paulice==0.3.0"
POLL_SECONDS = 30


def _client(name):
    from qiskit_ibm_catalog import QiskitServerless

    return QiskitServerless(name=name)


def upload(name=None):
    """Upload ``serverless/source_files/`` as a Qiskit Function. Safe to repeat.

    ``name`` is the saved account to use; ``None`` means the default one.
    """
    from qiskit_ibm_catalog import QiskitFunction

    _client(name).upload(
        QiskitFunction(
            title=FUNCTION_TITLE,
            entrypoint="add_checks_remote.py",
            working_dir=str(SOURCE_DIR),
            dependencies=[PAULICE],
        )
    )
    print(f"uploaded {FUNCTION_TITLE!r}")


def add_checks(
    isa_circuit,
    target_qubits,
    ancilla_qubits,
    noise_model,
    *,
    seed=None,
    name=None,
    **options,
):
    """Run the check search remotely, returning one CheckedCircuit per check count.

    Parameters
    ----------
    isa_circuit, target_qubits, ancilla_qubits, noise_model
        As for ``add_pauli_checks``. ``noise_model`` must have uniform gate noise
        (``NoiseModel.from_backend(..., uniform_gate_noise=True)``).
    seed
        Seed for the search, so the result is reproducible.
    name
        The saved account to use, as in ``upload``.
    options
        Forwarded unchanged to ``add_pauli_checks``.
    """
    noise = _noise_fields(noise_model)
    function = _client(name).load(FUNCTION_TITLE)
    if function is None:
        raise RuntimeError(
            f"No function {FUNCTION_TITLE!r} found. Run serverless.upload() first."
        )
    job = function.run(
        isa_circuit=isa_circuit,
        target_qubits=[int(q) for q in target_qubits],
        ancilla_qubits=[int(q) for q in ancilla_qubits],
        noise_model=noise,
        seed=seed,
        options=options,
    )
    print(f"serverless job submitted: {job.job_id}")
    status = job.status()
    print(f"  {status}")
    t0 = time.time()
    while status not in ("DONE", "ERROR", "CANCELED"):
        time.sleep(POLL_SECONDS)
        new_status = job.status()
        if new_status != status:  # print changes only, not every poll
            status = new_status
            print(f"  {status} after {time.time() - t0:.0f}s")
    if status != "DONE":
        raise RuntimeError(f"serverless job ended as {status}:\n{job.logs()}")
    print(job.logs())
    return _rebuild(job.result())


def _noise_fields(noise_model):
    """The noise model as plain floats, since a NoiseModel cannot be sent as is."""
    fields = {
        "gate_noise": noise_model.gate_noise,
        "readout_noise": noise_model.readout_noise,
        "idling_noise": noise_model.idling_noise,
    }
    if not all(v is None or isinstance(v, (int, float)) for v in fields.values()):
        raise ValueError(
            "Only a uniform noise model can be sent to Serverless. Build it with "
            "NoiseModel.from_backend(backend, layout, uniform_gate_noise=True)."
        )
    return {k: None if v is None else float(v) for k, v in fields.items()}


def _rebuild(result):
    """Reassemble CheckedCircuit objects from the six fields the function returned.

    CheckedCircuit is a frozen dataclass that rebuilds its post-selection decoder
    in __post_init__, so nothing is lost on the way back.
    """
    from qiskit_paulice import CheckedCircuit

    return [
        CheckedCircuit(
            circuit=circuit,
            target_qubits=tuple(targets),
            check_qubits=tuple(checks),
            check_support=tuple(tuple(s) for s in support),
            cost=cost,
            cost_metric=metric,
        )
        for circuit, targets, checks, support, cost, metric in zip(
            result["circuits"],
            result["target_qubits"],
            result["check_qubits"],
            result["check_support"],
            result["cost"],
            result["cost_metric"],
            strict=True,
        )
    ]
