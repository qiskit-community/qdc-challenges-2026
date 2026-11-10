"""Provided Sampler and readout-calibration execution for the Floquet challenge.

The notebook supplies its active backend, service, ledger and hardware switches
explicitly. Importing this module neither connects to IBM Quantum nor submits a
job. Bookkeeping is delegated to :mod:`utils.runtime_helpers` so recovered jobs
retain their original settings and each measured QPU charge is recorded once.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from utils.floquet_helpers import QPUBudget
from utils.runtime_helpers import charge_qpu_time, run_program

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


def run_counts(
    isa_circuits: list[QuantumCircuit], label: str, settings: Mapping[str, Any], *,
    use_hardware: bool, allow_submit: bool, backend_name: str, service: Any,
    ledger_path: str | Path, backend: Any, shots: int,
) -> tuple[list[dict[str, int]], str, float]:
    """Recover or submit a standard-target Sampler job and return its counts.

    ``isa_circuits`` must already be transpiled, measured circuits for
    ``backend``. ``label`` identifies the experiment in the ledger; ``settings``
    records its size, depths, folding factor and mode. The service is used only
    to retrieve jobs, and ``allow_submit=False`` permits recovery without a new
    submission. ``use_hardware=False`` raises before contacting the QPU.

    Return count dictionaries in circuit order, the job ID, and the ledger's
    cumulative measured QPU seconds. Sampling uses ``shots`` per circuit and the
    workflow's fixed seed of 2026. Submission, result retrieval and accounting
    failures propagate rather than silently substituting local data.
    """
    if not use_hardware:
        raise RuntimeError("Sampler jobs require both hardware switches.")
    from qiskit_ibm_runtime import Batch
    from qiskit_ibm_runtime.executor_sampler import Sampler

    estimate = QPUBudget.estimate(len(isa_circuits), shots=shots)

    def submit() -> Any:
        """Submit the measured ISA circuits in one standard-backend batch."""
        with Batch(backend=backend) as batch:
            return Sampler(mode=batch).run(isa_circuits, shots=shots, dry_run=False)

    key = f"sampler-{label}-standard"
    job = run_program(
        key, backend_name=backend_name,
        settings={"shots": shots, "circuits": len(isa_circuits), **settings},
        seed=2026, submit=submit, retrieve=service.job,
        estimated_qpu_seconds=estimate,
        ledger_path=ledger_path, allow_submit=allow_submit,
    )
    result = job.result()
    try:
        charged = float(job.metrics()["usage"]["qpu_charge_time_seconds"])
    except (AttributeError, KeyError, TypeError, ValueError):
        charged = float(job.usage())
    job_id = job.job_id() if callable(job.job_id) else job.job_id
    total = charge_qpu_time(
        key, job_id=job_id, qpu_seconds=charged, ledger_path=ledger_path,
    )
    return [item.data.meas.get_counts() for item in result], job_id, total


@contextmanager
def tracked_readout_calibration(
    mitigator: Any, qubits: list[int], *, use_hardware: bool,
    allow_submit: bool, backend_name: str, service: Any,
    ledger_path: str | Path, n_qubits: int,
) -> Iterator[dict[str, Any]]:
    """Track one balanced mthree calibration and yield its calibration settings.

    ``mitigator`` is an initialized ``mthree.M3Mitigation`` for the standard
    backend. ``qubits`` is the sorted union of physical measurement qubits;
    ``n_qubits`` identifies the logical challenge size in the ledger and saved
    calibration filename. The notebook calls ``mitigator.cals_from_system``
    inside this context with the yielded settings. Calibration is synchronous,
    balanced, and uses at most 10,000 shots per circuit.

    The temporary backend-run wrapper recovers or submits one job through the
    active ledger. It is always restored on exit, and a recovered or submitted
    job is charged even if processing the calibration fails. Multiple-job
    calibrations are rejected before submission because each job must be charged
    before another runs. Hardware, missing-qubit, missing-charge and calibration
    failures remain visible; importing or binding this context submits no job.
    """
    if not use_hardware:
        raise RuntimeError("Readout calibration requires both hardware switches.")
    if not qubits:
        raise ValueError("Provide the physical measurement qubits before calibrating.")

    from mthree.mitigation import HadamardGenerator
    from runningman.job import RunningManJob

    # mthree submits all jobs before returning; require one chargeable job.
    n_cal_circuits = HadamardGenerator(len(qubits)).length
    if n_cal_circuits > mitigator.system_info["max_circuits"]:
        raise RuntimeError("mthree calibration needs multiple jobs; split and track them before running.")

    calibration_shots = min(int(mitigator.system_info["max_shots"]), 10000)
    calibration_path = (
        Path(ledger_path).parent
        / f"walkthrough_mthree_{backend_name}_{n_qubits}q_standard.json"
    )
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    original_run = mitigator.system.run
    tracked_jobs: list[Any] = []
    key = f"mthree-readout-{n_qubits}q-standard"

    def tracked_run(
        circuits: Sequence[QuantumCircuit], shots: int | None = None,
        rep_delay: float | None = None, job_tags: list[str] | None = None,
    ) -> Any:
        """Intercept mthree's backend call and recover or submit its sole job."""
        if tracked_jobs:
            raise RuntimeError("A second mthree job would run before the first is charged.")
        settings = {
            "size": n_qubits, "physical_qubits": qubits,
            "method": "balanced", "calibration_shots": calibration_shots,
            "circuits": len(circuits), "shots": shots,
            "rep_delay": rep_delay, "job_tags": job_tags,
        }
        job = run_program(
            key, backend_name=backend_name, settings=settings, seed=2026,
            submit=lambda: original_run(circuits, shots=shots, rep_delay=rep_delay,
                                        job_tags=job_tags),
            retrieve=lambda job_id: RunningManJob(service.job(job_id)),
            estimated_qpu_seconds=QPUBudget.estimate(len(circuits), shots=shots),
            ledger_path=ledger_path, allow_submit=allow_submit,
        )
        tracked_jobs.append(job)
        return job

    mitigator.system.run = tracked_run
    try:
        yield {
            "shots": calibration_shots, "method": "balanced",
            "cals_file": str(calibration_path), "async_cal": False,
        }
    finally:
        mitigator.system.run = original_run
        # Processing failures must not hide a charge for an existing job.
        if tracked_jobs:
            job = tracked_jobs[0]
            job_id = job.job_id() if callable(job.job_id) else job.job_id
            try:
                charged = float(job.metrics()["usage"]["qpu_charge_time_seconds"])
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("Inspect the mthree job charge before another submission.") from exc
            total = charge_qpu_time(
                key, job_id=job_id, qpu_seconds=charged, ledger_path=ledger_path,
            )
            mitigator.calibration_job_id = job_id
            print(f"mthree: {len(qubits)} physical qubits | job {job_id} | total QPU {total:.1f} s")

    if len(tracked_jobs) != 1:
        raise RuntimeError("mthree did not submit or recover one tracked calibration job.")
    if mitigator._job_error is not None:
        raise RuntimeError("mthree could not build its readout calibration.") from mitigator._job_error
