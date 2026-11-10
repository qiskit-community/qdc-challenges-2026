"""Qiskit Serverless entrypoint: run the spacetime Pauli check search remotely.

Uploaded under the title ``add_checks_remote`` and called from Steps 3 and 4 of
the notebook. The search needs no QPU and is the longest classical step there, so
it is the natural thing to move off a laptop.

**You should not need to edit this file.** The search itself lives in
``check_search.py``, which both this entrypoint and the notebook import, so the
same code runs either way. What is left here is plumbing, and two details of it
matter:

* **No backend here.** The function runs under the Serverless account, which need
  not see your device, so the backend stays on your side.
* **The noise model arrives as plain numbers.** ``paulice``'s ``NoiseModel`` is a
  dataclass and does not survive as a function argument, so the caller sends its
  three fields (uniform gate noise only) and this file rebuilds it.

The return value is decomposed into the six fields of ``CheckedCircuit``. That
class is a frozen dataclass which re-derives its post-selection decoder in
``__post_init__``, so the notebook rebuilds the objects losslessly from these
lists: the decoder, ``dope()`` and ``estimate_fault_rates()`` all come back.
"""

from importlib.metadata import version

from check_search import add_checks
from qiskit_paulice.noise_models import NoiseModel
from qiskit_serverless import get_arguments, save_result

arguments = get_arguments()

isa_circuit = arguments.get("isa_circuit")
target_qubits = [int(q) for q in arguments.get("target_qubits")]
ancilla_qubits = [int(q) for q in arguments.get("ancilla_qubits")]
noise_model = NoiseModel(**arguments.get("noise_model"))
seed = arguments.get("seed")
options = arguments.get("options") or {}

print(f"qiskit-paulice {version('qiskit-paulice')}")
print(
    f"{len(target_qubits)} target qubits, {len(ancilla_qubits)} ancillas, "
    f"{noise_model}, options={options}"
)

checked_circuits = add_checks(
    isa_circuit, target_qubits, ancilla_qubits, noise_model, seed=seed, **options
)

print(f"found {len(checked_circuits)} checked circuits")

save_result(
    {
        "circuits": [cc.circuit for cc in checked_circuits],
        "target_qubits": [list(cc.target_qubits) for cc in checked_circuits],
        "check_qubits": [list(cc.check_qubits) for cc in checked_circuits],
        "check_support": [
            [list(support) for support in cc.check_support] for cc in checked_circuits
        ],
        "cost": [cc.cost for cc in checked_circuits],
        "cost_metric": [cc.cost_metric for cc in checked_circuits],
    }
)
