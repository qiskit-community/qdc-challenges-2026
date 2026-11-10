# QDC 2026 Challenge Installation Guide

Please complete these steps before the challenges begin.

> **Note:** Windows is not fully supported. `mthree`, `ffsim` and `pyscf` do not have Windows wheels, so they are skipped on Windows and the Proteasome challenge cannot run there natively. Windows users should work under [WSL](https://learn.microsoft.com/windows/wsl/install), or on macOS / Linux.

## Step 0: Create your IBM Quantum account

[Create an IBM Quantum account](https://quantum.cloud.ibm.com/signin) if you haven't already. **Sign up using the *same email address* you registered with for this event.**

## Step 1: Clone the repository

In your terminal, navigate to a folder where you would like to place the QDC 2026 Challenge repository, then run:
```
git clone https://github.com/qiskit-community/qdc-challenges-2026.git
cd qdc-challenges-2026
```

## Step 2: Create a virtual environment

The challenges are built and tested with **Python 3.14**. If you do not have it installed, follow the instructions [here](https://www.python.org/downloads/). Then create and activate a new virtual environment named **`qdc2026-venv`**.

macOS / Linux:
```
python3.14 -m venv qdc2026-venv
source qdc2026-venv/bin/activate
```

Windows (PowerShell):
```
py -3.14 -m venv qdc2026-venv
.\qdc2026-venv\Scripts\Activate.ps1
```

## Step 3: Install the required packages

A single [`requirements.txt`](./requirements.txt) at the root of this repository covers all eight challenges. With the virtual environment active, run:
```
python -m pip install -r requirements.txt
```

- Whenever a challenge notebook asks you to install `requirements.txt`, it means this file. You only need to install it once.
- `mthree` is built from source, so a C compiler is required (Xcode Command Line Tools on macOS, `gcc` on Linux).
- On Windows, `mthree`, `ffsim` and `pyscf` are not available and are skipped. The Proteasome challenge needs them, so run it under [WSL](https://learn.microsoft.com/windows/wsl/install) (or on macOS / Linux).

## Step 4: Register the virtual environment as a Jupyter kernel

Run the following lines to register your virtual environment as a Jupyter kernel and launch Jupyter:
```
python -m ipykernel install --user --name=qdc2026-venv --display-name "QDC 2026"
jupyter lab
```

Open each challenge notebook from inside its own folder, then click **`Kernel > Change Kernel...`** and select **`QDC 2026`**. The notebooks load their helper modules and data with paths relative to that folder.

## Step 5: Save your IBM Quantum account

The notebooks connect with `QiskitRuntimeService()`, which uses your default saved account. Save your event credentials once by running the following, replacing the placeholders with your API key and the instance for QDC 2026:
```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    token="<YOUR_API_KEY>",
    instance="<YOUR_INSTANCE>",
    set_as_default=True,
    overwrite=True,
)
```

The Proteasome challenge additionally reads its token, instance and backend from a local `.env` file: copy `.env.example` to `.env` in that challenge's folder and fill it in. Do not commit this file.

See the [README](./README.md#installation-and-setup) for how to get your team access credentials at the event.

You're ready to get started! Good luck :)
