"""
Stage 1 Entrypoint — Hartree-Fock (any molecule)

Runs PySCF restricted HF for a fully specified calculation, validates the result,
base64-encodes mf_data.pkl, and returns it with the provenance record in save_result.

The calculation is defined only by the ``spec`` argument; no molecule, path or
file in this bundle is assumed.

Arguments
---------
spec : dict   ``Stage1Spec.to_dict()`` — embedded XYZ text, units, charge, spin and
              the HF configuration (method, basis, conv_tol, max_cycle, init_guess,
              level_shift, damp, verbose).

Returns (via save_result)
-------------------------
The ``stage1-hf-result/1`` record (see steps/stage1_spec.build_result_record), plus
mf_pkl_b64 — base64-encoded mf_data.pkl bytes.
"""

import sys
import base64
import pathlib
import subprocess
import importlib
import importlib.util
import importlib.metadata as _meta

# ── Must be first: purge any stale/bundled ffsim ─────────────────────────────
_bundle_dir = pathlib.Path(__file__).parent
_bad_ffsim_dir = _bundle_dir / "ffsim"
sys.path[:] = [
    p for p in sys.path
    if not (pathlib.Path(p) == _bad_ffsim_dir or "/ffsim" in p or "\\ffsim" in p)
]
for _k in [k for k in list(sys.modules) if k == "ffsim" or k.startswith("ffsim.")]:
    del sys.modules[_k]

try:
    import ffsim as _ffsim_test
    print(
        f"[stage1] ffsim OK: version={getattr(_ffsim_test, '__version__', 'unknown')}  "
        f"path={getattr(_ffsim_test, '__file__', 'unknown')}",
        flush=True,
    )
    del _ffsim_test
except Exception as _e:
    print(f"[stage1] WARNING: ffsim smoke-test failed: {_e}", flush=True)
for _k in [k for k in list(sys.modules) if k == "ffsim" or k.startswith("ffsim.")]:
    del sys.modules[_k]

sys.path.insert(0, str(_bundle_dir))
from pathlib import Path

# ── Vendor injection: qiskit-addon-sqd ───────────────────────────────────────
_vendor_sqd = _bundle_dir / "vendor" / "qiskit-addon-sqd"
if _vendor_sqd.exists():
    sys.path.insert(0, str(_vendor_sqd))
    for _k in [k for k in list(sys.modules)
               if k == "qiskit_addon_sqd" or k.startswith("qiskit_addon_sqd.")]:
        del sys.modules[_k]
    try:
        import qiskit_addon_sqd as _sqd_pkg
        _sqd_ver = getattr(_sqd_pkg, "__version__", None) or _meta.version("qiskit-addon-sqd")
    except Exception:
        _sqd_ver = "unknown"
    print(f"[stage1] vendored qiskit-addon-sqd={_sqd_ver}", flush=True)
else:
    print("[stage1] WARNING: vendor/qiskit-addon-sqd not found", flush=True)

# ── quantum-fragment-methods install ─────────────────────────────────────────
if importlib.util.find_spec("quantum_fragment_methods") is None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--no-deps", "--no-build-isolation",
        str(_bundle_dir)
    ])
    print("[stage1] quantum-fragment-methods installed")

from steps import run_meanfield
from qiskit_serverless import get_arguments, save_result

args = get_arguments()

from steps import Stage1Spec

SPEC = Stage1Spec.from_dict(args["spec"])  # validates; raises on any unsupported request
RUN_NAME = SPEC.tag
DATA_DIR = f"/tmp/qfm/{RUN_NAME}/data"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

print(f"[stage1] {SPEC.molecule.describe()}  {SPEC.config.method}/{SPEC.config.basis}  "
      f"digest={SPEC.digest()[:16]}", flush=True)

record = run_meanfield(SPEC, DATA_DIR)

print(
    f"[stage1] DONE  hf_energy={record['hf_energy']:.10f} Ha  "
    f"converged={record['hf_converged']}  n_ao={record['n_ao']}",
    flush=True,
)

# ── Base64-encode mf_data.pkl for hand-off to Stage 2 ────────────────────────
_pkl_path = Path(DATA_DIR) / "mf_data.pkl"
with open(_pkl_path, "rb") as _fh:
    _mf_bytes = _fh.read()
_mf_b64 = base64.b64encode(_mf_bytes).decode("ascii")
print(f"[stage1] mf_data.pkl  size={len(_mf_bytes):,} bytes  b64={len(_mf_b64):,} chars", flush=True)

save_result({**record, "mf_pkl_b64": _mf_b64})
