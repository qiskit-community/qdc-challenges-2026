"""
Stage 2 Entrypoint — EWF Fragmentation (Proteasome Ligand)

Receives mf_data.pkl as base64 from the notebook (Stage 1 result),
decodes it to /tmp, runs Vayesta EWF fragmentation, then returns
embedding_data.pkl and ewf_dumpfile.h5 as base64 for Stage 3.

Arguments
---------
mf_pkl_b64    : base64-encoded mf_data.pkl bytes  (from Stage 1 result)
run_name      : must match Stage 1                 (from Stage 1 result)
bath_type     : Vayesta bath type                  (default: "mp2")
truncation    : MP2 bath truncation threshold      (default: 1e-4)
fragmentation : IAO fragmentation type             (default: "iao")
verbose       : Vayesta verbosity                  (default: 3)

Returns (via save_result)
-------------------------
stage, run_name, n_fragments,
embedding_pkl_b64  — base64-encoded embedding_data.pkl for Stage 3
dumpfile_b64       — base64-encoded ewf_dumpfile.h5 for Stage 3
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
        f"[stage2] ffsim OK: version={getattr(_ffsim_test, '__version__', 'unknown')}  "
        f"path={getattr(_ffsim_test, '__file__', 'unknown')}",
        flush=True,
    )
    del _ffsim_test
except Exception as _e:
    print(f"[stage2] WARNING: ffsim smoke-test failed: {_e}", flush=True)
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
    print(f"[stage2] vendored qiskit-addon-sqd={_sqd_ver}", flush=True)
else:
    print("[stage2] WARNING: vendor/qiskit-addon-sqd not found", flush=True)

# ── Vendor injection: Vayesta (EWF fragmentation, Apache-2.0) ────────────────
# Package root is vendor/vayesta/ (it contains the ``vayesta`` package).  Only the path is
# prepared here: importing ``vayesta`` parses sys.argv, which is patched further below.
_vendor_vayesta = _bundle_dir / "vendor" / "vayesta"
if not (_vendor_vayesta / "vayesta" / "__init__.py").is_file():
    raise FileNotFoundError(
        f"[stage2] vendored Vayesta missing: {_vendor_vayesta}/vayesta/__init__.py "
        "(redeploy the full serverless_bundle/)"
    )
sys.path.insert(0, str(_vendor_vayesta))
for _k in [k for k in list(sys.modules) if k == "vayesta" or k.startswith("vayesta.")]:
    del sys.modules[_k]
_vay_spec = importlib.util.find_spec("vayesta")
if _vay_spec is None or not str(_vay_spec.origin).startswith(str(_vendor_vayesta)):
    raise ImportError(f"[stage2] vayesta does not resolve to the vendored copy: {_vay_spec}")
print(f"[stage2] vendored vayesta from {_vendor_vayesta}", flush=True)

# ── quantum-fragment-methods install ─────────────────────────────────────────
if importlib.util.find_spec("quantum_fragment_methods") is None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--no-deps", "--no-build-isolation",
        str(_bundle_dir)
    ])
    print("[stage2] quantum-fragment-methods installed")

# ── Patch sys.argv so Vayesta writes logs to /tmp, not CWD (which is / in Fleets) ──
import sys as _sys
if "--output-dir" not in _sys.argv:
    _sys.argv += ["--output-dir", "/tmp"]

from steps import run_fragments
from qiskit_serverless import get_arguments, save_result

args = get_arguments()

RUN_NAME      = args.get("run_name",      "ixazomib_ligand_sto-3g")
EWF_CONFIG    = args.get("ewf_config")
BATH_TYPE     = args.get("bath_type",     "mp2")
TRUNCATION    = args.get("truncation",    1e-4)
FRAGMENTATION = args.get("fragmentation", "atomic")
VERBOSE       = args.get("verbose",       3)

# ── Decode mf_data.pkl from Stage 1 ──────────────────────────────────────────
MF_PKL_B64 = args.get("mf_pkl_b64")
if not MF_PKL_B64:
    raise ValueError("[stage2] mf_pkl_b64 argument is required — pass Stage 1 result['mf_pkl_b64']")

DATA_DIR = f"/tmp/qfm/{RUN_NAME}/data"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

_pkl_path  = Path(DATA_DIR) / "mf_data.pkl"
_pkl_bytes = base64.b64decode(MF_PKL_B64)
with open(_pkl_path, "wb") as _fh:
    _fh.write(_pkl_bytes)
print(f"[stage2] decoded mf_data.pkl  {len(_pkl_bytes):,} bytes → {_pkl_path}", flush=True)

print(
    f"[stage2] run_name={RUN_NAME}  ewf_config={EWF_CONFIG is not None}",
    flush=True,
)

frags = run_fragments(
    DATA_DIR, RUN_NAME,
    ewf_config=EWF_CONFIG,
    bath_type=BATH_TYPE,
    truncation=TRUNCATION,
    fragmentation=FRAGMENTATION,
    verbose=VERBOSE,
)

print(f"[stage2] DONE  n_fragments={frags['n_fragments']}", flush=True)

# ── Gzip-compress then base64-encode outputs for hand-off to Stage 3 ─────────
import gzip as _gzip

with open(frags["dumpfile"], "rb") as _fh:
    _h5_bytes = _fh.read()
_h5_gz     = _gzip.compress(_h5_bytes, compresslevel=1)
_h5_b64    = base64.b64encode(_h5_gz).decode("ascii")
print(f"[stage2] ewf_dumpfile.h5     raw={len(_h5_bytes):,} B  gz={len(_h5_gz):,} B  b64={len(_h5_b64):,} chars", flush=True)

result_payload = {
    "stage":        "fragmentation",
    "run_name":     RUN_NAME,
    "n_fragments":  frags["n_fragments"],
    "dumpfile_b64": _h5_b64,
    "sidecar":      frags.get("sidecar"),
    "diagnostics":  frags.get("diagnostics"),
}

save_result(result_payload)
