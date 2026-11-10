"""
Stage 3 Entrypoint — Post-Hartree-Fock Solve + Energy Reconstruction (Proteasome Ligand)

Receives mf_data.pkl (Stage 1), embedding_data.pkl, and ewf_dumpfile.h5
(Stage 2) as base64 arguments, decodes them to /tmp, runs per-fragment
FCI/CCSD/SQD solvers, and reconstructs the total EWF correlation energy.

Arguments
---------
mf_pkl_b64         : base64-encoded mf_data.pkl       (from Stage 1 result)
embedding_pkl_b64  : base64-encoded embedding_data.pkl (from Stage 2 result)
dumpfile_b64       : base64-encoded ewf_dumpfile.h5    (from Stage 2 result)
run_name           : must match Stages 1 & 2           (from Stage 1 result)
results_dir        : where to write summary.json        (default: /tmp/qfm/<run_name>/results)
fci_threshold      : n_orb <= this → FCI               (default: 10)
sqd_threshold      : n_orb >= this → quantum solver    (default: 15)
qpu_backend        : IBM backend name (None → CCSD)    (default: None)
n_shots            : shots per SQD QPU circuit          (default: 100_000)
n_reps             : LUCJ circuit repetitions           (default: 1)
sqd_config         : dict of extra SQD classical opts   (default: None)
solver_mode        : "sqd" | "trimsqd" | "both"        (default: "sqd")
trim_config        : TrimSQD-specific config dict       (default: None)
resume_job_ids     : {str(frag_id): qpu_job_id}        (default: None)
"""

import sys
import pathlib
import subprocess
import importlib
import importlib.util
import importlib.metadata as _meta

# ── Must be first: purge any stale/bundled ffsim ─────────────────────────────
_bundle_dir = pathlib.Path(__file__).parent
_bad_ffsim_dir = _bundle_dir / "ffsim"
if _bad_ffsim_dir.exists():
    sys.path[:] = [
        p for p in sys.path
        if not (pathlib.Path(p) == _bad_ffsim_dir or "/ffsim" in p or "\\ffsim" in p)
    ]
    for _k in [k for k in list(sys.modules) if k == "ffsim" or k.startswith("ffsim.")]:
        del sys.modules[_k]

try:
    import ffsim as _ffsim_test
    print(
        f"[stage3] ffsim OK: version={getattr(_ffsim_test, '__version__', 'unknown')}  "
        f"path={getattr(_ffsim_test, '__file__', 'unknown')}",
        flush=True,
    )
    del _ffsim_test
except Exception as _e:
    print(f"[stage3] WARNING: ffsim smoke-test failed: {_e}", flush=True)

sys.path.insert(0, str(_bundle_dir))
from pathlib import Path

# ── Vendor injection: qiskit-addon-sqd (trim-sqd branch, v0.13+) ─────────────
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
    print(f"[stage3] vendored qiskit-addon-sqd={_sqd_ver} from {_vendor_sqd}", flush=True)
else:
    print(
        "[stage3] WARNING: vendor/qiskit-addon-sqd not found — TrimSQD will be unavailable",
        flush=True,
    )

# ── quantum-fragment-methods install ─────────────────────────────────────────
if importlib.util.find_spec("quantum_fragment_methods") is None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--no-deps", "--no-build-isolation",
        str(_bundle_dir)
    ])
    print("[stage3] quantum-fragment-methods installed")

# ── Patch sys.argv so Vayesta writes logs to /tmp, not CWD (which is / in Fleets) ──
import sys as _sys
if "--output-dir" not in _sys.argv:
    _sys.argv += ["--output-dir", "/tmp"]

from steps import run_solve, run_reconstruct
from qiskit_serverless import get_arguments, save_result

args = get_arguments()

RUN_NAME    = args.get("run_name",    "ixazomib_ligand_sto-3g")
DATA_DIR    = f"/tmp/qfm/{RUN_NAME}/data"
RESULTS_DIR = args.get("results_dir", f"/tmp/qfm/{RUN_NAME}/results")
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# ── Download artefacts from the Serverless file API ───────────────────────────
# The notebook uploaded them via serverless.file_upload() before submitting
# this job. Each file is stored as "{run_name}__{filename}.tar.gz".
import os as _os
from qiskit_serverless import ServerlessClient
from qiskit_serverless.core.constants import (
    ENV_JOB_GATEWAY_TOKEN, ENV_JOB_GATEWAY_HOST, ENV_JOB_GATEWAY_INSTANCE
)
from qiskit_serverless.core.function import QiskitFunction as _QiskitFunction

_gw_token    = _os.environ.get(ENV_JOB_GATEWAY_TOKEN)
_gw_host     = _os.environ.get(ENV_JOB_GATEWAY_HOST)
_gw_instance = _os.environ.get(ENV_JOB_GATEWAY_INSTANCE)
_fn_handle   = _QiskitFunction(title="qfm-proteasome-stage3-solve")

_client = ServerlessClient(host=_gw_host, token=_gw_token, instance=_gw_instance)

import tarfile as _tarfile

for _filename in ["mf_data.pkl", "embedding_data.pkl", "ewf_dumpfile.h5"]:
    _remote_name = f"{RUN_NAME}__{_filename}"
    _tgz_name    = _remote_name + ".tar.gz"
    # Download the .tar.gz wrapper
    _client.file_download(
        file=_tgz_name,
        function=_fn_handle,
        target_name=_tgz_name,
        download_location=DATA_DIR,
    )
    # Extract the inner file
    _tgz_path = Path(DATA_DIR) / _tgz_name
    with _tarfile.open(_tgz_path, "r:gz") as _tar:
        _tar.extract(_remote_name, path=DATA_DIR)
    # Rename from {run_name}__{filename} → {filename}
    (Path(DATA_DIR) / _remote_name).rename(Path(DATA_DIR) / _filename)
    _tgz_path.unlink()
    _size = Path(DATA_DIR, _filename).stat().st_size
    print(f"[stage3] downloaded+extracted {_filename}  {_size:,} bytes → {DATA_DIR}/{_filename}", flush=True)

FCI_THRESHOLD  = args.get("fci_threshold",   10)
SQD_THRESHOLD  = args.get("sqd_threshold",   15)
QPU_BACKEND    = args.get("qpu_backend",     None)
N_SHOTS        = args.get("n_shots",         100_000)
N_REPS         = args.get("n_reps",          1)
SOLVER_MODE    = args.get("solver_mode",     "sqd")
TRIM_CONFIG    = args.get("trim_config",     None)
SQD_CONFIG     = args.get("sqd_config",      None)
RESUME_JOB_IDS = args.get("resume_job_ids",  None)
BASIS          = args.get("basis",           None)   # passed to run_solve for job tag construction
# stage3_config: optional dict or Stage3Config.  When given, its per-solver
# settings override the individual sqd_config / trim_config / n_reps args.
STAGE3_CONFIG  = args.get("stage3_config",   None)

# ── Challenge-notebook extensions (all optional; defaults = original behaviour) ──
#   fragment_ids          : solve ONLY these fragment IDs (single-fragment lab)
#   quantum_fragment_ids  : ONLY these IDs may use SQD/TrimSQD; other quantum-sized
#                           fragments fall back to CCSD (full EWF with one QPU fragment)
#   reconstruct           : False -> skip EWF reconstruction, return per-fragment data
#   counts_file           : file-API name of a frozen counts.npy to replay (no QPU sampling)
#   counts_fragment_id    : fragment the replayed counts belong to
BUNDLE_API           = 3   # bumped when challenge-notebook arguments/result fields change
FRAGMENT_IDS         = args.get("fragment_ids",         None)
QUANTUM_FRAGMENT_IDS = args.get("quantum_fragment_ids", None)
#   solver_assignments    : explicit {fragment_id: solver} routing map (JSON keys are strings)
#   use_threshold_fallback: False -> solver_assignments is mandatory and must be complete
SOLVER_ASSIGNMENTS     = args.get("solver_assignments",     None)
USE_THRESHOLD_FALLBACK = bool(args.get("use_threshold_fallback", True))
RECONSTRUCT          = bool(args.get("reconstruct",     True))
COUNTS_FILE          = args.get("counts_file",          None)
COUNTS_FRAGMENT_ID   = args.get("counts_fragment_id",   None)
#   measurement_plan      : {str(fragment_id): {mode: live|replay, solver, n_qubits, backend,
#                            hamiltonian_sha256, counts_file, counts_sha256}} -- fragment-specific
#                           measurement routing (Step 3 validates it; the legacy counts_file /
#                           counts_fragment_id pair is a different, single-fragment mechanism)
MEASUREMENT_PLAN     = args.get("measurement_plan",     None)
MEASUREMENT_PLAN_API = 1
#   shared_protocol       : single-source TrimSQD protocol (Task 4B.2).  Step 3 validates it, applies the
#                           SAME frozen settings to every TrimSQD fragment, and proves each solver
#                           recorded them; the result echoes `quantum_protocol`.
SHARED_PROTOCOL      = args.get("shared_protocol",      None)
SHARED_PROTOCOL_API  = 1

if SHARED_PROTOCOL is not None and not isinstance(SHARED_PROTOCOL, dict):
    raise ValueError("[stage3] shared_protocol must be an object")

if MEASUREMENT_PLAN is not None:
    if COUNTS_FILE:
        raise ValueError("[stage3] give either measurement_plan or counts_file, not both")
    if not isinstance(MEASUREMENT_PLAN, dict) or not MEASUREMENT_PLAN:
        raise ValueError("[stage3] measurement_plan must be a non-empty {fragment_id: entry} object")
    _staged_names = {}
    for _key, _entry in MEASUREMENT_PLAN.items():
        _pf = int(_key)
        if not isinstance(_entry, dict) or _entry.get("mode") not in ("live", "replay"):
            raise ValueError(f"[stage3] measurement_plan[{_pf}] needs an explicit mode 'live' or 'replay'")
        if _entry["mode"] != "replay":
            continue
        _remote = _entry.get("counts_file")
        if not _remote:
            raise ValueError(f"[stage3] measurement_plan[{_pf}] is a replay without counts_file")
        if _remote in _staged_names:
            raise ValueError(
                f"[stage3] counts file {_remote!r} is assigned to fragments {_staged_names[_remote]} "
                f"and {_pf}; one measurement file is never shared between fragments")
        _staged_names[_remote] = _pf
        _pdir = Path(DATA_DIR) / f"fragment_{_pf}"
        _pdir.mkdir(parents=True, exist_ok=True)
        _ptgz = _remote + ".tar.gz"
        _client.file_download(file=_ptgz, function=_fn_handle, target_name=_ptgz, download_location=DATA_DIR)
        with _tarfile.open(Path(DATA_DIR) / _ptgz, "r:gz") as _tar:
            _tar.extract(_remote, path=DATA_DIR)
        (Path(DATA_DIR) / _remote).rename(_pdir / "counts.npy")
        (Path(DATA_DIR) / _ptgz).unlink()
        print(f"[stage3] planned counts for fragment {_pf} staged -> {_pdir / 'counts.npy'}", flush=True)

if COUNTS_FILE:
    if COUNTS_FRAGMENT_ID is None:
        raise ValueError("[stage3] counts_file requires counts_fragment_id")
    _frag_dir = Path(DATA_DIR) / f"fragment_{int(COUNTS_FRAGMENT_ID)}"
    _frag_dir.mkdir(parents=True, exist_ok=True)
    _counts_tgz = COUNTS_FILE + ".tar.gz"
    _client.file_download(
        file=_counts_tgz,
        function=_fn_handle,
        target_name=_counts_tgz,
        download_location=DATA_DIR,
    )
    with _tarfile.open(Path(DATA_DIR) / _counts_tgz, "r:gz") as _tar:
        _tar.extract(COUNTS_FILE, path=DATA_DIR)
    (Path(DATA_DIR) / COUNTS_FILE).rename(_frag_dir / "counts.npy")
    (Path(DATA_DIR) / _counts_tgz).unlink()
    print(f"[stage3] frozen counts staged -> {_frag_dir / 'counts.npy'}", flush=True)

print(
    f"[stage3] run_name={RUN_NAME}  solver_mode={SOLVER_MODE}  "
    f"qpu_backend={QPU_BACKEND}  n_shots={N_SHOTS}  n_reps={N_REPS}  "
    f"fragment_ids={FRAGMENT_IDS}  quantum_fragment_ids={QUANTUM_FRAGMENT_IDS}  "
    f"reconstruct={RECONSTRUCT}",
    flush=True,
)

# ── Post-HF solve (FCI / CCSD / SQD per fragment) ────────────────────────────
# stage3_config (when given) overrides the individual sqd_config / trim_config /
# n_reps kwargs and also propagates FCI/CCSD numerical settings to the solvers.
solv = run_solve(
    DATA_DIR, RUN_NAME,
    fci_threshold=FCI_THRESHOLD,
    sqd_threshold=SQD_THRESHOLD,
    qpu_backend=QPU_BACKEND,
    n_shots=N_SHOTS,
    n_reps=N_REPS,
    sqd_config=SQD_CONFIG,
    solver_mode=SOLVER_MODE,
    trim_config=TRIM_CONFIG,
    resume_job_ids=RESUME_JOB_IDS,
    compute_rdms=True,
    basis=BASIS,
    fragment_ids=FRAGMENT_IDS,
    quantum_fragment_ids=QUANTUM_FRAGMENT_IDS,
    solver_assignments=SOLVER_ASSIGNMENTS,
    use_threshold_fallback=USE_THRESHOLD_FALLBACK,
    # Replay job: only this fragment's counts are staged.  Step 3 then refuses any other
    # quantum fragment instead of sampling the QPU for it, and validates the staged counts.
    replay_fragment_ids=[int(COUNTS_FRAGMENT_ID)] if COUNTS_FILE else None,
    **({"measurement_plan": MEASUREMENT_PLAN} if MEASUREMENT_PLAN is not None else {}),
    **({"shared_protocol": SHARED_PROTOCOL} if SHARED_PROTOCOL is not None else {}),
    **({"stage3_config": STAGE3_CONFIG} if STAGE3_CONFIG is not None else {}),
)

# ── Single-fragment lab: no reconstruction, return the per-fragment record ────
if not RECONSTRUCT:
    import base64 as _b64, gzip as _gz, hashlib as _hl
    import h5py as _h5
    _fid = int(FRAGMENT_IDS[0]) if FRAGMENT_IDS else None
    _summ = solv["fragment_summaries"].get(_fid) if _fid is not None else None
    if _summ is None:
        raise RuntimeError("[stage3] reconstruct=False requires exactly one valid fragment_ids entry")
    with _h5.File(str(Path(DATA_DIR) / "ewf_dumpfile.h5"), "r") as _hf:
        _g = _hf[f"fragment_{_fid}"]
        _sha = _hl.sha256(_g["heff"][:].tobytes() + _g["eris"][:].tobytes()).hexdigest()
        _nocc = int(_g.attrs["nocc"])
    _label = _summ["solver"].lower()
    if _label == "sqd" and _summ.get("trimsqd_energy_ha") is not None:
        _label = "both"
    _counts_b64 = None
    if _summ.get("counts_file"):
        _counts_b64 = _b64.b64encode(_gz.compress(Path(_summ["counts_file"]).read_bytes())).decode("ascii")
    _single = {
        "stage":                  "single_fragment",
        "bundle_api":             BUNDLE_API,
        "execution_mode":         "single_fragment",
        "run_name":               RUN_NAME,
        "fragment_id":            _fid,
        "method_used":            _label,
        "solver_mode":            SOLVER_MODE,
        "qpu_backend":            QPU_BACKEND,
        "n_orb":                  _summ["n_orb"],
        "n_elec":                 2 * _nocc,
        "energy_ha":              _summ["energy_ha"],
        "e_corr_ha":              _summ["e_corr_ha"],
        "qpu_job_id":             _summ["qpu_job_id"],
        "hamiltonian_sha256":     _sha,
        "software":               solv.get("software"),
        "counts_b64":             _counts_b64,
    }
    if MEASUREMENT_PLAN is not None:
        _single["measurement_plan_api"] = MEASUREMENT_PLAN_API
        _single["measurement"] = _summ.get("measurement")
    if SHARED_PROTOCOL is not None:
        _single["shared_protocol_api"] = SHARED_PROTOCOL_API
        _single["quantum_protocol"] = solv.get("quantum_protocol")
        _single["protocol"] = _summ.get("protocol")
        _single["sqd_config_hash"] = _summ.get("sqd_config_hash")
    if _label == "ccsd":
        _single["ccsd_energy_ha"] = _summ["energy_ha"]
    if _label in ("sqd", "both"):
        _single["sqd_energy_ha"] = _summ["energy_ha"]
    if _summ.get("trimsqd_energy_ha") is not None:
        _single["trimsqd_energy_ha"] = _summ["trimsqd_energy_ha"]
    print(f"[stage3] single-fragment DONE  fragment={_fid}  method={_label}  E={_summ['energy_ha']:.8f} Ha", flush=True)
    save_result(_single)

# ── Reconstruct total EWF energy (primary solver) ────────────────────────────
else:
    recon = run_reconstruct(DATA_DIR, RUN_NAME, RESULTS_DIR)

    # ── Reconstruct TrimSQD total energy if mode=="both" ─────────────────────────
    e_total_trimsqd = None
    if solv.get("solver_results_trim_pkl"):
        from steps.step4_reconstruct import QuantumReconstructionError as _QRE
        try:
            from steps.step4_reconstruct import run_reconstruct as _recon
            recon_trim = _recon(
                DATA_DIR, RUN_NAME + "_trim", RESULTS_DIR,
                solver_results_pkl=solv["solver_results_trim_pkl"],
            )
            e_total_trimsqd = recon_trim.get("e_total")
            print(f"[stage3] E(EWF+TrimSQD) = {e_total_trimsqd:.8f} Ha", flush=True)
        except _QRE:
            raise      # a quantum RDM that cannot be integrated is never "skipped"
        except Exception as _e:
            print(f"[stage3] TrimSQD reconstruction skipped: {_e}", flush=True)

    print(
        f"[stage3] DONE  e_total={recon['e_total']:.8f} Ha  "
        f"n_fci={solv['n_fci']}  n_ccsd={solv['n_ccsd']}  "
        f"n_sqd={solv['n_sqd']}  n_trimsqd={solv.get('n_trimsqd', 0)}",
        flush=True,
    )

    result_dict = {
        "stage":        "solve_reconstruct",
        "bundle_api":   BUNDLE_API,
        "run_name":     RUN_NAME,
        "solver_mode":  SOLVER_MODE,
        "e_hf":         recon["e_hf"],
        "e_corr":       recon["e_corr"],
        "e_total":      recon["e_total"],
        "e_corr_sqd":   solv.get("e_corr_total"),
        "n_fragments":  solv["n_fragments"],
        "n_fci":        solv["n_fci"],
        "n_ccsd":       solv["n_ccsd"],
        "n_sqd":        solv["n_sqd"],
        "n_trimsqd":    solv.get("n_trimsqd", 0),
        "hf_converged": True,   # HF convergence checked in Stage 1
        "qpu_backend":  QPU_BACKEND,
        "fragments":    recon.get("fragments", {}),
    }

    # ── Quantum-fragment accounting (validated by the challenge notebook) ────────
    # Count UNIQUE fragments that actually ran a quantum solver; a CCSD fallback does
    # not count.  quantum_rdm_integrated is True only if the cumulant reconstruction
    # consumed that fragment's own RDMs (not the e_corr placeholder).
    _summ_all = solv.get("fragment_summaries", {})
    _quantum_ids = sorted(f for f, s in _summ_all.items() if s["solver"] in ("SQD", "TrimSQD"))
    _recon_frags = recon.get("fragments", {})
    _rdm_integrated = all(
        _recon_frags.get(f, _recon_frags.get(str(f), {})).get("rdm_integrated", False)
        for f in _quantum_ids
    )
    result_dict.update({
        "execution_mode":          "full_ewf",
        "n_quantum_fragments":     len(_quantum_ids),
        "quantum_fragment_ids":    _quantum_ids,
        "quantum_fragment_id":     _quantum_ids[0] if len(_quantum_ids) == 1 else None,
        "quantum_rdm_integrated":  bool(_quantum_ids) and _rdm_integrated,
        "fragment_summaries":      _summ_all,
        "software":                solv.get("software"),
        "solver_config":           solv.get("solver_config"),
        "solver_assignments_used": (
            {str(k): v for k, v in SOLVER_ASSIGNMENTS.items()} if SOLVER_ASSIGNMENTS else None
        ),
        "use_threshold_fallback":  USE_THRESHOLD_FALLBACK,
    })
    if SHARED_PROTOCOL is not None:
        result_dict["shared_protocol_api"] = SHARED_PROTOCOL_API
        result_dict["quantum_protocol"] = solv.get("quantum_protocol")
    if MEASUREMENT_PLAN is not None:
        # New counts of the fragments this job sampled live (the client stores each as its own
        # write-once artifact); replayed counts are not echoed back.
        import base64 as _b64m, gzip as _gzm
        _new_counts = {}
        for _f, _s in _summ_all.items():
            _m = _s.get("measurement")
            if _m and _m["mode"] == "live" and _s.get("counts_file"):
                _new_counts[str(_f)] = _b64m.b64encode(
                    _gzm.compress(Path(_s["counts_file"]).read_bytes())).decode("ascii")
        result_dict["measurement_plan_api"] = MEASUREMENT_PLAN_API
        result_dict["counts_b64_by_fragment"] = _new_counts

    if solv.get("e_corr_total_trimsqd") is not None:
        result_dict["e_corr_trimsqd"] = solv["e_corr_total_trimsqd"]
    if e_total_trimsqd is not None:
        result_dict["e_total_trimsqd"] = e_total_trimsqd

    save_result(result_dict)
