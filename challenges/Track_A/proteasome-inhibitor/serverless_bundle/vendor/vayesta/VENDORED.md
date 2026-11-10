# Vendored Vayesta

| | |
|---|---|
| Upstream | https://github.com/BoothGroup/Vayesta |
| Upstream commit | `e9e48c7090b1ac3b92e9384de45cab88dcb140a4` (2026-09-04, merge of PR #203 `cp_mol`) |
| Reported version | `vayesta.__version__ == "1.0.1"`. The `v1.0.1` tag itself (`0c36c53c`) is **not** this code: the commit above is 950 commits newer. |
| Licence | Apache-2.0, see `LICENSE` (byte copy of upstream `LICENSE` at the commit above) |
| Copyright / authors | see `pyproject.upstream.toml` (upstream `[project]` metadata). Three files carry PySCF Apache-2.0 headers (`vayesta/core/vpyscf/*`), kept unchanged. |
| Package root | this directory contains the `vayesta/` package, so put **this directory** on `sys.path` |
| Modifications | none. Every shipped file is byte-identical to the upstream tree at the commit above (see `FILES.sha256`). |
| Omitted from upstream | `vayesta/tests/**`, `vayesta/misc/gmtkn55/**` (benchmark data), `vayesta/misc/molecules/data/*.dat`, `vayesta/ewf/TODO.txt`; repo-level files (docs, examples, setup.py, MANIFEST.in, uv.lock) |
| Native library | the macOS arm64 `vayesta/libs/libcore.dylib` found in the historical copy is deliberately **not** shipped. C sources and CMake files are included. |

## Native component (`libcore`)

`vayesta.libs.libcore` is used only by `vayesta.core.ao2mo.kao2gmo.kao2gmo_cderi`. All callers in the
quantum-embedding code are guarded by `emb.kdf is not None` (periodic k-point density fitting). The
molecular RHF -> IAO/EWF path used by Stage 2 never reaches it. When the library is absent
`vayesta.libs.libcore` is `None` and `kao2gmo_cderi` falls back to a Python driver.

To build a Linux `libcore.so` from these sources (needs CMake, a C compiler, OpenMP and a BLAS such as libopenblas):

    cmake -S vayesta/libs -B /tmp/vayesta_build -DCMAKE_BUILD_TYPE=Release
    cmake --build /tmp/vayesta_build -j        # writes vayesta/libs/libcore.so

Upstream `CMakeLists.txt` adds `-march=native -Ofast`; build for the target CPU baseline only.

## Runtime dependencies (upstream `pyproject.toml`)

Required: numpy>=1.19, scipy>=1.10, h5py>=3, pyscf>=2.8. cvxpy is declared upstream but is only needed by
`vayesta.dmet`; `vayesta.ewf` imports without it. Optional and not used here: mpi4py, dyson, ebcc, pygnme.

## Integrity

`FILES.sha256` lists SHA-256 of every vendored file except itself and this file
(`shasum -a 256 -c FILES.sha256` from this directory).
