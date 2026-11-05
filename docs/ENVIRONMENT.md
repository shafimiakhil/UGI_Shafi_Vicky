# ENVIRONMENT.md - Python environment for the Córdoba pipeline

Created 2026-10-25 (Phase 1, step 3). Local machine: Windows 11 Pro 10.0.26200, 1.9 TB system disk with 62 GB free at creation time.

## Environment

| Item | Value |
|---|---|
| Kind | conda environment `cordoba` |
| Location | `C:\Users\SHAFI\anaconda3\envs\cordoba` |
| conda | 25.5.1 (Anaconda3 base, platform win-64) |
| Channel used | conda-forge only (`--override-channels`) |
| Python | 3.11.17 (conda-forge build `hb12b558_2_cpython`) |
| pip | 26.2.2 (conda-forge) |
| Size on disk | 749 MB after the first install (without torch); 3.8 GB after the torch install |
| Build log | scratchpad `build_env.log`, 2026-10-25 00:32:20 to 00:33:45 (+02:00), all steps exit 0; torch install `install_torch.log`, 2026-10-25 08:30:16 to 08:33:39 (+01:00, clocks changed from CEST to CET overnight), pip exit 0 |

Creation commands (run from Git Bash on Windows):

```bash
conda create -y -n cordoba -c conda-forge --override-channels python=3.11 pip
/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe -m pip install --upgrade pip
/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe -m pip install numpy pandas scipy scikit-learn pyyaml tqdm matplotlib seaborn rasterio geopandas shapely affine pyproj pyogrio joblib
/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe -m pip install --no-cache-dir "torch==2.14.1" "torchvision==0.29.1" --index-url https://download.pytorch.org/whl/cu130
```

Activate with `conda activate cordoba`, or call the interpreter directly: `C:\Users\SHAFI\anaconda3\envs\cordoba\python.exe`.

## Installed packages (pip freeze, 2026-10-25, after the torch install)

| Package | Version | Requested by CLAUDE.md Section 6 step 3 |
|---|---|---|
| numpy | 2.4.6 | yes |
| pandas | 3.1.0 | yes |
| scipy | 1.17.1 | yes |
| scikit-learn | 1.9.1 | yes |
| PyYAML | 6.0.3 | yes |
| tqdm | 4.70.1 | yes |
| matplotlib | 3.11.2 | yes |
| seaborn | 0.13.2 | yes |
| rasterio | 1.4.4 | yes |
| geopandas | 1.2.0 | yes |
| shapely | 2.2.0 | yes |
| affine | 3.0.1 | yes |
| torch | 2.14.1+cu130 | yes (installed 2026-10-25 after approval of the download) |
| torchvision | 0.29.1+cu130 | yes |
| pyproj | 3.7.2 | dependency (geopandas) |
| pyogrio | 0.13.0 | dependency (geopandas I/O) |
| joblib | 1.6.0 | dependency (scikit-learn); needed to load `prisma_pca_fold*.joblib` |
| pillow | 13.0.0 | dependency (matplotlib) |
| attrs 26.1.0, certifi 2026.10.23, click 8.5.0, click-plugins 1.1.1.2, cligj 0.7.2, cloudpickle 3.1.2, colorama 0.4.6, contourpy 1.3.3, cycler 0.12.1, fonttools 4.67.0, kiwisolver 1.5.1, narwhals 2.27.1, packaging 26.3, pyparsing 3.3.3, python-dateutil 3.0.0, six 1.17.0, threadpoolctl 3.7.0, tzdata 2026.5 | as listed | transitive dependencies (first install) |
| filelock 3.32.3, fsspec 2026.7.0, Jinja2 3.1.6, MarkupSafe 3.0.3, mpmath 1.3.0, networkx 3.6.1, sympy 1.14.0, typing_extensions 4.16.0 | as listed | transitive dependencies (torch install) |

conda-level packages: python 3.11.17, pip 26.2.2, setuptools 84.2.1, wheel 0.48.0, openssl 3.6.5, libsqlite 3.54.1, tk 8.6.13, bzip2 1.0.8, liblzma 5.8.3, libzlib 1.3.2, libexpat 2.8.5, libffi 3.7.0, ucrt 10.0.26100.0, vc 14.5, vc14_runtime 14.51.36247, vcomp14 14.51.36247.

## Geospatial library stack

| Library | Bundled GDAL | Bundled PROJ |
|---|---|---|
| rasterio 1.4.4 (pip wheel) | 3.10.3 | 9.6.0 |
| pyogrio 0.13.0 (pip wheel) | 3.12.4 | (own) |
| pyproj 3.7.2 (pip wheel) | - | 9.5.1 |

Two private GDAL copies (rasterio vs pyogrio) is the normal situation for pip wheels on Windows. There is no system GDAL: `gdalinfo` is not on PATH and `osgeo` is not importable. Reading rasters and the shapefile directly inside the zip archives works with `zip://C:/path/archive.zip!member.tif` paths (forward slashes required; backslashes break the vsizip path).

## GPU and CUDA

`nvidia-smi` on 2026-10-25:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, WDDM driver model |
| Driver | 610.88 |
| CUDA driver API | 13.3 |

Torch-level check, run with `tools/check_torch.py` on 2026-10-25 (values in `docs/environment/torch_check.csv`):

| Key | Value |
|---|---|
| torch | 2.14.1+cu130 |
| torchvision | 0.29.1+cu130 |
| torch.version.cuda | 13.0 |
| cuDNN | 92400 |
| `torch.cuda.is_available()` | **True** |
| Device | NVIDIA GeForce RTX 4090, compute capability 8.9, 23.99 GiB |
| 2048 x 2048 fp32 matmul on GPU | finite result (first-call time in the CSV) |
| bf16 supported | True |
| `resnet18(weights=None)` | builds, 11,689,512 parameters (the ImageNet weights `resnet18-f37072fd.pth`, 46,830,571 B, were already in `C:/Users/SHAFI/.cache/torch/hub/checkpoints/` from 2026-06-09; `weights="DEFAULT"` loads them from there, no download) |

Re-run: `/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe -I tools/check_torch.py`.

The CUDA build was installed only after explicit approval (CLAUDE.md requires asking before any download above 500 MB). Wheel sizes read from the PyTorch index with HTTP HEAD on 2026-10-25 (Python 3.11, win_amd64), kept as the record of that decision:

| Wheel | Bytes | GB |
|---|---|---|
| torch 2.14.1+cu130 | 1,990,594,787 | 1.99 |
| torch 2.9.1+cu128 | 2,862,053,760 | 2.86 |
| torch 2.14.1+cpu | 123,978,347 | 0.12 |
| torch 2.9.1+cpu | 110,888,878 | 0.11 |

Latest releases on PyPI on 2026-10-25: torch 2.14.1, torchvision 0.29.1. A driver with CUDA 13.3 can run the cu130 and cu128 builds; the cu130 build was chosen and installed (downloaded size reported by pip: 1990.6 MB).

Other conda environments on this machine that already hold a CUDA torch (not used for this project, listed for reference): `gpu` (Python 3.10.19, torch 2.5.1+cu121, torchvision 0.20.1+cu121, `torch.cuda.is_available()` True) and `torchgpu` (Python 3.11.15, torch 2.6.0+cu124, CUDA available True, torchvision import fails with a DLL load error).

## Added 27 Oct 2026 (Shafi's instruction: install earthengine-api into the cordoba env)

- Command: `C:/Users/SHAFI/anaconda3/envs/cordoba/python.exe -m pip install earthengine-api`.
- Installed: earthengine-api 1.7.50, plus google-api-core 2.43.0, google-api-python-client 2.201.0, google-auth 2.62.0, google-cloud-storage 3.17.1, protobuf 7.36.2, cryptography 50.0.2, httplib2 0.32.0 and other dependencies, as listed in pip's output.
- The CLI is `C:/Users/SHAFI/anaconda3/envs/cordoba/Scripts/earthengine.exe`, which is not on PATH.
- google.api_core warns that Python 3.11 support ends 2027-10-24.

## Added 2 Nov 2026 (stage A9 of CLAUDE.md Section 11.5: NISAR search with `asf_search`)

- Command: `C:/Users/SHAFI/anaconda3/envs/cordoba/python.exe -m pip install asf_search==14.1.0`, after a `--dry-run` that listed the same packages.
- Installed at 08:26 (+01:00): asf_search 14.1.0, dateparser 1.4.3, pytz 2026.6, regex 2026.9.29, tenacity 9.2.1, tzlocal 5.4.5. Its other requirements (numpy, python-dateutil, requests 2.34.2, shapely) were already present.
- Used only for the read-only catalogue search in `work_ms/scripts/a9_nisar_search.py`; no NISAR product was downloaded.
- asf_search 14.1.0 sends an automatic error report to ASF when a search comes back incomplete (`asf_search.REPORT_ERRORS`, True by default). The script sets it to False since 2 Nov 10:14 UTC; reports sent before that are listed in `docs/NISAR_SEARCH_2026-11-02.md`.
- The script also reads CMR's json listing with `requests` (already in the environment) to cross-check the asf_search results.

## Added 2 Nov 2026 (P2 of Shafi's stage-B instruction: NISAR streaming)

- Command: `C:/Users/SHAFI/anaconda3/envs/cordoba/python.exe -m pip install earthaccess h5py`, after a `--dry-run` that listed the same packages. Installed 15:26:30-15:26:37 (+01:00).
- New: earthaccess 0.17.0, h5py 3.16.0 (bundled HDF5 2.0.0), aiobotocore 3.10.0, aiohappyeyeballs 2.7.1, aiohttp 3.14.5, aioitertools 0.13.0, aiosignal 1.4.0, botocore 1.43.117, bounded-pool-executor 0.0.3, frozenlist 1.8.0, importlib-resources 7.1.0, jmespath 1.1.0, multidict 7.2.1, multimethod 2.1, pqdm 0.2.0, propcache 0.5.4, python-cmr 0.13.0, s3fs 2026.9.0, tinynetrc 1.3.1, wrapt 2.5.2, yarl 1.25.1.
- Changed: **python-dateutil 3.0.0 → 2.9.0.post0** (downgrade required by the new packages); fsspec 2026.7.0 → 2026.9.0 (pinned by s3fs).
- Import check after the install: rasterio 1.4.4 (GDAL 3.10.3), h5py, earthaccess, asf_search, pandas and torch 2.14.1+cu130 (CUDA available) import together in one process.
- The Earthdata login is stored by Shafi himself (`earthaccess.login(persist=True)`, CLAUDE.md 12.6); no credential passes through Claude Code.

## Pending

- Lab GPU machine: [to be verified] (CLAUDE.md header).
- Luigi's software versions (scikit-learn 1.3.2 is known from the PCA pickles; torch and torchvision versions [to be verified], see docs/CODE_VS_PAPER.md Section 8).

## Added 3 Nov 2026

- `scikit-image` 0.26.0, with `imageio` 2.38.1, `lazy-loader` 0.6 and `tifffile` 2026.3.3 (pip, into the `cordoba` env).
  - Needed for the skeletons of the centreline-completeness score (CLAUDE.md 13.4 rule 7).
  - numpy 2.4.6 and scipy 1.17.1 are unchanged.
- `psutil` 7.2.2 (pip, into the `cordoba` env; 13.5 night CPU task 3): counts the bytes received while `scripts/s1_2021_2022.py` reads. `planetary-computer` and `pystac-client` are not needed: that script uses `requests` for the STAC search and the anonymous SAS token.
