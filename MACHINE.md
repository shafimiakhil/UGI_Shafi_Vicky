# Machine and measured throughput

Probe `work_ms/scripts/machine_probe.py`; values and per-run numbers `docs/machine_probe.json`, probed 2026-11-06. Paths from the project root.

## Hardware and software

| Item | Value | Source |
|---|---|---|
| CPU | 13th Gen Intel(R) Core(TM) i9-13900KF | `Get-CimInstance Win32_Processor`, Name |
| Physical cores | 24 | `Win32_Processor`, NumberOfCores |
| Logical threads | 32 | `Win32_Processor`, NumberOfLogicalProcessors |
| RAM installed | 128 GiB, 4 modules of 32 GiB, 5600 MT/s | `Get-CimInstance Win32_PhysicalMemory`, Capacity, Speed |
| RAM visible to Windows | 127.8 GiB | `Get-CimInstance Win32_ComputerSystem`, TotalPhysicalMemory |
| System disk | NVMe WD_BLACK SN850X 2000GB, NVMe, SSD, 2,000,398,934,016 bytes | `Win32_DiskDrive` associated with the system volume, Model; `MSFT_PhysicalDisk`, BusType, MediaType, Size |
| System volume | C:, NTFS, 1,999,291,543,552 bytes (1.82 TiB) | `Get-CimInstance Win32_LogicalDisk`, Size, FileSystem |
| GPU | NVIDIA GeForce RTX 4090 | `nvidia-smi --query-gpu=name` |
| GPU memory | 24,564 MiB | `nvidia-smi --query-gpu=memory.total` |
| NVIDIA driver | 610.88 | `nvidia-smi --query-gpu=driver_version` |
| CUDA runtime | 13.3, from the installed driver | `nvidia-smi -q -d COMPUTE`, CUDA UMD Version |
| Windows | Microsoft Windows 11 Pro, 64-bit | `Get-CimInstance Win32_OperatingSystem`, Caption, OSArchitecture |
| Windows build | 26300 (version 10.0.26300) | `Win32_OperatingSystem`, BuildNumber, Version |
| Last boot | 2026-11-02 13:15:28, before the first run below (2026-11-03 15:00:11) | `Win32_OperatingSystem`, LastBootUpTime; `run_log.md` |
| Python | 3.11.17 | `platform.python_version()` |
| torch | 2.14.1+cu130 | `torch.__version__` |
| torchvision | 0.29.1+cu130 | `torchvision.__version__` |
| CUDA of torch | 13.0, compile-time | `torch.version.cuda` |
| cuDNN | 92400; compile-time 9.24.0; library loaded by torch 9.24.0 | `torch.backends.cudnn.version()`; `torch._C._cudnn.getCompileVersion()`, `getRuntimeVersion()` |
| Environment of the runs below | python 3.11.17, torch 2.14.1+cu130, device NVIDIA GeForce RTX 4090, in all 19 runs | `run_log.md` of each run |

## Measured throughput of the baseline

Median (minimum-maximum) over the runs of a set. Per-run values in `docs/machine_probe.json`, `throughput.runs`.

| Seed set | Runs | Train tiles | Internal-validation tiles | Tiles per s | Seconds per epoch | Minutes per fold with inference | of which inference, s | Peak GPU memory, MiB | Source |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 4 | 1,267-1,440 | 224-256 | 48.8 (48.6-50.0) | 34.3 (32.0-34.5) | 37.2 (35.5-37.8) | 59.5 (34.2-72.2) | 3,159 (3,159-3,159) | `work_ms/runs_itareo/ugi/ugi_r1_none/fold<k>_seed<s>/`, k 1-4, s 0 |
| seed 1 | 4 | 1,267-1,440 | 224-256 | 50.2 (49.9-50.3) | 33.0 (31.2-34.4) | 35.3 (33.8-36.5) | 41.4 (34.2-51.1) | 3,159 (3,159-3,159) | `work_ms/runs_itareo/ugi/ugi_r1_none/fold<k>_seed<s>/`, k 1-4, s 1 |
| seed 2 | 4 | 1,267-1,440 | 224-256 | 50.4 (50.3-50.6) | 32.8 (30.6-34.4) | 35.1 (33.1-36.5) | 41.5 (34.2-49.9) | 3,159 (3,159-3,159) | `work_ms/runs_itareo/ugi/ugi_r1_none/fold<k>_seed<s>/`, k 1-4, s 2 |
| seed 3 | 4 | 1,267-1,440 | 224-256 | 50.4 (49.9-50.6) | 32.8 (31.3-34.4) | 35.0 (33.9-36.4) | 41.3 (33.9-50.8) | 3,159 (3,159-3,159) | `work_ms/runs_itareo/ugi/ugi_r1_none/fold<k>_seed<s>/`, k 1-4, s 3 |
| fold C, seeds 0-2 | 3 | 1,538 | 272 | 49.0 (47.5-49.6) | 38.6 (38.5-41.4) | 41.5 (41.4-44.0) | 20.4 (20.2-30.3) | 3,159 (3,159-3,159) | `work_ms/runs_itareo/ugi/ugi_foldc_sq8km/fold<k>_seed<s>/`, k 5, s 0-2 |

| Column | Definition | Source |
|---|---|---|
| Train and internal-validation tiles | `train_tiles`, `val_tiles` | `split.json` |
| Tiles per s | `train_tiles` / median `seconds_train`, epoch 1 excluded (worker start); batches of 8 with `drop_last`, so at most 7 tiles per epoch are unused | `split.json`, `epochs.csv`, `work_ms/itareo/ugi_seg/train.py` |
| Seconds per epoch | median `seconds_epoch`, epoch 1 excluded; training, internal validation and checkpoint | `epochs.csv`, `train.py` |
| Minutes per fold with inference | total seconds of the run / 60; set-up, 60 epochs, inference and scoring | `run_log.md`, `train.py` |
| of which inference | `seconds`; prediction over the held-out region, writing `prob_oof.tif` and `oof_groups.csv`; number of windows not stated in the file | `prob_oof.json`, `work_ms/itareo/ugi_seg/oof.py` |
| Peak GPU memory | maximum of `gpu_max_alloc_mb`, which is `torch.cuda.max_memory_allocated()` / 2^20 (MiB), reset each epoch; training and internal validation only, inference not recorded | `epochs.csv`, `train.py` |
| Whole card, for comparison | `nvidia-smi` `memory.used`, one job alone, F4, 3 epochs, v1 mosaic, not the baseline runs; 1,323 MiB before it starts, peak 6,199 MiB up to the end of the epochs, peak 12,526 MiB from inference on; two jobs at once, peak 23,393 MiB | `work_ms/runs_itareo/ugi/ugi_f4_throughput/nvidia_smi.csv`, `work_ms/runs_itareo/ugi/ugi_f4_throughput/phases.txt`, `work_ms/runs_itareo/ugi/ugi_f4_throughput/alone_fold1_seed0/epochs.csv (file time)`, `work_ms/runs_itareo/ugi/ugi_f4_throughput/f4_throughput.json` |


