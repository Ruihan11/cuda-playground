# CLAUDE.md — kernel-profiling-harness

## Project Overview

A Modal-based CUDA kernel profiling harness targeting RTX5090 (local, sm_120) + H200 (Modal, sm_90). Used daily for GPU kernel engineering study. Primary workflow: write kernel → profile on H200 → fetch `.ncu-rep` → analyze in Nsight Compute UI.

## Working Mode

**User writes all code.** Claude's role is pair programmer / design advisor. Do not write complete files. Provide:
- Design specs (what a file needs to contain and why)
- Short snippets (≤10 lines) for non-obvious API usage
- Review of code the user writes

## Project Structure

```
kernel-profiling-harness/
├── harness/
│   ├── app.py          # Modal app, image, volume — single source of truth for config
│   ├── run_ncu.py      # ncu profiling entrypoint
│   ├── run_nsys.py     # nsys timeline entrypoint
│   └── run_bench.py    # H200 benchmark: hand-rolled versions + sdpa + FA4
├── kernels/
│   └── {kernel_name}/
│       ├── kernel.toml     # build config, correctness thresholds, input spec
│       ├── {version}.cu    # CUDA kernel with standalone main()
│       ├── {version}.py    # (alt) PyTorch / Triton kernel
│       ├── reference.py    # (M3) reference impl + generate_input(**kwargs)
│       └── run_local.py    # per-kernel local validation + benchmark (ctypes)
├── scripts/
│   └── fetch_reports.sh    # wraps `modal volume get`
├── temp/                   # compiled .so files — gitignored
├── reports/                # local synced reports — gitignored
├── Progress.md
└── CLAUDE.md
```

## Key Design Decisions

### Image
- Base: `nvidia/cuda:13.0.0-devel-ubuntu24.04` (has nvcc, cuda-gdb)
- ncu installed separately via apt (`nsight-compute-2026.x.x`) because devel image ncu may be stale or not support sm_100
- ncu binary path: `glob("/opt/nvidia/nsight-compute/*/ncu")` — do NOT hardcode
- GPU type: `"H200:1"` (not `modal.gpu.H200()`) — B200 dropped

### kernels/ mounting
- Use `image.add_local_dir("kernels", remote_path=KERNELS_PATH)` — `modal.Mount` was removed in Modal 1.x
- `add_local_dir` on the image object uploads on each `modal run` without baking into the image layer

### Volume
- Name: `kernel-profiling-reports`, mounted at `/reports`
- Must call `volume.commit()` after writing reports — writes are not auto-persisted
- Report path: `/reports/{kernel_name}/{timestamp}/{version}.ncu-rep`

### kernel.toml fields
```toml
[build]    # arch, std, extra_flags
[correctness]  # max_error, mean_error, dtype
[input]    # kwargs passed to generate_input() in M3
[profile]  # kernel_name_regex, warmup_iters, profile_iters
```

### ncu profile modes
- `full`: `--set full` (~30-60s)
- `quick`: SpeedOfLight + MemoryWorkloadAnalysis + WarpStateStats (~10-20s)
- `custom`: `--sections A,B,C` (user-specified)

### .cu kernel interface (M1)
Standalone `main(int argc, char* argv[])`. Accepts optional `argv[1]` as input size. Correctness check (M3) will require a separate Python wrapper or shared-lib approach — defer to M3.

### Failure handling
- Hard failures (compile error, correctness fail, ncu crash): skip version, print warning, continue
- Summary shows `✗ COMPILE_ERROR` / `✗ CORRECTNESS (max_err=X)` for failed versions
- `modal run` exits non-zero if any version failed

### Reference profiling (M3)
- `reference.py` wraps reference call in `torch.cuda.nvtx.range_push/pop`
- ncu uses `--nvtx --nvtx-include "reference"` to isolate those kernels
- `.py` kernels always get `--target-processes all` (Python spawns child CUDA procs)

## Milestones

See Progress.md.

## Common Commands

```bash
# Profile (ncu) — hand-rolled .cu kernels
modal run harness/run_ncu.py --kernel vecsum --versions v1_naive --mode quick
modal run harness/run_ncu.py --kernel softmax --versions v1_naive,v2_online --mode quick

# Profile FA4 specifically (Python kernel, uses --target-processes all internally)
modal run harness/run_ncu.py::fa4 --mode quick

# Profile (nsys)
modal run harness/run_nsys.py --kernel vecsum --versions v1_naive

# Benchmark hand-rolled versions vs sdpa vs fa4 on H200
modal run harness/run_bench.py --kernel atten --versions v4_half_fa2,v5_wmma_fa2
# Baseline-only (fa4 vs sdpa, no hand-rolled versions)
modal run harness/run_bench.py::baselines

# Local build (RTX5090, sm_120)
nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_120 -Wno-deprecated-gpu-targets \
  -o temp/v5_atten.so kernels/atten/v5_wmma_fa2.cu

# Local validation + benchmark (atten kernel)
python kernels/atten/run_local.py

# Local debug binary
nvcc -g -G -O0 -o softmax_debug kernels/softmax/v1_softmax_naive.cu
cuda-gdb ./softmax_debug

# Fetch reports
modal volume ls kernel-profiling-reports
```

## CuTe Python Kernels

- Use `@cute.jit` for the launcher, `@cute.kernel` for the GPU kernel
- Validate via `validate_cute()` in `run_local.py` (not ctypes)
- Benchmark via `benchmark_sdp()` using CUDA graphs
- FA4 import: `from flash_attn.cute import flash_attn_func`

## Kernel Naming Convention

- `v1_naive.cu` / `v1_softmax_naive.cu` — single-thread baseline
- `v2_online.cu` — algorithmic improvement (online scan)
- `v3_parallel.cu` — parallel reduction with shared memory
- `ref_kernel.py` — torch reference + ctypes correctness test in `main()`
