"""
Benchmark kernel versions on H200 via Modal.

Examples:
    modal run harness/run_bench.py --kernel atten --versions v4_half_fa2,v5_wmma_fa2
    modal run harness/run_bench.py --kernel atten --versions v4_half_fa2 --dtype fp32
"""

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.app import GPU_TYPE, KERNELS_PATH, app, cuda_image

_image = cuda_image.add_local_dir("harness", remote_path="/root/harness").add_local_dir(
    "kernels", remote_path=KERNELS_PATH
)

# (batch, heads, seq, d)
_DEFAULT_CONFIGS = [
    (2, 16, 512, 128),
    (2, 16, 1024, 128),
    (2, 16, 2048, 128),
    (2, 16, 4096, 128),
]


@app.function(image=_image, gpu=GPU_TYPE, timeout=300)
def bench_single(
    kernel_name: str,
    version: str,
    configs: list,
    warmup: int,
    iters: int,
    dtype_str: str,
) -> dict:
    import ctypes
    import subprocess
    import tomllib
    import torch

    kernel_dir = Path(KERNELS_PATH) / kernel_name
    toml_path = kernel_dir / "kernel.toml"

    if not toml_path.exists():
        return {"status": "error", "error": f"kernel.toml not found in {kernel_dir}"}

    with open(toml_path, "rb") as f:
        meta = tomllib.load(f)

    build_cfg = meta.get("build", {})
    arch = build_cfg.get("arch", "sm_90")
    std = build_cfg.get("std", "c++17")

    cu_path = kernel_dir / f"{version}.cu"
    if not cu_path.exists():
        return {"status": "error", "error": f"Source not found: {cu_path}"}

    so_path = f"/tmp/{kernel_name}_{version}.so"
    compile_cmd = [
        "nvcc",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        f"-arch={arch}",
        f"-std={std}",
        "-O3",  # always O3 for benchmarking
        str(cu_path),
        "-o",
        so_path,
    ]
    r = subprocess.run(compile_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"status": "compile_error", "error": r.stderr}

    lib = ctypes.CDLL(so_path)
    lib.benchmark_launch.restype = ctypes.c_float
    lib.benchmark_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 6

    dtype = torch.float16 if dtype_str == "fp16" else torch.float32
    results = []

    for batch, heads, seq, d in configs:
        Q = torch.randn(batch, heads, seq, d, dtype=dtype, device="cuda").contiguous()
        K = torch.randn(batch, heads, seq, d, dtype=dtype, device="cuda").contiguous()
        V = torch.randn(batch, heads, seq, d, dtype=dtype, device="cuda").contiguous()
        out = torch.zeros_like(Q)

        ms = lib.benchmark_launch(
            Q.data_ptr(),
            K.data_ptr(),
            V.data_ptr(),
            out.data_ptr(),
            ctypes.c_int64(batch),
            ctypes.c_int64(heads),
            ctypes.c_int64(seq),
            ctypes.c_int64(d),
            ctypes.c_int64(warmup),
            ctypes.c_int64(iters),
        )
        flops = 4 * batch * heads * seq * seq * d  # QK^T + PV, fwd only
        tflops = flops / (ms * 1e-3) / 1e12
        results.append({"seq": seq, "d": d, "ms": float(ms), "tflops": tflops})

    return {"status": "ok", "version": version, "results": results}


@app.function(image=_image, gpu=GPU_TYPE, timeout=300)
def bench_sdpa(configs: list, warmup: int, iters: int) -> dict:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention

    results = []
    for batch, heads, seq, d in configs:
        Q = torch.randn(batch, heads, seq, d, dtype=torch.float16, device="cuda")
        K = torch.randn(batch, heads, seq, d, dtype=torch.float16, device="cuda")
        V = torch.randn(batch, heads, seq, d, dtype=torch.float16, device="cuda")

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for _ in range(warmup):
                scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()

            # use CUDAGraph for low-overhead timing
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                scaled_dot_product_attention(Q, K, V)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                g.replay()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / iters

        flops = 4 * batch * heads * seq * seq * d
        tflops = flops / (ms * 1e-3) / 1e12
        results.append({"seq": seq, "d": d, "ms": float(ms), "tflops": tflops})

    return {"status": "ok", "results": results}


@app.local_entrypoint()
def main(
    kernel: str,
    versions: str = "v4_tiled_fa2",
    warmup: int = 10,
    iters: int = 100,
    dtype: str = "fp16",
):
    version_list = [v.strip() for v in versions.split(",")]

    print(f"\nkernel  : {kernel}")
    print(f"versions: {version_list}")
    print(f"dtype   : {dtype}   warmup/iters: {warmup}/{iters}\n")

    def col(s):
        return f"{s:>13}"

    all_cols = version_list + ["sdpa"]
    header = f"{'seq':>6}  {'d':>4}  " + "  ".join(col(v) for v in all_cols)
    sep = "-" * len(header)

    # launch all in parallel — kernel versions + sdpa
    futures = {
        version: bench_single.remote(
            kernel, version, _DEFAULT_CONFIGS, warmup, iters, dtype
        )
        for version in version_list
    }
    sdpa_future = bench_sdpa.remote(_DEFAULT_CONFIGS, warmup, iters)

    all_results: dict[tuple, dict[str, float]] = {}
    for version, result in futures.items():
        if result["status"] != "ok":
            print(f"  {version}: FAILED ({result['status']})")
            print(result.get("error", "")[:800])
            continue
        for r in result["results"]:
            key = (r["seq"], r["d"])
            all_results.setdefault(key, {})[version] = r["tflops"]

    sdpa_result = sdpa_future
    if sdpa_result["status"] == "ok":
        for r in sdpa_result["results"]:
            key = (r["seq"], r["d"])
            all_results.setdefault(key, {})["sdpa"] = r["tflops"]
    else:
        print(f"  sdpa: FAILED")

    print(header)
    print(sep)
    for _, _, seq, d in _DEFAULT_CONFIGS:
        key = (seq, d)
        row = f"{seq:>6}  {d:>4}"
        for version in all_cols:
            val = all_results.get(key, {}).get(version)
            row += f"  {col(f'{val:.4f} TF/s' if val else 'FAILED')}"
        print(row)
