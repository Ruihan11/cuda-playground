"""
nvcc -shared -Xcompiler -fPIC -O3 -o temp/v1_atten.so kernels/atten/v1_naive_atten.cu -Wno-deprecated-gpu-targets
nvcc -shared -Xcompiler -fPIC -O3 -o temp/v2_atten.so kernels/atten/v2_fa2.cu -Wno-deprecated-gpu-targets
nvcc -shared -Xcompiler -fPIC -O3 -o temp/v3_atten.so kernels/atten/v3_tiled_fa2.cu -Wno-deprecated-gpu-targets
nvcc -shared -Xcompiler -fPIC -O3 -o temp/v4_atten.so kernels/atten/v4_warpReduce_fa2.cu -Wno-deprecated-gpu-targets
"""

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
import ctypes


def ptr(t: torch.Tensor) -> ctypes.POINTER(ctypes.c_float):
    return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_float))


def kernel_validation(path: str):
    torch.manual_seed(42)
    batch, heads, seq, d = 10, 10, 128, 64
    Q = torch.randn(batch, heads, seq, d, dtype=torch.float32)
    K = torch.randn(batch, heads, seq, d, dtype=torch.float32)
    V = torch.randn(batch, heads, seq, d, dtype=torch.float32)
    Q_d = Q.half().cuda()
    K_d = K.half().cuda()
    V_d = V.half().cuda()

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ref = scaled_dot_product_attention(Q_d, K_d, V_d).float().cpu()

    lib = ctypes.CDLL(path)
    lib.atten_launch.restype = None
    lib.atten_launch.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4 + [
        ctypes.c_int64
    ] * 4

    Q_c, K_c, V_c = (
        Q.float().contiguous().cpu(),
        K.float().contiguous().cpu(),
        V.float().contiguous().cpu(),
    )
    cuda_out = torch.zeros_like(Q_c)
    lib.atten_launch(
        ptr(Q_c),
        ptr(K_c),
        ptr(V_c),
        ptr(cuda_out),
        ctypes.c_int64(batch),
        ctypes.c_int64(heads),
        ctypes.c_int64(seq),
        ctypes.c_int64(d),
    )

    cuda_out = cuda_out.float()
    assert torch.allclose(ref, cuda_out, atol=1e-2), (
        f"FAIL max_err={(ref - cuda_out).abs().max()}"
    )
    print(f"PASS max_err={(ref - cuda_out).abs().max()}")


def benchmark(
    so_path: str,
    batch: int,
    heads: int,
    seq: int,
    d: int,
    warmup: int = 10,
    iters: int = 100,
) -> float:
    lib = ctypes.CDLL(so_path)
    lib.benchmark_launch.restype = ctypes.c_float
    lib.benchmark_launch.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4 + [
        ctypes.c_int64
    ] * 6
    Q = torch.randn(batch, heads, seq, d).cuda().contiguous()
    K = torch.randn(batch, heads, seq, d).cuda().contiguous()
    V = torch.randn(batch, heads, seq, d).cuda().contiguous()
    out = torch.zeros_like(Q)
    return lib.benchmark_launch(
        ptr(Q),
        ptr(K),
        ptr(V),
        ptr(out),
        ctypes.c_int64(batch),
        ctypes.c_int64(heads),
        ctypes.c_int64(seq),
        ctypes.c_int64(d),
        ctypes.c_int64(warmup),
        ctypes.c_int64(iters),
    )


def benchmark_sdp(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_validation(versions: dict[str, str]):
    for name, path in versions.items():
        print(f"{name}: ", end="")
        kernel_validation(path)


def run_benchmark(versions: dict[str, str], configs: list[tuple[int, int, int, int]]):

    ver_names = list(versions.keys())

    def col(s):
        return f"{s:>10}"

    header = f"{'seq':>6}  {'d':>4}  " + "  ".join(
        col(f"{n} TF/s") for n in ver_names + ["sdpa"]
    )
    print(f"\n{header}")
    print("-" * len(header))

    for batch, heads, seq, d in configs:
        row = f"{seq:>6}  {d:>4}"
        flops = 4 * batch * heads * seq * seq * d

        for path in versions.values():
            ms = benchmark(path, batch, heads, seq, d)
            if ms <= 0:
                row += f"  {col('OOM')}"
            else:
                tflops = flops / (ms * 1e-3) / 1e12
                row += f"  {col(f'{tflops:.4f}')}"

        q = torch.randn(batch, heads, seq, d, dtype=torch.float16).cuda()
        k = torch.randn(batch, heads, seq, d, dtype=torch.float16).cuda()
        v = torch.randn(batch, heads, seq, d, dtype=torch.float16).cuda()

        def fa_fn():
            return scaled_dot_product_attention(q, k, v, scale=1 / d**0.5)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            ms_fa = benchmark_sdp(fa_fn)
        tflops_fa = flops / (ms_fa * 1e-3) / 1e12
        row += f"  {col(f'{tflops_fa:.4f}')}"

        print(row)


def main():
    versions = {
        # "v1": "temp/v1_atten.so",
        # "v2": "temp/v2_atten.so",
        "v3": "temp/v3_atten.so",
        "v4": "temp/v4_atten.so",
    }

    configs = [
        (2, 16, 512, 128),
        # (2, 16, 1024, 128),
        # (2, 16, 2048, 128),
        # (2, 16, 4096, 128),
    ]

    run_validation(versions)

    run_benchmark(versions, configs)


if __name__ == "__main__":
    main()
