"""
nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_120 -o temp/v1_atten.so kernels/atten/v1_naive_atten.cu -Wno-deprecated-gpu-targets
nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_120 -o temp/v2_atten.so kernels/atten/v2_fa2.cu -Wno-deprecated-gpu-targets
nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_120 -o temp/v3_atten.so kernels/atten/v3_tiled_fa2.cu -Wno-deprecated-gpu-targets
nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_120 -o temp/v4_atten.so kernels/atten/v4_half_fa2.cu -Wno-deprecated-gpu-targets
nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_120 -o temp/v5_atten.so kernels/atten/v5_wmma_fa2.cu -Wno-deprecated-gpu-targets
"""

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
import ctypes
from kernels.atten.v6_cute import FlashAttention as fa_v6

try:
    from flash_attn.cute import flash_attn_func as _fa4_fn
    _FA4_AVAILABLE = True
except ImportError:
    _FA4_AVAILABLE = False


def ptr(t: torch.Tensor) -> ctypes.POINTER(ctypes.c_float):
    return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_float))


def kernel_validation(path: str, dtype: torch.dtype = torch.float32):
    torch.manual_seed(666)
    batch, heads, seq, d = 2, 16, 512, 128
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

    Q_c = Q.to(dtype).contiguous().cpu()
    K_c = K.to(dtype).contiguous().cpu()
    V_c = V.to(dtype).contiguous().cpu()
    cuda_out = torch.zeros(batch, heads, seq, d, dtype=dtype)
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
    dtype: torch.dtype = torch.float32,
    warmup: int = 10,
    iters: int = 100,
) -> float:
    lib = ctypes.CDLL(so_path)
    lib.benchmark_launch.restype = ctypes.c_float
    lib.benchmark_launch.argtypes = [ctypes.POINTER(ctypes.c_float)] * 4 + [
        ctypes.c_int64
    ] * 6
    Q = torch.randn(batch, heads, seq, d).to(dtype).cuda().contiguous()
    K = torch.randn(batch, heads, seq, d).to(dtype).cuda().contiguous()
    V = torch.randn(batch, heads, seq, d).to(dtype).cuda().contiguous()
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


def benchmark_fa4(batch: int, heads: int, seq: int, d: int, warmup: int = 10, iters: int = 100) -> float:
    Q = torch.randn(batch, seq, heads, d, dtype=torch.float16).cuda()
    K = torch.randn(batch, seq, heads, d, dtype=torch.float16).cuda()
    V = torch.randn(batch, seq, heads, d, dtype=torch.float16).cuda()
    return benchmark_sdp(lambda: _fa4_fn(Q, K, V), warmup, iters)


def validate_cute(fa: fa_v6):
    torch.manual_seed(666)
    batch, heads, seq, d = 2, 16, 512, 128
    Q = torch.randn(batch, seq, heads, d, dtype=torch.float16).cuda()
    K = torch.randn(batch, seq, heads, d, dtype=torch.float16).cuda()
    V = torch.randn(batch, seq, heads, d, dtype=torch.float16).cuda()
    O = torch.zeros_like(Q)
    fa.run(Q, K, V, O, scale=d**-0.5)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ref = scaled_dot_product_attention(
            Q.permute(0, 2, 1, 3), K.permute(0, 2, 1, 3), V.permute(0, 2, 1, 3)
        ).permute(0, 2, 1, 3)
    assert torch.allclose(ref, O, atol=1e-2), f"FAIL max_err={(ref - O).abs().max()}"
    print(f"PASS max_err={(ref - O).abs().max()}")


def benchmark_cute(fa: fa_v6, batch: int, heads: int, seq: int, d: int) -> float:
    Q = torch.randn(batch, seq, heads, d, dtype=torch.float16).cuda()
    K = torch.randn(batch, seq, heads, d, dtype=torch.float16).cuda()
    V = torch.randn(batch, seq, heads, d, dtype=torch.float16).cuda()
    O = torch.zeros_like(Q)
    return benchmark_sdp(lambda: fa.run(Q, K, V, O, scale=d**-0.5))


def run_validation(
    versions: dict[str, tuple[str, torch.dtype]],
    cute_versions: dict[str, fa_v6] | None = None,
):
    for name, (path, dtype) in versions.items():
        print(f"{name}: ", end="")
        kernel_validation(path, dtype)
    for name, fa in (cute_versions or {}).items():
        print(f"{name}: ", end="")
        validate_cute(fa)


def run_benchmark(
    versions: dict[str, tuple[str, torch.dtype]],
    configs: list[tuple[int, int, int, int]],
    cute_versions: dict[str, fa_v6] | None = None,
):
    cute_versions = cute_versions or {}
    ver_names = list(versions.keys()) + list(cute_versions.keys())
    extra_cols = ["sdpa"] + (["fa4"] if _FA4_AVAILABLE else [])

    def col(s):
        return f"{s:>10}"

    header = f"{'seq':>6}  {'d':>4}  " + "  ".join(
        col(f"{n} TF/s") for n in ver_names + extra_cols
    )
    print(f"\n{header}")
    print("-" * len(header))

    for batch, heads, seq, d in configs:
        row = f"{seq:>6}  {d:>4}"
        flops = 4 * batch * heads * seq * seq * d

        for path, dtype in versions.values():
            ms = benchmark(path, batch, heads, seq, d, dtype)
            row += f"  {col('OOM' if ms <= 0 else f'{flops / (ms * 1e-3) / 1e12:.4f}')}"

        for fa in cute_versions.values():
            ms = benchmark_cute(fa, batch, heads, seq, d)
            row += f"  {col(f'{flops / (ms * 1e-3) / 1e12:.4f}')}"

        q = torch.randn(batch, heads, seq, d, dtype=torch.float16).cuda()
        k = torch.randn(batch, heads, seq, d, dtype=torch.float16).cuda()
        v = torch.randn(batch, heads, seq, d, dtype=torch.float16).cuda()
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            ms_sdpa = benchmark_sdp(
                lambda: scaled_dot_product_attention(q, k, v, scale=1 / d**0.5)
            )
        row += f"  {col(f'{flops / (ms_sdpa * 1e-3) / 1e12:.4f}')}"

        if _FA4_AVAILABLE:
            ms_fa4 = benchmark_fa4(batch, heads, seq, d)
            row += f"  {col(f'{flops / (ms_fa4 * 1e-3) / 1e12:.4f}')}"

        print(row)


def main():
    versions = {
        # "v1": ("temp/v1_atten.so", torch.float32),
        # "v2": ("temp/v2_atten.so", torch.float32),
        # "v3": ("temp/v3_atten.so", torch.float32),
        # "v4": ("temp/v4_atten.so", torch.float16),
        "v5": ("temp/v5_atten.so", torch.float16),
    }
    cute_versions = {
        # "v6": fa_v6(),
    }

    configs = [
        (2, 16, 128, 128),
        (2, 16, 256, 128),
        (2, 16, 512, 128),
        (2, 16, 1024, 128),
        (2, 16, 2048, 128),
        (2, 16, 4096, 128),
    ]

    run_validation(versions, cute_versions)

    run_benchmark(versions, configs, cute_versions)


if __name__ == "__main__":
    main()
