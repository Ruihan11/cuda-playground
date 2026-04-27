"""
nvcc -shared -Xcompiler -fPIC -O3 -o temp/v1_atten.so kernels/atten/v1_naive_atten.cu -Wno-deprecated-gpu-targets
nvcc -shared -Xcompiler -fPIC -O3 -o temp/v2_atten.so kernels/atten/v2_fa2_atten.cu -Wno-deprecated-gpu-targets
"""

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
import ctypes


def ptr(t: torch.Tensor) -> ctypes.POINTER(ctypes.c_float):
    return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_float))


def test_kernel(path: str):
    torch.manual_seed(42)
    batch, heads, seq, d = 10, 10, 40, 40
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
        ctypes.c_int
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
        ctypes.c_int(batch),
        ctypes.c_int(heads),
        ctypes.c_int(seq),
        ctypes.c_int(d),
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
        ctypes.c_int
    ] * 6
    Q = torch.randn(batch, heads, seq, d).contiguous()
    K = torch.randn(batch, heads, seq, d).contiguous()
    V = torch.randn(batch, heads, seq, d).contiguous()
    out = torch.zeros_like(Q)
    return lib.benchmark_launch(
        ptr(Q),
        ptr(K),
        ptr(V),
        ptr(out),
        ctypes.c_int(batch),
        ctypes.c_int(heads),
        ctypes.c_int(seq),
        ctypes.c_int(d),
        ctypes.c_int(warmup),
        ctypes.c_int(iters),
    )


def main():
    versions = {
        "v1": "temp/v1_atten.so",
        "v2": "temp/v2_atten.so",
        # "v3": "temp/v3_atten.so",
        # "v4": "temp/v4_atten.so",
    }

    for name, path in versions.items():
        print(f"{name}: ", end="")
        test_kernel(path)

    configs = [(1, 1, 512, 64), (1, 1, 1024, 64), (1, 1, 2048, 64)]

    print(f"\n{'seq':>6}  {'d':>4}  {'v1 GB/s':>10}  {'v2 GB/s':>10}")
    print("-" * 38)
    for batch, heads, seq, d in configs:
        row = f"{seq:>6}  {d:>4}"
        for name, path in versions.items():
            ms = benchmark(path, batch, heads, seq, d)
            total_bytes = 4 * batch * heads * seq * d * 4
            gbs = total_bytes / (ms * 1e-3) / 1e9
            row += f"  {gbs:>10.3f}"
        print(row)


if __name__ == "__main__":
    main()
