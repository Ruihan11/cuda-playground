"""
nvcc -shared -Xcompiler -fPIC -O3 -o temp/v1_softmax.so kernels/softmax/v1_naive_softmax.cu -Wno-deprecated-gpu-targets
nvcc -shared -Xcompiler -fPIC -O3 -o temp/v2_softmax.so kernels/softmax/v2_online_softmax.cu -Wno-deprecated-gpu-targets
nvcc -shared -Xcompiler -fPIC -O3 -o temp/v3_softmax.so kernels/softmax/v3_parallel_softmax.cu -Wno-deprecated-gpu-targets
nvcc -shared -Xcompiler -fPIC -O3 -o temp/v4_softmax.so kernels/softmax/v4_tiled_softmax.cu -Wno-deprecated-gpu-targets
"""

import torch
import torch.nn.functional as F
import ctypes


def ptr(t: torch.Tensor) -> ctypes.POINTER(ctypes.c_float):
    return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_float))


def test_kernel(path: str):
    torch.manual_seed(42)
    rows, cols = 4, 4
    x = torch.randn(rows, cols)
    ref = F.softmax(x, dim=1)

    lib = ctypes.CDLL(path)
    lib.softmax_launch.restype = None
    lib.softmax_launch.argtypes = [ctypes.POINTER(ctypes.c_float)] * 2 + [
        ctypes.c_int
    ] * 2

    x_c = x.contiguous().cpu()
    cuda_out = torch.zeros_like(x_c)
    lib.softmax_launch(ptr(x_c), ptr(cuda_out), ctypes.c_int(rows), ctypes.c_int(cols))

    assert torch.allclose(ref, cuda_out, atol=1e-5), (
        f"FAIL max_err={(ref - cuda_out).abs().max()}"
    )
    print(f"PASS max_err={(ref - cuda_out).abs().max()}")


def benchmark(
    so_path: str, x: torch.Tensor, warmup: int = 10, iters: int = 100
) -> float:
    rows, cols = x.shape
    lib = ctypes.CDLL(so_path)
    lib.benchmark_launch.restype = ctypes.c_float
    lib.benchmark_launch.argtypes = [ctypes.POINTER(ctypes.c_float)] * 2 + [
        ctypes.c_int
    ] * 4
    x_c = x.contiguous().cpu()
    out = torch.zeros_like(x_c)
    return lib.benchmark_launch(
        ptr(x_c),
        ptr(out),
        ctypes.c_int(rows),
        ctypes.c_int(cols),
        ctypes.c_int(warmup),
        ctypes.c_int(iters),
    )


def main():
    versions = {
        "v1": "temp/v1_softmax.so",
        "v2": "temp/v2_softmax.so",
        "v3": "temp/v3_softmax.so",
        "v4": "temp/v4_softmax.so",
    }

    for name, path in versions.items():
        print(f"{name}: ", end="")
        test_kernel(path)

    configs_1row = [(1, 512), (1, 2048), (1, 8192), (1, 32768), (1, 131072)]
    configs_256row = [(256, 512), (256, 1024), (256, 2048), (256, 4096), (256, 8192)]

    print(
        f"\n{'rows':>6}  {'cols':>6}  {'v1 GB/s':>10}  {'v2 GB/s':>10}  {'v3 GB/s':>10}  {'v4 GB/s':>10}"
    )

    print("-" * 66)
    for rows, cols in configs_256row:
        x = torch.randn(rows, cols)
        row = f"{rows:>6}  {cols:>6}"
        for path in versions.values():
            ms = benchmark(path, x)
            gbs = 2 * rows * cols * 4 / (ms * 1e-3) / 1e9
            row += f"  {gbs:>10.3f}"
        print(row)


if __name__ == "__main__":
    main()
