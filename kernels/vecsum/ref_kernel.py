# nvcc -shared -Xcompiler -fPIC -O3 -o vecsum.so v1_naive.cu -Wno-deprecated-gpu-targets
import torch
import ctypes


def ref_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 1 and y.dim() == 1 and x.numel() == y.numel(), (
        "x and y should be same shape 1D vector"
    )
    return x + y


def main():
    torch.manual_seed(42)
    n = 2048
    x = torch.randn(n, dtype=torch.float32)
    y = torch.randn(n, dtype=torch.float32)
    ref = ref_kernel(x, y)
    # print(ref)

    lib = ctypes.CDLL("./vecsum.so")
    lib.vecsum_launch.restype = None
    lib.vecsum_launch.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int]

    x_c, y_c = x.contiguous().cpu(), y.contiguous().cpu()
    cuda_out = torch.zeros(n, dtype=torch.float32)
    ptr = lambda t: ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_float))
    lib.vecsum_launch(ptr(x_c), ptr(y_c), ptr(cuda_out), ctypes.c_int(n))

    assert torch.allclose(ref, cuda_out, atol=1e-5), (
        f"FAIL max_err={(ref - cuda_out).abs().max()}"
    )
    print(f"PASS max_err={(ref - cuda_out).abs().max()}")


if __name__ == "__main__":
    main()
