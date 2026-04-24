# nvcc -shared -Xcompiler -fPIC -O3 -o softmax.so kernels/softmax/v1_softmax_naive.cu -Wno-deprecated-gpu-targets
# nvcc -shared -Xcompiler -fPIC -O3 -o softmax.so kernels/softmax/v2_online_softmax.cu -Wno-deprecated-gpu-targets
# nvcc -shared -Xcompiler -fPIC -O3 -o softmax.so kernels/softmax/v3_parallel_softmax.cu -Wno-deprecated-gpu-targets
import torch
import torch.nn.functional as F
import ctypes


def ref_kernel(x: torch.Tensor) -> torch.Tensor:
    x = F.softmax(x, 0)
    return x


def main():
    # torch.manual_seed(42)
    # n = 2048
    # x = torch.randn(n, dtype=torch.float32)
    x = torch.randn(10000)
    ref = ref_kernel(x)
    print(ref)

    lib = ctypes.CDLL("./softmax.so")
    lib.softmax_launch.restype = None
    lib.softmax_launch.argtypes = [ctypes.POINTER(ctypes.c_float)] * 2 + [ctypes.c_int]

    x_c = x.contiguous().cpu()
    cuda_out = torch.zeros(x.numel(), dtype=torch.float32)
    ptr = lambda t: ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_float))
    lib.softmax_launch(ptr(x_c), ptr(cuda_out), ctypes.c_int(x.numel()))

    assert torch.allclose(ref, cuda_out, atol=1e-5), (
        f"FAIL max_err={(ref - cuda_out).abs().max()}"
    )
    print(f"PASS max_err={(ref - cuda_out).abs().max()}")


if __name__ == "__main__":
    main()
