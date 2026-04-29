import modal

APP_NAME = "kernel-profiling-harness"
REPORTS_PATH = "/reports"
KERNELS_PATH = "/workspace/kernels"
GPU_TYPE = "H200:1"

app = modal.App(APP_NAME)

volume = modal.Volume.from_name("kernel-profiling-reports", create_if_missing=True)

# nvidia/cuda devel image provides nvcc (needed to compile .cu files).
# nsight-compute is installed separately via apt because the image-bundled
# ncu version does not support sm_100 (Blackwell / B200).
# numpy before torch so torch's resolver sees it and picks a compatible version.
cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("wget", "gnupg")
    .run_commands(
        "wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub"
        " | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg]"
        " https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /'"
        " > /etc/apt/sources.list.d/cuda.list",
        "apt-get update && apt-get install -y nsight-compute-2026.1.0"
        " && NSYS=$(apt-cache search '^nsight-systems-20' | sort | tail -1 | awk '{print $1}')"
        " && echo \"installing nsys: $NSYS\" && apt-get install -y $NSYS",
    )
    .pip_install("numpy")
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
)
