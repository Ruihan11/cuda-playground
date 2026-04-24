import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.app import GPU_TYPE, KERNELS_PATH, REPORTS_PATH, app, cuda_image, volume

_image = (
    cuda_image
    .add_local_dir("harness", remote_path="/root/harness")
    .add_local_dir("kernels", remote_path=KERNELS_PATH)
)


@app.function(
    image=_image,
    volumes={REPORTS_PATH: volume},
    gpu=GPU_TYPE,
    timeout=600,
)
def nsys_single(
    kernel_name: str,
    version: str,
    trace: str,
    timestamp: str,
) -> dict:
    import glob
    import tomllib

    kernel_dir = Path(KERNELS_PATH) / kernel_name
    toml_path = kernel_dir / "kernel.toml"

    if not toml_path.exists():
        return {"status": "error", "error": f"kernel.toml not found in {kernel_dir}"}

    with open(toml_path, "rb") as f:
        meta = tomllib.load(f)

    build_cfg = meta.get("build", {})
    arch = build_cfg.get("arch", "sm_100")
    std = build_cfg.get("std", "c++17")
    extra_flags = build_cfg.get("extra_flags", ["-O3"])

    cu_path = kernel_dir / f"{version}.cu"
    if not cu_path.exists():
        return {"status": "error", "error": f"Source not found: {cu_path}"}

    # Compile
    binary = Path(f"/tmp/{kernel_name}_{version}")
    compile_cmd = ["nvcc", f"-arch={arch}", f"-std={std}", *extra_flags, str(cu_path), "-o", str(binary)]
    r = subprocess.run(compile_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"status": "compile_error", "error": r.stderr}

    # Profile with nsys
    report_dir = Path(REPORTS_PATH) / kernel_name / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / version  # nsys appends .nsys-rep automatically

    # Find nsys binary — installed to /opt/nvidia/nsight-systems/*/bin/nsys
    nsys_candidates = glob.glob("/opt/nvidia/nsight-systems/*/bin/nsys")
    nsys = sorted(nsys_candidates)[-1] if nsys_candidates else "nsys"

    nsys_cmd = [
        nsys, "profile",
        "--output", str(report_path),
        "--force-overwrite", "true",
        "--trace", trace,
        "--export", "sqlite",   # also export .sqlite for programmatic access
        str(binary),
    ]

    r = subprocess.run(nsys_cmd, capture_output=True, text=True)

    volume.commit()

    report_file = Path(str(report_path) + ".nsys-rep")
    if r.returncode != 0 or not report_file.exists():
        return {"status": "nsys_error", "error": r.stderr, "stdout": r.stdout}

    return {
        "status": "ok",
        "report_remote": str(report_file),
        "report_bytes": report_file.read_bytes(),
        "nsys_stdout": r.stdout,
    }


@app.local_entrypoint()
def main(
    kernel: str,
    versions: str = "v1_naive",
    trace: str = "cuda,nvtx,osrt",
):
    """Profile one or more kernel versions with nsys on B200.

    Examples:
        modal run harness/run_nsys.py --kernel vecsum
        modal run harness/run_nsys.py --kernel vecsum --versions v1_naive,v2_grid
        modal run harness/run_nsys.py --kernel vecsum --trace cuda,nvtx
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    version_list = [v.strip() for v in versions.split(",")]

    print(f"\nkernel  : {kernel}")
    print(f"versions: {version_list}")
    print(f"trace   : {trace}")
    print(f"run id  : {timestamp}\n")

    any_failed = False
    for version in version_list:
        print(f"  profiling {version} ...", end=" ", flush=True)
        result = nsys_single.remote(kernel, version, trace, timestamp)
        status = result["status"]

        if status == "ok":
            local_path = Path(f"reports/{kernel}/{timestamp}/{version}.nsys-rep")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(result["report_bytes"])
            print(f"ok → {local_path}")
        else:
            any_failed = True
            print(f"FAILED ({status})")
            err = result.get("error", "")
            if err:
                print(f"    {err[:300].strip()}")

    if any_failed:
        raise SystemExit(1)
