import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.app import GPU_TYPE, KERNELS_PATH, REPORTS_PATH, app, cuda_image, volume

# Modal puts the entrypoint at /root/profile.py so /root is on sys.path.
# harness/ at /root/harness/ makes `from harness.app import ...` work remotely.
# kernels/ is the last layer so editing .cu files only rebuilds that thin layer.
_image = (
    cuda_image
    .add_local_dir("harness", remote_path="/root/harness")
    .add_local_dir("kernels", remote_path=KERNELS_PATH)
)


def _build_ncu_flags(mode: str, sections: str) -> list[str]:
    if mode == "full":
        return ["--set", "full"]
    if mode == "quick":
        return [
            "--section", "SpeedOfLight",
            "--section", "MemoryWorkloadAnalysis",
            "--section", "WarpStateStats",
        ]
    if mode == "custom":
        return [flag for s in sections.split(",") for flag in ("--section", s.strip())]
    raise ValueError(f"Unknown mode: {mode!r}. Choose full | quick | custom")


@app.function(
    image=_image,
    volumes={REPORTS_PATH: volume},
    gpu=GPU_TYPE,
    timeout=600,
)
def profile_single(
    kernel_name: str,
    version: str,
    mode: str,
    sections: str,
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

    # Profile
    report_dir = Path(REPORTS_PATH) / kernel_name / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / version  # ncu appends .ncu-rep automatically

    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]

    ncu_cmd = [
        ncu,
        *_build_ncu_flags(mode, sections),
        "--export", str(report_path),
        "--force-overwrite",
    ]

    profile_cfg = meta.get("profile", {})
    kernel_regex = profile_cfg.get("kernel_name_regex", "")
    if kernel_regex:
        ncu_cmd += ["--kernel-name", kernel_regex]

    ncu_cmd.append(str(binary))

    r = subprocess.run(ncu_cmd, capture_output=True, text=True)

    volume.commit()

    report_file = Path(str(report_path) + ".ncu-rep")
    if r.returncode != 0 or not report_file.exists():
        return {"status": "ncu_error", "error": r.stderr, "stdout": r.stdout}

    # Return bytes directly — avoids slow volume.read_file() on the local side.
    return {
        "status": "ok",
        "report_remote": str(report_file),
        "report_bytes": report_file.read_bytes(),
        "ncu_stdout": r.stdout,
    }


@app.local_entrypoint()
def main(
    kernel: str,
    versions: str = "v1_naive",
    mode: str = "full",
    sections: str = "",
):
    """Profile one or more kernel versions with ncu on B200.

    Examples:
        modal run harness/run_ncu.py --kernel vecsum
        modal run harness/run_ncu.py --kernel vecsum --versions v1_naive,v2_grid --mode quick
        modal run harness/run_ncu.py --kernel vecsum --mode custom --sections SpeedOfLight,Occupancy
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    version_list = [v.strip() for v in versions.split(",")]

    print(f"\nkernel  : {kernel}")
    print(f"versions: {version_list}")
    print(f"mode    : {mode}" + (f" ({sections})" if sections else ""))
    print(f"run id  : {timestamp}\n")

    any_failed = False
    for version in version_list:
        print(f"  profiling {version} ...", end=" ", flush=True)
        result = profile_single.remote(kernel, version, mode, sections, timestamp)
        status = result["status"]

        if status == "ok":
            local_path = Path(f"reports/{kernel}/{timestamp}/{version}.ncu-rep")
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
