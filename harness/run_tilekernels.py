"""
Run TileKernels pytest suite on H200 via Modal.

Examples:
    modal run harness/run_tilekernels.py --module transpose
    modal run harness/run_tilekernels.py --module quant
    modal run harness/run_tilekernels.py --module moe
    modal run harness/run_tilekernels.py --module transpose --benchmark
"""

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.app import GPU_TYPE, app, tk_image

_TK_PATH = "/workspace/TileKernels"

_image = (
    tk_image
    .add_local_dir("harness", remote_path="/root/harness")
    .add_local_dir("submodule/TileKernels", remote_path=_TK_PATH)
)


@app.function(image=_image, gpu=GPU_TYPE, timeout=600)
def run_pytest(module: str, benchmark: bool, workers: int) -> tuple[int, str]:
    import subprocess
    import os

    env = {**os.environ, "TILELANG_PRINT_ON_COMPILATION": "0"}

    cmd = ["pytest", f"{_TK_PATH}/tests/{module}", "-v", "--tb=short"]
    if benchmark:
        cmd.append("--run-benchmark")
    else:
        cmd += ["-n", str(workers)]

    r = subprocess.run(cmd, capture_output=True, text=True, cwd=_TK_PATH, env=env)
    return r.returncode, r.stdout + r.stderr


@app.local_entrypoint()
def main(module: str = "transpose", benchmark: bool = False, workers: int = 4):
    print(f"module: {module}  benchmark: {benchmark}  workers: {workers}\n")
    returncode, output = run_pytest.remote(module, benchmark, workers)
    print(output)
    if returncode != 0:
        raise SystemExit(returncode)
