# cuda-playground

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install modal numpy
pip install torch --index-url https://download.pytorch.org/whl/cu130

modal setup
```

## Local correctness test (CPU-side, no Modal)
```bash
# compile kernel as shared lib, then run ref_kernel.py validation
nvcc -shared -Xcompiler -fPIC -O3 -o temp/atten.so    kernels/atten/v3_tiled_fa2.cu
nvcc -shared -Xcompiler -fPIC -O3 -o temp/softmax.so  kernels/softmax/v3_parallel_softmax.cu
nvcc -shared -Xcompiler -fPIC -O3 -o temp/vecsum.so   kernels/vecsum/v1_naive.cu

python kernels/atten/ref_kernel.py
python kernels/softmax/ref_kernel.py
python kernels/vecsum/ref_kernel.py
```

## Local benchmark (all versions, TFLOP/s)
```bash
python kernels/atten/ref_kernel.py     # prints per-version TFLOP/s table
python kernels/softmax/ref_kernel.py   # prints per-version GB/s table
```

## Remote profiling on B200 (Modal)
```bash
# full profile — max data, source annotation, single launch
modal run harness/run_ncu.py --kernel atten   --versions v3_tiled_fa2                          --mode full
modal run harness/run_ncu.py --kernel atten   --versions v1_naive_atten,v2_fa2,v3_tiled_fa2,v4_warpReduce_fa2  --mode full
modal run harness/run_ncu.py --kernel softmax --versions v1_naive_softmax,v2_online_softmax,v3_parallel_softmax --mode full
modal run harness/run_ncu.py --kernel vecsum  --versions v1_naive                              --mode full

# quick profile — SpeedOfLight + MemoryWorkload + WarpStateStats only (~3x faster)
modal run harness/run_ncu.py --kernel atten --versions v3_tiled_fa2 --mode quick

# custom sections
modal run harness/run_ncu.py --kernel atten --versions v3_tiled_fa2 --mode custom --sections SpeedOfLight,Occupancy
```

## Fetch reports locally
```bash
modal volume ls kernel-profiling-reports
bash scripts/fetch_reports.sh          # syncs /reports to local reports/
```

## Local debug binary (cuda-gdb)
```bash
nvcc -g -G -O0 -o temp/atten_debug kernels/atten/v3_tiled_fa2.cu
cuda-gdb ./temp/atten_debug
```
