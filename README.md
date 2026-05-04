# cuda-playground

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install modal numpy
pip install torch --index-url https://download.pytorch.org/whl/cu130

modal setup
```

## Local benchmark (all versions, TFLOP/s)
```bash
python -m kernels.atten.run_local     # prints per-version TFLOP/s table
```

## Remote profiling on H200 (Modal)
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

## Local debug binary (cuda-gdb)
```bash
nvcc -g -G -O0 -o temp/atten_debug kernels/atten/v3_tiled_fa2.cu
cuda-gdb ./temp/atten_debug
```
