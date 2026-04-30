# Progress

## Learning Path

**Phase 1 — Hand-rolled fundamentals (done):**
vecsum -> softmax (v1->v2->v3) -> naive attention -> hand-rolled FA2 (v3/v4/v5)

**Phase 2 — Source code study (current):**
FA2 paper + source -> FA3 (Hopper WGMMA/TMA) -> FA4

Rationale: hand-rolling past v5 has diminishing returns. Real FA2 needs `cp.async`
pipelining, larger warp tiles, and mask handling that take weeks per step to get right.
Reading and annotating production source (Tri Dao's flash-attention, FlashInfer) gives
direct exposure to the design decisions, with selective re-implementation of small
pieces for understanding.

---

## Kernels (Phase 1, hand-rolled)

### vecsum — complete
- `v1_naive.cu` — 1 kernel, `<<<CEIL_DIV(n,block), block>>>`, global memory only

### softmax — complete
- `v1_naive_softmax.cu` — 3-pass single-thread baseline
- `v2_online_softmax.cu` — 2-pass online `(m,s)` joint update
- `v3_parallel_softmax.cu` — parallel reduction + shared memory + tree reduce
- `v4_tiled_softmax.cu` — tiled / block-level

### attention — complete (Phase 1 endpoint)
- `v1_naive_atten.cu` — direct QK^T then softmax then PV, fp32
- `v2_fa2.cu` — FA2 algorithm, single-thread per row, fp32
- `v3_tiled_fa2.cu` — tiled FA2, online softmax, shared memory, fp32
- `v4_half_fa2.cu` — fp16 storage + fp32 compute, float4 vectorized loads
- `v5_wmma_fa2.cu` — Tensor Core WMMA for QK^T, scalar PV

H200 results (TF/s, b=2 h=16 d=128):

| seq  | v4_half | v5_wmma | sdpa (FA) |
|------|---------|---------|-----------|
| 512  | 3.16    | 6.24    | 219.14    |
| 1024 | 3.80    | 6.35    | 317.92    |
| 2048 | 3.41    | 5.85    | 340.06    |
| 4096 | 3.86    | 5.76    | 350.34    |

Observations:
- v5 is 2x v4 — WMMA raises arithmetic intensity beyond L2 bandwidth bound
- v5 drops at seq=2048+ — KV exceeds H200 L2 (~50MB), DRAM bound kicks in
- ~50x gap to sdpa — missing async pipeline, larger tiles, PV WMMA

---

## Phase 2 plan — Source code study

### Stage 1: FA2 paper + reference implementation
- [ ] Re-read FA2 paper Algorithm 1, hand-derive online softmax rescale
- [ ] Read `flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h` (forward only)
- [ ] Annotate: tile sizes, warp partition, `cp.async` stages, softmax fusion
- [ ] Map each section back to v5 — identify the gap

### Stage 2: CUTLASS primitives (as needed)
- [ ] Skim CUTLASS `Mma`, `Copy`, `Pipeline` abstractions used by FA2
- [ ] Goal: read FA source fluently, not write CUTLASS GEMM from scratch

### Stage 3: FA3 (Hopper-specific, matches H200 target)
- [ ] WGMMA (warpgroup MMA) vs WMMA — what changed
- [ ] TMA (tensor memory accelerator) — async bulk copy
- [ ] Ping-pong scheduling between warpgroups
- [ ] Read FA3 forward kernel, compare to FA2

### Stage 4: FA4
- [ ] Survey what FA4 changes (when released / available)

### Optional small reimplementations
- v6: add `cp.async` pipeline to v5 (pure CUDA, no CUTLASS)
- v7: PV WMMA on top of v6
- These are exercises, not the goal — the goal is reading comprehension

---

## Harness — complete

- `harness/app.py` — Modal app, H200 GPU, cuda:13.0.0 + nsight-compute apt
- `harness/run_ncu.py` — ncu profiling entrypoint
- `harness/run_nsys.py` — nsys timeline entrypoint
- `harness/run_bench.py` — H200 benchmark + sdpa baseline comparison

---

## Session Log

### 2026-04-19
- Harness scaffolded, Modal smoke test passed (ncu vecsum)
- Fixed: modal.Mount removed in Modal 1.x -> `add_local_dir`; tomllib on Python 3.10

### 2026-04-23
- vecsum v1, softmax v1/v2 done; ctypes correctness PASS

### 2026-04-29
- atten v1-v5 done; H200 benchmark shows v5 = 2x v4, both ~50x behind sdpa
- Hardware target shifted: B200 -> H200 (Modal) + RTX5090 (local)
- v5 NCU on v4 confirmed L2 bandwidth bound; WMMA in v5 broke that ceiling
- Decision: pivot from hand-rolling to source code study
  - v6/v7 (cp.async pipeline + PV WMMA) deferred as optional exercises
  - Phase 2 starts with FA2 paper + Tri Dao source reading
