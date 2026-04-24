# Progress

## Learning Path

vecsum → softmax (v1→v2→v3) → naive attention → FlashAttention 2/3/4

---

## Kernels

### vecsum ✅ complete
- `v1_naive.cu` — 1 kernel, `<<<CEIL_DIV(n,block), block>>>`, global memory only
- `ref_kernel.py` — torch reference + ctypes correctness test (PASS)

### softmax ⬜ in progress
- `v1_naive.cu` ✅ — 3-pass (`find_max`, `compute_expsum`, `softmax_naive`), `<<<1,1>>>`, single thread
- `v2_online.cu` ✅ — 2-pass, online `(m,s)` joint update, `<<<1,1>>>`
- `v3_parallel.cu` ⬜ — 2-pass, parallel reduction with shared memory + tree reduce, next up
- `ref_kernel.py` — torch.softmax reference + ctypes test (PASS on v1, v2)

### naive attention ⬜ not started
### FlashAttention 2 ⬜ not started

---

## Harness

- `harness/app.py` ✅ — Modal app, image (cuda:13.0.0 + nsight-compute-2026.1.0 + nsys dynamic apt), volume
- `harness/run_ncu.py` ✅ — ncu profiling entrypoint, `add_local_dir` mount, glob ncu binary
- `harness/run_nsys.py` ✅ — nsys timeline entrypoint (nsys apt install dynamic, not yet smoke tested)
- Modal smoke test (ncu on vecsum) ✅ passed, `.ncu-rep` confirmed

---

## Session Log

### 2026-04-19
- Harness scaffolded, Modal smoke test passed on B200 (ncu vecsum)
- Fixed: modal.Mount removed in Modal 1.x → `add_local_dir`; tomllib import on local Python 3.10; nsys dynamic apt search

### 2026-04-23
- Pivoted kernel target: reduction → vecsum (simpler warmup)
- vecsum v1_naive: fixed 3 bugs (CEIL_DIV, n vs block, printf), ctypes test PASS
- softmax v1_naive: 3-pass single-thread, ctypes test PASS
- softmax v2_online: online (m,s) scan, ctypes test PASS
- Next: softmax v3_parallel (tree reduction + shared memory)
