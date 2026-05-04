# PROJECT_STATE.md

## Current Focus

**TileKernel** — getting it running locally (RTX5090, sm_120).

FA4 source study is paused; resume after TileKernel is working.

---

## Phase 2 Queue

| Task | Status | Notes |
|------|--------|-------|
| TileKernel — run locally | 🔄 active | starting now |
| FA4 source study (sm_90) | ⏳ paused | resume after TileKernel |
| FA4 Blackwell (sm_120) | ⏳ blocked | depends on FA4 sm_90 |

---

## FA4 Study Checkpoint (paused 2026-05-04)

Reading order when resuming:
1. `flash_fwd_sm90.py` — main loop, focus on TMA issue + WGMMA gemm calls
2. `pipeline.py` — barrier mechanics
3. `flash_fwd_sm120.py` — Blackwell delta (UMMA, tmem)
