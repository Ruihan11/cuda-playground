import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass import Float16, Float32

LOG2E = Float32(1.442695041)  # log2(e), converts natural log to exp2


class FlashAttention:
    def __init__(self, tile_m=64, tile_n=64, tile_d=128, num_threads=128):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.num_threads = num_threads

    def run(self, Q, K, V, output, scale):
        b, s, h, d = Q.shape

        def prep(t):
            # (b, s, h, d) → (s, d, b*h): batch*head last matches quack's indexing convention
            return t.permute(1, 3, 0, 2).reshape(s, d, b * h).contiguous()

        def prep_v(t):
            # (b, s, h, d) → (d, s, b*h): V stored transposed so partition_B sees (N=d, K=s)
            return t.permute(3, 1, 0, 2).reshape(d, s, b * h).contiguous()

        O_buf = torch.zeros(s, d, b * h, dtype=output.dtype, device=output.device)
        grid = (s // self.tile_m, b * h, 1)
        self._launch(
            from_dlpack(prep(Q)),
            from_dlpack(prep(K)),
            from_dlpack(prep_v(V)),
            from_dlpack(O_buf),
            Float32(scale),
            grid,
        )
        output.copy_(O_buf.reshape(s, d, b, h).permute(2, 0, 3, 1))

    @cute.jit
    def _launch(self, Q, K, V, output, scale, grid):
        from cutlass.cute.nvgpu import warp

        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(Float16, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(Float16, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, self.tile_d, 16),
        )

        self.kernel(Q, K, V, output, scale, tiled_mma_qk, tiled_mma_pv).launch(
            grid=grid, block=[self.num_threads, 1, 1]
        )

    @cute.kernel
    def kernel(self, mQ, mK, mV, mO, scale, tiled_mma_qk, tiled_mma_pv):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # mQ/K/V/O: (s, d, b*h); select this CTA's batch*head via last-dim index
        gQ_2d = mQ[None, None, bidy]
        gK_2d = mK[None, None, bidy]
        gV_2d = mV[None, None, bidy]
        gO_2d = mO[None, None, bidy]

        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(
            mQ.element_type,
            cute.make_ordered_layout((self.tile_m, self.tile_d), order=(1, 0)),
            byte_alignment=16,
        )
        sK = smem.allocate_tensor(
            mK.element_type,
            cute.make_ordered_layout((self.tile_n, self.tile_d), order=(1, 0)),
            byte_alignment=16,
        )
        sV = smem.allocate_tensor(
            mV.element_type,
            cute.make_ordered_layout((self.tile_d, self.tile_n), order=(1, 0)),
            byte_alignment=16,
        )
        # sP: staging buffer for softmax output (C-layout → smem → A-layout for PV gemm)
        sP = smem.allocate_tensor(
            mQ.element_type,
            cute.make_ordered_layout((self.tile_m, self.tile_n), order=(0, 1)),
            byte_alignment=16,
        )

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mQ.element_type)
        tiled_copy = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_ordered_layout((32, 4), order=(1, 0)),
            cute.make_layout((1, 8)),
        )
        thr_copy = tiled_copy.get_slice(tidx)

        # Load Q tile once (reused across n-loop)
        gQ = cute.local_tile(gQ_2d, (self.tile_m, self.tile_d), (bidx, 0))
        cute.copy(tiled_copy, thr_copy.partition_S(gQ), thr_copy.partition_D(sQ))
        cute.arch.sync_threads()

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)

        smem_copy_A_qk = cute.make_tiled_copy_A(copy_atom, tiled_mma_qk).get_slice(tidx)
        smem_copy_B_qk = cute.make_tiled_copy_B(copy_atom, tiled_mma_qk).get_slice(tidx)
        smem_copy_A_pv = cute.make_tiled_copy_A(copy_atom, tiled_mma_pv).get_slice(tidx)
        smem_copy_B_pv = cute.make_tiled_copy_B(copy_atom, tiled_mma_pv).get_slice(tidx)

        acc_qk = cute.make_rmem_tensor(
            thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )
        acc_o = cute.make_rmem_tensor(
            thr_mma_pv.partition_shape_C((self.tile_m, self.tile_d)), Float32
        )
        acc_o.fill(0.0)

        # Load Q smem → registers one K-slice at a time; reused across n-loop
        tCsQ = thr_mma_qk.partition_A(sQ)
        tCrQ = thr_mma_qk.make_fragment_A(tCsQ)
        tCrQ_view = smem_copy_A_qk.retile(tCrQ)
        for k in cutlass.range_constexpr(cute.size(tCsQ.shape[2])):
            cute.copy(smem_copy_A_qk, tCsQ[None, None, k], tCrQ_view[None, None, k])

        # Online softmax state: one entry per M-fragment-row this thread owns
        n_m = cute.size(acc_qk, mode=[0])
        n_qk = cute.size(acc_qk) // n_m  # N elements per M-row in acc_qk
        n_o = cute.size(acc_o) // n_m  # D elements per M-row in acc_o

        row_max = cute.make_rmem_tensor((n_m,), Float32)
        row_sum = cute.make_rmem_tensor((n_m,), Float32)
        row_max.fill(Float32(-1e9))
        row_sum.fill(0.0)

        seq = mK.shape[0]  # mK is (s, d, b*h); shape[0]=s is the sequence length

        for n in cutlass.range(seq // self.tile_n):
            # ── Load K tile → sK ────────────────────────────────────────────
            gK = cute.local_tile(gK_2d, (self.tile_n, self.tile_d), (n, 0))
            cute.copy(tiled_copy, thr_copy.partition_S(gK), thr_copy.partition_D(sK))
            cute.arch.sync_threads()

            # ── QK^T gemm ───────────────────────────────────────────────────
            acc_qk.fill(0.0)
            tCsK = thr_mma_qk.partition_B(sK)
            tCrK = thr_mma_qk.make_fragment_B(tCsK)
            tCrK_view = smem_copy_B_qk.retile(tCrK)
            for k in cutlass.range_constexpr(cute.size(tCsK.shape[2])):
                cute.copy(smem_copy_B_qk, tCsK[None, None, k], tCrK_view[None, None, k])
            cute.gemm(tiled_mma_qk, acc_qk, tCrQ, tCrK, acc_qk)

            # Scale by attention scale factor
            for i in cutlass.range_constexpr(cute.size(acc_qk)):
                acc_qk[i] = acc_qk[i] * scale

            # ── Online softmax update ────────────────────────────────────────
            # For each M-fragment row: find new_max, rescale acc_o, compute exp
            for mi in cutlass.range_constexpr(n_m):
                # Local max over this thread's N-elements for this M-row
                local_max = acc_qk[mi * n_qk]
                for ni in cutlass.range_constexpr(n_qk):
                    local_max = cute.arch.fmax(local_max, acc_qk[mi * n_qk + ni])

                # Warp reduce: threads sharing the same M-row agree on max
                # threads_in_group=4: within SM80 MMA m16n8k16, 4 threads per N-column share a row
                local_max = cute.arch.warp_reduction_max(local_max, threads_in_group=4)

                new_max = cute.arch.fmax(row_max[mi], local_max)

                # Rescale previous acc_o and row_sum by exp(old_max - new_max)
                alpha = cute.math.exp2((row_max[mi] - new_max) * LOG2E, fastmath=True)
                row_sum[mi] = row_sum[mi] * alpha
                for oi in cutlass.range_constexpr(n_o):
                    acc_o[mi * n_o + oi] = acc_o[mi * n_o + oi] * alpha

                # Compute softmax numerators, accumulate local row_sum
                local_sum = Float32(0.0)
                for ni in cutlass.range_constexpr(n_qk):
                    e = cute.math.exp2(
                        (acc_qk[mi * n_qk + ni] - new_max) * LOG2E, fastmath=True
                    )
                    acc_qk[mi * n_qk + ni] = e
                    local_sum = local_sum + e

                # Warp reduce local_sum: same threads-in-group as max reduction
                local_sum = cute.arch.warp_reduction_sum(local_sum, threads_in_group=4)
                row_max[mi] = new_max
                row_sum[mi] = row_sum[mi] + local_sum

            # ── Write P to smem via C-layout so PV gemm can read as A-layout ──
            # C-layout ≠ A-layout; smem is the neutral ground between both.
            rP_fp16 = cute.make_rmem_tensor_like(acc_qk, Float16)
            rP_fp16.store(acc_qk.load().to(Float16))
            cute.autovec_copy(rP_fp16, thr_mma_qk.partition_C(sP))

            # ── Load V tile → sV (overlap with sP write, no sync yet) ───────
            gV = cute.local_tile(gV_2d, (self.tile_d, self.tile_n), (0, n))
            cute.copy(tiled_copy, thr_copy.partition_S(gV), thr_copy.partition_D(sV))
            cute.arch.sync_threads()

            # ── PV gemm: load P and V smem → registers ──────────────────────
            tCsP = thr_mma_pv.partition_A(sP)
            tCrP = thr_mma_pv.make_fragment_A(tCsP)
            tCrP_view = smem_copy_A_pv.retile(tCrP)
            for k in cutlass.range_constexpr(cute.size(tCsP.shape[2])):
                cute.copy(smem_copy_A_pv, tCsP[None, None, k], tCrP_view[None, None, k])

            tCsV = thr_mma_pv.partition_B(sV)
            tCrV = thr_mma_pv.make_fragment_B(tCsV)
            tCrV_view = smem_copy_B_pv.retile(tCrV)
            for k in cutlass.range_constexpr(cute.size(tCsV.shape[2])):
                cute.copy(smem_copy_B_pv, tCsV[None, None, k], tCrV_view[None, None, k])

            cute.gemm(tiled_mma_pv, acc_o, tCrP, tCrV, acc_o)

        # ── Normalize acc_o by row_sum ───────────────────────────────────────
        for mi in cutlass.range_constexpr(n_m):
            inv = Float32(1.0) / row_sum[mi]
            for oi in cutlass.range_constexpr(n_o):
                acc_o[mi * n_o + oi] = acc_o[mi * n_o + oi] * inv

        # ── Write O: fp32 acc_o → fp16 in registers → global memory ──
        rO = cute.make_rmem_tensor_like(acc_o, Float16)
        rO.store(acc_o.load().to(Float16))

        gO = cute.local_tile(gO_2d, (self.tile_m, self.tile_d), (bidx, 0))
        # partition_C maps this thread's rO elements to the correct gO positions
        cute.autovec_copy(rO, thr_mma_pv.partition_C(gO))


if __name__ == "__main__":
    B, S, H, D = 2, 512, 16, 128
    Q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    K = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    V = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    output = torch.zeros(B, S, H, D, dtype=torch.float16, device="cuda")

    fa = FlashAttention()
    fa.run(Q, K, V, output, scale=D**-0.5)
    print("DONE")
    print(output[0, 0, 0, :8])
