/*
nvcc -g -G -O0 -o temp/atten_debug kernels/atten/v5_wmma_fa2.cu
*/
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda::wmma;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define Br 64
#define Bc 64
#define D 128

__global__ void fa2_kernel(const __half *Q, const __half *K, const __half *V,
                           __half *output, const int64_t batch,
                           const int64_t heads, const int64_t seq,
                           const int64_t dimension, const float scale) {

  __shared__ __half s_K[Bc][D];
  __shared__ __half s_V[Bc][D];
  __shared__ union {
    float f[Br][Bc];
    __half h[Br][Bc];
  } s_score;

  int64_t bh = blockIdx.x;
  int64_t b = bh / heads;
  int64_t h = bh % heads;
  int64_t stride_b = (int64_t)heads * seq * dimension;
  int64_t stride_h = (int64_t)seq * dimension;
  int64_t stride_s = (int64_t)dimension;

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  int64_t q_row_start = (int64_t)blockIdx.y * Br + warp_id * 16;
  bool warp_valid = (q_row_start < seq);
  // scalar per-thread state — each thread (lane_id < 16) owns one Q-row
  float m_state = -FLT_MAX, l_state = 0.0f;
  float o[D] = {};

  fragment<matrix_a, 16, 16, 16, __half, row_major> q_frag[D / 16];
  // all warps load q_frag uniformly — invalid warps use row 0 (safe), results
  // discarded
  {
    int64_t safe_row = warp_valid ? q_row_start : 0;
    const __half *q_base =
        Q + b * stride_b + h * stride_h + safe_row * stride_s;
#pragma unroll
    for (int i = 0; i < D / 16; i++)
      load_matrix_sync(q_frag[i], q_base + i * 16, D);
  }

  // declare outside j-loop so compiler keeps fragments in registers across
  // iterations
  fragment<accumulator, 16, 16, 16, float> s_frag[Bc / 16];

  for (int64_t j = 0; j < seq / Bc; j++) {
    for (int e = threadIdx.x; e < Bc * D; e += blockDim.x) {
      int r = e / D, c = e % D;
      int64_t kv = j * Bc + r;
      s_K[r][c] = kv < seq ? K[b * stride_b + h * stride_h + kv * stride_s + c]
                           : __half(0);
      s_V[r][c] = kv < seq ? V[b * stride_b + h * stride_h + kv * stride_s + c]
                           : __half(0);
    }
    __syncthreads();
    // all warps execute WMMA uniformly — invalid warps write garbage to their
    // s_score section
    {
      // QK^T: warp computes [16 x Bc] score block
#pragma unroll
      for (int kk = 0; kk < Bc / 16; kk++) {
        fill_fragment(s_frag[kk], 0.0f);
#pragma unroll
        for (int k = 0; k < D / 16; k++) {
          fragment<matrix_b, 16, 16, 16, __half, col_major> k_frag;
          load_matrix_sync(k_frag, &s_K[kk * 16][k * 16], D);
          mma_sync(s_frag[kk], q_frag[k], k_frag, s_frag[kk]);
        }
        store_matrix_sync(&s_score.f[warp_id * 16][kk * 16], s_frag[kk], Bc,
                          mem_row_major);
      }
      __syncwarp();
    }

    // softmax + PV: only valid lanes (scalar, no WMMA divergence issue)
    if (lane_id < 16 && (q_row_start + lane_id) < seq) {
      int ri = lane_id;
      float m_new = m_state;
#pragma unroll
      for (int c = 0; c < Bc; c++)
        if (j * Bc + c < seq)
          m_new = fmaxf(m_new, s_score.f[warp_id * 16 + ri][c] * scale);

      float rescale = __expf(m_state - m_new), sum = 0.f;
#pragma unroll
      for (int k = 0; k < D; k++)
        o[k] *= rescale;
// fuse softmax + PV: keep p as float to avoid half round-trip precision loss
#pragma unroll
      for (int c = 0; c < Bc; c++) {
        float p = (j * Bc + c < seq)
                      ? __expf(s_score.f[warp_id * 16 + ri][c] * scale - m_new)
                      : 0.f;
        sum += p;
#pragma unroll
        for (int k = 0; k < D; k++)
          o[k] += p * __half2float(s_V[c][k]);
      }
      m_state = m_new;
      l_state = l_state * rescale + sum;
    }

    __syncthreads();
  }
  // load back with denominator
  if (lane_id < 16 && (q_row_start + lane_id) < seq) {
    int ri = lane_id;
#pragma unroll
    for (int k = 0; k < D; k++)
      output[b * stride_b + h * stride_h + (q_row_start + ri) * stride_s + k] =
          __float2half(o[k] / l_state);
  }
}

int main() {

  int64_t batch = 2, heads = 16, seq = 1024, d = 128;
  float scale = 1.0f / sqrt(float(d));
  __half *h_q = (__half *)malloc(batch * heads * seq * d * sizeof(__half));
  __half *h_k = (__half *)malloc(batch * heads * seq * d * sizeof(__half));
  __half *h_v = (__half *)malloc(batch * heads * seq * d * sizeof(__half));
  __half *output = (__half *)malloc(batch * heads * seq * d * sizeof(__half));
  for (int64_t i = 0; i < batch; i++) {
    for (int64_t j = 0; j < heads; j++) {
      for (int64_t k = 0; k < seq; k++) {
        for (int64_t l = 0; l < d; l++) {
          h_q[i * heads * seq * d + j * seq * d + k * d + l] = __half(1);
          h_k[i * heads * seq * d + j * seq * d + k * d + l] = __half(1);
          h_v[i * heads * seq * d + j * seq * d + k * d + l] = __half(1);
        }
      }
    }
  }

  __half *d_q, *d_k, *d_v, *d_r;
  cudaMalloc(&d_q, batch * heads * seq * d * sizeof(__half));
  cudaMalloc(&d_k, batch * heads * seq * d * sizeof(__half));
  cudaMalloc(&d_v, batch * heads * seq * d * sizeof(__half));
  cudaMalloc(&d_r, batch * heads * seq * d * sizeof(__half));
  cudaMemcpy(d_q, h_q, batch * heads * seq * d * sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k, batch * heads * seq * d * sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, batch * heads * seq * d * sizeof(__half),
             cudaMemcpyHostToDevice);

  dim3 grid(batch * heads, CEIL_DIV(seq, Br));
  dim3 block(Br * 2);
  fa2_kernel<<<grid, block>>>(d_q, d_k, d_v, d_r, batch, heads, seq, d, scale);
  cudaDeviceSynchronize();
  cudaMemcpy(output, d_r, batch * heads * seq * d * sizeof(__half),
             cudaMemcpyDeviceToHost);
  printf("%f\n", __half2float(output[0]));
  printf("%f\n", __half2float(output[1]));
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_r);

  return 0;
}

extern "C" void atten_launch(const __half *q, const __half *k, const __half *v,
                             __half *result, const int64_t batch,
                             const int64_t heads, const int64_t seq,
                             const int64_t d) {
  __half *d_q, *d_k, *d_v, *d_r;
  float scale = 1.0f / sqrt(float(d));
  cudaMalloc(&d_q, batch * heads * seq * d * sizeof(__half));
  cudaMalloc(&d_k, batch * heads * seq * d * sizeof(__half));
  cudaMalloc(&d_v, batch * heads * seq * d * sizeof(__half));
  cudaMalloc(&d_r, batch * heads * seq * d * sizeof(__half));
  cudaMemcpy(d_q, q, batch * heads * seq * d * sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, k, batch * heads * seq * d * sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, batch * heads * seq * d * sizeof(__half),
             cudaMemcpyHostToDevice);
  dim3 grid(batch * heads, CEIL_DIV(seq, Br));
  dim3 block(Br * 2);
  fa2_kernel<<<grid, block>>>(d_q, d_k, d_v, d_r, batch, heads, seq, d, scale);
  cudaDeviceSynchronize();
  cudaMemcpy(result, d_r, batch * heads * seq * d * sizeof(__half),
             cudaMemcpyDeviceToHost);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_r);
}

static void run_kernels(const __half *d_q, const __half *d_k, const __half *d_v,
                        __half *d_r, int64_t batch, int64_t heads, int64_t seq,
                        int64_t d) {
  float scale = 1.0f / sqrtf((float)d);
  dim3 grid(batch * heads, CEIL_DIV(seq, Br));
  dim3 block(Br * 2);
  fa2_kernel<<<grid, block>>>(d_q, d_k, d_v, d_r, batch, heads, seq, d, scale);
}

// q, k, v, result must be device pointers (allocated by caller)
extern "C" float benchmark_launch(const __half *q, const __half *k,
                                  const __half *v, __half *result,
                                  const int64_t batch, const int64_t heads,
                                  const int64_t seq, const int64_t d,
                                  int64_t warmup, int64_t iters) {
  for (int64_t i = 0; i < warmup; i++)
    run_kernels(q, k, v, result, batch, heads, seq, d);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int64_t i = 0; i < iters; i++)
    run_kernels(q, k, v, result, batch, heads, seq, d);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return (float)(ms / iters);
}
