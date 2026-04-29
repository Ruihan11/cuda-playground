/*
nvcc -g -G -O0 -o temp/atten_debug kernels/atten/v4_half_fa2.cu
*/
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define Br 64
#define Bc 32
#define D 128

__global__ void fa2_kernel(const __half *Q, const __half *K, const __half *V,
                           __half *output, const int64_t batch,
                           const int64_t heads, const int64_t seq,
                           const int64_t dimension, const float scale) {

  __shared__ float s_K[Bc][D];
  __shared__ float s_V[Bc][D];

  int64_t row = blockIdx.y * Br + threadIdx.x;
  int64_t bh = blockIdx.x;
  int64_t b = bh / heads;
  int64_t stride_b = (int64_t)heads * seq * dimension;
  int64_t stride_h = (int64_t)seq * dimension;
  int64_t stride_s = (int64_t)dimension;

  int64_t h = bh % heads;

  float m = -FLT_MAX;
  float l = 0.0f;
  float o[D] = {0};
  bool valid = (row < seq);

  float q_cache[D];
  if (valid) {
    for (int k = 0; k < D; k++) {
      q_cache[k] =
          __half2float(Q[b * stride_b + h * stride_h + row * stride_s + k]);
    }
  }
  for (int64_t j = 0; j < seq / Bc; j++) {

    for (int elem = threadIdx.x; elem < Bc * D / 8; elem += blockDim.x) {
      int base = elem * 8, r = base / D, c = base % D;
      int64_t kv = j * Bc + r;
      uint4 raw_k, raw_v;
      memcpy(&raw_k, &K[b * stride_b + h * stride_h + kv * stride_s + c], 16);
      memcpy(&raw_v, &V[b * stride_b + h * stride_h + kv * stride_s + c], 16);

      if (kv < seq) {
        __half h8k[8], h8v[8];
        memcpy(h8k, &raw_k, 16);
        memcpy(h8v, &raw_v, 16);
        for (int i = 0; i < 8; i++)
          s_K[r][c + i] = __half2float(h8k[i]);
        for (int i = 0; i < 8; i++)
          s_V[r][c + i] = __half2float(h8v[i]);
      } else {
        for (int i = 0; i < 8; i++)
          s_K[r][c + i] = s_V[r][c + i] = 0.0f;
      }
    }

    __syncthreads();
    if (valid) {
      for (int64_t jj = 0; jj < Bc; jj++) {
        if (j * Bc + jj >= seq)
          break;

        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int64_t k = 0; k < D; k += 4) {
          acc0 += q_cache[k] * s_K[jj][k];
          acc1 += q_cache[k + 1] * s_K[jj][k + 1];
          acc2 += q_cache[k + 2] * s_K[jj][k + 2];
          acc3 += q_cache[k + 3] * s_K[jj][k + 3];
        }
        float acc = acc0 + acc1 + acc2 + acc3;
        acc *= scale;

        // new maximum and denominator
        float m_new = fmaxf(m, acc);
        float e_rescue = __expf(m - m_new);
        float e_new = __expf(acc - m_new);
        float l_new = l * e_rescue + e_new;

        // O_i^new = diag(e^(m_i - m_i_new)) * O_i + e^(S_ij - m_i_new) @ V_j
        for (int64_t k = 0; k < dimension; k++)
          o[k] = e_rescue * o[k] + e_new * s_V[jj][k];

        m = m_new;
        l = l_new;
      }
    }
    __syncthreads();
  }
  // load back with denominator
  if (valid) {
    for (int64_t k = 0; k < dimension; k++)
      output[b * stride_b + h * stride_h + row * stride_s + k] =
          __float2half(o[k] / l);
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
  dim3 block(Br);
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
  dim3 block(Br);
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
  dim3 block(Br);
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
