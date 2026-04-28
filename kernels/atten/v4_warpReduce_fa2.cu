/*
nvcc -g -G -O0 -o temp/atten_debug kernels/atten/v4_warpReduce_fa2.cu
*/
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define Br 32
#define Bc 32
#define D 64

__global__ void fa2_kernel(const float *Q, const float *K, const float *V,
                           float *output, const int64_t batch,
                           const int64_t heads, const int64_t seq,
                           const int64_t dimension, const float scale) {

  __shared__ float s_K[Bc][D];
  __shared__ float s_V[Bc][D];

  int64_t row = blockIdx.y;
  int64_t bh = blockIdx.x;
  int64_t b = bh / heads;
  int64_t stride_b = (int64_t)heads * seq * dimension;
  int64_t stride_h = (int64_t)seq * dimension;
  int64_t stride_s = (int64_t)dimension;

  int64_t h = bh % heads;

  float m = -FLT_MAX;
  float l = 0.0f;
  bool valid = (row < seq);
  float o0 = 0.0f, o1 = 0.0f;

  float q_reg[2];
  q_reg[0] = Q[b * stride_b + h * stride_h + row * stride_s + threadIdx.x];
  q_reg[1] = Q[b * stride_b + h * stride_h + row * stride_s + threadIdx.x + 32];

  for (int64_t j = 0; j < seq / Bc; j++) {

    for (int64_t r = 0; r < Bc; r++) {
      int64_t actual_kv = j * Bc + r;
      for (int c = threadIdx.x; c < D; c += 32) {
        s_K[r][c] =
            actual_kv < seq
                ? K[b * stride_b + h * stride_h + actual_kv * stride_s + c]
                : 0.0f;
        s_V[r][c] =
            actual_kv < seq
                ? V[b * stride_b + h * stride_h + actual_kv * stride_s + c]
                : 0.0f;
      }
    }
    __syncthreads();
    if (valid) {
      for (int64_t jj = 0; jj < Bc; jj++) {
        if (j * Bc + jj >= seq)
          break;

        float partial = q_reg[0] * s_K[jj][threadIdx.x] +
                        q_reg[1] * s_K[jj][threadIdx.x + 32];
        for (int mask = 16; mask > 0; mask >>= 1)
          partial += __shfl_xor_sync(0xffffffff, partial, mask);

        float acc = partial * scale;
        acc = __shfl_sync(0xffffffff, acc, 0);

        // new maximum and denominator
        float m_new = fmaxf(m, acc);
        float decay = expf(m - m_new), weight = expf(acc - m_new);
        float l_new = l * decay + weight;

        // O_i^new = diag(e^(m_i - m_i_new)) * O_i + e^(S_ij - m_i_new) @ V_j
        o0 = decay * o0 + weight * s_V[jj][threadIdx.x];
        o1 = decay * o1 + weight * s_V[jj][threadIdx.x + 32];
        m = m_new;
        l = l_new;
      }
    }
    __syncthreads();
  }
  // load back with denominator
  if (valid) {
    output[b * stride_b + h * stride_h + row * stride_s + threadIdx.x] = o0 / l;
    output[b * stride_b + h * stride_h + row * stride_s + threadIdx.x + 32] =
        o1 / l;
  }
}

int main() {

  int64_t batch = 1, heads = 1, seq = 32, d = 64;
  float scale = 1.0f / sqrt(float(d));
  float *h_q = (float *)malloc(batch * heads * seq * d * sizeof(float));
  float *h_k = (float *)malloc(batch * heads * seq * d * sizeof(float));
  float *h_v = (float *)malloc(batch * heads * seq * d * sizeof(float));
  float *output = (float *)malloc(batch * heads * seq * d * sizeof(float));
  for (int64_t i = 0; i < batch; i++) {
    for (int64_t j = 0; j < heads; j++) {
      for (int64_t k = 0; k < seq; k++) {
        for (int64_t l = 0; l < d; l++) {
          h_q[i * heads * seq * d + j * seq * d + k * d + l] = float(1);
          h_k[i * heads * seq * d + j * seq * d + k * d + l] = float(1);
          h_v[i * heads * seq * d + j * seq * d + k * d + l] = float(1);
        }
      }
    }
  }

  float *d_q, *d_k, *d_v, *d_r;
  cudaMalloc(&d_q, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&d_k, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&d_v, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&d_r, batch * heads * seq * d * sizeof(float));
  cudaMemcpy(d_q, h_q, batch * heads * seq * d * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k, batch * heads * seq * d * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, batch * heads * seq * d * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 grid(batch * heads, seq);
  dim3 block(32);
  fa2_kernel<<<grid, block>>>(d_q, d_k, d_v, d_r, batch, heads, seq, d, scale);
  cudaDeviceSynchronize();
  cudaMemcpy(output, d_r, batch * heads * seq * d * sizeof(float),
             cudaMemcpyDeviceToHost);
  // printf("%f\n", output[0]);
  // printf("%f\n", output[1]);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_r);

  return 0;
}

extern "C" void atten_launch(const float *q, const float *k, const float *v,
                             float *result, const int64_t batch,
                             const int64_t heads, const int64_t seq,
                             const int64_t d) {
  float *d_q, *d_k, *d_v, *d_r;
  float scale = 1.0f / sqrt(float(d));
  cudaMalloc(&d_q, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&d_k, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&d_v, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&d_r, batch * heads * seq * d * sizeof(float));
  cudaMemcpy(d_q, q, batch * heads * seq * d * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, k, batch * heads * seq * d * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, batch * heads * seq * d * sizeof(float),
             cudaMemcpyHostToDevice);
  dim3 grid(batch * heads, seq);
  dim3 block(32);
  fa2_kernel<<<grid, block>>>(d_q, d_k, d_v, d_r, batch, heads, seq, d, scale);
  cudaDeviceSynchronize();
  cudaMemcpy(result, d_r, batch * heads * seq * d * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_r);
}

static void run_kernels(const float *d_q, const float *d_k, const float *d_v,
                        float *d_r, int64_t batch, int64_t heads, int64_t seq,
                        int64_t d) {
  float scale = 1.0f / sqrtf((float)d);
  dim3 grid(batch * heads, seq);
  dim3 block(32);
  fa2_kernel<<<grid, block>>>(d_q, d_k, d_v, d_r, batch, heads, seq, d, scale);
}

// q, k, v, result must be device pointers (allocated by caller)
extern "C" float benchmark_launch(const float *q, const float *k,
                                  const float *v, float *result,
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
  return ms / iters;
}
