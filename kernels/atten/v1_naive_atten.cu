/*
nvcc -g -G -O0 -o temp/atten_debug kernels/atten/v1_naive_atten.cu
*/
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void qk_matmul(const float *Q, const float *K, float *output,
                          const int64_t batch, const int heads, const int seq,
                          const int64_t dimension) {

  int64_t b = blockIdx.x / heads;
  int64_t h = blockIdx.x % heads;
  int64_t i = blockIdx.y;
  int64_t j = blockIdx.z;
  float scale = 1.0f / sqrtf((float)dimension);

  float acc = 0.0f;
  for (int64_t k = 0; k < dimension; k++) {
    acc += Q[b * heads * seq * dimension + h * seq * dimension + i * dimension +
             k] *
           K[b * heads * seq * dimension + h * seq * dimension + j * dimension +
             k];
  }

  output[b * heads * seq * seq + h * seq * seq + i * seq + j] = acc * scale;
}

__global__ void sv_matmul(const float *S, const float *V, float *output,
                          const int64_t batch, const int heads, const int seq,
                          const int64_t dimension) {

  int64_t b = blockIdx.x / heads;
  int64_t h = blockIdx.x % heads;
  int64_t i = blockIdx.y;
  int64_t j = blockIdx.z;

  float acc = 0.0f;
  for (int64_t k = 0; k < seq; k++) {
    acc += S[b * heads * seq * seq + h * seq * seq + i * seq + k] *
           V[b * heads * seq * dimension + h * seq * dimension + k * dimension +
             j];
  }

  output[b * heads * seq * dimension + h * seq * dimension + i * dimension +
         j] = acc;
}
__global__ void tiled_scan(const float *input, const int64_t rows,
                           const int cols, float *max_val, float *exp_sum) {
  __shared__ float s_max[256];
  __shared__ float s_sum[256];
  int64_t tid = threadIdx.x;
  int64_t row = blockIdx.x;
  int64_t pid = row * cols + tid;
  s_max[tid] = (tid < cols) ? input[pid] : -FLT_MAX;
  s_sum[tid] = (tid < cols) ? 1.0f : 0.0f;
  __syncthreads();

  for (int64_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float m_new = fmaxf(s_max[tid], s_max[tid + stride]);
      s_sum[tid] = s_sum[tid] * expf(s_max[tid] - m_new) +
                   s_sum[tid + stride] * expf(s_max[tid + stride] - m_new);
      s_max[tid] = m_new;
    }
    __syncthreads();
  }
  if (tid == 0) {
    max_val[blockIdx.x] = s_max[0];
    exp_sum[blockIdx.x] = s_sum[0];
  }
}

__global__ void tiled_softmax(const float *input, const int64_t rows,
                              const int64_t cols, float *max_val,
                              float *exp_sum, float *output) {
  int64_t tid = threadIdx.x;
  int64_t row = blockIdx.x;
  int64_t pid = row * cols + tid;
  if (tid < cols) {
    output[pid] = expf(input[pid] - max_val[row]) / exp_sum[row];
  }
}
int main() {

  int64_t batch = 1, heads = 1, seq = 4, d = 4;
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

  float *d_q, *d_k, *d_v, *d_s, *d_o, *d_r, *max_val, *exp_sum;
  int64_t softmax_rows = batch * heads * seq;
  cudaMalloc(&d_q, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&d_k, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&d_v, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&d_s, batch * heads * seq * seq * sizeof(float));
  cudaMalloc(&d_o, batch * heads * seq * seq * sizeof(float));
  cudaMalloc(&d_r, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&max_val, softmax_rows * sizeof(float));
  cudaMalloc(&exp_sum, softmax_rows * sizeof(float));
  cudaMemcpy(d_q, h_q, batch * heads * seq * d * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k, batch * heads * seq * d * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, batch * heads * seq * d * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 grid(batch * heads, seq, seq);
  dim3 block(1, 1, 1);
  qk_matmul<<<grid, block>>>(d_q, d_k, d_s, batch, heads, seq, d);
  cudaDeviceSynchronize();
  tiled_scan<<<softmax_rows, 256>>>(d_s, softmax_rows, seq, max_val, exp_sum);
  cudaDeviceSynchronize();
  tiled_softmax<<<softmax_rows, 256>>>(d_s, softmax_rows, seq, max_val, exp_sum,
                                       d_o);
  cudaDeviceSynchronize();
  dim3 sv_grid(batch * heads, seq, d);
  sv_matmul<<<sv_grid, block>>>(d_o, d_v, d_r, batch, heads, seq, d);
  cudaDeviceSynchronize();
  cudaMemcpy(output, d_r, batch * heads * seq * d * sizeof(float),
             cudaMemcpyDeviceToHost);
  // printf("%f\n", output[0]);
  // printf("%f\n", output[1]);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_s);
  cudaFree(d_o);
  cudaFree(d_r);
  cudaFree(max_val);
  cudaFree(exp_sum);

  return 0;
}

extern "C" void atten_launch(const float *q, const float *k, const float *v,
                             float *result, const int64_t batch,
                             const int heads, const int64_t seq, const int d) {
  float *d_q, *d_k, *d_v, *d_s, *d_o, *d_r, *max_val, *exp_sum;
  int64_t softmax_rows = batch * heads * seq;
  cudaMalloc(&d_q, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&d_k, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&d_v, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&d_s, batch * heads * seq * seq * sizeof(float));
  cudaMalloc(&d_o, batch * heads * seq * seq * sizeof(float));
  cudaMalloc(&d_r, batch * heads * seq * d * sizeof(float));
  cudaMalloc(&max_val, softmax_rows * sizeof(float));
  cudaMalloc(&exp_sum, softmax_rows * sizeof(float));
  cudaMemcpy(d_q, q, batch * heads * seq * d * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, k, batch * heads * seq * d * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, batch * heads * seq * d * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 grid(batch * heads, seq, seq);
  dim3 block(1, 1, 1);
  qk_matmul<<<grid, block>>>(d_q, d_k, d_s, batch, heads, seq, d);
  cudaDeviceSynchronize();
  tiled_scan<<<softmax_rows, 256>>>(d_s, softmax_rows, seq, max_val, exp_sum);
  cudaDeviceSynchronize();
  tiled_softmax<<<softmax_rows, 256>>>(d_s, softmax_rows, seq, max_val, exp_sum,
                                       d_o);
  cudaDeviceSynchronize();
  dim3 sv_grid(batch * heads, seq, d);
  sv_matmul<<<sv_grid, block>>>(d_o, d_v, d_r, batch, heads, seq, d);
  cudaDeviceSynchronize();
  cudaMemcpy(result, d_r, batch * heads * seq * d * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_s);
  cudaFree(d_o);
  cudaFree(d_r);
  cudaFree(max_val);
  cudaFree(exp_sum);
}

static void run_kernels(const float *d_q, const float *d_k, const float *d_v,
                        float *d_s, float *d_o, float *d_r, float *max_val,
                        float *exp_sum, int64_t batch, int heads, int seq,
                        int d) {
  int64_t softmax_rows = batch * heads * seq;
  dim3 grid(batch * heads, seq, seq), block(1, 1, 1);
  qk_matmul<<<grid, block>>>(d_q, d_k, d_s, batch, heads, seq, d);
  tiled_scan<<<softmax_rows, 256>>>(d_s, softmax_rows, seq, max_val, exp_sum);
  tiled_softmax<<<softmax_rows, 256>>>(d_s, softmax_rows, seq, max_val, exp_sum,
                                       d_o);
  dim3 sv_grid(batch * heads, seq, d);
  sv_matmul<<<sv_grid, block>>>(d_o, d_v, d_r, batch, heads, seq, d);
}

// q, k, v, result must be device pointers (allocated by caller)
extern "C" float benchmark_launch(const float *q, const float *k,
                                  const float *v, float *result,
                                  const int64_t batch, const int heads,
                                  const int64_t seq, const int d, int warmup,
                                  int64_t iters) {
  int64_t softmax_rows = batch * heads * seq;
  float *d_s, *d_o, *max_val, *exp_sum;
  cudaMalloc(&d_s, batch * heads * seq * seq * sizeof(float));
  cudaMalloc(&d_o, batch * heads * seq * seq * sizeof(float));
  cudaMalloc(&max_val, softmax_rows * sizeof(float));
  cudaMalloc(&exp_sum, softmax_rows * sizeof(float));

  for (int64_t i = 0; i < warmup; i++)
    run_kernels(q, k, v, d_s, d_o, result, max_val, exp_sum, batch, heads, seq,
                d);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int64_t i = 0; i < iters; i++)
    run_kernels(q, k, v, d_s, d_o, result, max_val, exp_sum, batch, heads, seq,
                d);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_s);
  cudaFree(d_o);
  cudaFree(max_val);
  cudaFree(exp_sum);
  return ms / iters;
}
