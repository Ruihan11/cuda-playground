/*
 * nvcc -g -G -O0 -o temp/softmax_debug kernels/softmax/v3_parallel_softmax.cu
 * */
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void parallel_scan(const float *input, float *max_val,
                              float *exp_sum, int n) {
  __shared__ float s_max[256];
  __shared__ float s_sum[256];
  int tid = threadIdx.x;
  int pid = blockIdx.x * blockDim.x + tid;
  s_max[tid] = (pid < n) ? input[pid] : -FLT_MAX;
  s_sum[tid] = (pid < n) ? 1.0f : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
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

__global__ void online_scan(const float *in_max, const float *in_sum,
                            float *max_val, float *exp_sum, int n) {
  float m = -FLT_MAX, s = 0.0f;
  for (int i = 0; i < n; i++) {
    float m_new = fmaxf(m, in_max[i]);
    s = s * expf(m - m_new) + in_sum[i] * expf(in_max[i] - m_new);
    m = m_new;
  }
  *max_val = m;
  *exp_sum = s;
}

__global__ void parallel_softmax(const float *input, float *max_val,
                                 float *exp_sum, float *output, int n) {
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid < n) {
    output[pid] = expf(input[pid] - *max_val) / *exp_sum;
  }
}

int main() {
  int n = 1000;
  size_t bytes = n * sizeof(float);
  float *d_a, *d_b, *d_c, *d_d;
  float *h_a = (float *)malloc(bytes);
  float *output = (float *)malloc(bytes);
  int block = 256;
  int grid = CEIL_DIV(n, block);
  for (int i = 0; i < n; i++)
    h_a[i] = float(i);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, grid * sizeof(float));
  cudaMalloc(&d_c, grid * sizeof(float));
  cudaMalloc(&d_d, bytes);
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

  parallel_scan<<<grid, block>>>(d_a, d_b, d_c, n);
  cudaDeviceSynchronize();
  float *d_max, *d_sum;
  cudaMalloc(&d_max, sizeof(float));
  cudaMalloc(&d_sum, sizeof(float));
  online_scan<<<1, 1>>>(d_b, d_c, d_max, d_sum, grid);
  cudaDeviceSynchronize();
  parallel_softmax<<<grid, block>>>(d_a, d_max, d_sum, d_d, n);
  cudaDeviceSynchronize();

  cudaMemcpy(output, d_d, bytes, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
    printf("%f\n", output[i]);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_d);

  return 0;
}

extern "C" void softmax_launch(const float *a, float *out, int rows, int cols) {
  size_t bytes = cols * sizeof(float);
  int block = 256;
  int grid = CEIL_DIV(cols, block);
  float *d_a, *d_b, *d_c, *d_d, *d_max, *d_sum;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, grid * sizeof(float));
  cudaMalloc(&d_c, grid * sizeof(float));
  cudaMalloc(&d_d, bytes);
  cudaMalloc(&d_max, sizeof(float));
  cudaMalloc(&d_sum, sizeof(float));
  for (int row = 0; row < rows; row++) {
    cudaMemcpy(d_a, a + row * cols, bytes, cudaMemcpyHostToDevice);
    parallel_scan<<<grid, block>>>(d_a, d_b, d_c, cols);
    cudaDeviceSynchronize();
    online_scan<<<1, 1>>>(d_b, d_c, d_max, d_sum, grid);
    cudaDeviceSynchronize();
    parallel_softmax<<<grid, block>>>(d_a, d_max, d_sum, d_d, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(out + row * cols, d_d, bytes, cudaMemcpyDeviceToHost);
  }
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_d);
  cudaFree(d_max);
  cudaFree(d_sum);
}

extern "C" float benchmark_launch(const float *a, float *out, int rows,
                                  int cols, int warmup, int iters) {
  for (int i = 0; i < warmup; i++)
    softmax_launch(a, out, rows, cols);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++)
    softmax_launch(a, out, rows, cols);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iters;
}
