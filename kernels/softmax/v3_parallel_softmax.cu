/*
 * nvcc -g -G -O0 -o softmax_debug kernels/softmax/v3_parallel_softmax.cu
 * */
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void parallel_scan(const float *input, float *max_val, float *expsum,
                              int n) {
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
    expsum[blockIdx.x] = s_sum[0];
  }
}

__global__ void parallel_softmax(const float *input, float *max_val,
                                 float *expsum, float *output, int n) {
  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid < n) {
    output[pid] = expf(input[pid] - *max_val) / *expsum;
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
  parallel_softmax<<<grid, block>>>(d_a, d_b, d_c, d_d, n);
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

extern "C" void softmax_launch(const float *a, float *out, int n) {

  float *d_a, *d_b, *d_c, *d_d;
  size_t bytes = n * sizeof(float);
  // int block = 256;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, sizeof(float));
  cudaMalloc(&d_c, sizeof(float));
  cudaMalloc(&d_d, bytes);
  cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
  parallel_scan<<<1, 1>>>(d_a, d_b, d_c, n);
  cudaDeviceSynchronize();
  parallel_softmax<<<1, 1>>>(d_a, d_b, d_c, d_d, n);
  cudaDeviceSynchronize();
  cudaMemcpy(out, d_d, bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_d);
}
