/*
 * nvcc -g -G -O0 -o temp/softmax_debug kernels/softmax/v4_tiled_softmax.cu
 * */
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void tiled_scan(const float *input, const int rows, const int cols,
                           float *max_val, float *exp_sum) {
  __shared__ float s_max[256];
  __shared__ float s_sum[256];
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int pid = row * cols + tid;
  s_max[tid] = (tid < cols) ? input[pid] : -FLT_MAX;
  s_sum[tid] = (tid < cols) ? 1.0f : 0.0f;
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

__global__ void tiled_softmax(const float *input, const int rows,
                              const int cols, float *max_val, float *exp_sum,
                              float *output) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  int pid = row * cols + tid;
  if (tid < cols) {
    output[pid] = expf(input[pid] - max_val[row]) / exp_sum[row];
  }
}

int main() {
  int rows = 4, cols = 4;
  float *d_a, *d_b, *d_c, *d_d;
  float *h_a = (float *)malloc(rows * cols * sizeof(float));
  float *output = (float *)malloc(rows * cols * sizeof(float));
  int block = 256;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      h_a[i * cols + j] = float(j);
    }
  }

  cudaMalloc(&d_a, rows * cols * sizeof(float));
  cudaMalloc(&d_b, rows * sizeof(float));
  cudaMalloc(&d_c, rows * sizeof(float));
  cudaMalloc(&d_d, rows * cols * sizeof(float));
  cudaMemcpy(d_a, h_a, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

  tiled_scan<<<rows, block>>>(d_a, rows, cols, d_b, d_c);
  cudaDeviceSynchronize();
  tiled_softmax<<<rows, block>>>(d_a, rows, cols, d_b, d_c, d_d);
  cudaDeviceSynchronize();

  cudaMemcpy(output, d_d, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < rows; i++) {
  //   for (int j = 0; j < cols; j++) {
  //     printf("%f\n", output[i * cols + j]);
  //   }
  // }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_d);

  return 0;
}

extern "C" void softmax_launch(const float *input, float *output,
                               const int rows, const int cols) {
  float *d_a, *d_b, *d_c, *d_d;
  int block = 256;
  cudaMalloc(&d_a, rows * cols * sizeof(float));
  cudaMalloc(&d_b, rows * sizeof(float));
  cudaMalloc(&d_c, rows * sizeof(float));
  cudaMalloc(&d_d, rows * cols * sizeof(float));
  cudaMemcpy(d_a, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
  tiled_scan<<<rows, block>>>(d_a, rows, cols, d_b, d_c);
  cudaDeviceSynchronize();
  tiled_softmax<<<rows, block>>>(d_a, rows, cols, d_b, d_c, d_d);
  cudaDeviceSynchronize();
  cudaMemcpy(output, d_d, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_d);
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
