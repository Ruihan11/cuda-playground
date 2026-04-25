/*
 * nvcc -g -G -O0 -o temp/softmax_debug kernels/softmax/v1_naive_softmax.cu
 * */
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void find_max(const float *input, float *output, int n) {
  float m = -FLT_MAX;
  for (int i = 0; i < n; i++) {
    m = fmaxf(m, input[i]);
  }
  *output = m;
}

__global__ void compute_expsum(const float *input, float *max_val,
                               float *output, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) {
    s += expf(input[i] - *max_val);
  }
  *output = s;
}

__global__ void naive_softmax(const float *input, float *max_val, float *expsum,
                              float *output, int n) {
  for (int i = 0; i < n; i++) {
    output[i] = expf(input[i] - *max_val) / *expsum;
  }
}

int main() {
  int n = 4;
  float *d_a, *d_b, *d_c, *d_d;
  size_t bytes = n * sizeof(float);
  float *h_a = (float *)malloc(bytes);
  float *output = (float *)malloc(bytes);
  for (int i = 0; i < n; i++)
    h_a[i] = float(i);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, sizeof(float));
  cudaMalloc(&d_c, sizeof(float));
  cudaMalloc(&d_d, bytes);
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

  // int block = 256;
  find_max<<<1, 1>>>(d_a, d_b, n);
  cudaDeviceSynchronize();
  compute_expsum<<<1, 1>>>(d_a, d_b, d_c, n);
  cudaDeviceSynchronize();
  naive_softmax<<<1, 1>>>(d_a, d_b, d_c, d_d, n);
  cudaDeviceSynchronize();

  cudaMemcpy(output, d_d, bytes, cudaMemcpyDeviceToHost);
  printf("%f\n", output[0]);
  printf("%f\n", output[1]);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_d);

  return 0;
}

extern "C" void softmax_launch(const float *a, float *out, int rows, int cols) {
  size_t bytes = cols * sizeof(float);
  float *d_a, *d_b, *d_c, *d_d;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, sizeof(float));
  cudaMalloc(&d_c, sizeof(float));
  cudaMalloc(&d_d, bytes);
  for (int row = 0; row < rows; row++) {
    cudaMemcpy(d_a, a + row * cols, bytes, cudaMemcpyHostToDevice);
    find_max<<<1, 1>>>(d_a, d_b, cols);
    cudaDeviceSynchronize();
    compute_expsum<<<1, 1>>>(d_a, d_b, d_c, cols);
    cudaDeviceSynchronize();
    naive_softmax<<<1, 1>>>(d_a, d_b, d_c, d_d, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(out + row * cols, d_d, bytes, cudaMemcpyDeviceToHost);
  }
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
