/*
 * nvcc -g -G -O0 -o temp/softmax_debug kernels/softmax/v2_online_softmax.cu
 * */
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void online_scan(const float *input, float *max_val, float *expsum,
                            int n) {
  float m = -FLT_MAX, s = 0.0f;
  for (int i = 0; i < n; i++) {
    float m_new = fmaxf(m, input[i]);
    s = s * expf(m - m_new) + expf(input[i] - m_new);
    m = m_new;
  }
  *max_val = m;
  *expsum = s;
}

__global__ void online_softmax(const float *input, float *max_val,
                               float *expsum, float *output, int n) {
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
  online_scan<<<1, 1>>>(d_a, d_b, d_c, n);
  cudaDeviceSynchronize();
  online_softmax<<<1, 1>>>(d_a, d_b, d_c, d_d, n);
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
    online_scan<<<1, 1>>>(d_a, d_b, d_c, cols);
    cudaDeviceSynchronize();
    online_softmax<<<1, 1>>>(d_a, d_b, d_c, d_d, cols);
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
