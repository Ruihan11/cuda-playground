/*
 * nvcc -g -G -O0 -o temp/atten_debug kernels/atten/v1_naive_atten.cu
 * */
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void fa2_kernel(const float *Q, const float *K, const float *V,
                           float *output, const int batch, const int heads,
                           const int seq, const int dimension,
                           const float scale) {

  int i = blockIdx.y; // #seq - #rows
  int bh = blockIdx.x;
  int b = bh / heads;
  int h = bh % heads;

  float m = -FLT_MAX;
  float l = 0.0f;
  float o[64] = {0};

  for (int j = 0; j < seq; j++) {

    float acc = 0.0f;
    for (int k = 0; k < dimension; k++) {
      acc += Q[b * heads * seq * dimension + h * seq * dimension +
               i * dimension + k] *
             K[b * heads * seq * dimension + h * seq * dimension +
               j * dimension + k];
    }
    acc *= scale;

    float m_new = fmaxf(m, acc);
    float l_new = l * expf(m - m_new) + expf(acc - m_new);

    for (int k = 0; k < dimension; k++)
      o[k] = expf(m - m_new) * o[k] +
             expf(acc - m_new) * V[b * heads * seq * dimension +
                                   h * seq * dimension + j * dimension + k];

    m = m_new;
    l = l_new;
  }

  for (int k = 0; k < dimension; k++)
    output[b * heads * seq * dimension + h * seq * dimension + i * dimension +
           k] = o[k] / l;
}

int main() {

  int batch = 1, heads = 1, seq = 4, d = 4;
  float scale = 1.0f / sqrt(float(d));
  float *h_q = (float *)malloc(batch * heads * seq * d * sizeof(float));
  float *h_k = (float *)malloc(batch * heads * seq * d * sizeof(float));
  float *h_v = (float *)malloc(batch * heads * seq * d * sizeof(float));
  float *output = (float *)malloc(batch * heads * seq * d * sizeof(float));
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < heads; j++) {
      for (int k = 0; k < seq; k++) {
        for (int l = 0; l < d; l++) {
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
  dim3 block(1);
  fa2_kernel<<<grid, block>>>(d_q, d_k, d_v, d_r, batch, heads, seq, d, scale);
  cudaDeviceSynchronize();
  cudaMemcpy(output, d_r, batch * heads * seq * d * sizeof(float),
             cudaMemcpyDeviceToHost);
  printf("%f\n", output[0]);
  printf("%f\n", output[1]);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_v);

  return 0;
}

extern "C" void atten_launch(const float *q, const float *k, const float *v,
                             float *result, const int batch, const int heads,
                             const int seq, const int d) {
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
  dim3 block(1);
  fa2_kernel<<<grid, block>>>(d_q, d_k, d_v, d_r, batch, heads, seq, d, scale);
  cudaDeviceSynchronize();
  cudaMemcpy(result, d_r, batch * heads * seq * d * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_r);
}

extern "C" float benchmark_launch(const float *q, const float *k,
                                  const float *v, float *result,
                                  const int batch, const int heads,
                                  const int seq, const int d, int warmup,
                                  int iters) {
  for (int i = 0; i < warmup; i++)
    atten_launch(q, k, v, result, batch, heads, seq, d);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; i++)
    atten_launch(q, k, v, result, batch, heads, seq, d);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / iters;
}
