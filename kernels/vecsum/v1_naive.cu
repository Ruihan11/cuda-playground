#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV(x, y)(((x)+(y)-1)/(y))

__global__ void vecsum_naive(const float* a, const float* b, float* c, int n) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid < n){
        c[pid] = a[pid] + b[pid];
    }
}


extern "C" void vecsum_launch(const float* a, const float* b, float* c, int n)
{

    float *d_a, *d_b, *d_c;
    size_t bytes = n * sizeof(float);
    int block = 256;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
    vecsum_naive<<<CEIL_DIV(n, block), block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}
