#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 16


__global__ void matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 



int main(int argc, char const *argv[])
{
    int m, n, k;
    srand(3333);
    m = 9000;
    n = 9000;
    k = 9000;

    int *a, *b;
    cudaMallocHost((void **) &a, sizeof(int)*m*n);
    cudaMallocHost((void **) &b, sizeof(int)*n*k);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = rand();
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            b[i * k + j] = rand();
        }
    }

    int *device_a, *device_b, *device_c;
    cudaMalloc((void **) &device_a, sizeof(int)*m*n);
    cudaMalloc((void **) &device_b, sizeof(int)*n*k);
    cudaMalloc((void **) &device_c, sizeof(int)*m*k);

    cudaMemcpy(device_a, a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    float elapsed_time_ms;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    matrix_mult<<<dimGrid, dimBlock>>>(device_a, device_b, device_c, m, n, k);    
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, elapsed_time_ms);
    

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    return 0;
}