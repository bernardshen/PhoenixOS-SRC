#include <iostream>
#include <cuda_runtime.h>

#define N 512  // Define the size of the matrix (N x N)

// CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float* A, float* B, float* C, int n) {
    // Calculate the row index of the C element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of the C element
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the indices are within matrix bounds
    if (row < n && col < n) {
        float value = 0.0;
        for (int k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

int main() {
    // Allocate host memory for matrices
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize matrices A and B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0;  // Example values
        h_B[i] = 1.0;  // Example values
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy matrices from host memory to device memory
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threadsPerBlock = 16;
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim((N + threadsPerBlock - 1) / threadsPerBlock, (N + threadsPerBlock - 1) / threadsPerBlock);

    // Launch the kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify the result (for small matrices)
    bool error = false;
    for (int i = 0; i < N * N; ++i) {
        if (h_C[i] != N) {
            error = true;
            break;
        }
    }
    if (!error) {
        std::cout << "Matrix multiplication completed successfully!" << std::endl;
    } else {
        std::cout << "Error in matrix multiplication!" << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
