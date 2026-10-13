// Part 2/3: naive CUDA matrix multiplication — one thread per output element.
// Usage: ./matrix_gpu N [reps]
#include "common.cuh"

#define BLOCK 16

__global__ void matrixMultiplyGPU(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

static void launch_naive(const float *dA, const float *dB, float *dC, int N) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    matrixMultiplyGPU<<<grid, block>>>(dA, dB, dC, N);
}

int main(int argc, char **argv) {
    return run_matmul_benchmark(argc, argv, "naive_cuda", launch_naive);
}
