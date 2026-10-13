// Part 4: shared-memory tiled CUDA matrix multiplication.
// Each 16x16 block stages a tile of A and a tile of B in shared memory, so every
// value fetched from global memory is reused TILE_WIDTH times.
// Usage: ./matrix_tiled N [reps]
#include "common.cuh"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16      // build with -DTILE_WIDTH=32 for the 32x32 variant
#endif
#ifndef IMPL_NAME
#define IMPL_NAME "tiled_cuda"
#endif

__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0.0f;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;
        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        __syncthreads();
    }
    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

static void launch_tiled(const float *dA, const float *dB, float *dC, int N) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    matrixMultiplyTiled<<<grid, block>>>(dA, dB, dC, N);
}

int main(int argc, char **argv) {
    return run_matmul_benchmark(argc, argv, IMPL_NAME, launch_tiled);
}
