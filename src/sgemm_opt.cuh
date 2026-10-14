// Optimized SGEMM kernel (Part 6.2) — see matrix_opt.cu for the explanation.
// Shared by matrix_opt.cu and matrix_lib.cu.
// Requires N to be a multiple of 128 (callers zero-pad).
#pragma once
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NTHREADS ((BM * BN) / (TM * TN))   // 256

__global__ void __launch_bounds__(NTHREADS)
sgemm_2d_blocktile(const float *__restrict__ A, const float *__restrict__ B,
                   float *__restrict__ C, int N) {
    __shared__ float As[BK * BM];   // transposed: As[k * BM + m]
    __shared__ float Bs[BK * BN];   //             Bs[k * BN + n]

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // Position of this thread's 8x8 output patch inside the 128x128 block tile.
    const int threadCol = threadIdx.x % (BN / TN);   // 0..15
    const int threadRow = threadIdx.x / (BN / TN);   // 0..15

    // Move pointers to the start of this block's row panel of A, column panel
    // of B, and output tile of C.
    A += (size_t)cRow * BM * N;
    B += (size_t)cCol * BN;
    C += (size_t)cRow * BM * N + (size_t)cCol * BN;

    // Load mapping: the A tile is 128 rows x 8 cols = 256 float4 (one per thread);
    // the B tile is 8 rows x 128 cols = 256 float4 (one per thread).
    const int innerRowA = threadIdx.x / (BK / 4);    // 0..127
    const int innerColA = threadIdx.x % (BK / 4);    // 0..1
    const int innerRowB = threadIdx.x / (BN / 4);    // 0..7
    const int innerColB = threadIdx.x % (BN / 4);    // 0..31

    float acc[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int bk = 0; bk < N; bk += BK) {
        // Global -> shared. A is transposed on the way in.
        float4 a4 = reinterpret_cast<const float4 *>(&A[(size_t)innerRowA * N + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = a4.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = a4.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = a4.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = a4.w;

        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4 *>(&B[(size_t)innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        A += BK;                 // next 8 columns of A
        B += (size_t)BK * N;     // next 8 rows of B

        // Outer-product accumulation out of registers.
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[k * BM + threadRow * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[k * BN + threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    // Registers -> global, 4 floats per store.
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4 v = make_float4(acc[i * TN + j], acc[i * TN + j + 1],
                                   acc[i * TN + j + 2], acc[i * TN + j + 3]);
            reinterpret_cast<float4 *>(
                &C[(size_t)(threadRow * TM + i) * N + threadCol * TN + j])[0] = v;
        }
    }
}
