#pragma once
#include <cuda_runtime.h>

#define CONV_TILE 16
#define MAX_FILTER 31                         // up to 31x31 filters in constant memory
#define CONV_PIXEL_MAX 255u

__constant__ float c_filter[MAX_FILTER * MAX_FILTER];

__device__ __forceinline__ unsigned int conv_to_pixel(float s) {
    s = fabsf(s) + 0.5f;
    return (s >= (float)CONV_PIXEL_MAX) ? CONV_PIXEL_MAX : (unsigned int)s;
}

// Naive: every thread reads its F*F neighbours straight from global memory.
__global__ void conv2d_naive(const unsigned int *__restrict__ in, unsigned int *__restrict__ out,
                             int W, int H, int F) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int L = F - 1 - F / 2;
    float sum = 0.0f;
    for (int a = 0; a < F; a++) {
        int iy = y + a - L;
        if (iy < 0 || iy >= H) continue;
        for (int b = 0; b < F; b++) {
            int ix = x + b - L;
            if (ix < 0 || ix >= W) continue;
            sum += c_filter[a * F + b] * (float)in[iy * W + ix];
        }
    }
    out[y * W + x] = conv_to_pixel(sum);
}

// Tiled: each 16x16 block cooperatively loads its (16+F-1)^2 input patch,
// halo included, into shared memory once; then every output pixel reads its
// F*F neighbourhood from shared memory. Filter lives in constant memory, which
// is broadcast to all threads of a warp since they read the same coefficient.
// Launch with dynamic shared memory = (CONV_TILE+F-1)^2 * sizeof(unsigned int).
__global__ void conv2d_tiled(const unsigned int *__restrict__ in, unsigned int *__restrict__ out,
                             int W, int H, int F) {
    extern __shared__ unsigned int tile[];
    const int L  = F - 1 - F / 2;
    const int TW = CONV_TILE + F - 1;
    const int ox = blockIdx.x * CONV_TILE;
    const int oy = blockIdx.y * CONV_TILE;
    const int tid = threadIdx.y * CONV_TILE + threadIdx.x;

    for (int i = tid; i < TW * TW; i += CONV_TILE * CONV_TILE) {
        int ty = i / TW, tx = i % TW;
        int gy = oy + ty - L, gx = ox + tx - L;
        tile[i] = (gy >= 0 && gy < H && gx >= 0 && gx < W) ? in[gy * W + gx] : 0u;
    }
    __syncthreads();

    int x = ox + threadIdx.x, y = oy + threadIdx.y;
    if (x >= W || y >= H) return;
    float sum = 0.0f;
    for (int a = 0; a < F; a++) {
        const unsigned int *row = &tile[(threadIdx.y + a) * TW + threadIdx.x];
        for (int b = 0; b < F; b++)
            sum += c_filter[a * F + b] * (float)row[b];
    }
    out[y * W + x] = conv_to_pixel(sum);
}