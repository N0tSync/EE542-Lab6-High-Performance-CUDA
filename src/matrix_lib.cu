// Parts 7 & 8.3: shared library (libmatrix.so) callable from Python via ctypes.
//
// Build: nvcc -O3 -Xcompiler -fPIC -shared matrix_lib.cu -o libmatrix.so -lcublas
//
// Exported C functions (all return 0 on success, a CUDA/cuBLAS error code otherwise;
// all take HOST pointers to row-major, C-contiguous arrays):
//
//   int   gpu_matrix_multiply       (const float *A, const float *B, float *C, int N)  // tiled (Part 4)
//   int   gpu_matrix_multiply_opt   (const float *A, const float *B, float *C, int N)  // register-blocked (Part 6.2)
//   int   gpu_matrix_multiply_cublas(const float *A, const float *B, float *C, int N)  // cuBLAS SGEMM
//   int   gpu_convolve(const unsigned *in, unsigned *out, int W, int H,
//                      const float *filter, int F, int tiled)                         // Part 8
//   int   gpu_convolve_batch(const unsigned *in, unsigned *out, int W, int H,
//                            const float *filters, int F, int nfilters)               // many filters, 1 upload
//   float gpu_last_kernel_ms(void)   // GPU time of the kernel(s) in the most recent call
//   float gpu_last_total_ms(void)    // full time of the most recent call (alloc+copies+kernel)
//   int   gpu_device_name(char *buf, int len)
//   int   gpu_warmup(void)           // create the CUDA context / cuBLAS handle up front
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include "sgemm_opt.cuh"
#include "conv_kernels.cuh"

#define TILE_WIDTH 16

// ---------------------------------------------------------------- kernels
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

// ---------------------------------------------------------------- helpers
static float g_kernel_ms = 0.0f;
static float g_total_ms  = 0.0f;
static cublasHandle_t g_cublas = nullptr;

static double wall_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Bail out of the current exported function with the CUDA error code.
#define TRY(call)                                                           \
    do {                                                                    \
        cudaError_t e__ = (call);                                           \
        if (e__ != cudaSuccess) {                                           \
            fprintf(stderr, "libmatrix: %s (%s:%d)\n",                      \
                    cudaGetErrorString(e__), __FILE__, __LINE__);           \
            rc = (int)e__; goto done;                                       \
        }                                                                   \
    } while (0)

// Times a stretch of GPU work with CUDA events.
struct EventTimer {
    cudaEvent_t a, b;
    EventTimer()  { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~EventTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start()  { cudaEventRecord(a); }
    float stop()  { cudaEventRecord(b); cudaEventSynchronize(b);
                    float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms; }
};

enum MatmulKind { MM_TILED, MM_OPT, MM_CUBLAS };

static int matmul_common(const float *h_A, const float *h_B, float *h_C, int N, MatmulKind kind) {
    int rc = 0;
    double t0 = wall_ms();
    // The optimized kernel needs N padded to a multiple of 128 (zero padding keeps
    // the product exact).
    int Np = (kind == MM_OPT) ? ((N + BM - 1) / BM) * BM : N;
    size_t pbytes = (size_t)Np * Np * sizeof(float);
    size_t hp = N * sizeof(float), dp = Np * sizeof(float);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    EventTimer t;

    TRY(cudaMalloc((void **)&d_A, pbytes));
    TRY(cudaMalloc((void **)&d_B, pbytes));
    TRY(cudaMalloc((void **)&d_C, pbytes));
    if (Np != N) {
        TRY(cudaMemset(d_A, 0, pbytes));
        TRY(cudaMemset(d_B, 0, pbytes));
    }
    TRY(cudaMemcpy2D(d_A, dp, h_A, hp, hp, N, cudaMemcpyHostToDevice));
    TRY(cudaMemcpy2D(d_B, dp, h_B, hp, hp, N, cudaMemcpyHostToDevice));

    t.start();
    if (kind == MM_TILED) {
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
        matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    } else if (kind == MM_OPT) {
        dim3 grid(Np / BN, Np / BM);
        sgemm_2d_blocktile<<<grid, NTHREADS>>>(d_A, d_B, d_C, Np);
    } else {
        if (!g_cublas) {
            if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) { rc = -1; goto done; }
            cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH);
        }
        const float alpha = 1.0f, beta = 0.0f;
        // Row-major trick: compute C^T = B^T A^T in column-major (see matrix_cublas.cu).
        if (cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                        d_B, N, d_A, N, &beta, d_C, N) != CUBLAS_STATUS_SUCCESS) { rc = -2; goto done; }
    }
    TRY(cudaGetLastError());
    g_kernel_ms = t.stop();

    TRY(cudaMemcpy2D(h_C, hp, d_C, dp, hp, N, cudaMemcpyDeviceToHost));

done:
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    g_total_ms = (float)(wall_ms() - t0);
    return rc;
}

// ---------------------------------------------------------------- exports
extern "C" {

int gpu_matrix_multiply(const float *h_A, const float *h_B, float *h_C, int N) {
    return matmul_common(h_A, h_B, h_C, N, MM_TILED);
}

int gpu_matrix_multiply_opt(const float *h_A, const float *h_B, float *h_C, int N) {
    return matmul_common(h_A, h_B, h_C, N, MM_OPT);
}

int gpu_matrix_multiply_cublas(const float *h_A, const float *h_B, float *h_C, int N) {
    return matmul_common(h_A, h_B, h_C, N, MM_CUBLAS);
}

int gpu_convolve(const unsigned int *h_in, unsigned int *h_out, int W, int H,
                 const float *h_filter, int F, int tiled) {
    int rc = 0;
    double t0 = wall_ms();
    size_t bytes = (size_t)W * H * sizeof(unsigned int);
    unsigned int *d_in = nullptr, *d_out = nullptr;
    EventTimer t;
    if (F < 1 || F > MAX_FILTER) { fprintf(stderr, "libmatrix: filter size %d unsupported (1..%d)\n", F, MAX_FILTER); return -1; }

    TRY(cudaMalloc((void **)&d_in, bytes));
    TRY(cudaMalloc((void **)&d_out, bytes));
    TRY(upload_flipped_filter(h_filter, F));
    TRY(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    t.start();
    launch_conv(tiled != 0, d_in, d_out, W, H, F);
    TRY(cudaGetLastError());
    g_kernel_ms = t.stop();
    TRY(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
done:
    cudaFree(d_in); cudaFree(d_out);
    g_total_ms = (float)(wall_ms() - t0);
    return rc;
}

// Apply nfilters F x F filters (stored back to back) to one image. The image is
// copied to the GPU once, so this amortises the transfer overhead — the main
// lesson of Part 5 applied to the library API. out holds nfilters images.
int gpu_convolve_batch(const unsigned int *h_in, unsigned int *h_out, int W, int H,
                       const float *h_filters, int F, int nfilters) {
    int rc = 0;
    double t0 = wall_ms();
    size_t bytes = (size_t)W * H * sizeof(unsigned int);
    unsigned int *d_in = nullptr, *d_out = nullptr;
    float kms = 0;
    if (F < 1 || F > MAX_FILTER) { fprintf(stderr, "libmatrix: filter size %d unsupported (1..%d)\n", F, MAX_FILTER); return -1; }

    TRY(cudaMalloc((void **)&d_in, bytes));
    TRY(cudaMalloc((void **)&d_out, bytes * nfilters));
    TRY(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    for (int f = 0; f < nfilters; f++) {
        EventTimer t;
        TRY(upload_flipped_filter(h_filters + (size_t)f * F * F, F));
        t.start();
        launch_conv(true, d_in, d_out + (size_t)f * W * H, W, H, F);
        TRY(cudaGetLastError());
        kms += t.stop();
    }
    g_kernel_ms = kms;
    TRY(cudaMemcpy(h_out, d_out, bytes * nfilters, cudaMemcpyDeviceToHost));
done:
    cudaFree(d_in); cudaFree(d_out);
    g_total_ms = (float)(wall_ms() - t0);
    return rc;
}

float gpu_last_kernel_ms(void) { return g_kernel_ms; }
float gpu_last_total_ms(void)  { return g_total_ms; }

int gpu_device_name(char *buf, int len) {
    cudaDeviceProp p;
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e == cudaSuccess) e = cudaGetDeviceProperties(&p, dev);
    if (e != cudaSuccess) return (int)e;
    snprintf(buf, len, "%s", p.name);
    return 0;
}

int gpu_warmup(void) {
    cudaError_t e = cudaFree(0);   // creates the context
    if (e != cudaSuccess) return (int)e;
    if (!g_cublas && cublasCreate(&g_cublas) == CUBLAS_STATUS_SUCCESS)
        cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH);
    return 0;
}

}  // extern "C"
