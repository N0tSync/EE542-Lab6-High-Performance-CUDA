// Part 6.1: cuBLAS SGEMM.
// cuBLAS is column-major. Our matrices are row-major, and a row-major matrix
// is the transpose of the same buffer read column-major. Since
//   C = A*B   <=>   C^T = B^T * A^T
// we ask cuBLAS for B*A on the raw buffers, which yields C in row-major order
// with no explicit transposes (this is why d_B is passed before d_A).
// Usage: ./matrix_cublas N [reps]
#include "common.cuh"
#include <cublas_v2.h>

static cublasHandle_t handle;

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t s__ = (call);                                         \
        if (s__ != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s__,          \
                    __FILE__, __LINE__);                                     \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static void init_cublas() {
    double t0 = wall_ms();
    CUBLAS_CHECK(cublasCreate(&handle));
    // Keep FP32 math (no TF32 tensor-core shortcut) so the comparison with the
    // hand-written FP32 kernels is apples to apples.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    fprintf(stderr, "(cublasCreate took %.1f ms — one-time cost, excluded from timings)\n",
            wall_ms() - t0);
}

static void cleanup_cublas() { cublasDestroy(handle); }

static void launch_cublas(const float *dA, const float *dB, float *dC, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N, &alpha,
                             dB, N,
                             dA, N,
                             &beta, dC, N));
}

int main(int argc, char **argv) {
    return run_matmul_benchmark(argc, argv, "cublas", launch_cublas, 1,
                                init_cublas, cleanup_cublas);
}
