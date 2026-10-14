// Shared benchmark harness for the GPU matrix-multiply programs.
// Each .cu file defines a launcher and calls run_matmul_benchmark().
//
// Two times are reported for every run:
//   kernel_ms : GPU compute only (cudaEvent around the kernel, averaged over reps)
//   total_ms  : everything a caller pays for one multiply:
//               cudaMalloc + H2D copy + kernel + D2H copy + cudaFree
// total_ms - kernel_ms is the PCIe / allocation overhead discussed in Part 5.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// launcher(dA, dB, dC, n): n is the (possibly padded) leading dimension.
typedef void (*matmul_launcher)(const float *dA, const float *dB, float *dC, int n);

static inline double wall_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// pad_to > 1 makes the device buffers a multiple of pad_to in each dimension
// (zero filled) so kernels that need aligned sizes still give exact results.
static int run_matmul_benchmark(int argc, char **argv, const char *impl,
                                matmul_launcher launch, int pad_to = 1,
                                void (*init)() = nullptr, void (*cleanup)() = nullptr) {
    int N    = (argc > 1) ? atoi(argv[1]) : 1024;
    int reps = (argc > 2) ? atoi(argv[2]) : 5;
    int Np   = ((N + pad_to - 1) / pad_to) * pad_to;

    size_t bytes  = (size_t)N * N * sizeof(float);
    size_t pbytes = (size_t)Np * Np * sizeof(float);
    float *A = (float *)malloc(bytes), *B = (float *)malloc(bytes), *C = (float *)malloc(bytes);
    srand(42);
    for (size_t i = 0; i < (size_t)N * N; i++) {
        A[i] = rand() % 100 / 100.0f;
        B[i] = rand() % 100 / 100.0f;
    }

    CUDA_CHECK(cudaFree(0));            // force context creation outside timing
    if (init) init();

    cudaDeviceProp prop;
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    auto copy_in = [&](float *dst, const float *src) {
        if (Np == N) CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
        else CUDA_CHECK(cudaMemcpy2D(dst, Np * sizeof(float), src, N * sizeof(float),
                                     N * sizeof(float), N, cudaMemcpyHostToDevice));
    };
    auto copy_out = [&](float *dst, const float *src) {
        if (Np == N) CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
        else CUDA_CHECK(cudaMemcpy2D(dst, N * sizeof(float), src, Np * sizeof(float),
                                     N * sizeof(float), N, cudaMemcpyDeviceToHost));
    };

    // ---------- End-to-end time (one full call, like a library user sees) ----------
    // A warm-up pass first so one-time costs (module load, cuBLAS kernel selection)
    // don't land in the measurement.
    double total_ms = 0;
    for (int pass = 0; pass < 2; pass++) {
        double t0 = wall_ms();
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, pbytes));
        CUDA_CHECK(cudaMalloc(&dB, pbytes));
        CUDA_CHECK(cudaMalloc(&dC, pbytes));
        if (Np != N) { CUDA_CHECK(cudaMemset(dA, 0, pbytes)); CUDA_CHECK(cudaMemset(dB, 0, pbytes)); }
        copy_in(dA, A);
        copy_in(dB, B);
        launch(dA, dB, dC, Np);
        CUDA_CHECK(cudaGetLastError());
        copy_out(C, dC);                   // cudaMemcpy is synchronous
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        total_ms = wall_ms() - t0;
    }

    // ---------- Kernel-only time ----------
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, pbytes));
    CUDA_CHECK(cudaMalloc(&dB, pbytes));
    CUDA_CHECK(cudaMalloc(&dC, pbytes));
    if (Np != N) { CUDA_CHECK(cudaMemset(dA, 0, pbytes)); CUDA_CHECK(cudaMemset(dB, 0, pbytes)); }
    copy_in(dA, A);
    copy_in(dB, B);
    launch(dA, dB, dC, Np);             // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; r++) launch(dA, dB, dC, Np);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ev_ms; CUDA_CHECK(cudaEventElapsedTime(&ev_ms, e0, e1));
    double kernel_ms = ev_ms / reps;

    // ---------- Correctness: sample entries vs double-precision dot products ----------
    copy_out(C, dC);
    double maxrel = 0.0;
    for (int s = 0; s < 64; s++) {
        int i = rand() % N, j = rand() % N;
        double ref = 0.0;
        for (int k = 0; k < N; k++) ref += (double)A[(size_t)i * N + k] * B[(size_t)k * N + j];
        double rel = (ref != 0.0) ? fabs((C[(size_t)i * N + j] - ref) / ref) : 0.0;
        if (rel > maxrel) maxrel = rel;
    }
    bool ok = maxrel < 1e-3;

    double gflops = 2.0 * N * (double)N * N / (kernel_ms * 1e6);
    printf("[%s] GPU: %s | N=%d%s\n", impl, prop.name, N,
           (Np != N) ? " (padded)" : "");
    printf("  kernel: %10.4f ms  (%.1f GFLOP/s)\n", kernel_ms, gflops);
    printf("  total : %10.4f ms  (alloc + H2D + kernel + D2H + free)\n", total_ms);
    printf("  overhead: %.4f ms (%.1f%% of total)\n", total_ms - kernel_ms,
           100.0 * (total_ms - kernel_ms) / total_ms);
    printf("  check : max rel err %.2e -> %s\n", maxrel, ok ? "PASS" : "FAIL");
    printf("RESULT,%s,%d,%.4f,%.4f,%.3f,%.2e\n", impl, N, kernel_ms, total_ms, gflops, maxrel);

    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    if (cleanup) cleanup();
    free(A); free(B); free(C);
    return ok ? 0 : 2;
}
