// Part 8.2: 2D convolution on the GPU (naive and shared-memory tiled kernels).
// Same file formats and the same math as conv_cpu.c, so the outputs can be
// compared byte-for-byte (allowing +-1 for float summation-order rounding).
//
// Usage: ./conv_gpu image.bin filter.txt out.bin [reps] [naive|tiled]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "conv_io.h"
#include "conv_kernels.cuh"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static double wall_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s image.bin filter.txt out.bin [reps] [naive|tiled]\n", argv[0]);
        return 1;
    }
    int reps = (argc > 4) ? atoi(argv[4]) : 10;
    bool tiled = !(argc > 5 && strcmp(argv[5], "naive") == 0);

    int W, H, F;
    unsigned int *in = read_image(argv[1], &W, &H);
    float *k = read_filter(argv[2], &F);
    if (F > MAX_FILTER) { fprintf(stderr, "filter too large (max %d)\n", MAX_FILTER); return 1; }
    size_t bytes = (size_t)W * H * sizeof(unsigned int);
    unsigned int *out = (unsigned int *)malloc(bytes);

    CUDA_CHECK(cudaFree(0));   // create context outside timing

    // End-to-end: alloc + filter upload + H2D + kernel + D2H + free (2nd pass after warm-up)
    double total_ms = 0;
    for (int pass = 0; pass < 2; pass++) {
        double t0 = wall_ms();
        unsigned int *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        CUDA_CHECK(upload_flipped_filter(k, F));
        CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));
        launch_conv(tiled, d_in, d_out, W, H, F);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
        total_ms = wall_ms() - t0;
    }

    // Kernel only
    unsigned int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));
    launch_conv(tiled, d_in, d_out, W, H, F);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; r++) launch_conv(tiled, d_in, d_out, W, H, F);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ev; CUDA_CHECK(cudaEventElapsedTime(&ev, e0, e1));
    double kernel_ms = ev / reps;

    write_image(argv[3], out, W, H);
    const char *impl = tiled ? "conv_gpu_tiled" : "conv_gpu_naive";
    printf("%s: image %dx%d, filter %dx%d: kernel %.4f ms, total %.4f ms\n",
           impl, W, H, F, F, kernel_ms, total_ms);
    printf("RESULT,%s,%d,%d,%.4f,%.4f\n", impl, W, F, kernel_ms, total_ms);

    cudaFree(d_in); cudaFree(d_out);
    free(in); free(out); free(k);
    return 0;
}
