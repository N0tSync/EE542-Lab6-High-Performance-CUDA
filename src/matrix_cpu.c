// Part 1: CPU matrix multiplication (baseline)
// Usage: ./matrix_cpu N [ijk|ikj]
//   ijk = textbook loop order from the lab handout (default)
//   ikj = cache-friendly loop order (optional extra data point)
// Prints a human-readable line and a machine-readable RESULT line:
//   RESULT,<impl>,<N>,<kernel_ms>,<total_ms>,<gflops>,<check>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void matrixMultiplyCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Same math, loop order changed so the inner loop walks B and C row-wise
// (unit stride) instead of striding down a column of B.
void matrixMultiplyCPU_ikj(float *A, float *B, float *C, int N) {
    memset(C, 0, (size_t)N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    const char *order = (argc > 2) ? argv[2] : "ijk";
    size_t size = (size_t)N * N * sizeof(float);

    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);
    if (!A || !B || !C) { fprintf(stderr, "malloc failed\n"); return 1; }

    srand(42);
    for (size_t i = 0; i < (size_t)N * N; i++) {
        A[i] = rand() % 100 / 100.0f;
        B[i] = rand() % 100 / 100.0f;
    }

    double t0 = now_ms();
    if (strcmp(order, "ikj") == 0) matrixMultiplyCPU_ikj(A, B, C, N);
    else                           matrixMultiplyCPU(A, B, C, N);
    double ms = now_ms() - t0;

    // Spot-check a few entries with a double-precision dot product.
    double maxrel = 0.0;
    for (int s = 0; s < 16; s++) {
        int i = rand() % N, j = rand() % N;
        double ref = 0.0;
        for (int k = 0; k < N; k++) ref += (double)A[i * N + k] * B[k * N + j];
        double rel = (ref != 0.0) ? ((C[i * N + j] - ref) / ref) : 0.0;
        if (rel < 0) rel = -rel;
        if (rel > maxrel) maxrel = rel;
    }

    double gflops = 2.0 * N * (double)N * N / (ms * 1e6);
    printf("CPU (%s) execution time (N=%d): %f seconds  [%.2f GFLOP/s, max rel err %.2e]\n",
           order, N, ms / 1e3, gflops, maxrel);
    printf("RESULT,cpu_%s,%d,%.4f,%.4f,%.3f,%.2e\n", order, N, ms, ms, gflops, maxrel);

    free(A); free(B); free(C);
    return 0;
}
