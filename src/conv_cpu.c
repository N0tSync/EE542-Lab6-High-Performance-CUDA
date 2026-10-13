#include <string.h>
#include <time.h>
#include "conv_io.h"

void convolve_cpu(const unsigned int *in, unsigned int *out, int W, int H,
                  const float *k, int F) {
    int R = F / 2;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sum = 0.0f;
            for (int i = 0; i < F; i++) {
                int iy = y - i + R;
                if (iy < 0 || iy >= H) continue;
                for (int j = 0; j < F; j++) {
                    int ix = x - j + R;
                    if (ix < 0 || ix >= W) continue;
                    sum += k[i * F + j] * (float)in[iy * W + ix];
                }
            }
            out[y * W + x] = to_pixel(sum);
        }
    }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s image.bin filter.txt out.bin [reps]\n", argv[0]);
        return 1;
    }
    int reps = (argc > 4) ? atoi(argv[4]) : 1;
    int W, H, F;
    unsigned int *in = read_image(argv[1], &W, &H);
    float *k = read_filter(argv[2], &F);
    unsigned int *out = (unsigned int *)malloc((size_t)W * H * sizeof(unsigned int));

    double t0 = now_ms();
    for (int r = 0; r < reps; r++) convolve_cpu(in, out, W, H, k, F);
    double ms = (now_ms() - t0) / reps;

    write_image(argv[3], out, W, H);
    printf("CPU convolution: image %dx%d, filter %dx%d: %.3f ms\n", W, H, F, F, ms);
    // RESULT,impl,M,F,kernel_ms,total_ms
    printf("RESULT,conv_cpu,%d,%d,%.4f,%.4f\n", W, F, ms, ms);
    free(in); free(out); free(k);
    return 0;
}
