// Image / filter file I/O shared by conv_cpu.c and conv_gpu.cu.
//
// Image file (.bin, produced by tools/img_tools.py):
//   int32 width, int32 height, then width*height uint32 pixels, row-major.
//   Pixels are grayscale intensities 0..255 stored as 32-bit unsigned ints.
// Filter file (.txt, produced by tools/make_filters.py):
//   first token F (filter is F x F), then F*F floats, row-major.
#ifndef CONV_IO_H
#define CONV_IO_H
#include <stdio.h>
#include <stdlib.h>

#define PIXEL_MAX 255u

static unsigned int *read_image(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    int hdr[2];
    if (fread(hdr, sizeof(int), 2, f) != 2) { fprintf(stderr, "bad header in %s\n", path); exit(1); }
    *w = hdr[0]; *h = hdr[1];
    size_t n = (size_t)(*w) * (*h);
    unsigned int *img = (unsigned int *)malloc(n * sizeof(unsigned int));
    if (fread(img, sizeof(unsigned int), n, f) != n) { fprintf(stderr, "short read in %s\n", path); exit(1); }
    fclose(f);
    return img;
}

static void write_image(const char *path, const unsigned int *img, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    int hdr[2] = {w, h};
    fwrite(hdr, sizeof(int), 2, f);
    fwrite(img, sizeof(unsigned int), (size_t)w * h, f);
    fclose(f);
}

static float *read_filter(const char *path, int *F) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    if (fscanf(f, "%d", F) != 1 || *F <= 0) { fprintf(stderr, "bad filter %s\n", path); exit(1); }
    float *k = (float *)malloc((size_t)(*F) * (*F) * sizeof(float));
    for (int i = 0; i < (*F) * (*F); i++)
        if (fscanf(f, "%f", &k[i]) != 1) { fprintf(stderr, "short filter %s\n", path); exit(1); }
    fclose(f);
    return k;
}

// Convert a float response to an output pixel: absolute value (so edge filters
// keep negative edges too), rounded, clamped to 0..PIXEL_MAX.
static inline unsigned int to_pixel(float s) {
    if (s < 0) s = -s;
    s += 0.5f;
    return (s >= (float)PIXEL_MAX) ? PIXEL_MAX : (unsigned int)s;
}
#endif
