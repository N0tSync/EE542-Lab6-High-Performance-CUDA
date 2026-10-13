// Part 6.2: further-optimized SGEMM — 2D register blocking + vectorized loads.
//
// What changed versus the 16x16 shared-memory tiled kernel (Part 4):
//
//  1. Bigger block tiles (BM x BN = 128 x 128 outputs per block, BK = 8).
//     Each element brought into shared memory is reused 128 times instead of 16,
//     so global-memory traffic per FLOP drops ~8x.
//
//  2. Register blocking (thread coarsening): each of the 256 threads computes an
//     8x8 patch of C, held in 64 registers. In the inner loop a thread loads 8
//     values of A and 8 of B from shared memory and does 64 FMAs with them, so
//     shared-memory loads per FMA drop from 2 (tiled kernel) to 0.25. The tiled
//     kernel is bound by shared-memory bandwidth; this one is much closer to
//     being bound by the FMA units.
//
//  3. float4 (128-bit) global loads: one instruction moves 4 floats, so fewer
//     load instructions and fully coalesced transactions.
//
//  4. A is stored transposed in shared memory (As[k][m]) so the per-thread reads
//     of a column of A in the inner loop are contiguous, mostly broadcast reads.
//
// The kernel needs N to be a multiple of 128 (for full tiles and 16-byte
// alignment). The harness pads the matrices with zeros up to the next multiple
// of 128, which leaves the product exact; the padding copy is included in the
// "total" time.
//
// Usage: ./matrix_opt N [reps]
#include "common.cuh"

#include "sgemm_opt.cuh"

static void launch_opt(const float *dA, const float *dB, float *dC, int N) {
    // N here is already padded to a multiple of 128 by the harness.
    dim3 grid(N / BN, N / BM);
    sgemm_2d_blocktile<<<grid, NTHREADS>>>(dA, dB, dC, N);
}

int main(int argc, char **argv) {
    return run_matmul_benchmark(argc, argv, "optimized_cuda", launch_opt, BM);
}
