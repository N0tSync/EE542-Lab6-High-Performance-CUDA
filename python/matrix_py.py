#!/usr/bin/env python3
"""Part 7: call the CUDA matrix multiply in libmatrix.so from Python.

    python3 python/matrix_py.py                   # N=1024, tiled kernel (the lab's example)
    python3 python/matrix_py.py --sizes 512 1024 2048 4096 --impl tiled opt cublas --csv results/matmul_python.csv

For each call it reports the time Python saw (wall clock around the ctypes call)
next to the GPU kernel time measured inside the library, and checks the result
against NumPy.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from cudalib import CudaLib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024])
    ap.add_argument("--impl", nargs="+", default=["tiled"], choices=["tiled", "opt", "cublas"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--csv", help="append results to this CSV")
    ap.add_argument("--lib", help="path to libmatrix.so")
    args = ap.parse_args()

    lib = CudaLib(args.lib)
    lib.warmup()
    print(f"GPU: {lib.device_name}")
    rows = []
    rng = np.random.default_rng(0)
    for N in args.sizes:
        A = rng.random((N, N), dtype=np.float32)
        B = rng.random((N, N), dtype=np.float32)
        t0 = time.perf_counter()
        ref = A @ B                                   # NumPy (multi-threaded BLAS on the CPU)
        numpy_ms = (time.perf_counter() - t0) * 1e3
        for impl in args.impl:
            lib.matmul(A, B, impl)                    # warm-up
            best = None
            for _ in range(args.reps):
                start = time.time()
                C = lib.matmul(A, B, impl)
                end = time.time()
                wall = (end - start) * 1e3
                if best is None or wall < best[0]:
                    best = (wall, lib.last_kernel_ms, lib.last_total_ms)
            wall, kms, tms = best
            err = float(np.max(np.abs(C - ref)) / np.max(np.abs(ref)))
            print(f"Python call to CUDA library ({impl:6s}, N={N}) completed in {wall / 1e3:.4f} seconds"
                  f"  | kernel {kms:8.3f} ms, inside-lib total {tms:8.3f} ms,"
                  f" Python/ctypes overhead {wall - tms:6.3f} ms | rel err {err:.1e}"
                  f" | NumPy {numpy_ms:8.2f} ms")
            rows.append(dict(impl=f"python_{impl}", N=N, kernel_ms=round(kms, 4), lib_total_ms=round(tms, 4),
                             python_wall_ms=round(wall, 4), numpy_ms=round(numpy_ms, 4), rel_err=f"{err:.2e}"))

    if args.csv:
        p = Path(args.csv); p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists()
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new: w.writeheader()
            w.writerows(rows)
        print(f"appended {len(rows)} rows to {p}")


if __name__ == "__main__":
    main()
