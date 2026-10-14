#!/usr/bin/env python3
"""Part 8.3: a Python program accelerated by the CUDA convolution in libmatrix.so.

Two modes:

  edges  — the "program": an edge-detection pipeline (Gaussian blur -> Sobel X and Y
           -> gradient magnitude) on every image given. Runs it once with the CUDA
           library and once with pure NumPy, checks they match, saves PNGs, and
           reports the speedup.

      python3 python/conv_py.py edges data/testcard_1024.bin data/photo_2048.bin --out results/edges

  bench  — times single-filter convolutions so they can be compared row-for-row
           with ./build/conv_cpu (C) and ./build/conv_gpu (CUDA executable).
           Use the same image and filter files; scripts/bench_conv.sh does all three.

      python3 python/conv_py.py bench --images data/*.bin --filters filters/box3.txt filters/box9.txt \\
              --csv results/conv.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from img_tools import read_bin, write_bin, to_png       # noqa: E402
from cudalib import CudaLib, convolve_numpy              # noqa: E402


def read_filter(path):
    toks = Path(path).read_text().split()
    F = int(toks[0])
    return np.array(toks[1:1 + F * F], dtype=np.float32).reshape(F, F)


def gauss5():
    r = 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(x * x + y * y) / 2.0)
    return (k / k.sum()).astype(np.float32)


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
SOBEL_Y = SOBEL_X.T.copy()


def edge_pipeline(img, conv, conv_batch=None):
    blurred = conv(img, gauss5())
    if conv_batch is not None:
        gx, gy = conv_batch(blurred, [SOBEL_X, SOBEL_Y])     # one upload, two filters
    else:
        gx, gy = conv(blurred, SOBEL_X), conv(blurred, SOBEL_Y)
    mag = np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)
    return blurred, np.minimum(mag, 255).astype(np.uint32)


def cmd_edges(args, lib):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {lib.device_name}")
    print(f"{'image':28s} {'size':>9s} {'NumPy ms':>10s} {'CUDA lib ms':>12s} {'speedup':>8s}  match")
    for path in args.images:
        img = read_bin(path)
        edge_pipeline(img, lib.convolve, lib.convolve_batch)            # warm-up

        t0 = time.perf_counter()
        blur_np, edges_np = edge_pipeline(img, convolve_numpy)
        t_np = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        blur_gpu, edges_gpu = edge_pipeline(img, lib.convolve, lib.convolve_batch)
        t_gpu = (time.perf_counter() - t0) * 1e3

        maxdiff = int(np.abs(edges_np.astype(np.int64) - edges_gpu.astype(np.int64)).max())
        stem = Path(path).stem
        to_png(img, out / f"{stem}_0_input.png")
        to_png(blur_gpu, out / f"{stem}_1_blur.png")
        to_png(edges_gpu, out / f"{stem}_2_edges.png")
        write_bin(out / f"{stem}_edges.bin", edges_gpu)
        print(f"{stem:28s} {img.shape[1]:>4d}x{img.shape[0]:<4d} {t_np:10.2f} {t_gpu:12.2f} "
              f"{t_np / t_gpu:7.1f}x  {'OK' if maxdiff <= 2 else f'DIFF {maxdiff}'}")
    print(f"PNGs written to {out}/")


def cmd_bench(args, lib):
    rows = []
    print(f"GPU: {lib.device_name}")
    for ipath in args.images:
        img = read_bin(ipath)
        for fpath in args.filters:
            k = read_filter(fpath)
            lib.convolve(img, k)                                          # warm-up
            walls = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                res = lib.convolve(img, k)
                walls.append((time.perf_counter() - t0) * 1e3)
            wall = min(walls)
            kms, tms = lib.last_kernel_ms, lib.last_total_ms
            M, F = img.shape[1], k.shape[0]
            line = (f"python+CUDA lib: image {M}x{img.shape[0]}, filter {F}x{F}: python wall {wall:.4f} ms "
                    f"(kernel {kms:.4f} ms, lib total {tms:.4f} ms)")
            row = dict(impl="conv_python_cuda", M=M, F=F, kernel_ms=round(kms, 4), total_ms=round(wall, 4))
            if args.numpy:
                t0 = time.perf_counter()
                ref = convolve_numpy(img, k)
                t_np = (time.perf_counter() - t0) * 1e3
                d = int(np.abs(ref.astype(np.int64) - res.astype(np.int64)).max())
                line += f" | numpy {t_np:.2f} ms, max diff {d}"
                rows.append(dict(impl="conv_numpy", M=M, F=F, kernel_ms=round(t_np, 4), total_ms=round(t_np, 4)))
            print(line)
            rows.append(row)
            if args.save:
                Path(args.save).mkdir(parents=True, exist_ok=True)
                write_bin(Path(args.save) / f"{Path(ipath).stem}_{Path(fpath).stem}_py.bin", res)

    if args.csv:
        p = Path(args.csv); p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists()
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["impl", "M", "F", "kernel_ms", "total_ms"])
            if new: w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", help="path to libmatrix.so")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("edges"); e.add_argument("images", nargs="+"); e.add_argument("--out", default="results/edges")
    b = sub.add_parser("bench")
    b.add_argument("--images", nargs="+", required=True)
    b.add_argument("--filters", nargs="+", required=True)
    b.add_argument("--reps", type=int, default=5)
    b.add_argument("--numpy", action="store_true", help="also time the pure-NumPy version and verify")
    b.add_argument("--save", help="directory to save outputs")
    b.add_argument("--csv")
    args = ap.parse_args()

    lib = CudaLib(args.lib)
    lib.warmup()
    {"edges": cmd_edges, "bench": cmd_bench}[args.cmd](args, lib)


if __name__ == "__main__":
    main()
