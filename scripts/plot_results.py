#!/usr/bin/env python3
"""Turn the benchmark CSVs into report figures and tables.

    python3 scripts/plot_results.py            # reads results/*.csv, writes results/figures/*.png + results/tables.md

Figures
  matmul_time.png          execution time vs N, all implementations (log-log)    <- video log plot
  matmul_gflops.png        throughput vs N
  matmul_overhead.png      share of a GPU call spent outside the kernel (alloc + PCIe copies)
  matmul_speedup.png       speedup over the CPU baseline
  conv_time_F<k>.png       convolution time vs image size M, one chart per filter size
Tables (Markdown, paste into the report)
  Part 5/6 table (kernel time and end-to-end time), speedups, convolution comparison.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os

ROOT = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("LAB6_RESULTS", ROOT / "results")).resolve()
FIG = RES / "figures"

# Fixed color + marker per implementation (color follows the entity, never its rank).
# Validated categorical palette; marker shapes are the second encoding so identity
# never depends on color alone.
STYLE = {
    "cpu_ijk":           ("CPU (C, ijk)",            "#2a78d6", "o"),
    "cpu_ikj":           ("CPU (C, ikj loop order)", "#6250d6", "v"),
    "naive_cuda":        ("Naive CUDA",              "#eb6834", "s"),
    "tiled_cuda":        ("Tiled CUDA (16x16)",      "#1baf7a", "D"),
    "tiled32_cuda":      ("Tiled CUDA (32x32)",      "#e87ba4", "P"),
    "optimized_cuda":    ("Optimized CUDA (6.2)",    "#e34948", "^"),
    "cublas":            ("cuBLAS",                  "#eda100", "X"),
    "python_tiled":      ("Python -> lib (tiled)",   "#008300", "h"),
    "conv_cpu":          ("C (CPU)",                 "#2a78d6", "o"),
    "conv_gpu_naive":    ("CUDA exe (naive)",        "#eb6834", "s"),
    "conv_gpu_tiled":    ("CUDA exe (tiled)",        "#1baf7a", "D"),
    "conv_python_cuda":  ("Python -> CUDA lib",      "#eda100", "X"),
    "conv_numpy":        ("Python (NumPy, no GPU)",  "#e87ba4", "P"),
}
ORDER = list(STYLE)

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def read_csv(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def series(rows, key_x, key_y, filt=None):
    out = defaultdict(dict)
    for r in rows:
        if filt and not filt(r):
            continue
        try:
            out[r["impl"]][int(r[key_x])] = float(r[key_y])
        except (ValueError, KeyError):
            pass
    return out


def plot_lines(data, title, xlabel, ylabel, path, logy=True, logx=True):
    data = {k: v for k, v in data.items() if v}
    if not data:
        print(f"skip {path.name} (no data yet)")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for impl in sorted(data, key=lambda k: ORDER.index(k) if k in ORDER else 99):
        pts = sorted(data[impl].items())
        if not pts:
            continue
        label, color, marker = STYLE.get(impl, (impl, "#6b6a64", "o"))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker,
                label=label, markeredgecolor="#fcfcfb", markeredgewidth=1.2)
    if logx: ax.set_xscale("log", base=2)
    if logy: ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    xs = sorted({x for d in data.values() for x in d})
    if logx and xs:
        ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs], rotation=45 if len(xs) > 8 else 0)
        ax.minorticks_off()
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def fmt_ms(v):
    if v is None:
        return "—"
    if v >= 1000:
        return f"{v / 1000:.2f} s"
    return f"{v:.3f} ms" if v < 10 else f"{v:.1f} ms"


def md_table(data, sizes, impls, title):
    lines = [f"**{title}**", "", "| Implementation | " + " | ".join(f"N={n}" for n in sizes) + " |",
             "|---|" + "---|" * len(sizes)]
    for impl in impls:
        if impl not in data:
            continue
        lines.append(f"| {STYLE.get(impl, (impl,))[0]} | " +
                     " | ".join(fmt_ms(data[impl].get(n)) for n in sizes) + " |")
    return "\n".join(lines) + "\n"


def speedup_table(data, base, sizes, impls, title):
    lines = [f"**{title}**", "", "| Implementation | " + " | ".join(f"N={n}" for n in sizes) + " |",
             "|---|" + "---|" * len(sizes)]
    for impl in impls:
        if impl not in data or impl == base:
            continue
        cells = []
        for n in sizes:
            b, v = data.get(base, {}).get(n), data[impl].get(n)
            cells.append(f"{b / v:,.1f}x" if b and v else "—")
        lines.append(f"| {STYLE.get(impl, (impl,))[0]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    md = ["# Lab 6 results\n"]

    # ------------------------------------------------------------ matrix multiply
    mm = read_csv(RES / "matmul.csv")
    py = read_csv(RES / "matmul_python.csv")
    for r in py:
        r["kernel_ms"], r["total_ms"] = r["kernel_ms"], r["python_wall_ms"]
    if mm:
        kern = series(mm, "N", "kernel_ms")
        total = series(mm, "N", "total_ms")
        gfl = series(mm, "N", "gflops")
        if py:
            ptot = series(py, "N", "total_ms", lambda r: r["impl"] == "python_tiled")
            total.update(ptot)
        impls = [i for i in ORDER if i in kern]
        gpu_impls = [i for i in impls if not i.startswith("cpu")]

        plot_lines(kern, "Matrix multiply: execution time vs N", "Matrix size N",
                   "Time (ms, log scale) — GPU: kernel only", FIG / "matmul_time.png")
        plot_lines(total, "Matrix multiply: end-to-end time vs N (incl. alloc + PCIe copies)",
                   "Matrix size N", "Time (ms, log scale)", FIG / "matmul_time_total.png")
        plot_lines({k: v for k, v in gfl.items()}, "Matrix multiply: throughput",
                   "Matrix size N", "GFLOP/s (log scale)", FIG / "matmul_gflops.png")

        over = {}
        for i in gpu_impls:
            over[i] = {n: 100 * (total[i][n] - kern[i][n]) / total[i][n]
                       for n in kern[i] if n in total.get(i, {}) and total[i][n] > 0}
        plot_lines(over, "Share of each GPU call spent outside the kernel",
                   "Matrix size N", "Overhead (% of end-to-end time)", FIG / "matmul_overhead.png", logy=False)

        base = "cpu_ijk" if "cpu_ijk" in kern else None
        if base:
            sp = {i: {n: kern[base][n] / kern[i][n] for n in kern[i] if n in kern[base]}
                  for i in impls if i != base}
            plot_lines(sp, "Speedup over CPU (C, ijk), kernel time", "Matrix size N",
                       "Speedup (x, log scale)", FIG / "matmul_speedup.png")

        sizes = sorted({n for d in kern.values() for n in d})
        table_sizes = [n for n in sizes if n >= 256] or sizes
        md.append("## Matrix multiplication (Parts 1–6)\n")
        md.append(md_table(kern, table_sizes, impls, "Kernel / compute time (GPU rows: kernel only, cudaEvent)"))
        md.append(md_table(total, table_sizes, impls + (["python_tiled"] if py else []),
                           "End-to-end time per call (alloc + H2D + kernel + D2H + free)"))
        if base:
            md.append(speedup_table(kern, base, table_sizes, impls, "Speedup = CPU time / GPU kernel time"))
            md.append(speedup_table(total, base, table_sizes, impls, "Speedup = CPU time / GPU end-to-end time"))
        if "cublas" in kern:
            md.append(speedup_table(kern, "cublas", table_sizes, gpu_impls,
                                    "Fraction of cuBLAS speed (cuBLAS time / kernel time; 1.0x = parity)"))

    # ------------------------------------------------------------ convolution
    cv = read_csv(RES / "conv.csv") + read_csv(RES / "conv_python.csv")
    if cv:
        md.append("## Convolution (Part 8)\n")
        for F in sorted({int(r["F"]) for r in cv}):
            rows = [r for r in cv if int(r["F"]) == F]
            kern = series(rows, "M", "kernel_ms")
            total = series(rows, "M", "total_ms")
            plot_lines(total, f"Convolution, {F}x{F} filter: time vs image size",
                       "Image size M (M x M pixels)", "Time per image (ms, log scale)",
                       FIG / f"conv_time_F{F}.png")
            Ms = sorted({n for d in total.values() for n in d})
            impls = [i for i in ORDER if i in total]
            md.append(md_table(total, Ms, impls, f"{F}x{F} filter — end-to-end time per image").replace("N=", "M="))
            md.append(md_table(kern, Ms, [i for i in impls if i.startswith("conv_gpu") or i == "conv_python_cuda"],
                               f"{F}x{F} filter — GPU kernel time only").replace("N=", "M="))
            if "conv_cpu" in total:
                md.append(speedup_table(total, "conv_cpu", Ms, impls,
                                        f"{F}x{F} filter — speedup over C (end-to-end)").replace("N=", "M="))

    (RES / "tables.md").write_text("\n".join(md))
    print(f"wrote {RES / 'tables.md'}")


if __name__ == "__main__":
    main()
