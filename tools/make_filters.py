#!/usr/bin/env python3
"""Writes the image-processing filters used in Part 8 to filters/*.txt.

File format: first line F, then F rows of F floats.

Filters (see https://en.wikipedia.org/wiki/Kernel_(image_processing)):
  identity3         sanity check: output == input
  box3, box9, box15 mean blur (also give us 3 different filter sizes N for timing)
  gauss5, gauss7    Gaussian blur
  sharpen3          unsharp-style sharpen
  edge_laplace3     Laplacian edge detector (8-neighbour)
  sobel_x3, sobel_y3  Sobel gradient (vertical / horizontal edges)
  emboss3           emboss
"""
from pathlib import Path
import numpy as np


def gaussian(F, sigma):
    r = F // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return k / k.sum()


FILTERS = {
    "identity3": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
    "box3": np.ones((3, 3)) / 9,
    "box9": np.ones((9, 9)) / 81,
    "box15": np.ones((15, 15)) / 225,
    "gauss5": gaussian(5, 1.0),
    "gauss7": gaussian(7, 1.5),
    "sharpen3": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "edge_laplace3": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], float),
    "sobel_x3": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "sobel_y3": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "emboss3": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], float),
}


def write_filter(path, k):
    F = k.shape[0]
    with open(path, "w") as f:
        f.write(f"{F}\n")
        for row in k:
            f.write(" ".join(f"{v:.8g}" for v in row) + "\n")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "filters"
    out.mkdir(exist_ok=True)
    for name, k in FILTERS.items():
        write_filter(out / f"{name}.txt", k)
        print(f"wrote filters/{name}.txt ({k.shape[0]}x{k.shape[0]})")
