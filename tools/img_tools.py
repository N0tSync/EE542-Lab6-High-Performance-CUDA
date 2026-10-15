import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_bin(path):
    with open(path, "rb") as f:
        w, h = np.frombuffer(f.read(8), dtype=np.int32)
        data = np.frombuffer(f.read(), dtype=np.uint32)
    return data.reshape(int(h), int(w))


def write_bin(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.uint32)
    h, w = arr.shape
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.array([w, h], dtype=np.int32).tobytes())
        f.write(arr.tobytes())


def load_gray(path, size=None):
    img = Image.open(path).convert("L")
    if size:
        # center-crop to square, then resize to size x size (M x M)
        s = min(img.size)
        l, t = (img.width - s) // 2, (img.height - s) // 2
        img = img.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint32)


def to_png(arr, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L").save(path)


def synthetic(size, seed=0):
    """A test card: gradient, circles, rectangles, text-like bars, plus noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = (x + y) / (2 * size) * 120 + 40
    img = Image.fromarray(base.astype(np.uint8), "L")
    d = ImageDraw.Draw(img)
    s = size
    d.ellipse([s * .08, s * .08, s * .42, s * .42], fill=230)
    d.ellipse([s * .16, s * .16, s * .34, s * .34], fill=30)
    d.rectangle([s * .55, s * .10, s * .90, s * .40], fill=200)
    d.polygon([(s * .10, s * .90), (s * .30, s * .55), (s * .50, s * .90)], fill=10)
    for i in range(8):
        x0 = s * (.58 + .04 * i)
        d.rectangle([x0, s * .55, x0 + s * .02, s * .90], fill=250 if i % 2 else 60)
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 6, (size, size))
    return np.clip(arr, 0, 255).astype(np.uint32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("to_bin"); a.add_argument("src"); a.add_argument("dst"); a.add_argument("--size", type=int)
    a = sub.add_parser("to_png"); a.add_argument("src"); a.add_argument("dst")
    a = sub.add_parser("samples"); a.add_argument("outdir")
    a.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    a.add_argument("--from-image", help="use this photo (resized) instead of the synthetic test card")
    a = sub.add_parser("diff"); a.add_argument("a"); a.add_argument("b")
    a = sub.add_parser("grid"); a.add_argument("dst"); a.add_argument("bins", nargs="+")
    args = p.parse_args()

    if args.cmd == "to_bin":
        arr = load_gray(args.src, args.size)
        write_bin(args.dst, arr)
        print(f"wrote {args.dst}: {arr.shape[1]}x{arr.shape[0]}")
    elif args.cmd == "to_png":
        to_png(read_bin(args.src), args.dst)
        print(f"wrote {args.dst}")
    elif args.cmd == "samples":
        out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
        stem = Path(args.from_image).stem if args.from_image else "testcard"
        for s in args.sizes:
            arr = load_gray(args.from_image, s) if args.from_image else synthetic(s)
            write_bin(out / f"{stem}_{s}.bin", arr)
            to_png(arr, out / f"{stem}_{s}.png")
            print(f"wrote {out / f'{stem}_{s}.bin'}")
    elif args.cmd == "diff":
        a, b = read_bin(args.a).astype(np.int64), read_bin(args.b).astype(np.int64)
        if a.shape != b.shape:
            print(f"shape mismatch {a.shape} vs {b.shape}"); sys.exit(1)
        d = np.abs(a - b)
        print(f"max |diff| = {d.max()}, pixels differing = {(d > 0).sum()} / {d.size}")
        sys.exit(0 if d.max() <= 1 else 1)   # +-1 allowed for float rounding order
    elif args.cmd == "grid":
        imgs = [read_bin(b) for b in args.bins]
        h = max(i.shape[0] for i in imgs)
        gap = 8
        canvas = np.full((h, sum(i.shape[1] for i in imgs) + gap * (len(imgs) - 1)), 255, np.uint32)
        x = 0
        for i in imgs:
            canvas[: i.shape[0], x : x + i.shape[1]] = i
            x += i.shape[1] + gap
        to_png(canvas, args.dst)
        print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
