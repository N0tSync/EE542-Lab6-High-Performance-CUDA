#!/usr/bin/env bash
# Part 8.1 / 8.2 demonstration for the report: apply every filter in filters/ to
# one image on the CPU and on the GPU, verify they agree, and save PNGs plus a
# side-by-side figure (input | CPU result | GPU result) per filter.
#
#   scripts/demo_filters.sh                       # uses data/testcard_1024.bin
#   scripts/demo_filters.sh data/photo_1024.bin   # your own image (make it with img_tools.py to_bin)
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=${1:-data/testcard_1024.bin}
STEM=$(basename "$IMG" .bin)
OUT=results/demo/$STEM
mkdir -p "$OUT"

for flt in filters/*.txt; do
    name=$(basename "$flt" .txt)
    build/conv_cpu "$IMG" "$flt" "$OUT/${name}_cpu.bin" > /dev/null
    if [[ -x build/conv_gpu ]]; then
        build/conv_gpu "$IMG" "$flt" "$OUT/${name}_gpu.bin" 1 > /dev/null
        printf "%-15s " "$name"; python3 tools/img_tools.py diff "$OUT/${name}_cpu.bin" "$OUT/${name}_gpu.bin"
        python3 tools/img_tools.py grid "$OUT/${name}_compare.png" "$IMG" "$OUT/${name}_cpu.bin" "$OUT/${name}_gpu.bin" > /dev/null
    else
        python3 tools/img_tools.py grid "$OUT/${name}_compare.png" "$IMG" "$OUT/${name}_cpu.bin" > /dev/null
        echo "$name (CPU only)"
    fi
    python3 tools/img_tools.py to_png "$OUT/${name}_cpu.bin" "$OUT/${name}.png" > /dev/null
done
echo "PNGs in $OUT/"
