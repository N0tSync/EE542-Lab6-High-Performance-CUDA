#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGES=${IMAGES:-"data/testcard_512.bin data/testcard_1024.bin data/testcard_2048.bin data/testcard_4096.bin"}
FILTERS=${FILTERS:-"filters/box3.txt filters/box9.txt filters/box15.txt"}
OUT=results/conv.csv
mkdir -p results out

for f in $IMAGES; do [[ -f $f ]] || { echo "missing $f — run: python3 tools/img_tools.py samples data/"; exit 1; }; done

echo "impl,M,F,kernel_ms,total_ms,process_ms" > "$OUT"
rm -f results/conv_python.csv

ms_now() { date +%s%N | awk '{printf "%.3f", $1/1e6}'; }

run() {   # run <name-for-log> <cmd...>; records the RESULT line plus whole-process wall time
    local t0 t1 line
    t0=$(ms_now)
    line=$("${@:2}" | grep '^RESULT,') || { echo "!! failed: ${*:2}" >&2; return; }
    t1=$(ms_now)
    echo "${line#RESULT,},$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')" >> "$OUT"
    echo "  $1: ${line#RESULT,}"
}

fails=0
for img in $IMAGES; do
    for flt in $FILTERS; do
        tag="$(basename "$img" .bin)_$(basename "$flt" .txt)"
        echo "== $tag =="
        run "C       " build/conv_cpu "$img" "$flt" "out/${tag}_cpu.bin"
        if [[ -x build/conv_gpu ]]; then
            run "CUDA nai" build/conv_gpu "$img" "$flt" "out/${tag}_gpu_naive.bin" 10 naive
            run "CUDA til" build/conv_gpu "$img" "$flt" "out/${tag}_gpu.bin" 10 tiled
            python3 tools/img_tools.py diff "out/${tag}_cpu.bin" "out/${tag}_gpu.bin" || fails=$((fails+1))
        fi
    done
done

if [[ -f build/libmatrix.so ]]; then
    echo "== Python -> libmatrix.so =="
    # shellcheck disable=SC2086
    python3 python/conv_py.py bench --images $IMAGES --filters $FILTERS --numpy \
        --save out --csv results/conv_python.csv
    for img in $IMAGES; do for flt in $FILTERS; do
        tag="$(basename "$img" .bin)_$(basename "$flt" .txt)"
        python3 tools/img_tools.py diff "out/${tag}_cpu.bin" "out/${tag}_py.bin" || fails=$((fails+1))
    done; done
fi

echo
[[ $fails == 0 ]] && echo "All outputs match (|diff| <= 1)." || echo "!! $fails output comparisons FAILED"
echo "Wrote $OUT and results/conv_python.csv. Next: python3 scripts/plot_results.py"
