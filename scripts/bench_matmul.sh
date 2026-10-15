#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

CPU_SIZES=${CPU_SIZES:-"64 128 256 384 512 768 1024 1536 2048"}
GPU_SIZES=${GPU_SIZES:-"64 128 256 384 512 768 1024 1536 2048 3072 4096 6144 8192"}
REPS=${REPS:-5}
OUT=${OUT:-results/matmul.csv}

mkdir -p results
echo "impl,N,kernel_ms,total_ms,gflops,max_rel_err" > "$OUT"

run() {   # run <binary> <args...>  -> append its RESULT line
    local line
    if line=$("$@" | tee /dev/stderr | grep '^RESULT,'); then
        echo "${line#RESULT,}" >> "$OUT"
    else
        echo "!! failed: $*" >&2
    fi
}

echo "== CPU (C) =="
for N in $CPU_SIZES; do run build/matrix_cpu "$N" ijk; done
if [[ "${CPU_IKJ:-1}" == 1 ]]; then
    for N in $CPU_SIZES; do run build/matrix_cpu "$N" ikj; done
fi

for bin in matrix_gpu matrix_tiled matrix_tiled32 matrix_opt matrix_cublas; do
    [[ -x build/$bin ]] || { echo "skipping $bin (not built)"; continue; }
    echo "== $bin =="
    for N in $GPU_SIZES; do run "build/$bin" "$N" "$REPS"; done
done

echo
echo "Wrote $OUT. Next: python3 python/matrix_py.py --sizes 512 1024 2048 4096 --csv results/matmul_python.csv"
echo "      then:  python3 scripts/plot_results.py"
