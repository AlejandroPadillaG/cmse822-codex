#!/usr/bin/env bash

module load NVHPC/23.7-CUDA-12.1.1   # adjust for your environment
module load Python                   # python3 + pandas + matplotlib

set -euo pipefail

# ---------------------- user tunables ---------------------------------------
SIZES=(16 32 64 128 256 512 1024 2048 4096)
CSV="jacobi_bench.csv"
NVC="nvc++"
CPU_SRC="jacobi_serial.cpp"
GPU_SRC="jacobi_gpu.cpp"
CPU_EXE="jacobi_cpu"
GPU_EXE="jacobi_gpu"
TIMEOUT=60

# ---------------------- build ------------------------------------------------
printf "[build] compiling CPU variant …\n" >&2
$NVC -std=c++20 -O3 -mp=multicore "$CPU_SRC" -o "$CPU_EXE"

printf "[build] compiling GPU variant …\n" >&2
$NVC -std=c++20 -O2 -mp=gpu -gpu=ccall "$GPU_SRC" -o "$GPU_EXE"
printf "[build] done.\n" >&2

# ---------------------- runtime env -----------------------------------------
export OMP_NUM_THREADS=$(nproc)
export OMP_TARGET_OFFLOAD=DEFAULT

# ---------------------- benchmark helper ------------------------------------
regex_time='in[[:space:]]+([0-9eE+.-]+)[[:space:]]+s'
regex_iter='Converged after[[:space:]]+([0-9]+)'

declare -a failures=()

run_case() {
    local exe=$1 variant=$2 size=$3
    printf "[run] %-3s  N=%d … " "$variant" "$size"
    if ! out=$(timeout ${TIMEOUT}s ./$exe "$size" 2>&1); then
        echo "failed"; failures+=("$variant:$size"); return 1
    fi
    local time iters
    time=$(grep -Eo "$regex_time" <<< "$out" | awk '{print $(NF-1)}' || true)
    iters=$(grep -Eo "$regex_iter" <<< "$out" | awk '{print $3}' || true)

    if [[ -z "$time" ]]; then
        echo "no-time"; failures+=("$variant:$size"); return 1
    fi
    [[ -z "$iters" ]] && iters="0"
    echo "$time s, $iters iters"
    printf "%d,%s,%s,%s\n" "$size" "$variant" "$time" "$iters" >> "$CSV"
}

# ---------------------- run loop -------------------------------------------
printf "Size,Variant,Time_s,Iter\n" > "$CSV"
for N in "${SIZES[@]}"; do
    run_case "$CPU_EXE" CPU "$N" || true
    run_case "$GPU_EXE" GPU "$N" || true
    echo
done

echo "[info] timings saved to $CSV" >&2
if ((${#failures[@]})); then
    echo "[warn] failures: ${failures[*]}" >&2
fi


echo "[done] benchmark complete."
