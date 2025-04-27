#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Benchmark driver for serial (CPU), OpenMP, and MPI+OpenMP variants
# using NVHPC for single‐rank builds and GNU mpicxx for the hybrid build.
#
# Produces sl_bench.csv with columns:
#    Ranks,Nx,Variant,Time_s
# -------------------------------------------------------------------------
module load NVHPC/23.7-CUDA-12.1.1     # adjust for your cluster

set -euo pipefail

# ------------ user tunables -------------------------------------------------
SIZES=(16 32 64 128 256 512 1024 2048 4096)   # for example; strong‐scaling sizes
RANKS=(1 2 4 8 16)          # numbers of MPI ranks to test
CSV="sl_bench.csv"
NVC="nvc++"
TIMEOUT=1200

CXXFLAGS="-std=c++20 -O3"
OMPFLAG="-mp=multicore"

SERIAL_EXE="SL_cpu"
OMP_EXE="SL_omp"
MPI_EXE="SL_mpi"

IMPL_SRC="SL_methods_local5.cpp SL_methods_local5_omp.cpp"
COMMON_SRC="compute_weno5_Qiu2.cpp ghost_ext.cpp"
SERIAL_SRC="$IMPL_SRC weno5_perf_serial_vs_omp.cpp"
OMP_SRC="$IMPL_SRC weno5_perf_serial_vs_omp.cpp"
MPI_SRC="$IMPL_SRC SL_methods_local5_mpi.cpp weno5_perf_mpi.cpp"

# ------------ build ---------------------------------------------------------
echo "[build] SERIAL …"
$NVC $CXXFLAGS                  $SERIAL_SRC $COMMON_SRC -o $SERIAL_EXE

echo "[build] OpenMP …"
$NVC $CXXFLAGS $OMPFLAG         $OMP_SRC    $COMMON_SRC -o $OMP_EXE

echo "[build] MPI+OpenMP …"
mpicxx $CXXFLAGS -fopenmp       $MPI_SRC    $COMMON_SRC -o $MPI_EXE

echo "[build] done."
# ----------------------------------------------------------------------------

export OMP_NUM_THREADS=1         # serial/OpenMP each on 1 rank
regex='runtime[[:space:]]*=[[:space:]]*([0-9eE.+-]+)'

run_case () {
    local ranks=$1
    local exe=$2
    local variant=$3
    local nx=$4
    local extra_arg=${5-}          # default = empty
    printf "[run] %-3s Ranks=%-2d Nx=%-5d … " \
           "$variant" "$ranks" "$nx"

    if [[ "$variant" == "MPI" ]]; then
        cmd=(mpirun -np "$ranks" ./"$exe" "$nx")
    else
        if [[ -n "$extra_arg" ]]; then
            cmd=(./"$exe" "$nx" "$extra_arg")
        else
            cmd=(./"$exe" "$nx")
        fi
    fi

    if ! out=$(timeout ${TIMEOUT}s "${cmd[@]}" 2>&1); then
        echo "timeout/err"
        return
    fi
    if [[ $out =~ $regex ]]; then
        t=${BASH_REMATCH[1]}
        echo "$t s"
        printf "%d,%d,%s,%s\n" \
               "$ranks" "$nx" "$variant" "$t" >> "$CSV"
    else
        echo "no-match"
        echo "$out" >&2
    fi
}

# ------------ run loop ------------------------------------------------------
printf "Ranks,Nx,Variant,Time_s\n" > "$CSV"
for ranks in "${RANKS[@]}"; do
  for nx in "${SIZES[@]}"; do

    # SERIAL: only sensible when ranks=1
    if [[ $ranks -eq 1 ]]; then
      run_case 1 "$SERIAL_EXE" "CPU" "$nx"
      run_case 1 "$OMP_EXE"    "OMP" "$nx" "omp"
    fi

    # MPI: run on given number of ranks
    run_case "$ranks" "$MPI_EXE" "MPI" "$nx"

    echo
  done
done

echo "[info] timings saved to $CSV"
echo "[done] benchmark complete."
