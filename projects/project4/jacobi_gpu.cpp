/**
 * jacobi_gpu.cpp  —  OpenMP target-offload Jacobi solver (GPU).
 *
 * Build:   nvc++ -std=c++20 -O3 -mp=gpu -gpu=native jacobi_gpu.cpp -o jacobi_gpu
 * Run  :   ./jacobi_gpu [N]
 */
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include "mm_utils.hpp"

constexpr double TOLERANCE = 1.0e-3;
constexpr int    DEF_SIZE  = 1000;
constexpr int    MAX_ITERS = 100000;
constexpr double LARGE     = 1.0e30;

int main(int argc, char **argv)
{
    const int Ndim = (argc == 2) ? std::atoi(argv[1]) : DEF_SIZE;
    std::cout << "Matrix dimension (Ndim) = " << Ndim << '\n';

    std::vector<TYPE> A(Ndim * Ndim);
    std::vector<TYPE> b(Ndim);
    std::vector<TYPE> xold(Ndim, 0.0);
    std::vector<TYPE> xnew(Ndim, 0.0);

    initDiagDomNearIdentityMatrix(Ndim, A.data());
    for (int i = 0; i < Ndim; ++i)
        b[i] = static_cast<TYPE>(std::rand() % 51) * 0.01 * 0.1;

    TYPE conv = LARGE;
    int  iters = 0;
    const double t0 = omp_get_wtime();

    /*---------------------- persistent device data region ------------------*/
    #pragma omp target data map(to:   A[0:Ndim*Ndim], b[0:Ndim]) \
                            map(tofrom: xold[0:Ndim], xnew[0:Ndim])
    {
        while (conv > TOLERANCE && iters < MAX_ITERS) {
            ++iters;

            /* Jacobi sweep (device) --------------------------------------- */
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < Ndim; ++i) {
                TYPE sigma = 0.0;
                for (int j = 0; j < Ndim; ++j) {
                    // branchless form to avoid divergence
                    sigma += A[i * Ndim + j] * xold[j] * static_cast<TYPE>(i != j);
                }
                xnew[i] = (b[i] - sigma) / A[i * Ndim + i];
            }

            /* Convergence (device) ---------------------------------------- */
            conv = 0.0;
            #pragma omp target teams distribute parallel for map(tofrom:conv) reduction(+:conv)
            for (int i = 0; i < Ndim; ++i) {
                TYPE diff = xnew[i] - xold[i];
                conv += diff * diff;
            }
            conv = std::sqrt(conv);

            /* Copy xnew → xold inside device ------------------------------ */
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < Ndim; ++i)
                xold[i] = xnew[i];
        }
    }

    const double elapsed = omp_get_wtime() - t0;
    std::cout << "Converged after " << iters << " iterations in "
              << elapsed << " s (‖Δx‖₂ = " << conv << ")\n";

    /* Host-side verification ----------------------------------------------- */
    TYPE err = 0.0, chksum = 0.0;
    #pragma omp parallel for reduction(+:err,chksum)
    for (int i = 0; i < Ndim; ++i) {
        TYPE Ax_i = 0.0;
        for (int j = 0; j < Ndim; ++j)
            Ax_i += A[i * Ndim + j] * xnew[j];
        TYPE diff = Ax_i - b[i];
        err    += diff * diff;
        chksum += xnew[i];
    }
    err = std::sqrt(err);
    std::cout << "Verification → ‖Ax−b‖₂ = " << err
              << ", checksum(x) = " << chksum << '\n';
    if (err > TOLERANCE)
        std::cerr << "WARNING: error exceeds tolerance!\n";
}
