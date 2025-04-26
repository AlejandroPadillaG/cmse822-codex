/**
 * jacobi_serial.cpp  —  Pure‑CPU Jacobi iterative solver (OpenMP threads).
 *
 * Build:   nvc++ -std=c++20 -O3 -mp=multicore jacobi_serial.cpp -o jacobi_cpu
 * Run  :   ./jacobi_cpu [N]
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

    double t0 = omp_get_wtime();
    TYPE conv = LARGE;
    int  iters = 0;

    while (conv > TOLERANCE && iters < MAX_ITERS) {
        ++iters;

        /* Jacobi sweep ----------------------------------------------------- */
        #pragma omp parallel for
        for (int i = 0; i < Ndim; ++i) {
            TYPE sigma = 0.0;
            for (int j = 0; j < Ndim; ++j) {
                if (i != j)
                    sigma += A[i * Ndim + j] * xold[j];
            }
            xnew[i] = (b[i] - sigma) / A[i * Ndim + i];
        }

        /* Convergence ------------------------------------------------------ */
        conv = 0.0;
        #pragma omp parallel for reduction(+:conv)
        for (int i = 0; i < Ndim; ++i) {
            TYPE diff = xnew[i] - xold[i];
            conv += diff * diff;
        }
        conv = std::sqrt(conv);

        std::swap(xold, xnew);
    }
    double elapsed = omp_get_wtime() - t0;

    std::cout << "Converged after " << iters << " iterations in "
              << elapsed << " s (‖Δx‖₂ = " << conv << ")\n";

    /* Verification --------------------------------------------------------- */
    TYPE err = 0.0, chksum = 0.0;
    #pragma omp parallel for reduction(+:err,chksum)
    for (int i = 0; i < Ndim; ++i) {
        TYPE Ax_i = 0.0;
        for (int j = 0; j < Ndim; ++j)
            Ax_i += A[i * Ndim + j] * xold[j];
        TYPE diff = Ax_i - b[i];
        err    += diff * diff;
        chksum += xold[i];
    }
    err = std::sqrt(err);

    std::cout << "Verification → ‖Ax−b‖₂ = " << err
              << ", checksum(x) = " << chksum << '\n';
    if (err > TOLERANCE)
        std::cerr << "WARNING: error exceeds tolerance!\n";
}
