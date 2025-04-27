#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

std::vector<double> ghost_ext(int Nx, int p,
                              const std::vector<double> &f_in)
{
    // f_in has length Nx
    // We'll return Nx + 2p
    if (static_cast<int>(f_in.size()) != Nx) {
        throw std::runtime_error("ghost_ext: f_in.size() != Nx");
    }
    std::vector<double> f_out(Nx + 2*p, 0.0);

    // main domain: f_out[p..p+Nx-1] = f_in[0..Nx-1]
    for (int i = 0; i < Nx; ++i) {
        f_out[p + i] = f_in[i];
    }

    // Periodic BC: left ghost cells
    //   f_out[0..p-1] = f_in[Nx-p.. Nx-1]
    for (int i = 0; i < p; ++i) {
        f_out[i] = f_in[Nx - p + i];
    }

    // Periodic BC: right ghost cells
    //   f_out[Nx+p.. Nx+2p-1] = f_in[0.. p-1]
    for (int i = 0; i < p; ++i) {
        f_out[Nx + p + i] = f_in[i];
    }

    return f_out;
}