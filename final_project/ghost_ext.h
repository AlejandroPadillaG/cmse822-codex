#ifndef GHOST_EXT_H
#define GHOST_EXT_H

#include <vector>

/**
 * @brief Extends the solution array with ghost cells for periodic BC.
 *
 * Equivalent to the MATLAB function:
 *    f_curr_new = ghost_ext(Nx, p, f_curr)
 *
 * @param Nx   Number of physical grid points (without ghost cells).
 * @param p    Number of ghost cells on each side.
 * @param f_curr A std::vector<double> of length Nx.
 * @return A std::vector<double> of length Nx + 2*p with periodic boundary cells.
 */
std::vector<double> ghost_ext(int Nx, int p, const std::vector<double> &f_curr);

#endif // GHOST_EXT_H