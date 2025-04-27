/*─────────────────────────────────────────────────────────────────────────────
  exchange_ghost_cells.hpp
  ---------------------------------------------------------------------------
  Periodic halo exchange for a 1-D domain–decomposed array.

    fld        : vector length  Nx_loc + 2*p
                 [ left ghosts | interior | right ghosts ]
    Nx_loc     : # interior cells on *this* rank
    p          : # ghost cells on each side
    rank,size  : usual MPI rank info

  After the call:
      fld[0 .. p-1]             ← right-most p interiors of left neighbour
      fld[p+Nx_loc .. end]      ← left-most  p interiors of right neighbour
───────────────────────────────────────────────────────────────────────────*/
#pragma once
#include <mpi.h>
#include <vector>

inline void
exchange_ghost_cells(std::vector<double>& fld,
                     int Nx_loc, int p,
                     int rank, int size)
{
    const int left  = (rank - 1 + size) % size;
    const int right = (rank + 1) % size;

    /* send left-most interior cells → left neighbour
       receive right ghost cells     ← right neighbour */
    MPI_Sendrecv(&fld[p],                  p, MPI_DOUBLE, left,  0,
                 &fld[p + Nx_loc],         p, MPI_DOUBLE, right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* send right-most interior cells → right neighbour
       receive left ghost cells       ← left neighbour */
    MPI_Sendrecv(&fld[p + Nx_loc - p],     p, MPI_DOUBLE, right, 1,
                 &fld[0],                  p, MPI_DOUBLE, left,  1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
