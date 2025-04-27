#include "SL_methods_local5_omp.h"
#include "ghost_ext.h"
#include <mpi.h>
#include <cmath>
#include <vector>

/* domain-decomposed SL-update (OpenMP inside each rank) */
void SL_step_MPI(std::vector<double>& fld,   // size Nx_loc+2p
                 int Nx_loc,int p,double v,double dx,
                 double dt,double eps,
                 int rank,int size)
{
    /* ---- exchange ghost cells (periodic) ---- */
    int L=(rank-1+size)%size,R=(rank+1)%size;
    MPI_Request reqs[4];
    MPI_Isend(&fld[p],           p,MPI_DOUBLE,L,0,MPI_COMM_WORLD,&reqs[0]);           // leftmost p -> left
    MPI_Isend(&fld[p+Nx_loc-p],  p,MPI_DOUBLE,R,1,MPI_COMM_WORLD,&reqs[1]);           // rightmost p -> right
    MPI_Irecv(&fld[0],           p,MPI_DOUBLE,L,1,MPI_COMM_WORLD,&reqs[2]);           // fill left ghosts
    MPI_Irecv(&fld[p+Nx_loc],    p,MPI_DOUBLE,R,0,MPI_COMM_WORLD,&reqs[3]);           // fill right ghosts
    MPI_Waitall(4,reqs,MPI_STATUSES_IGNORE);

    /* ---- local OpenMP SL update ---- */
    std::vector<double> interior(fld.begin()+p,fld.begin()+p+Nx_loc);
    interior=SL_methods_local5_omp(fld,fld,v,p,dt,dx,Nx_loc,eps);
    fld=std::move(interior);                    // interior+ghosts rebuilt
}
