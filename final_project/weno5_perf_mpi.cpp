#include "SL_methods_local5_omp.h"
#include "exchange_ghost_cells.hpp"
#include "ghost_ext.h"
#include <mpi.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

/* run a single MPI solve and return wall-time (seconds, rank-0) */
double run_one_mpi(int Nx, int rank, int size)
{
    const double v=1.0, L=2*M_PI, t_final=20.0, CFL=22.1;
    const int    p=3;
    double dx=L/Nx;

    /* local mesh -------------------------------------------------------- */
    int Nx_loc = Nx / size;                       // assume Nx%size==0
    std::vector<double> fld(Nx_loc+2*p);
    double x0 = rank*Nx_loc*dx;
    for(int i=0;i<Nx_loc;++i) fld[p+i]=std::sin(x0+i*dx);
    fld = ghost_ext(Nx_loc,p,{fld.begin()+p,fld.begin()+p+Nx_loc});

    auto tic = std::chrono::steady_clock::now();

    /* one simple SL march (same as before) ------------------------------ */
    double t=0.0;
    while(t<t_final){
        double dt=CFL*dx/std::abs(v);
        if(t+dt>t_final) dt=t_final-t;
        exchange_ghost_cells(fld,Nx_loc,p,rank,size);   // your existing helper
        fld = SL_methods_local5_omp(fld,fld,v,p,dt,dx,Nx_loc,1e-12);
        t += dt;
    }
    double local   = std::chrono::duration<double>
                     (std::chrono::steady_clock::now()-tic).count();
    double global;
    MPI_Reduce(&local,&global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    return global;          // valid only on rank 0
}

int main(int argc,char**argv)
{
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int Nx = (argc>1)? std::atoi(argv[1]) : 256;
    double t = run_one_mpi(Nx,rank,size);

    if(rank==0)
        std::cout << "[time] runtime = " << t << std::endl;

    MPI_Finalize();
    return 0;
}
