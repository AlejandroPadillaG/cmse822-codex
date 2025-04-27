#include "SL_methods_local5.h"
#include "SL_methods_local5_omp.h"
#include "ghost_ext.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

double run_one(int Nx, bool useOMP)
{
    const double v=1.0, L=2*M_PI, t_final=20.0, CFL=22.1, eps=1e-12;
    int p=3;
    double dx=L/Nx;

    /* grid and initial field */
    std::vector<double> x_full(Nx+2*p);
    for(int i=0;i<Nx;++i) x_full[p+i]=i*dx;
    std::vector<double> f0(Nx);
    for(int i=0;i<Nx;++i) f0[i]=std::sin(i*dx);
    auto fld = ghost_ext(Nx,p,f0);

    auto now = std::chrono::steady_clock::now;
    auto t0  = now();

    double t=0.0;
    while(t<t_final){
        double dt=CFL*dx/std::abs(v);
        if(t+dt>t_final) dt=t_final-t;
        fld = useOMP ?
              SL_methods_local5_omp(fld,x_full,v,p,dt,dx,Nx,eps)
            : SL_methods_local5    (fld,x_full,v,p,dt,dx,Nx,eps);
        t  += dt;
    }
    return std::chrono::duration<double>(now()-t0).count();
}

int main(int argc, char** argv)
{
    // read grid size from command line
    int Nx = (argc > 1) ? std::atoi(argv[1]) : 256;
    // pass "omp" as second argument to invoke the OpenMP variant
    bool useOMP = (argc > 2 && std::string(argv[2]) == "omp");

    double t = run_one(Nx, useOMP);
    std::cout << "[time] runtime = " << t << std::endl;
    return 0;
}
