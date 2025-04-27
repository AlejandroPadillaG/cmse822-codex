#include "SL_methods_local5_omp.h"
#include "compute_weno5_Qiu2.h"
#include "ghost_ext.h"
#include <array>
#include <cmath>
#include <stdexcept>
#include <omp.h>

std::vector<double>
SL_methods_local5_omp(const std::vector<double>& f_prev,
                      const std::vector<double>& x,
                      double v,int p,double dt,double dx,
                      int Nx,double eps)
{
    if ((int)f_prev.size()!=Nx+2*p|| (int)x.size()!=Nx+2*p)
        throw std::runtime_error("size mismatch");

    std::vector<std::array<double,2>> f_m12(Nx);
    double xshift=v*dt/dx;
    int x_index=0;
    while(xshift> 0.5){xshift-=1.0;--x_index;}
    while(xshift<-0.5){xshift+=1.0;++x_index;}
    double xs_pos=std::abs(xshift);

    /* ---------------- reconstruction ---------------- */
#pragma omp parallel for
    for(int i=0;i<Nx;++i){
        int j;
        if(xshift>=0.0){
            j=i+x_index;
            while(j<0) j+=Nx;
            while(j>=Nx)j-=Nx;
            std::array<double,5> fv;
            for(int off=-3;off<=1;++off)
                fv[off+3]=f_prev[j+p+off];
            double fr=compute_weno5_Qiu2(fv,xs_pos,eps);
            f_m12[i][0]=fr;
            f_m12[i][1]=j;
        }else{
            j=i+1+x_index;
            while(j<0) j+=Nx;
            while(j>=Nx)j-=Nx;
            std::array<double,5> tmp,fv;
            for(int off=-3;off<=1;++off)
                tmp[off+3]=f_prev[j+p+off];
            for(int k=0;k<5;++k) fv[4-k]=tmp[k];
            j=(j-1+Nx)%Nx;
            double fr=compute_weno5_Qiu2(fv,xs_pos,eps);
            f_m12[i][0]=fr;
            f_m12[i][1]=j;
        }
    }

    /* ---------------- update ---------------- */
    std::vector<double> f_new(Nx);
#pragma omp parallel for
    for(int i=0;i<Nx;++i){
        int j=(int)f_m12[i][1];
        int ip1=(i+1)%Nx;
        double f_next=f_m12[ip1][0];
        double f_curr=f_m12[i][0];
        f_new[i]=f_prev[j+p]-xshift*(f_next-f_curr);
    }
    return ghost_ext(Nx,p,f_new);
}
