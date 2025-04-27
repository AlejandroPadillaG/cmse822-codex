/*  SL_methods_local5.cpp  –  serial Semi-Lagrangian update
 *  (8-argument version that matches SL_methods_local5.h)
 */
#include "SL_methods_local5.h"

#include <vector>
#include <stdexcept>
#include <array>
#include <cmath>

#include "ghost_ext.h"
#include "compute_weno5_Qiu2.h"

/* ------------------------------------------------------------------ */
std::vector<double>
SL_methods_local5(const std::vector<double>& f_prev,
                  const std::vector<double>& x,      /* only for size check */
                  double  v,
                  int     p,
                  double  dt,
                  double  dx,
                  int     Nx,
                  double  epsilon)
{
    /* --- sanity checks ------------------------------------------------- */
    if ((int)f_prev.size() != Nx + 2*p)
        throw std::runtime_error("SL_methods_local5: f_prev wrong size");
    if ((int)x.size()      != Nx + 2*p)
        throw std::runtime_error("SL_methods_local5: x wrong size");

    /* --- storage for reconstructed interface values -------------------- */
    std::vector<std::array<double,2>> f_m12(Nx);

    /* --- integer / fractional shift ------------------------------------ */
    double xshift = v * dt / dx;
    int    x_index = 0;
    while (xshift >  0.5) { xshift -= 1.0; --x_index; }
    while (xshift < -0.5) { xshift += 1.0; ++x_index; }
    double xs_pos = std::fabs(xshift);

    /* ====================== reconstruction loop ======================== */
    for (int i = 0; i < Nx; ++i)
    {
        int j;
        if (xshift >= 0.0) {                /* left-biased */
            j = i + x_index;
            while (j < 0)   j += Nx;
            while (j >= Nx) j -= Nx;

            std::array<double,5> fv;
            for (int off=-3; off<=1; ++off)
                fv[off+3] = f_prev[j + p + off];

            f_m12[i][0] = compute_weno5_Qiu2(fv,xs_pos,epsilon);
            f_m12[i][1] = j;
        } else {                            /* right-biased */
            j = i + 1 + x_index;
            while (j < 0)   j += Nx;
            while (j >= Nx) j -= Nx;

            std::array<double,5> tmp,fv;
            for (int off=-3; off<=1; ++off)
                tmp[off+3] = f_prev[j + p + off];
            for (int k=0;k<5;++k) fv[4-k]=tmp[k];

            j = (j-1+Nx)%Nx;                /* update index after flip */

            f_m12[i][0] = compute_weno5_Qiu2(fv,xs_pos,epsilon);
            f_m12[i][1] = j;
        }
    }

    /* =========================== update ================================= */
    std::vector<double> f_new(Nx);
    for (int i=0;i<Nx;++i){
        int ip1=(i+1)%Nx;
        int j   = static_cast<int>(f_m12[i][1]);
        double fR= f_m12[ip1][0];
        double fL= f_m12[i  ][0];
        f_new[i]= f_prev[j+p] - xshift*(fR-fL);
    }

    /* return array with ghost cells for next step ----------------------- */
    return ghost_ext(Nx,p,f_new);
}
