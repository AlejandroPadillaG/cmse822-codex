#pragma once
#include <vector>

std::vector<double>
SL_methods_local5_omp(const std::vector<double>& f_prev,
                      const std::vector<double>& x,
                      double v,int p,double dt,double dx,
                      int Nx,double eps);
