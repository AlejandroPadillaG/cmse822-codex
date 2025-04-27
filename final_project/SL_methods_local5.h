/*  SL_methods_local5.h
 *  prototypes for both serial and OpenMP versions
 */
#pragma once
#include <vector>

/* ------------ serial version ---------------- */
std::vector<double>
SL_methods_local5(const std::vector<double>& f_prev,
                  const std::vector<double>& x,
                  double v, int p,
                  double dt, double dx,
                  int Nx,
                  double epsilon);

/* ------------ OpenMP version ---------------- */
std::vector<double>
SL_methods_local5_omp(const std::vector<double>& f_prev,
                      const std::vector<double>& x,
                      double v, int p,
                      double dt, double dx,
                      int Nx,
                      double epsilon);
