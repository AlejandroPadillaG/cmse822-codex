#ifndef COMPUTE_WENO5_QIU2_H
#define COMPUTE_WENO5_QIU2_H

#include <array>

// Computes the WENO5-Qiu reconstruction for a set of 5 function values
// and a shift parameter xshift using Kahan (compensated) summation.
// 
// Parameters:
//   f_values : std::array<double, 5>
//       Array of 5 function values: f_{i-3}, f_{i-2}, f_{i-1}, f_i, f_{i+1}.
//   xshift : double
//       Shift parameter (e.g., v * dt / dx), which must lie in [0, 0.5].
//   epsilon : double (default: 1e-6)
//       A small constant to avoid division by zero in the smoothness indicators.
// 
// Returns:
//   double : The reconstructed value at the cell interface.
double compute_weno5_Qiu2(const std::array<double, 5>& f_values,
                         double xshift,
                         double epsilon = 1e-6);

#endif // COMPUTE_WENO5_QIU2_H