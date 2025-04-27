// compute_weno5_Qiu.cpp

#include <array>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <numeric> 

/**
 * @brief Compute the WENO5-Qiu reconstruction for a set of 5 function values
 *        and shift xshift.
 *
 * @param f_values Array of 5 function values: f_{i-3}, f_{i-2}, f_{i-1}, f_i, f_{i+1}.
 * @param xshift   Shift parameter (e.g. v * dt / dx), must be in [0, 0.5].
 * @param epsilon  Small constant to avoid division by zero (default 1e-6).
 * @return Reconstructed value at the cell interface.
 *
 */
double compute_weno5_Qiu2(const std::array<double, 5>& f_values,
                         double xshift,
                         double epsilon = 1e-6)
{
    if (xshift < 0.0 || xshift > 0.5) {
        throw std::runtime_error("xshift must be within [0, 0.5].");
    }

    // Coefficient matrices and vectors as defined in your algorithm
    std::array<std::array<double, 5>, 5> C = {{
        {{  1.0/30.0,    0.0,          -1.0/24.0,     0.0,           1.0/120.0 }},
        {{ -13.0/60.0,  -1.0/24.0,      1.0/4.0,      1.0/24.0,     -1.0/30.0  }},
        {{  47.0/60.0,   5.0/8.0,      -1.0/3.0,     -1.0/8.0,       1.0/20.0  }},
        {{  9.0/20.0,   -5.0/8.0,       1.0/12.0,     1.0/8.0,      -1.0/30.0  }},
        {{ -1.0/20.0,    1.0/24.0,      1.0/24.0,    -1.0/24.0,      1.0/120.0 }}
    }};

    std::array<std::array<double, 5>, 3> ci = {{
        {{  1.0/3.0,  -7.0/6.0,  11.0/6.0,  0.0,       0.0    }},
        {{  0.0,      -1.0/6.0,  5.0/6.0,   1.0/3.0,   0.0    }},
        {{  0.0,       0.0,      1.0/3.0,   5.0/6.0,  -1.0/6.0}}
    }};

    std::array<double, 3> di = {{0.1, 0.6, 0.3}};

    std::array<std::array<double, 5>, 3> D1 = {{
        {{ 1.0, -4.0,  3.0,  0.0,  0.0 }},
        {{ 0.0,  1.0,  0.0, -1.0,  0.0 }},
        {{ 0.0,  0.0,  3.0, -4.0,  1.0 }}
    }};
    std::array<std::array<double, 5>, 3> D2 = {{
        {{ 1.0, -2.0,  1.0,  0.0,  0.0 }},
        {{ 0.0,  1.0, -2.0,  1.0,  0.0 }},
        {{ 0.0,  0.0,  1.0, -2.0,  1.0 }}
    }};

    std::array<double, 3> D1f, D2f, beta;

    // Compute D1f and D2f using Kahan summation
    for (int row = 0; row < 3; ++row) {
        double sumD1 = 0.0, cD1 = 0.0;
        double sumD2 = 0.0, cD2 = 0.0;
        for (int col = 0; col < 5; ++col) {
            double term1 = D1[row][col] * f_values[col];
            double y1 = term1 - cD1;
            double t1 = sumD1 + y1;
            cD1 = (t1 - sumD1) - y1;
            sumD1 = t1;

            double term2 = D2[row][col] * f_values[col];
            double y2 = term2 - cD2;
            double t2 = sumD2 + y2;
            cD2 = (t2 - sumD2) - y2;
            sumD2 = t2;
        }
        D1f[row] = sumD1;
        D2f[row] = sumD2;
    }

    // Compute beta[row] = (D1f[row])^2 + (13/3)*(D2f[row])^2
    for (int row = 0; row < 3; ++row) {
        double sqD1 = D1f[row] * D1f[row];
        double sqD2 = D2f[row] * D2f[row];
        beta[row] = sqD1 + (13.0/3.0) * sqD2;
    }

    // Compute weights: wi = di / ((beta + epsilon)^2)
    std::array<double, 3> wi;
    for (int i = 0; i < 3; ++i) {
        double denom = (beta[i] + epsilon);
        denom = denom * denom;  // (beta + epsilon)^2
        wi[i] = di[i] / denom;
    }
    // Normalize the weights using Kahan summation
    double sum_wi = 0.0, c_wi = 0.0;
    for (int i = 0; i < 3; ++i) {
        double y = wi[i] - c_wi;
        double t = sum_wi + y;
        c_wi = (t - sum_wi) - y;
        sum_wi = t;
    }
    for (int i = 0; i < 3; ++i) {
        wi[i] /= sum_wi;
    }

    // Compute rowVector[k] = sum_{i=0}^{2} wi[i]*ci[i][k] for k = 0,...,4 using Kahan summation
    std::array<double, 5> rowVector;
    for (int k = 0; k < 5; ++k) {
        double sumVal = 0.0, c_val = 0.0;
        for (int i = 0; i < 3; ++i) {
            double term = wi[i] * ci[i][k];
            double y = term - c_val;
            double t = sumVal + y;
            c_val = (t - sumVal) - y;
            sumVal = t;
        }
        rowVector[k] = sumVal;
    }

    // Replace first column of C with rowVector (transposed)
    for (int r = 0; r < 5; ++r) {
        C[r][0] = rowVector[r];
    }

    // Define the polynomial vector
    std::array<double, 5> poly = {{
        1.0,
        xshift,
        xshift * xshift,
        xshift * xshift * xshift,
        xshift * xshift * xshift * xshift
    }};

    // Compute z[i] = sum_{k=0}^{4} C[i][k] * poly[k] for i = 0,...,4 using Kahan summation
    std::array<double, 5> z;
    for (int i = 0; i < 5; ++i) {
        double sumVal = 0.0, c_val = 0.0;
        for (int k = 0; k < 5; ++k) {
            double term = C[i][k] * poly[k];
            double y = term - c_val;
            double t = sumVal + y;
            c_val = (t - sumVal) - y;
            sumVal = t;
        }
        z[i] = sumVal;
    }

    // Compute f_m = sum_{i=0}^{4} f_values[i] * z[i] using Kahan summation
    double f_m = 0.0, c_fm = 0.0;
    for (int i = 0; i < 5; ++i) {
        double term = f_values[i] * z[i];
        double y = term - c_fm;
        double t = f_m + y;
        c_fm = (t - f_m) - y;
        f_m = t;
    }

    return f_m;
}