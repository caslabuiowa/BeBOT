#include "bernstein_helpers.hpp"
#include "common.hpp"

#include <numeric>

namespace bebot::bernstein::internal {
double binom(int n, int k) {
    // Stolen from cppreference :P
    // TODO: No idea how efficient this is
    return 1 / ((n + 1) * std::beta(n - k + 1, k + 1));
}

Eigen::MatrixXd bezier_coefficients(int n) {
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(n + 1, n + 1);
    for (auto k : range(0, n + 1)) {
        for (auto i : range(k, n + 1)) {
            coefficients(i, k) = pow(-1, i - k) * binom(n, i) * binom(i, k);
        }
    }
    return coefficients;
}

// For computing norm
Eigen::MatrixXd bezier_product_matrix(int n) {
    Eigen::MatrixXd prod_matrix(2 * n + 1, (n + 1) * (n + 1));
    for (auto j : range(2 * n + 1)) {
        auto denominator = binom(2 * n, j);
        for (auto i : range(std::max(0, j - n), std::min(n, j) + 1)) {
            if (n >= i && n >= j - i && 2 * n >= j && j - i >= 0) {
                prod_matrix(j, n * i + j) = (binom(n, i) * binom(n, j - i)) / denominator;
            }
        }
    }
    return prod_matrix;
}

}  // namespace bebot::bernstein::internal