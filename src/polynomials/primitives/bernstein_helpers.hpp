#include <Eigen/Dense>

#include <vector>

namespace bebot::bernstein::internal
{

double binom(int n, int k);

Eigen::MatrixXd bezier_coefficients(int n);
Eigen::MatrixXd bezier_product_matrix(int n);

}  // namespace bebot::bernstein::internal