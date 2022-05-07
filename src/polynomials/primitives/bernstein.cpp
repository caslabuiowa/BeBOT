#include <bebot/bernstein.hpp>

using namespace bebot::bernstein;

namespace {

Eigen::MatrixXd deCasteljau_evaluate(Eigen::MatrixXd control_points, std::vector<double> t) {
    Eigen::MatrixXd curve{{1,2,3},{4,5,6}};
    return curve;
}

} // namespace

Bernstein::Bernstein(Eigen::MatrixXd control_points, double initial_time, double final_time)
    : control_points_(control_points), initial_time_(initial_time), final_time_(final_time)
{
}

int Bernstein::dimension()
{
    return control_points_.rows();
}

int Bernstein::degree()
{
    return control_points_.cols() - 1;
}

Eigen::VectorXd Bernstein::operator()(double t)
{
    Eigen::VectorXd result(dimension());
    return result;
}
