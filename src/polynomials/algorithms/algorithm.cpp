#include "bebot/algorithm.hpp"
#include "numeric"
#include <iostream>

// Python-like range functions
std::vector<int> range(const int& start, const int& end)
{
    std::vector<int> out(end - start);
    std::iota(out.begin(), out.end(), start);
    return out;
}

std::vector<int> range(const int& end)
{
    return range(0, end);
}

// de Casteljau algorithm implementations
// Might move this to /algorithms?
// 1D, 1 time point
double deCasteljau_1d(const Eigen::VectorXd& control_points, double t)
{
    Eigen::VectorXd control_points_new = control_points;
    while (control_points_new.size() > 1) {
        Eigen::VectorXd lower_order_points = Eigen::VectorXd::Zero(control_points_new.size() - 1);
        for (auto i : range(lower_order_points.size())) {
            lower_order_points(i) = (1 - t) * control_points_new(i) + t * control_points_new(i + 1);
        }
        control_points_new = lower_order_points;
    }
    return control_points_new(0);
}

// 1D, T time points
std::vector<double> deCasteljau_1d(const Eigen::VectorXd& control_points, std::vector<double> t)
{
    std::vector<double> out;
    std::transform(t.begin(), t.end(), std::back_inserter(out), [control_points](auto t) {
        return deCasteljau_1d(control_points, t);
    });
    return out;
}

// N-D, 1 time point
Eigen::VectorXd deCasteljau_Nd(const Eigen::MatrixXd& control_points, double t)
{
    Eigen::VectorXd NdVec(control_points.rows());
    for (auto& row_num : range(control_points.rows())) {
        NdVec(row_num) = deCasteljau_1d(control_points.row(row_num), t);
    }
    return NdVec;
}

// N-D, T time points
std::vector<Eigen::VectorXd>
deCasteljau_Nd(const Eigen::MatrixXd& control_points, std::vector<double> t)
{
    std::vector<Eigen::VectorXd> out;
    std::transform(t.begin(), t.end(), std::back_inserter(out), [control_points](auto t) {
        return deCasteljau_Nd(control_points, t);
    });
    return out;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
deCasteljau_split(const Eigen::MatrixXd& control_points, double t)
{
    auto control_points_left = Eigen::MatrixXd(control_points.rows(), control_points.cols());
    auto control_points_right = Eigen::MatrixXd(control_points.rows(), control_points.cols());
    auto index = 0;
    auto new_control_points = control_points;

    // TODO: implement this
    return { control_points_left, control_points_right };
}
