#include "bebot/algorithm.hpp"
#include "bebot/bernstein.hpp"
#include "common.hpp"

#include <iostream>

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

std::pair<Eigen::VectorXd, Eigen::VectorXd>
deCasteljau_split_1d(const Eigen::VectorXd& control_points, double t) {
    auto control_points_left = Eigen::VectorXd(control_points.size());
    auto control_points_right = Eigen::VectorXd(control_points.size());
    auto index = 0;
    auto new_control_points = control_points;
    auto temp_control_points = control_points;

    while (new_control_points.size() > 1) {
        control_points_left(index) = temp_control_points(0);
        control_points_right(index) = temp_control_points(Eigen::last);
        index++;

        temp_control_points = Eigen::VectorXd::Zero(new_control_points.size() - 1);
        for (auto i : range(temp_control_points.size())) {
            temp_control_points(i) = (1 - t) * new_control_points(i) + t * new_control_points(i + 1);
        }
        new_control_points = temp_control_points;
    }

    control_points_left(index) = new_control_points(0);
    control_points_right(index) = new_control_points(0);

    return { control_points_left, control_points_right };
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
deCasteljau_split(const Eigen::MatrixXd& control_points, double t)
{
    auto control_points_left = Eigen::MatrixXd(control_points.rows(), control_points.cols());
    auto control_points_right = Eigen::MatrixXd(control_points.rows(), control_points.cols());
    auto index = 0;
    auto new_control_points = control_points;
    auto temp_control_points = control_points;

    // TODO: implement this
    while (new_control_points.size() > 1) {
        control_points_left(index) = temp_control_points(0);
    }
    return { control_points_left, control_points_right };
}
