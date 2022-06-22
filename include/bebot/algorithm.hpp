#pragma once
/*
    Main algorithm implementations that aren't inherent to basic properties themselves
    Algorithms will assume normalised time inputs
*/

#include <Eigen/Dense>
#include <utility>


// de Casteljau evaluation algorithm
// 1D, 1 time point
double deCasteljau_1d(const Eigen::VectorXd& control_points, double t);
// 1D, T time points
std::vector<double> deCasteljau_1d(const Eigen::VectorXd& control_points, std::vector<double> t);
// N-D, 1 time point
Eigen::VectorXd deCasteljau_Nd(const Eigen::MatrixXd& control_points, double t);
// N-D, T time points
std::vector<Eigen::VectorXd>
deCasteljau_Nd(const Eigen::VectorXd& control_points, std::vector<double> t);

// de Casteljau split algorithm
std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
deCasteljau_split(const Eigen::VectorXd& control_points, double t);
