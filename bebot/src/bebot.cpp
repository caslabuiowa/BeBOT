#include <bebot/bebot.hpp>
#include <Eigen/Dense>
#include <iostream>

using namespace bebot;

BeBOT::BeBOT(const Eigen::Ref<Eigen::MatrixXd>& val) {
    cpts = val;
}

Eigen::MatrixXd BeBOT::get_cpts() {
    return cpts;
}