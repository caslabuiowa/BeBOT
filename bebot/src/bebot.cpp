#include <bebot/bebot.hpp>
#include <Eigen/Dense>
#include <iostream>

using namespace bebot;

BeBOT::BeBOT(const Eigen::Ref<Eigen::MatrixXd>& cpts) {
    set_cpts(cpts);
    set_t0(0.0);
    set_tf(1.0);
}

BeBOT::BeBOT(const Eigen::Ref<Eigen::MatrixXd>& cpts, const double tf) {
    set_cpts(cpts);
    set_t0(0.0);
    set_tf(tf);
}

BeBOT::BeBOT(const Eigen::Ref<Eigen::MatrixXd>& cpts, const double t0, const double tf) {
    set_cpts(cpts);
    set_t0(t0);
    set_tf(tf);
}

void BeBOT::set_cpts(const Eigen::Ref<Eigen::MatrixXd>& cpts) {
    _cpts = cpts;
}

Eigen::MatrixXd BeBOT::get_cpts() {
    return _cpts;
}

void BeBOT::set_t0(const double t0) {
    _t0 = t0;
}

double BeBOT::get_t0() {
    return _t0;
}

void BeBOT::set_tf(const double tf) {
    _tf = tf;
}

double BeBOT::get_tf() {
    return _tf;
}