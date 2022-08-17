#pragma once

#include "algorithm.hpp"
#include "bernstein.hpp"
#include "rational_bernstein.hpp"

#include <Eigen/Dense>

namespace bebot {
class BeBOT {
  private:
    double _t0, _tf;
    Eigen::MatrixXd _cpts;

  public:
    BeBOT() = default;
    ~BeBOT() = default;

    BeBOT(const Eigen::Ref<Eigen::MatrixXd>& cpts);
    BeBOT(const Eigen::Ref<Eigen::MatrixXd>& cpts, const double tf);
    BeBOT(const Eigen::Ref<Eigen::MatrixXd>& cpts, const double t0, const double tf);

    void set_cpts(const Eigen::Ref<Eigen::MatrixXd>& cpts);
    Eigen::MatrixXd get_cpts();

    void set_t0(const double t0);
    double get_t0();

    void set_tf(const double tf);
    double get_tf();
};
}  // namespace bebot