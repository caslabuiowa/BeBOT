#pragma once

#include <array>
#include <Eigen/Dense>

namespace bebot {
    class BeBOT {

        Eigen::MatrixXd cpts; 

        public:
            // BeBOT();
            BeBOT(const Eigen::Ref<Eigen::MatrixXd>& val);
            // ~BeBOT();

            void set_cpts(Eigen::MatrixXd);
            Eigen::MatrixXd get_cpts();
    };
}