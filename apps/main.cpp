#include <algorithm.hpp>
#include <bebot.hpp>
#include <Eigen/Dense>
#include <iostream>

// using namespace std;

int main() {
    Eigen::MatrixXd cpts{
        {0, 1, 2, 3, 4, 5},
        {3, 6, 2, 9, 6, 8},
        {3, 4, 6, 4, 7, 9}
    };

    bebot::BeBOT c(cpts);

    std::cout << cpts << std::endl << "---" << std::endl;
    std::cout << c.get_cpts() << std::endl << "---" << std::endl;

    Eigen::MatrixXd new_cpts{
        {0, 1, 2},
        {3, 4, 2}
    };
    c.set_cpts(new_cpts);
    std::cout << c.get_cpts() << std::endl;

    Eigen::VectorXd cpts_1d = c.get_cpts().row(0);
    std::cout << cpts_1d << std::endl;

    auto c_left_right = deCasteljau_split_1d(c.get_cpts().row(0), 0.5);
    std::cout << c_left_right.first << std::endl;
}