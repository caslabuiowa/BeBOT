#include <bebot/bebot.hpp>
#include <Eigen/Dense>
#include <iostream>

using namespace std;

int main() {
    Eigen::MatrixXd cpts{
        {0, 1, 2, 3, 4, 5},
        {3, 6, 2, 9, 6, 8},
        {3, 4, 6, 4, 7, 9}
    };

    bebot::BeBOT c(cpts);

    cout << cpts << endl << "---" << endl;
    cout << c.get_cpts() << endl;
}