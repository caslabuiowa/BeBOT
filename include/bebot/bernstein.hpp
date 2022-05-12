#pragma once
#include <Eigen/Dense>
#include <utility>
#include <vector>

namespace bebot::bernstein
{

class Bernstein
{
  public:
    // Constructors
    Bernstein(Eigen::MatrixXd control_points, double initial_time = 0, double final_time = 1);

    // Information
    int dimension();
    int degree();

    // Evaluate at time/s t
    Eigen::VectorXd operator()(double t);               // Evaluate the polynomial at t
    Eigen::VectorXd operator()(std::vector<double> t);  // Evaluate the polynomial at t

    // Arithmetic operations
    Bernstein operator+(Bernstein& b);
    Bernstein operator-(Bernstein& b);
    Bernstein operator*(Bernstein& b);
    Bernstein operator/(Bernstein& b);

    double mininum(int dimension, double tolerance = 1e-6);

    // Bernstein operations
    void elevate(int n = 1);
    std::pair<Bernstein, Bernstein> split(double partition_time);

    // Calculus operations
    Bernstein derivative();
    Bernstein integral();

  private:
    Eigen::MatrixXd control_points_;
    double initial_time_;
    double final_time_;
};

}  // namespace bebot::bernstein