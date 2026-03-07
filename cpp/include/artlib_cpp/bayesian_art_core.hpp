#pragma once

#include <cstddef>
#include <vector>

namespace artlib_cpp {

struct BayesianARTParams {
    double rho;
    std::vector<double> cov_init;  // flattened dim x dim
};

class BayesianARTCore {
public:
    explicit BayesianARTCore(BayesianARTParams params);

    void set_weights(const std::vector<std::vector<double>>& weights);

    void fit(const double* x, std::size_t rows, std::size_t cols);

    std::vector<int> predict(const double* x, std::size_t rows, std::size_t cols) const;

    const std::vector<int>& labels() const;
    const std::vector<std::vector<double>>& weights() const;

private:
    bool invert_and_det(const std::vector<double>& a, std::vector<double>& inv, double& det) const;
    double quadratic_form(const std::vector<double>& inv_cov, const double* dist) const;
    double category_choice(const double* sample, const std::vector<double>& w) const;
    double match_criterion(const double* sample, const std::vector<double>& w) const;
    std::vector<double> update_weight(const double* sample, const std::vector<double>& w) const;
    std::vector<double> new_weight(const double* sample) const;
    int step_fit(const double* sample);
    void validate_x(const double* x, std::size_t rows, std::size_t cols) const;

    BayesianARTParams params_;
    int dim_;
    std::vector<std::vector<double>> weights_;
    std::vector<int> labels_;
};

}  // namespace artlib_cpp
