#pragma once

#include <cstddef>
#include <vector>

namespace artlib_cpp {

struct GaussianARTParams {
    double rho;
    double alpha;
    std::vector<double> sigma_init;
};

class GaussianARTCore {
public:
    explicit GaussianARTCore(GaussianARTParams params);

    void set_weights(const std::vector<std::vector<double>>& weights);

    void fit(const double* x, std::size_t rows, std::size_t cols);

    std::vector<int> predict(const double* x, std::size_t rows, std::size_t cols) const;

    const std::vector<int>& labels() const;
    const std::vector<std::vector<double>>& weights() const;

private:
    double gaussian_exp(const double* x, const std::vector<double>& w) const;
    double category_choice(const double* sample, const std::vector<double>& w, double total_n) const;
    double match(const double* sample, const std::vector<double>& w) const;
    std::vector<double> update_weight(const std::vector<double>& i, const std::vector<double>& w) const;
    std::vector<double> new_weight(const std::vector<double>& i) const;
    int step_fit(const std::vector<double>& sample);
    void validate_x(const double* x, std::size_t rows, std::size_t cols) const;

    GaussianARTParams params_;
    int dim_;
    std::vector<std::vector<double>> weights_;
    std::vector<int> labels_;
};

}  // namespace artlib_cpp
