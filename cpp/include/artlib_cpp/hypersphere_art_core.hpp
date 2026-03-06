#pragma once

#include <cstddef>
#include <vector>

namespace artlib_cpp {

struct HypersphereARTParams {
    double rho;
    double alpha;
    double beta;
    double r_hat;
};

class HypersphereARTCore {
public:
    explicit HypersphereARTCore(HypersphereARTParams params);

    void set_weights(const std::vector<std::vector<double>>& weights);

    void fit(const double* x, std::size_t rows, std::size_t cols);

    std::vector<int> predict(const double* x, std::size_t rows, std::size_t cols) const;

    const std::vector<int>& labels() const;
    const std::vector<std::vector<double>>& weights() const;

private:
    double euclidean(const double* x, const std::vector<double>& w) const;
    double category_choice(const double* sample, const std::vector<double>& w) const;
    double match(const double* sample, const std::vector<double>& w) const;
    std::vector<double> update_weight(
        const std::vector<double>& i,
        const std::vector<double>& w,
        double i_radius,
        double max_radius
    ) const;
    std::vector<double> new_weight(const std::vector<double>& i) const;
    int step_fit(const std::vector<double>& sample);
    void validate_x(const double* x, std::size_t rows, std::size_t cols) const;

    HypersphereARTParams params_;
    int dim_;
    std::vector<std::vector<double>> weights_;
    std::vector<int> labels_;
};

}  // namespace artlib_cpp
