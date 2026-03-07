#pragma once

#include <cstddef>
#include <vector>

namespace artlib_cpp {

struct QuadraticNeuronARTParams {
    double rho;
    double s_init;
    double lr_b;
    double lr_w;
    double lr_s;
};

class QuadraticNeuronARTCore {
public:
    explicit QuadraticNeuronARTCore(QuadraticNeuronARTParams params);

    void set_weights(const std::vector<std::vector<double>>& weights);

    void fit(const double* x, std::size_t rows, std::size_t cols);

    std::vector<int> predict(const double* x, std::size_t rows, std::size_t cols) const;

    const std::vector<int>& labels() const;
    const std::vector<std::vector<double>>& weights() const;

private:
    double category_choice(
        const double* sample,
        const std::vector<double>& w,
        double* out_l2,
        std::vector<double>* out_z
    ) const;
    double match(double activation) const;
    std::vector<double> update_weight(
        const std::vector<double>& i,
        const std::vector<double>& w,
        double activation,
        double l2_z_b,
        const std::vector<double>& z
    ) const;
    std::vector<double> new_weight(const std::vector<double>& i) const;
    int step_fit(const std::vector<double>& sample);
    void validate_x(const double* x, std::size_t rows, std::size_t cols) const;

    QuadraticNeuronARTParams params_;
    int dim_;
    std::vector<std::vector<double>> weights_;
    std::vector<int> labels_;
};

}  // namespace artlib_cpp
