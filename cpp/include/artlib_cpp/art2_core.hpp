#pragma once

#include <cstddef>
#include <vector>

namespace artlib_cpp {

struct ART2Params {
    double rho;
    double alpha;
    double beta;
};

class ART2Core {
public:
    explicit ART2Core(ART2Params params);

    void set_weights(const std::vector<std::vector<double>>& weights);

    void fit(const double* x, std::size_t rows, std::size_t cols);

    std::vector<int> predict(const double* x, std::size_t rows, std::size_t cols) const;

    const std::vector<int>& labels() const;
    const std::vector<std::vector<double>>& weights() const;

private:
    double category_choice(const double* sample, const std::vector<double>& w) const;
    double match(const double* sample, double activation) const;
    std::vector<double> update_weight(const std::vector<double>& i, const std::vector<double>& w) const;
    std::vector<double> new_weight(const std::vector<double>& i) const;
    int step_fit(const std::vector<double>& sample);
    void validate_x(const double* x, std::size_t rows, std::size_t cols) const;

    ART2Params params_;
    int dim_;
    std::vector<std::vector<double>> weights_;
    std::vector<int> labels_;
};

}  // namespace artlib_cpp
