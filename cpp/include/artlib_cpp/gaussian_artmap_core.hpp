#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace artlib_cpp {

struct GaussianARTMAPParams {
    double rho;
    double alpha;
    std::vector<double> sigma_init;
    std::string mt;
    double epsilon;
};

class GaussianARTMAPCore {
public:
    explicit GaussianARTMAPCore(GaussianARTMAPParams params);

    void set_state(
        const std::vector<std::vector<double>>& weights,
        const std::vector<int>& cluster_labels
    );

    void fit(
        const double* x,
        std::size_t rows,
        std::size_t cols,
        const int* y,
        std::size_t y_len
    );

    std::pair<std::vector<int>, std::vector<int>> predict(
        const double* x, std::size_t rows, std::size_t cols
    );

    const std::vector<int>& labels_a() const;
    const std::vector<std::vector<double>>& weights() const;
    std::vector<int> cluster_labels() const;

private:
    void reset_rho();
    double gaussian_exp(const double* x, const std::vector<double>& w) const;
    double category_choice(
        const double* sample, const std::vector<double>& w, double total_n
    ) const;
    double match(const double* sample, const std::vector<double>& w) const;
    void update_weight_inplace(
        const double* sample, std::vector<double>& w, double& total_n
    ) const;
    std::vector<double> new_weight(const double* sample) const;
    bool match_tracking(double m);
    bool matches_vigilance(double m) const;
    int step_fit(const double* sample, int c_b, double& total_n);
    void validate_x(const double* x, std::size_t rows, std::size_t cols) const;

    GaussianARTMAPParams params_;
    int dim_;
    double rho_runtime_;
    std::vector<std::vector<double>> weights_;
    std::unordered_map<int, int> cluster_map_;
    std::vector<int> labels_a_;
};

}  // namespace artlib_cpp
