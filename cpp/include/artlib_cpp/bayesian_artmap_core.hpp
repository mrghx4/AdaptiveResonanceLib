#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace artlib_cpp {

struct BayesianARTMAPParams {
    double rho;
    std::vector<double> cov_init;  // flattened dim x dim
    std::string mt;
    double epsilon;
};

class BayesianARTMAPCore {
public:
    explicit BayesianARTMAPCore(BayesianARTMAPParams params);

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
    bool invert_and_det(const std::vector<double>& a, std::vector<double>& inv, double& det) const;
    double quadratic_form(const std::vector<double>& inv_cov, const double* dist) const;
    double category_choice(const double* sample, const std::vector<double>& w, double total_n) const;
    double match_criterion(const double* sample, const std::vector<double>& w) const;
    std::vector<double> update_weight(const double* sample, const std::vector<double>& w) const;
    std::vector<double> new_weight(const double* sample) const;
    bool match_tracking(double m);
    bool matches_vigilance(double m) const;
    int step_fit(const double* sample, int c_b);
    void validate_x(const double* x, std::size_t rows, std::size_t cols) const;

    BayesianARTMAPParams params_;
    int dim_;
    double rho_runtime_;
    std::vector<std::vector<double>> weights_;
    std::unordered_map<int, int> cluster_map_;
    std::vector<int> labels_a_;
};

}  // namespace artlib_cpp
