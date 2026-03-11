#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace artlib_cpp {

struct FuzzyARTMAPParams {
    double rho;
    double alpha;
    double beta;
    std::string mt;
    double epsilon;
};

class FuzzyARTMAPCore {
public:
    explicit FuzzyARTMAPCore(FuzzyARTMAPParams params);

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
    static double l1_and(const double* x, const std::vector<double>& w, int len);
    double category_choice(
        const double* sample, const std::vector<double>& w, double w_l1
    ) const;
    double match(const double* sample, const std::vector<double>& w) const;
    void update_weight_inplace(
        const double* sample,
        std::size_t sample_len,
        std::vector<double>& w,
        double& w_l1
    ) const;
    bool match_tracking(double m);
    bool matches_vigilance(double m) const;
    int step_fit(const double* sample, std::size_t sample_len, int c_b);
    void validate_x(const double* x, std::size_t rows, std::size_t cols) const;

    FuzzyARTMAPParams params_;
    int dim_original_;
    double rho_runtime_;
    std::vector<std::vector<double>> weights_;
    std::vector<double> weight_l1_;
    std::unordered_map<int, int> cluster_map_;
    std::vector<int> labels_a_;
};

}  // namespace artlib_cpp
