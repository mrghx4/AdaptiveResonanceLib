#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace artlib_cpp {

struct ART1MAPParams {
    double rho;
    double L;
    std::string match_tracking;
    double epsilon;
};

class ART1MAPCore {
public:
    explicit ART1MAPCore(ART1MAPParams params);

    void set_state(
        const std::vector<std::vector<double>>& weights,
        const std::vector<int>& cluster_labels
    );

    void fit(
        const std::int16_t* x,
        std::size_t rows,
        std::size_t cols,
        const int* y,
        std::size_t y_len
    );

    std::pair<std::vector<int>, std::vector<int>> predict(
        const std::int16_t* x, std::size_t rows, std::size_t cols
    );

    const std::vector<std::vector<double>>& weights() const;
    const std::vector<int>& labels_a() const;
    std::vector<int> cluster_labels() const;

private:
    bool match_tracking(double m);
    bool matches_vigilance(double m) const;
    void reset_rho();
    void validate_xy(
        const std::int16_t* x,
        std::size_t rows,
        std::size_t cols,
        const int* y,
        std::size_t y_len
    ) const;
    void validate_x(const std::int16_t* x, std::size_t rows, std::size_t cols) const;

    static int bit(std::int16_t v);
    double category_choice(const std::int16_t* sample, const std::vector<double>& w) const;
    double match(const std::int16_t* sample, const std::vector<double>& w) const;
    std::vector<double> update_weight(
        const std::int16_t* sample, const std::vector<double>& w
    ) const;
    std::vector<double> new_weight(const std::int16_t* sample) const;
    int step_fit(const std::int16_t* sample, int c_b);

    ART1MAPParams params_;
    std::size_t dim_;
    double rho_runtime_;
    std::vector<std::vector<double>> weights_;
    std::unordered_map<int, int> cluster_map_;
    std::vector<int> labels_a_;
};

}  // namespace artlib_cpp
