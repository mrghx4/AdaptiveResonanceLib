#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace artlib_cpp {

struct BinaryFuzzyARTMAPParams {
    double rho;
    std::string mt;
    uint32_t epsilon;
};

class BinaryFuzzyARTMAPCore {
public:
    explicit BinaryFuzzyARTMAPCore(BinaryFuzzyARTMAPParams params);

    void set_state(
        const std::vector<std::vector<uint32_t>>& weights,
        const std::vector<int>& cluster_labels
    );

    void fit(
        const int* x,
        std::size_t rows,
        std::size_t cols,
        const int* y,
        std::size_t y_len
    );

    std::pair<std::vector<int>, std::vector<int>> predict(
        const int* x, std::size_t rows, std::size_t cols
    );

    const std::vector<int>& labels_a() const;
    const std::vector<std::vector<uint32_t>>& weights() const;
    std::vector<int> cluster_labels() const;

private:
    void reset_rho();
    static uint32_t intersection_count(
        const std::vector<uint32_t>& i, const std::vector<uint32_t>& w
    );
    static uint32_t ones_count(const std::vector<uint32_t>& v);
    bool validate_hypothesis(uint32_t cluster_id, uint32_t c_b) const;
    static bool match_operator(const std::string& mt, uint32_t a, uint32_t b);
    bool match_tracking_integer(uint32_t m_int);
    static void update_inplace(std::vector<uint32_t>& w, const std::vector<uint32_t>& i);
    uint32_t step_fit(const std::vector<uint32_t>& sample, uint32_t c_b);
    void validate_x(const int* x, std::size_t rows, std::size_t cols) const;

    BinaryFuzzyARTMAPParams params_;
    uint32_t dim_original_;
    uint32_t rho_w1_;
    uint32_t rho_int_;
    std::vector<std::vector<uint32_t>> weights_;
    std::vector<uint32_t> w_count_cache_;
    std::unordered_map<uint32_t, uint32_t> cluster_map_;
    std::vector<int> labels_a_;
};

}  // namespace artlib_cpp
