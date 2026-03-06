#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace artlib_cpp {

class BinaryFuzzyARTCore {
public:
    explicit BinaryFuzzyARTCore(double rho);

    void set_weights(const std::vector<std::vector<uint32_t>>& weights);
    void fit(const std::uint8_t* x, std::size_t rows, std::size_t cols);
    std::vector<int> predict(const std::uint8_t* x, std::size_t rows, std::size_t cols) const;

    const std::vector<std::vector<uint32_t>>& weights() const;
    const std::vector<int>& labels() const;

private:
    static uint32_t intersection_count(
        const std::vector<uint32_t>& i, const std::vector<uint32_t>& w
    );
    static uint32_t ones_count(const std::vector<uint32_t>& v);
    static void update_inplace(std::vector<uint32_t>& w, const std::vector<uint32_t>& i);

    uint32_t step_fit(const std::vector<uint32_t>& sample);
    void validate_matrix(const std::uint8_t* x, std::size_t rows, std::size_t cols) const;

    double rho_;
    std::size_t dim_original_;
    uint32_t rho_int_;
    std::vector<std::vector<uint32_t>> weights_;
    std::vector<uint32_t> w_count_cache_;
    std::vector<int> labels_;
};

}  // namespace artlib_cpp
