#pragma once

#include "artlib_cpp/backend_interface.hpp"

#include <cstddef>
#include <vector>

namespace artlib_cpp {

struct FuzzyARTParams {
    double rho;
    double alpha;
    double beta;
};

class FuzzyARTCore final : public BackendModel {
public:
    explicit FuzzyARTCore(FuzzyARTParams params);

    std::string name() const override;
    void fit(const MatrixView& x) override;
    std::vector<int> predict(const MatrixView& x) const override;

    const std::vector<std::vector<double>>& weights() const;
    const std::vector<int>& labels() const;

private:
    int step_fit(const std::vector<double>& sample);
    double category_choice(const double* sample, const std::vector<double>& w) const;
    double match(const double* sample, const std::vector<double>& w) const;
    std::vector<double> update_weight(
        const std::vector<double>& sample,
        const std::vector<double>& w
    ) const;

    static double l1_and(const double* x, const std::vector<double>& w, int len);
    void validate_matrix(const MatrixView& x) const;

    FuzzyARTParams params_;
    std::size_t dim_original_;
    std::vector<std::vector<double>> weights_;
    std::vector<int> labels_;
};

}  // namespace artlib_cpp
