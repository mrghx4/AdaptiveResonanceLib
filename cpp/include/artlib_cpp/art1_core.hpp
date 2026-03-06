#pragma once

#include "artlib_cpp/backend_interface.hpp"

#include <string>
#include <vector>

namespace artlib_cpp {

struct ART1Params {
    double rho;
    double L;
};

class ART1Core final : public BackendModel {
public:
    explicit ART1Core(ART1Params params);

    std::string name() const override;
    void fit(const MatrixView& x) override;
    std::vector<int> predict(const MatrixView& x) const override;

    void set_weights(const std::vector<std::vector<double>>& weights);

    const std::vector<std::vector<double>>& weights() const;
    const std::vector<int>& labels() const;

private:
    double category_choice(const double* sample, const std::vector<double>& w) const;
    double match(const double* sample, const std::vector<double>& w) const;
    std::vector<double> update_weight(const double* sample, const std::vector<double>& w) const;
    std::vector<double> new_weight(const double* sample) const;
    int step_fit(const std::vector<double>& sample);
    void validate_matrix(const MatrixView& x) const;

    ART1Params params_;
    std::size_t dim_;
    std::vector<std::vector<double>> weights_;
    std::vector<int> labels_;
};

}  // namespace artlib_cpp
