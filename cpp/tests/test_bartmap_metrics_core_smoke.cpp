#include "artlib_cpp/bartmap_metrics_core.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    // 3x3 symmetric toy matrix with identical first two rows for cluster 1.
    const std::vector<double> X = {
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        3.0, 2.0, 1.0,
    };
    const std::vector<int> labels = {1, 1, 0};

    const double r = artlib_cpp::average_pearson_corr(
        X.data(),
        3,
        3,
        labels.data(),
        labels.size(),
        0,
        1
    );

    if (!std::isfinite(r)) {
        std::cerr << "Expected finite result" << std::endl;
        return 1;
    }
    if (std::abs(r - 1.0) > 1e-12) {
        std::cerr << "Unexpected correlation: " << r << std::endl;
        return 2;
    }

    return 0;
}
