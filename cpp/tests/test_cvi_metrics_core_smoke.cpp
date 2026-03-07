#include "artlib_cpp/cvi_metrics_core.hpp"

#include <cassert>
#include <vector>

int main() {
    // Two compact clusters in 2D
    std::vector<double> x = {
        0.0, 0.0,
        0.1, 0.0,
        1.0, 1.0,
        1.1, 1.0,
    };
    std::vector<int> y = {0, 0, 1, 1};

    const double ch = artlib_cpp::evaluate_cvi(
        x.data(), 4, 2, y.data(), artlib_cpp::CVIType::CalinskiHarabasz
    );
    const double db = artlib_cpp::evaluate_cvi(
        x.data(), 4, 2, y.data(), artlib_cpp::CVIType::DaviesBouldin
    );
    const double sil = artlib_cpp::evaluate_cvi(
        x.data(), 4, 2, y.data(), artlib_cpp::CVIType::Silhouette
    );

    assert(ch > 0.0);
    assert(db >= 0.0);
    assert(sil >= -1.0 && sil <= 1.0);
    return 0;
}
