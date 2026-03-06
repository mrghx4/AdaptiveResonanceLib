#include "artlib_cpp/gaussian_art_core.hpp"

#include <cassert>
#include <vector>

int main() {
    std::vector<double> x = {
        0.0, 0.0,
        0.05, 0.02,
        5.0, 5.0,
        5.1, 4.95
    };

    artlib_cpp::GaussianARTParams p{0.05, 1e-10, {0.5, 0.5}};
    artlib_cpp::GaussianARTCore model(p);
    model.fit(x.data(), 4, 2);

    auto y = model.predict(x.data(), 4, 2);
    assert(y.size() == 4);
    assert(!model.weights().empty());
    return 0;
}
