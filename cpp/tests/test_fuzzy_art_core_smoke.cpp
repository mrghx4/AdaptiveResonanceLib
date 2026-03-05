#include "artlib_cpp/fuzzy_art_core.hpp"

#include <cassert>
#include <vector>

int main() {
    const std::vector<double> raw = {
        0.9, 0.1, 0.1, 0.9,
        0.85, 0.15, 0.15, 0.85,
        0.1, 0.9, 0.9, 0.1
    };

    artlib_cpp::MatrixView x{raw.data(), 3, 4};
    artlib_cpp::FuzzyARTParams p{0.8, 1e-10, 1.0};
    artlib_cpp::FuzzyARTCore model(p);

    model.fit(x);
    const auto y = model.predict(x);

    assert(y.size() == 3);
    assert(model.labels().size() == 3);
    assert(!model.weights().empty());
    return 0;
}
