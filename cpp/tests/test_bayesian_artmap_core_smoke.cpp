#include "artlib_cpp/bayesian_artmap_core.hpp"

#include <cassert>
#include <vector>

int main() {
    std::vector<double> x = {
        0.0, 0.0,
        0.1, 0.1,
        5.0, 5.0,
        5.1, 5.2
    };
    std::vector<int> y = {0, 0, 1, 1};

    artlib_cpp::BayesianARTMAPParams p{0.7, {1.0, 0.0, 0.0, 1.0}, "MT+", 1e-10};
    artlib_cpp::BayesianARTMAPCore model(p);
    model.fit(x.data(), 4, 2, y.data(), y.size());

    auto [ya, yb] = model.predict(x.data(), 4, 2);
    assert(ya.size() == 4);
    assert(yb.size() == 4);
    assert(!model.weights().empty());
    return 0;
}
