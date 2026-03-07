#include "artlib_cpp/bayesian_art_core.hpp"

#include <cassert>
#include <vector>

int main() {
    std::vector<double> x = {
        0.0, 0.0,
        0.1, 0.1,
        5.0, 5.0,
        5.1, 5.2
    };

    artlib_cpp::BayesianARTParams p{0.7, {1.0, 0.0, 0.0, 1.0}};
    artlib_cpp::BayesianARTCore model(p);
    model.fit(x.data(), 4, 2);

    auto y = model.predict(x.data(), 4, 2);
    assert(y.size() == 4);
    assert(!model.weights().empty());
    return 0;
}
