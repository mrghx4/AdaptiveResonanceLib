#include "artlib_cpp/quadratic_neuron_art_core.hpp"

#include <cassert>
#include <vector>

int main() {
    std::vector<double> x = {
        0.0, 0.0,
        0.03, 0.02,
        0.9, 0.85,
        0.88, 0.92
    };

    artlib_cpp::QuadraticNeuronARTParams p{0.7, 0.5, 0.1, 0.1, 0.05};
    artlib_cpp::QuadraticNeuronARTCore model(p);
    model.fit(x.data(), 4, 2);

    auto y = model.predict(x.data(), 4, 2);
    assert(y.size() == 4);
    assert(!model.weights().empty());
    return 0;
}
