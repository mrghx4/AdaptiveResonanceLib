#include "artlib_cpp/fusion_core.hpp"

#include <cassert>
#include <random>
#include <stdexcept>
#include <vector>

int main() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (int iter = 0; iter < 200; ++iter) {
        const std::size_t rows = 1 + static_cast<std::size_t>(iter % 7);
        const std::size_t w0 = 1 + static_cast<std::size_t>(iter % 4);
        const std::size_t w1 = 2 + static_cast<std::size_t>(iter % 3);
        const std::size_t w2 = 1 + static_cast<std::size_t>((iter + 1) % 5);
        const std::size_t total = w0 + w1 + w2;

        std::vector<double> ch0(rows * w0);
        std::vector<double> ch2(rows * w2);
        for (double& v : ch0) v = dist(rng);
        for (double& v : ch2) v = dist(rng);

        const double* channels[] = {ch0.data(), ch2.data()};
        const std::size_t widths[] = {w0, w1, w2};
        const unsigned char present[] = {1, 0, 1};
        std::vector<double> joined(rows * total, 0.0);
        artlib_cpp::JoinChannelsWithFill(
            channels, widths, present, 3, rows, 0.5, joined.data());

        std::vector<double> out0(rows * w0, 0.0);
        std::vector<double> out2(rows * w2, 0.0);
        double* outputs[] = {out0.data(), out2.data()};
        artlib_cpp::ExtractPresentChannels(joined.data(), rows, widths, present, 3, outputs);
        assert(out0 == ch0);
        assert(out2 == ch2);

        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < w1; ++c) {
                assert(joined[r * total + w0 + c] == 0.5);
            }
        }
    }

    bool threw = false;
    try {
        const std::size_t widths[] = {1};
        const unsigned char present[] = {1};
        double out = 0.0;
        artlib_cpp::JoinChannelsWithFill(nullptr, widths, present, 1, 1, 0.0, &out);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    return 0;
}
