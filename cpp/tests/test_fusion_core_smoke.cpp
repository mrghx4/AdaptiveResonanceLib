#include "artlib_cpp/fusion_core.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void expect_invalid_argument(void (*fn)()) {
    bool threw = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void assert_close(double actual, double expected) {
    assert(std::abs(actual - expected) < 1e-12);
}

}  // namespace

int main() {
    const std::vector<double> activations = {
        0.1, 0.9,
        0.8, 0.2,
        0.6, 0.6,
    };
    const std::vector<double> gamma = {0.25, 0.75};
    const std::vector<unsigned char> no_skip = {0, 0};
    const std::vector<unsigned char> skip_second = {0, 1};

    assert(artlib_cpp::ArgmaxWeightedActivations(
        activations.data(), 3, 2, gamma.data(), no_skip.data()) == 0);
    assert(artlib_cpp::ArgmaxWeightedActivations(
        activations.data(), 3, 2, gamma.data(), skip_second.data()) == 1);

    const std::vector<double> channel_major = {
        0.1, 0.8, 0.6,
        0.9, 0.2, 0.6,
    };
    assert(artlib_cpp::ArgmaxWeightedChannelActivations(
        channel_major.data(), 2, 3, gamma.data(), no_skip.data()) == 0);

    const std::vector<double> state = {1.0, 2.0};
    const std::vector<double> actions = {3.0, 4.0, 5.0, 6.0};
    std::vector<double> query(2 * 6, 0.0);
    artlib_cpp::BuildStateActionRewardQuery(
        state.data(), 2, actions.data(), 2, 2, 2, 0.5, query.data());
    const std::vector<double> expected_query = {
        1.0, 2.0, 3.0, 4.0, 0.5, 0.5,
        1.0, 2.0, 5.0, 6.0, 0.5, 0.5,
    };
    assert(query == expected_query);

    const std::vector<double> ch0 = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> ch2 = {9.0, 10.0};
    const double* channels[] = {ch0.data(), ch2.data()};
    const std::size_t widths[] = {2, 3, 1};
    const unsigned char present[] = {1, 0, 1};
    std::vector<double> joined(2 * 6, 0.0);
    artlib_cpp::JoinChannelsWithFill(
        channels, widths, present, 3, 2, 0.5, joined.data());
    const std::vector<double> expected_joined = {
        1.0, 2.0, 0.5, 0.5, 0.5, 9.0,
        3.0, 4.0, 0.5, 0.5, 0.5, 10.0,
    };
    assert(joined == expected_joined);

    std::vector<double> out0(2 * 2, 0.0);
    std::vector<double> out2(2 * 1, 0.0);
    double* outputs[] = {out0.data(), out2.data()};
    artlib_cpp::ExtractPresentChannels(joined.data(), 2, widths, present, 3, outputs);
    assert(out0 == ch0);
    assert(out2 == ch2);

    expect_invalid_argument([]() {
        const double a = 1.0;
        const unsigned char s = 0;
        artlib_cpp::ArgmaxWeightedActivations(nullptr, 1, 1, &a, &s);
    });
    expect_invalid_argument([]() {
        const double a = 1.0;
        const double state = 1.0;
        artlib_cpp::BuildStateActionRewardQuery(&state, 1, &a, 0, 1, 1, 0.5, const_cast<double*>(&a));
    });
    expect_invalid_argument([]() {
        const double* channels_bad[] = {nullptr};
        const std::size_t widths_bad[] = {1};
        const unsigned char present_bad[] = {1};
        double out = 0.0;
        artlib_cpp::JoinChannelsWithFill(channels_bad, widths_bad, present_bad, 1, 1, 0.5, &out);
    });

    assert_close(query[4], 0.5);
    return 0;
}
