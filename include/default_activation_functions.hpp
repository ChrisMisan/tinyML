#pragma once

#include <activation_function.hpp>

namespace tinyML
{
    constexpr number_t relu_single(number_t x) noexcept
    {
        return std::max((number_t)0, x);
    }
    constexpr number_t relu_derivative_single(number_t x) noexcept
    {
        if (x < 0) return 0;
        else return 1;
    }

    inline void relu_vector(const vector_t& in, vector_t& out) noexcept
    {
        std::transform(std::begin(in), std::end(in), std::begin(out), relu_single);
    }
    inline void relu_derivative_vector(const vector_t& in, vector_t& out) noexcept
    {
        std::transform(std::begin(in), std::end(in), std::begin(out), relu_derivative_single);
    }

    inline void soft_max_vector(const vector_t& in, vector_t& out) noexcept
    {
        auto max = *std::max_element(std::begin(in), std::end(in));
        auto denominator = std::transform_reduce(std::begin(in), std::end(in), number_t(0), std::plus<>{},
            [&max](auto&& x) { return exp(x - max); });
        std::transform(std::begin(in), std::end(in), std::begin(out), [&denominator, &max](auto&& v) {return exp(v - max) / denominator; });
    }

    constexpr static auto relu = activation_f_t{relu_vector, relu_derivative_vector};
    constexpr auto soft_max = activation_f_t{soft_max_vector, nullptr};
}
