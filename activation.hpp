#pragma once

#include "common.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace mlp
{
    using function_t = void(*)(const vector_t& in, vector_t& out);

    struct activation_f_t
    {
        function_t activation_function;
        function_t activation_function_derivative;

        void operator()(const vector_t& in, vector_t& out) const
        {
            return activation_function(in, out);
        }

        void derivative(const vector_t& in, vector_t& out) const
        {
            return activation_function_derivative(in, out);
        }
    };

    number_t relu_single(number_t x) {
        return std::max((number_t)0, x);
    }
    number_t relu_d_single(number_t x) {
        if (x < 0) return 0;
        else return 1;
    }
    void relu_vector(const vector_t& in, vector_t& out) {
        std::transform(std::begin(in), std::end(in), std::begin(out), relu_single);
    }
    void relu_d_vector(const vector_t& in, vector_t& out) {
        std::transform(std::begin(in), std::end(in), std::begin(out), relu_d_single);
    }

    void soft_max_vector(const vector_t& in, vector_t& out)
    {
        auto max = *std::max_element(std::begin(in), std::end(in));
        auto denominator = std::transform_reduce(std::begin(in), std::end(in), number_t(0), std::plus<>{},
                                                 [&max](auto&& x){ return exp(x-max);});
        std::transform(std::begin(in), std::end(in), std::begin(out), [&denominator, &max](auto&& v) {return exp(v-max)/denominator;});
    }

    constexpr static auto relu = activation_f_t{relu_vector, relu_d_vector};
    constexpr static auto soft_max = activation_f_t{soft_max_vector, nullptr};
}