#pragma once

#include "common.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace tinyML
{
    using function_t = void(*)(const vector_t& in, vector_t& out);

    struct activation_f_t
    {
    public:
        function_t activation_function = nullptr;
        function_t activation_function_derivative = nullptr;
    public:
        constexpr void invoke(const vector_t& in, vector_t& out) const noexcept
        {
            return activation_function(in, out);
        }
        constexpr void derivative(const vector_t& in, vector_t& out) const noexcept
        {
            if(activation_function_derivative == nullptr)
                return;
            return activation_function_derivative(in, out);
        }
    public:
        constexpr void operator()(const vector_t& in, vector_t& out) const noexcept
        {
            return invoke(in, out);
        }
    public:
        constexpr bool operator==(const activation_f_t& other) const noexcept
        {
            return activation_function == other.activation_function &&
                activation_function_derivative == other.activation_function_derivative;
        }
    };
}
