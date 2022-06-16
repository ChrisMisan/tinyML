#pragma once

#include <common.hpp>
#include <activation_function.hpp>

namespace tinyML
{
    class layer_t
    {
        friend class multi_layer_perceptron_t;
    private:
        activation_f_t activation_f;
        vector_t activations;
        vector_t d_activations;
        vector_t deltas;
        vector_t deltas_temp;
    public:
        layer_t() noexcept = default;
        layer_t(const activation_f_t& activation_function, size_t layer_size);
        layer_t(const layer_t& other) = default;
        layer_t(layer_t&& other) noexcept = default;
    public:
        layer_t& operator=(const layer_t& other) = default;
        layer_t& operator=(layer_t&& other) noexcept = default;
    };
}
