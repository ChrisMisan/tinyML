#pragma once

#include <common.hpp>

namespace tinyML
{
    class layer_connection_t
    {
        friend class multi_layer_perceptron_t;
    private:
        matrix_t W;
        matrix_t gradient_temp;
        matrix_t W_transposed;
        vector_t B;
        vector_t out_temp;
        vector_t out_temp_internal;
    public:
        layer_connection_t() noexcept = default;
        layer_connection_t(size_t neuron_num_prev, size_t neuron_num_current);
        layer_connection_t(const layer_connection_t&) = default;
        layer_connection_t(layer_connection_t&&) noexcept = default;
    public:
        layer_connection_t& operator=(const layer_connection_t&) = default;
        layer_connection_t& operator=(layer_connection_t&&) noexcept = default;
    public:
        void forward_pass(const vector_t& input, vector_t& out) noexcept;
    };
}
