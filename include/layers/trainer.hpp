#pragma once

#include <common.hpp>

namespace tinyML
{
    class base_layer_connection_trainer_t
    {
    protected:
        const number_t learning_constant;
    public:
        constexpr base_layer_connection_trainer_t() = delete;
        constexpr explicit base_layer_connection_trainer_t(const number_t& learning_constant)
            : learning_constant(learning_constant)
        {}
    public:
        virtual ~base_layer_connection_trainer_t() = default;
    public:
        virtual const vector_t& activation_derivatives() const noexcept = 0;
        virtual void reinitialize() = 0;
        virtual void training_forward_pass(const vector_t& input, vector_t& output) = 0;
        virtual void training_backward_pass(const vector_t& input, const vector_t& input_deltas, vector_t& output_deltas) = 0;
    };
}
