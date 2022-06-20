#pragma once

#include <common.hpp>
#include <layers/trainer.hpp>
#include <layers/dense/connection.hpp>

namespace tinyML
{
    class dense_layer_connection_trainer_t
        : public base_layer_connection_trainer_t
    {
    public:
        dense_layer_connection_t& _trained_layer;
        matrix_t weights_transposed;

        vector_t activation_derivative_values;

        matrix_t weights_gradient;
        vector_t biases_gradient;
    public:
        dense_layer_connection_trainer_t(dense_layer_connection_t& trained_layer, const number_t& learning_constant);
    public:
        ~dense_layer_connection_trainer_t() override = default;
    public:
        virtual void reinitialize() override;
        const vector_t& activation_derivatives() const noexcept override;
        void training_forward_pass(const vector_t& input, vector_t& output) override;
        void training_backward_pass(const vector_t& input, const vector_t& input_deltas, vector_t& output_deltas) override;
    };
}
