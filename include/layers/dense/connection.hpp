#pragma once

#include <common.hpp>
#include <layers/connection.hpp>
#include <activation_function.hpp>

namespace tinyML
{
    class dense_layer_connection_t
        : public base_layer_connection_t
    {
        friend class dense_layer_connection_trainer_t;
    private:
        matrix_t weights;
        vector_t biases;
        activation_f_t activation_f;
    public:
        dense_layer_connection_t(size_t input_size, size_t output_size, const activation_f_t& activation_f);
    public:
        ~dense_layer_connection_t() override = default;
    public:
        void forward_pass(const vector_t& input, vector_t& output) const override;
        [[nodiscard]] std::unique_ptr<base_layer_connection_trainer_t> build_trainer(const number_t& learning_constant) override;
    public:
        [[nodiscard]] nlohmann::json serialize() const override;
        void deserialize(const nlohmann::json& data) override;
    };
}
