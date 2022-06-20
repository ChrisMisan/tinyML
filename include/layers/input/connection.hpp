#pragma once

#include <common.hpp>
#include <layers/connection.hpp>
#include <activation_function.hpp>

namespace tinyML
{
    class input_layer_connection_t
        : public base_layer_connection_t
    {
        friend class input_layer_connection_trainer_t;
    private:
        activation_f_t activation_f;
    public:
        constexpr input_layer_connection_t(size_t input_size, size_t output_size, const activation_f_t& activation_f)
            : base_layer_connection_t(input_size, output_size)
            , activation_f(activation_f)
        {
            assert(input_size == output_size);
        }
    public:
        ~input_layer_connection_t() override = default;
    public:
        void forward_pass(const vector_t& input, vector_t& output) const override;
        [[nodiscard]] std::unique_ptr<base_layer_connection_trainer_t> build_trainer(const number_t& learning_constant);
    public:
        [[nodiscard]] nlohmann::json serialize() const override;
        void deserialize(const nlohmann::json& data) override;
    };
}