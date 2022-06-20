#pragma once

#include <common.hpp>
#include <layers/definition.hpp>
#include <activation_function.hpp>

namespace tinyML
{
    class dense_layer_definition_t
        : public base_layer_definition_t
    {
    public:
        activation_f_t activation_f;
    public:
        constexpr dense_layer_definition_t(size_t output_size, const activation_f_t& activation_f) noexcept
            : base_layer_definition_t(output_size)
            , activation_f(activation_f)
        { }
    public:
        ~dense_layer_definition_t() override = default;
    public:
        [[nodiscard]] std::unique_ptr<base_layer_connection_t> build_kernel(size_t input_size) const override;
    };
}
