#pragma once

#include <common.hpp>

namespace tinyML
{
    class base_layer_definition_t
    {
    private:
        size_t _output_size = 0;
    public:
        constexpr base_layer_definition_t() = delete;
        constexpr base_layer_definition_t(size_t output_size) noexcept
            : _output_size(output_size)
        {}
    public:
        virtual ~base_layer_definition_t() = default;
    public:
        [[nodiscard]] constexpr size_t output_size() const noexcept
        {
            return _output_size;
        }
    public:
        [[nodiscard]] virtual std::unique_ptr<base_layer_connection_t> build_kernel(size_t input_size) const = 0;
    };
}
