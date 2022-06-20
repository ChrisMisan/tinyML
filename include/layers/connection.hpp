#pragma once

#include <common.hpp>
#include <serializable.hpp>

namespace tinyML
{
    class base_layer_connection_t
        : public serializable
    {
    private:
        size_t _input_size;
        size_t _output_size;
    public:
        constexpr base_layer_connection_t() = delete;
        constexpr base_layer_connection_t(size_t input_size, size_t output_size) noexcept
            : _input_size(input_size)
            , _output_size(output_size)
        {}
    public:
        virtual ~base_layer_connection_t() = default;
    public:
        [[nodiscard]] constexpr size_t input_size() const noexcept {return _input_size;};
        [[nodiscard]] constexpr size_t output_size() const noexcept {return _output_size;};
    public:
        virtual void forward_pass(const vector_t& input, vector_t& output) const = 0;
        [[nodiscard]] virtual std::unique_ptr<base_layer_connection_trainer_t> build_trainer(const number_t& learning_constant) = 0;
    };
}
