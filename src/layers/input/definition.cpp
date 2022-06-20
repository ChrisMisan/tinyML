#include <layers/input/definition.hpp>
#include <layers/input/connection.hpp>

namespace tinyML
{
    [[nodiscard]] std::unique_ptr<base_layer_connection_t> input_layer_definition_t::build_kernel(size_t input_size) const
    {
        return std::unique_ptr<base_layer_connection_t>(
                new input_layer_connection_t(input_size, output_size(), activation_f)
                );
    }
}