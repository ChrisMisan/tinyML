#include <layers/dense/definition.hpp>
#include <layers/dense/connection.hpp>

namespace tinyML
{
    std::unique_ptr<base_layer_connection_t> dense_layer_definition_t::build_kernel(size_t input_size) const
    {
        return std::unique_ptr<base_layer_connection_t>(
                new dense_layer_connection_t(input_size, output_size(), activation_f)
                );
    }
}
