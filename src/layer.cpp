#include <layer.hpp>

namespace tinyML
{
    layer_t::layer_t(const activation_f_t &activation_function, size_t layer_size)
        : activation_f(activation_function)
        , activations(layer_size, 0)
        , d_activations(layer_size, 0)
        , deltas(layer_size, 0)
        , deltas_temp(layer_size, 0)
    {}
}