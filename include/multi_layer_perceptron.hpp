#pragma once

#include <common.hpp>
#include <layer.hpp>
#include <layer_connection.hpp>
#include <default_activation_functions.hpp>

namespace tinyML
{
    class multi_layer_perceptron_t
    {
    private:
        std::vector<layer_t> layers;
        std::vector<layer_connection_t> weights_and_biases;
        number_t learning_constant;
    public:
        multi_layer_perceptron_t() = delete;
        template<typename It>
        multi_layer_perceptron_t(number_t learning_constant, It layer_sizes_b, It layer_sizes_e)
            : learning_constant(learning_constant)
            , layers(std::distance(layer_sizes_b, layer_sizes_e))
            , weights_and_biases(std::distance(layer_sizes_b, layer_sizes_e) - 1)
        {
            size_t layers_count = static_cast<size_t>(std::distance(layer_sizes_b, layer_sizes_e));
            for(size_t i=0;i<layers_count-1;++i)
            {
                auto next_layer_size_it = layer_sizes_b;
                size_t current_layer_size = *layer_sizes_b;
                size_t next_layer_size = *(++next_layer_size_it);
                layer_sizes_b = next_layer_size_it;
                layers[i] = layer_t(relu, current_layer_size);
                weights_and_biases[i] = layer_connection_t(current_layer_size, next_layer_size);
            }

            layers.back() = layer_t(soft_max, *layer_sizes_b);
        }
        multi_layer_perceptron_t(const multi_layer_perceptron_t& other) = default;
        multi_layer_perceptron_t(multi_layer_perceptron_t&& other) = default;
    public:
        multi_layer_perceptron_t& operator=(const multi_layer_perceptron_t& other) = default;
        multi_layer_perceptron_t& operator=(multi_layer_perceptron_t&& other) = default;
    public:
        void forward_pass(const vector_t& input) noexcept;
        void back_propagate(const vector_t& input) noexcept;
        void train(const matrix_t& X,
                   const matrix_t& Y,
                   size_t batch_size = 100,
                   size_t epochs = 100,
                   bool verbose=true) noexcept;
    };
}
