#pragma once

#include <common.hpp>
#include <layers/connection.hpp>

namespace tinyML
{
    class multi_layer_perceptron_t
    {
    private:
        std::vector<std::unique_ptr<base_layer_connection_t>> layers;
        std::vector<vector_t> layer_intermediate_outputs;
        number_t learning_constant;
    public:
        multi_layer_perceptron_t() = delete;
        template<typename It>
        multi_layer_perceptron_t(number_t learning_constant, It layer_definitions_b, It layer_definitions_e)
            : learning_constant(learning_constant)
        {
            size_t input_size = (*layer_definitions_b)->output_size();
            while(layer_definitions_b != layer_definitions_e)
            {
                layers.emplace_back((*layer_definitions_b)->build_kernel(input_size));
                input_size = (*layer_definitions_b)->output_size();
                ++layer_definitions_b;
            }

            for(size_t i=0;i<layers.size();i++)
            {
                layer_intermediate_outputs.emplace_back(vector_t(layers[i]->output_size()));
            }
        }
        multi_layer_perceptron_t(const multi_layer_perceptron_t& other) = default;
        multi_layer_perceptron_t(multi_layer_perceptron_t&& other) = default;
    public:
        multi_layer_perceptron_t& operator=(const multi_layer_perceptron_t& other) = default;
        multi_layer_perceptron_t& operator=(multi_layer_perceptron_t&& other) = default;
    public:
        void forward_pass(const vector_t& input, vector_t& output) noexcept;
    public:
        void train(const matrix_t& X,
                   const matrix_t& Y,
                   size_t batch_size = 100,
                   size_t epochs = 100,
                   bool verbose=true,
                   bool reinitialize=true);
    public:
        nlohmann::json serialize() const;
    public:
        friend multi_layer_perceptron_t deserialize_mlp(const nlohmann::json& data);
    };

    multi_layer_perceptron_t deserialize_mlp(const nlohmann::json& data);

    void save_to_file(const char* path, const multi_layer_perceptron_t& mlp);
    multi_layer_perceptron_t load_from_file(const char* path);
}
