#include <multi_layer_perceptron.hpp>
#include <layers/definition.hpp>
#include <layers/trainer.hpp>
#include <linalg.hpp>
#include <iostream>
#include <fstream>

#include <default_activation_functions.hpp>

#include <layers/input/definition.hpp>
#include <layers/dense/definition.hpp>

namespace tinyML
{
    void multi_layer_perceptron_t::forward_pass(const vector_t& input, vector_t& output) noexcept
    {
        const vector_t* current_v = &input;
        for(size_t i=0;i<layers.size();i++)
        {
            layers[i]->forward_pass(*current_v, layer_intermediate_outputs[i]);
            current_v = &layer_intermediate_outputs[i];
        }

        std::copy(
                std::begin(layer_intermediate_outputs.back()),
                std::end(layer_intermediate_outputs.back()),
                std::begin(output)
                );
    }

    void multi_layer_perceptron_t::train(
        const matrix_t& X,
        const matrix_t& Y,
        size_t batch_size,
        size_t epochs,
        bool verbose,
        bool reinitialize
        )
    {
        std::vector<std::unique_ptr<base_layer_connection_trainer_t>> layer_trainers;
        std::vector<vector_t> layer_intermediate_deltas;
        for(auto& layer : layers)
        {
            layer_trainers.emplace_back(layer->build_trainer(learning_constant));
            layer_intermediate_deltas.emplace_back(vector_t(layer->output_size()));
        }
        if(reinitialize)
        {
            for(auto& trainer : layer_trainers)
            {
                trainer->reinitialize();
            }
        }

        auto training_forward_pass = [&](const vector_t& input){
            const vector_t* current_v = &input;
            for(size_t i=0;i<layers.size();i++)
            {
                layer_trainers[i]->training_forward_pass(*current_v, layer_intermediate_outputs[i]);
                current_v = &layer_intermediate_outputs[i];
            }
        };

        auto training_backward_pass = [&](){
            for(size_t i=layers.size()-1;i>0;i--)
            {
                layer_trainers[i]->training_backward_pass(
                        layer_intermediate_outputs[i - 1],
                        layer_intermediate_deltas[i],
                        layer_intermediate_deltas[i-1]
                        );

                dot_mul_o(layer_intermediate_deltas[i-1], layer_trainers[i-1]->activation_derivatives(), layer_intermediate_deltas[i-1]);
            }
        };

        for(size_t epoch_number=0; epoch_number<epochs; epoch_number++)
        {
            size_t how_many_correct = 0;
            for(size_t batch_element_iterator=0; batch_element_iterator<batch_size; batch_element_iterator++)
            {
                size_t i = rand() % X.size();

                training_forward_pass(X[i]);

                auto& y = layer_intermediate_outputs.back();

                if (are_equal(y, Y[i])) how_many_correct++;

                dot_sub_o(y, Y[i], layer_intermediate_deltas.back());
                training_backward_pass();
            }

            if(verbose)
                std::cout << "EPOCH NUMBER: " << epoch_number << " ACCURACY: "<< static_cast<double>(how_many_correct)/batch_size <<std::endl;
        }
    }

    nlohmann::json multi_layer_perceptron_t::serialize() const
    {
        nlohmann::json j{};
        j["learning_constant"] = learning_constant;
        auto mlp_layers = nlohmann::json::array();
        for(auto& layer : layers)
        {
            mlp_layers.push_back(layer->serialize());
        }
        j["layers"] = mlp_layers;
        return j;
    }

    multi_layer_perceptron_t deserialize_mlp(const nlohmann::json& data)
    {
        std::vector<std::unique_ptr<base_layer_definition_t>> layers;
        for(auto& layer_data : data["layers"])
        {
            size_t output_size = layer_data["output_size"].get<size_t>();

            activation_f_t activation_f{};
            if(layer_data["activation_f"].get<std::string>()== "relu") activation_f = relu;
            else if(layer_data["activation_f"].get<std::string>() == "soft_max") activation_f = soft_max;

            if(layer_data["layer_type"].get<std::string>() == "input")
            {
                layers.emplace_back(new input_layer_definition_t(output_size, activation_f));
            }
            else if(layer_data["layer_type"].get<std::string>() == "dense")
            {
                layers.emplace_back(new dense_layer_definition_t(output_size, activation_f));
            }
        }
        number_t learning_constant = data["learning_constant"].get<number_t>();
        auto mlp = multi_layer_perceptron_t(learning_constant, std::begin(layers), std::end(layers));
        size_t i=0;
        for(auto& layer_data : data["layers"])
        {
            mlp.layers[i++]->deserialize(layer_data);
        }
        return mlp;
    }

    void save_to_file(const char* path, const multi_layer_perceptron_t& mlp)
    {
        auto cbor = nlohmann::json::to_cbor(mlp.serialize());

        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(cbor.data()), static_cast<std::streamsize>(cbor.size()));
    }
    multi_layer_perceptron_t load_from_file(const char* path)
    {
        std::ifstream file(path, std::ios::binary);

        file.seekg(0, std::ios::end);
        size_t length = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> cbor(length);
        file.read(reinterpret_cast<char*>(cbor.data()), static_cast<std::streamsize>(cbor.size()));
        return deserialize_mlp(nlohmann::json::from_cbor(cbor));
    }
}