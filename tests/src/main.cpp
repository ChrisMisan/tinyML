#include <numeric>
#include <array>
#include <iostream>

#include <dataset.hpp>
#include <multi_layer_perceptron.hpp>
#include <layers/input/definition.hpp>
#include <layers/dense/definition.hpp>
#include <default_activation_functions.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;
using namespace tinyML;

// HYPERPARAMETERS
constexpr number_t LEARNING_CONSTANT = 0.01;

// ARCHITECTURE
constexpr size_t layers_sizes[] = {784, 256, 10};
const auto input_layer = input_layer_definition_t(layers_sizes[0], relu);
const auto hidden_layer = dense_layer_definition_t(layers_sizes[1], relu);
const auto output_layer = dense_layer_definition_t(layers_sizes[2], soft_max);
const array<const base_layer_definition_t*, 3> nn_definition =
{
    &input_layer,
    &hidden_layer,
    &output_layer
};

int main() {
    std::vector<std::uint8_t> cbor;
    // initial training
    {
        auto mnist_dataset = load_mnist();

        multi_layer_perceptron_t network =
                multi_layer_perceptron_t(LEARNING_CONSTANT, begin(nn_definition), end(nn_definition));
        auto X = mnist_dataset.training_images;
        for(int i=0; i<X.size(); i++)
        {
            for (int j = 0; j < X[i].size(); j++) X[i][j] /= 255.0;
        }
        auto Xt = mnist_dataset.test_images;
        auto y = mnist_dataset.hot_encoded_training_labels;
        auto yt = mnist_dataset.hot_encoded_training_labels;
        network.train(X, y, 100, 100, true, true);
        cbor = json::to_cbor(network.serialize()); // serialize to cbor
    }
    std::cout << "something inbetween training sessions" << std::endl;
    // additional training
    {
        auto mnist_dataset = load_mnist();
        multi_layer_perceptron_t network = deserialize_mlp(json::from_cbor(cbor)); // create network from cbor
        auto X = mnist_dataset.training_images;
        for(int i=0; i<X.size(); i++)
        {
            for (int j = 0; j < X[i].size(); j++) X[i][j] /= 255.0;
        }
        auto Xt = mnist_dataset.test_images;
        auto y = mnist_dataset.hot_encoded_training_labels;
        auto yt = mnist_dataset.hot_encoded_training_labels;
        network.train(X, y, 100, 100, true, false);
        std::cout << "serializing to file cbor_test.bin" << std::endl;
        save_to_file("cbor_test.bin", network);
    }
    std::cout << "*restart of the device*" << std::endl;
    {
        auto mnist_dataset = load_mnist();
        multi_layer_perceptron_t network = load_from_file("cbor_test.bin"); // create network from cbor
        auto X = mnist_dataset.training_images;
        for(int i=0; i<X.size(); i++)
        {
            for (int j = 0; j < X[i].size(); j++) X[i][j] /= 255.0;
        }
        auto Xt = mnist_dataset.test_images;
        auto y = mnist_dataset.hot_encoded_training_labels;
        auto yt = mnist_dataset.hot_encoded_training_labels;
        network.train(X, y, 100, 100, true, false);
    }

	return 0;
}
