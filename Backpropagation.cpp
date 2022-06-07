#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <algorithm>
#include <numeric>
#include "mnist_reader_less.h"
#include "matrix_vector_algebra.h"
#include "types.h"
#include "activation.hpp"

using namespace std;

const number_t epsilon = 0.01f;
const number_t eta = 0.01f;
const number_t alpha = 0.01f;
const float LEARNING_CONSTANT = 0.1f;

template<typename It>
constexpr auto max_const(It b, It e)
{
    return *std::max_element(b, e);
}

constexpr int layers_sizes[] = {784, 256, 10};
constexpr int how_many_layers = size(layers_sizes);
constexpr int max_layer_size = max_const(begin(layers_sizes), end(layers_sizes));


using activation_f_t = number_t(*)(number_t v);
using Label = uint8_t;
using Pixel = uint8_t;


auto d_cross_entropy(const std::vector<number_t>& pred, const std::vector<number_t>& target) {
    std::vector<number_t> pred_copy(pred);
    std::transform(
        begin(pred_copy),
        end(pred_copy),
        begin(target),
        begin(pred_copy),
        [](auto p, auto t) {return (p-t);});
    return pred_copy;
}

template <typename T>
T get_random_number(T min, T max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

void print_vec(const std::vector<number_t>& vec)
{
    for (auto number : vec)
    {
        std::cout << number << " ";
    }
}

struct layer_t
{
    mlp::activation_f_t activation_f;
    std::vector<number_t> activations;
    std::vector<number_t> d_activations;
    std::vector<number_t> deltas;

    layer_t(mlp::activation_f_t activation_f, size_t length)
    {
        this->activation_f = activation_f;
        activations = std::vector<number_t>(length);
        d_activations = std::vector<number_t>(length);
        deltas = std::vector<number_t>(length);
    }


    std::vector<number_t> calculate_output_layer_gradient(const std::vector<number_t>& pred_values, const std::vector<number_t>& target_values)
    {
        std::vector<number_t> gradients(pred_values.size(), {});
        auto delta = d_cross_entropy(pred_values, target_values);

        activation_f.derivative(pred_values, gradients);

        for(size_t i=0; i<pred_values.size(); i++)
        {
            gradients[i] = delta[i] * gradients[i];
        }
        return gradients;
    }

    number_t sum_dow(const matrix_t& weights, size_t i, const std::vector<number_t>& next_layer_gradients)
    {
        number_t sum = 0.0;
        for(size_t j=0; j<weights.size(); j++)
        {
            sum += weights[j][i] * next_layer_gradients[j];
        }
        return sum;
    }

    std::vector<number_t> calculate_hidden_layer_gradient(
        const matrix_t& weights, const std::vector<number_t>& next_layer_gradients,
        const std::vector<number_t>& values)
    {
        std::vector<number_t> gradients(values.size(), {});
        activation_f.derivative(values, gradients);

        for(size_t i=0; i<values.size(); i++)
        {
            auto dow = sum_dow(weights, i, next_layer_gradients);
            gradients[i] = dow * gradients[i];
        }
        return gradients;
    }
};

struct layer_connection_t
{
    matrix_t W;
    matrix_t deltaW;
    std::vector<number_t> B;

    layer_connection_t(int neuron_num_next, int neuron_num_current) {
        for (int i = 0; i < neuron_num_next; i++) {
            W.push_back(std::vector<number_t>());
            deltaW.push_back(std::vector<number_t>());
            
            for (int y = 0; y < neuron_num_current; y++) {
                auto r = get_random_number((number_t)0, (number_t)1);
                W[i].push_back(r);
                deltaW[i].push_back(0);
            }
        }

        for (int y = 0; y < neuron_num_current; y++)
        {
            B.push_back(0);
        }
    }


    std::vector<number_t> forward_pass(const vector_t& input)
    {
        return vector_vector_add(matrix_multiply(input, W)[0], B);
    }

    void update_weigths(const std::vector<number_t>& prev_layer_values, const std::vector<number_t>& next_layer_gradient)
    {
        if (prev_layer_values.size()!=next_layer_gradient.size()) return;
        for(size_t i=0;i<W.size();i++)
        {
            for(size_t j=0;j<W[i].size();j++)
            {
                auto old_delta_weight = deltaW[i][j];
                auto new_delta_weight = eta * prev_layer_values[j] * next_layer_gradient[i] + alpha * old_delta_weight;
                deltaW[i][j] = new_delta_weight;
                W[i][j] = W[i][j] + deltaW[i][j];
            }
        }
    }
};

number_t convert_to_number(pixel_t pixel) {return pixel;}

number_t cross_entropy(std::vector<number_t>& a, std::vector<number_t>& b)
{
    std::transform(
        std::begin(a), 
        std::end(a),
        std::begin(b),
        std::begin(a),
        [](auto p, auto q){ return p*log(q);});
    return -std::accumulate(begin(a), end(a), a[0]);

}


struct mlp_t
{
    std::vector<layer_t> layers;
    std::vector<layer_connection_t> weights_and_biases;

    mlp_t(int number_of_layers, const int* layer_sizes) 
    {
        for(int i=0; i<number_of_layers-1;i++)
        {
            int current_layer_size = layer_sizes[i];
            int next_layer_size = layer_sizes[i+1];
            layers.push_back(layer_t(mlp::relu, layer_sizes[i]));
            weights_and_biases.push_back(layer_connection_t(current_layer_size, next_layer_size));
            
        }
        layers.push_back(layer_t(mlp::soft_max, layer_sizes[number_of_layers-1]));
    }

    void forward_pass(const std::vector<number_t>& input)
    {
        auto current_v = input;

        layers[0].activation_f(current_v, layers[0].activations);
        layers[0].activation_f.derivative(current_v, layers[0].d_activations);

        for(int i=1; i<layers.size(); i++)
        {
            current_v = weights_and_biases[i-1].forward_pass(current_v);
            layers[i].activation_f(current_v, layers[i].activations);
            if(i<layers.size()-1)layers[i].activation_f.derivative(current_v, layers[i].d_activations);
        }
    }
    void back_propagate(const vector_t& input){
        vector_t visible;
        for(int i=this->layers.size()-1; i>=0; i--)
        {
            auto& layer = layers[i];
            if(i>0)
            {
                auto prev_layer = this->layers[i-1];
                auto weights = this->weights_and_biases[i - 1].W;
                visible = prev_layer.activations;
                prev_layer.deltas = matrix_multiply(layer.deltas, transpose(weights))[0];
                for (int j = 0; j < prev_layer.deltas.size(); j++)
                {
                    prev_layer.deltas[j] *= prev_layer.d_activations[j];
                }
               
            }
            else
            {
                visible = input;
            }

            auto gradient = matrix_multiply(transpose(visible), layer.deltas);

            if (i > 0) {
                for (int j = 0; j < gradient.size(); j++)
                {
                    for (int k = 0; k < gradient[j].size(); k++)
                    {
                        weights_and_biases[i - 1].W[j][k] -= LEARNING_CONSTANT * gradient[j][k];
                    }
                }
            }

        }

    }

};

std::vector<std::vector<number_t>> hot_encode(const std::vector<number_t>& labels, int vector_size=10)
{
    std::vector<std::vector<number_t>> hot_encoded(labels.size(), std::vector<number_t>(vector_size, 0));
    for(unsigned int i=0; i<labels.size(); i++)
    {
        int label = labels[i];
        hot_encoded[i][label] = 1.0; 
    }
    return hot_encoded;
}
struct Dataset {
        std::vector<std::vector<number_t>> training_images;
        std::vector<std::vector<number_t>> test_images;     

        std::vector<std::vector<number_t>> hot_encoded_training_labels;
        std::vector<std::vector<number_t>> hot_encoded_test_labels;
};


auto read_mnist() {
    auto mnist_dataset = mnist::read_dataset<number_t, number_t>();
    Dataset dataset;
    dataset.training_images = mnist_dataset.training_images;
    dataset.test_images = mnist_dataset.test_images;
    dataset.hot_encoded_training_labels = hot_encode(mnist_dataset.training_labels);
    dataset.hot_encoded_test_labels = hot_encode(mnist_dataset.test_labels);
    return dataset;
}

auto get_xor(bool x1, bool x2) {
    std::vector<number_t> X(2, 0.0);
    if (x1)
        X[0] = 0.0;
    else
        X[0] = 1.0;

    if (x2)
        X[1] = 0.0;
    else
        X[1] = 1.0;


    std::vector<number_t> Y(2, 0.0);

    if (x1 ^ x2)
        Y[0] = 1.0;
    else
        Y[1] = 1.0;

    return std::make_pair(X, Y);
}

auto read_xor(unsigned int train_size=100, unsigned int test_size = 20) {
    Dataset dataset;

    std::vector<std::vector<number_t>> X;
    std::vector<std::vector<number_t>> Y;
    bool x1, x2;
    for (int i = 0; i < train_size; i++)
    {
        x1 = (rand() % 2) == 0;
        x2 = (rand() % 2) == 0;
        auto xor_element = get_xor(x1, x2);
        X.push_back(xor_element.first);
        Y.push_back(xor_element.second);
    }
    dataset.training_images = X;
    dataset.hot_encoded_training_labels = Y;
    
    X.clear();
    Y.clear();

    for (int i = 0; i < test_size; i++)
    {
        x1 = (rand() % 2) == 0;
        x2 = (rand() % 2) == 0;
        auto xor_element = get_xor(x1, x2);
        X.push_back(xor_element.first);
        Y.push_back(xor_element.second);
    }
    dataset.test_images = X;
    dataset.hot_encoded_test_labels = Y;

    return dataset;
}


auto accuracy(std::vector<Label> predictions, mnist::MNIST_dataset<number_t> mnist_dataset){
    int good_predictions = 0;
    int test_sample_size = mnist_dataset.test_images.size();

    for (int i = 0; i<test_sample_size; i++){
        if (mnist_dataset.test_labels[i] == predictions[i]){
            good_predictions += 1;
        }
    }

    return (good_predictions/test_sample_size*1.0);
}


auto train(mlp_t& network,
           const std::vector<std::vector<number_t>>& X,
           const std::vector<std::vector<number_t>>& Y,
           int batch_size=100, int epochs = 100)
{
    for(int epoch_number=0; epoch_number<epochs; epoch_number++){
        int how_many_correct = 0;
        for(int batch_element_iterator=0; batch_element_iterator<batch_size; batch_element_iterator++)
        {
            int i = rand() % X.size();
            network.forward_pass(X[i]);
            auto y = network.layers.back().activations;
            //network.layers.back().d_activations = d_cross_entropy(y, Y[i]);
            for (int j = 0; j < network.layers.back().activations.size(); j++)
            {
             //   std::cout << network.layers[3].activations[j]<<" ";
            }
            //std::cout << "   Y: ";
            
            for (int j = 0; j <Y[i].size(); j++)
            {
                // std::cout << Y[i][j] << " ";
            }
            //std::cout << std::endl;
            
            
            if (are_equal(y, Y[i])) how_many_correct++;
            network.layers.back().deltas = vector_vector_subtract(y, Y[i]);

            network.back_propagate(X[i]);
        }
        std::cout << "EPOCH NUMBER: " << epoch_number << " ACCURACY: "<< how_many_correct*1.0/batch_size <<std::endl;
    }
}



int main() {

    srand(time(NULL));

    auto mnist_dataset=read_mnist();
    
    mlp_t network = mlp_t(how_many_layers, layers_sizes);
    auto X = mnist_dataset.training_images;
    for(int i=0; i<X.size(); i++)
    {
        for (int j = 0; j < X[i].size(); j++) X[i][j] /= 255.0;
    }
    auto Xt = mnist_dataset.test_images;
    auto y = mnist_dataset.hot_encoded_training_labels;
    auto yt = mnist_dataset.hot_encoded_training_labels;
    train(network, X, y);
    
	return 0;
}
