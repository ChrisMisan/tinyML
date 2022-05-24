#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <numeric>
#include "mnist_reader_less.h"
#include "matrix_vector_algebra.h"
#include "types.h"

using namespace std;

const number_t epsilon = 0.01f;
const number_t eta = 0.01f;
const number_t alpha = 0.01f;

template<typename It>
constexpr auto max_const(It b, It e)
{
    return *std::max_element(b, e);
}

constexpr int layers_sizes[] = {784, 5, 7, 10};
constexpr int how_many_layers = size(layers_sizes);
constexpr int max_layer_size = max_const(begin(layers_sizes), end(layers_sizes));


using activation_f_t = number_t(*)(number_t v);
using Label = uint8_t;
using Pixel = uint8_t;

number_t relu(number_t x) {
    return max((number_t)0, x);
}


number_t relu_d(number_t x) {
    if (x < 0) return 0;
    else return 1;
}


auto d_cross_entropy(const std::vector<number_t>& pred, const std::vector<number_t>& target) {
    std::vector<number_t> pred_copy(pred);
    return std::transform(
        begin(pred_copy),
        end(pred_copy),
        begin(target),
        begin(pred_copy),
        [](auto a, auto b) {return -b / a; });

}

template <typename T>
T get_random_number(T min, T max) {
    double r=(double)rand() / (INT_MAX);
    return (number_t)r * max + min;
}

struct layer_t
{
   /* size_t neuron_count;
    std::vector<activation_f_t> neuron_activation_functions;
    std::vector<activation_f_t> neuron_activation_functions_derivatives;*/
    void run_activation_fs(std::vector<number_t>& in_out)
    {
        std::transform(
            std::begin(in_out), std::end(in_out), std::begin(in_out),
            [](auto v){return relu(v);});
    }

    std::vector<number_t> calculate_output_layer_gradient(const std::vector<number_t>& pred_values, const std::vector<number_t>& target_values)
    {
        std::vector<number_t> gradients(pred_values.size(), {});
        auto delta = d_cross_entropy(pred_values, target_values);

        for(size_t i=0; i<pred_values.size(); i++)
        {
            // TODO: generalize cost function //cost function derivative
            gradients[i] = delta[i] * relu_d(pred_values[i]);
        }
        return gradients;
    }

    number_t sum_dow(const std::vector<number_t>& output_weights, const std::vector<number_t>& next_layer_gradients)
    {
        number_t sum = 0.0;
        for(size_t i=0; i<output_weights.size(); i++)
        {
            sum += output_weights[i] * next_layer_gradients[i];
        }
        return sum;
    }

    std::vector<number_t> calculate_hidden_layer_gradient(
        const matrix_t& weights, const std::vector<number_t>& next_layer_gradients,
        const std::vector<number_t>& values)
    {
        std::vector<number_t> gradients(values.size(), {});
        for(size_t i=0; i<values.size(); i++)
        {
            auto dow = sum_dow(weights[i], next_layer_gradients);
            gradients[i] = dow * relu_d(values[i]);
        }
        return gradients;
    }
};

struct layer_connection_t
{
    matrix_t W;
    matrix_t deltaW;
    std::vector<number_t> B;

    layer_connection_t(int neuron_num_current, int neuron_num_next) {
        for (int i = 0; i < neuron_num_next; i++) {
            W.push_back(std::vector<number_t>());
            B.push_back(0);
            for (int y = 0; y < neuron_num_current; y++) {
                auto r = get_random_number((number_t)0, (number_t)1);
                W[i].push_back(r);

            }
        }
    }





    std::vector<number_t> forward_pass(const std::vector<number_t>& input)
    {
        return vector_vector_add(matrix_vector_multiply(W, input), B);
    }

    void update_weigths(const std::vector<number_t>& prev_layer_values, const std::vector<number_t>& next_layer_gradient)
    {
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
    std::vector<layer_t> layers = {how_many_layers, layer_t()};
    std::vector<layer_connection_t> weights_and_biases;
   


    mlp_t(int number_of_layers, const int* layer_sizes) 
    {
        for(int i=0; i<number_of_layers-1;i++)
        {
            int current_layer_size = layer_sizes[i];
            int next_layer_size = layer_sizes[i+1];
            
            weights_and_biases.push_back(layer_connection_t(current_layer_size, next_layer_size));
            
        }
    }

    std::vector<std::vector<number_t>> forward_pass(const std::vector<number_t>& input)
    {
        std::vector<std::vector<number_t>> activations;
        auto current_v = input;

        // auto current_layer = std::begin(layers);
        // ++current_layer;
        // auto current_wb = std::begin(weights_and_biases);

        // for(;current_layer != std::end(layers); current_layer++, current_wb++)
        // {
        //     current_v = current_wb->forward_pass(current_v);
        //     current_layer->run_activation_fs(current_v);
        //     activations.push_back(current_v);
        // }

        for(int i=1; i<layers.size(); i++)
        {
            current_v = weights_and_biases[i-1].forward_pass(current_v);
            layers[i].run_activation_fs(current_v);
            activations.push_back(current_v); 
        }
        
        return activations;
    }

    void back_propagate(const std::vector<number_t>& target_values, const std::vector<std::vector<number_t>>& layers_values)
    {
        //auto error = cross_entropy(target_values, output_values);
        //auto avg_error = error / (layers.back().size() - 1);

        //RMSerror_ = sqrt(error);
        // Implement a recent average measurement
        // recentAverageError_ =
        //    (recentAverageError_ * recentAverageSmoothingFactor_ + RMSerror_) /
        //    (recentAverageSmoothingFactor_ + 1.0);

        auto next_layer = layers.rbegin();
        auto next_layer_values = layers_values.rbegin();
        auto next_layer_gradient = next_layer->calculate_output_layer_gradient(*next_layer_values, target_values);
        // update weigths?

        auto current_layer = next_layer;
        auto current_layer_values = next_layer_values;
        ++current_layer;
        ++current_layer_values;
        auto current_w = weights_and_biases.rbegin();
        auto lrend = layers.rend();
        for(; current_layer != lrend; ++current_layer, ++next_layer, ++current_w, ++current_layer_values)
        {
            auto gradient = current_layer->calculate_hidden_layer_gradient(current_w->W, next_layer_gradient, *current_layer_values);
            current_w->update_weigths(*current_layer_values, next_layer_gradient);
            next_layer_gradient = gradient;
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

auto train(mlp_t& network, const std::vector<std::vector<number_t>>& X, const std::vector<std::vector<number_t>>& Y, int batch_size=10, int epochs = 10)
{
    for(int epoch_number=0; epoch_number<epochs; epochs++){
        
        for(int batch_element_iterator=0; batch_element_iterator<batch_size; batch_element_iterator++)
        {
            int i = rand() % X.size();
            auto activations = network.forward_pass(X[i]);
            network.back_propagate(Y[i], activations);
        }

    }
}


int main() {

    srand(time(NULL));

    auto dataset = read_xor(100, 20);

    //auto mnist_dataset=read_mnist();
    /*
    mlp_t network = mlp_t(how_many_layers, layers_sizes);
    //network.initialize_weights();
    //network.initialize_biases();
    auto X = mnist_dataset.training_images;
    auto y = mnist_dataset.hot_encoded_training_labels;
    train(network, X, y);
    */
	return 0;
}




number_t** initialize_weights(number_t* W[how_many_layers])
{
    for(int i=0; i<how_many_layers; i++)
    {
        for(int j=0; j<layers_sizes[i]; i++)
        {
            W[i][j] = get_random_number(0, 1);
        }
    }
    return W;
}

number_t* initialize_biases(number_t B[how_many_layers])
{
    for(int i=0; i<how_many_layers; i++)
    {
        B[i] = get_random_number(0, 1);
    }

    return B;
}


