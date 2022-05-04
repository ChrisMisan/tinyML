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

constexpr int layers_size[] = {2, 5, 7};
constexpr int how_many_layers = size(layers_size);
constexpr int max_layer_size = max_const(begin(layers_size), end(layers_size));


using activation_f_t = number_t(*)(number_t v);


struct layer_t
{
    size_t neuron_count;
    std::vector<activation_f_t> neuron_activation_functions;
    std::vector<activation_f_t> neuron_activation_functions_derivatives;
    void run_activation_fs(std::vector<number_t>& in_out)
    {
        std::transform(
            std::begin(in_out), std::end(in_out),
            std::begin(neuron_activation_functions), std::begin(in_out),
            [](auto v, auto act_f){return act_f(v);});
    }

    std::vector<number_t> calculate_output_layer_gradient(const std::vector<number_t>& values, const std::vector<number_t>& target_values)
    {
        std::vector<number_t> gradients(values.size(), {});
        for(size_t i=0; i<values.size(); i++)
        {
            // TODO: generalize cost function
            auto delta = target_values[i] - values[i]; //cost function derivative
            gradients[i] = delta * neuron_activation_functions_derivatives[i](values[i]);
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
            gradients[i] = dow * neuron_activation_functions_derivatives[i](values[i]);
        }
        return gradients;
    }
};

std::vector<number_t> matrix_vector_multiply(const matrix_t& A, const std::vector<number_t>& B);
std::vector<number_t> vector_vector_add(const std::vector<number_t>& A, const std::vector<number_t>& B);

struct layer_connection_t
{
    matrix_t W;
    matrix_t deltaW;
    std::vector<number_t> B;

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

number_t cross_entropy(const std::vector<number_t>& a, const std::vector<number_t>& b)
{
    /*return std::transform_reduce(
        std::begin(a), std::end(b),
        std::begin(b), number_t(0),
        [](auto&& a, auto&& b){ return a + b;},
        [](auto&& a, auto&& b) {
            auto delta = a - b;
            return delta * delta;
        }
        );*/return{};
}

struct mlp_t
{
    std::vector<layer_t> layers = {how_many_layers, layer_t()};
    std::vector<layer_connection_t> weights_and_biases = {how_many_layers-1, layer_connection_t()};

    std::vector<number_t> forward_pass(const std::vector<number_t>& input)
    {
        auto current_v = input;

        auto current_layer = std::begin(layers);
        ++current_layer;
        auto current_wb = std::begin(weights_and_biases);

        for(;current_layer != std::end(layers); ++current_layer, ++current_wb)
        {
            current_v = current_wb->forward_pass(current_v);
            current_layer->run_activation_fs(current_v);
        }
        
        return current_v;
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


auto read_mnist() {
    auto dataset = mnist::read_dataset<>();
    return dataset;

}

template <typename T>
T get_random_number(T min, T max) {
    double r=(double)rand() / (INT_MAX);
    return (number_t)r * max + min;
}


int main() {

    srand(time(NULL));

    auto mnist_dataset=read_mnist();

    mlp_t network = mlp_t();

    int value = 1;
    
    auto A = matrix_t(10, vector_t(10, value));
    auto v = vector_t(10, value);
    auto result = matrix_vector_multiplication(A, v);

    for(auto& el: result)
    {
        cout<<el<<" ";
    }


	return 0;
}


number_t** initialize_weights(number_t* W[how_many_layers])
{
    for(int i=0; i<how_many_layers; i++)
    {
        for(int j=0; j<layers_size[i]; i++)
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

number_t relu(number_t x) {
    return max((number_t)0, x);
}


number_t relu_d(number_t x) {
    if (x < 0) return 0;
    else return 1;
}

