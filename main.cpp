#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <numeric>

#include <common.hpp>
#include <activation.hpp>
#include <dataset.hpp>

using namespace std;
using namespace mlp;


// HYPERPARAMETERS
const float LEARNING_CONSTANT = 0.01f;


// ARCHITECTURE
constexpr int layers_sizes[] = {784, 256, 10};
constexpr int how_many_layers = size(layers_sizes);
constexpr int max_layer_size = max_const(begin(layers_sizes), end(layers_sizes));

namespace mlp
{
    struct layer_t
    {
        activation_f_t activation_f;
        vector_t activations;
        vector_t d_activations;
        vector_t deltas;
        vector_t deltas_temp;

        layer_t() = default;

        layer_t(activation_f_t activation_f, size_t length)
            : activation_f(activation_f)
            , activations(length, 0)
            , d_activations(length, 0)
            , deltas(length, 0)
            , deltas_temp(length, 0)
        {}
    };

    struct layer_connection_t
    {
        matrix_t W;
        matrix_t gradient_temp;
        matrix_t W_transposed;
        vector_t B;
        vector_t out_temp;
        vector_t out_temp_internal;

        layer_connection_t() = default;

        layer_connection_t(int neuron_num_next, int neuron_num_current)
            : W(neuron_num_next, vector_t(neuron_num_current, 0))
            , gradient_temp(neuron_num_next, vector_t(neuron_num_current, 0))
            , W_transposed(neuron_num_current, vector_t(neuron_num_next, 0))
            , B(neuron_num_current, 0)
            , out_temp(neuron_num_current, 0)
            , out_temp_internal(neuron_num_current, 0)
        {
            for(size_t i=0;i<height(W);i++)
            {
                for(size_t j=0;j<width(W);j++)
                {
                    W[i][j] = get_random_number(0.0f, 1.0f);
                }
            }

            for(size_t i=0;i<neuron_num_current;i++)
            {
                B[i]= get_random_number(0.0f, 1.0f);
            }
        }

        void forward_pass(const vector_t& input, vector_t& out)
        {
            vec_mat_mul_o(input, W, out_temp_internal);
            dot_add_o(out_temp_internal, B, out);
        }
    };

    struct mlp_t
    {
        std::vector<layer_t> layers;
        std::vector<layer_connection_t> weights_and_biases;

        mlp_t(int number_of_layers, const int* layer_sizes)
            : layers(number_of_layers)
            , weights_and_biases(number_of_layers - 1)
        {
            for(int i=0; i<number_of_layers-1;i++)
            {
                int current_layer_size = layer_sizes[i];
                int next_layer_size = layer_sizes[i+1];
                layers[i] = layer_t(mlp::relu, layer_sizes[i]);
                weights_and_biases[i] = layer_connection_t(current_layer_size, next_layer_size);
            }
            layers.back() = layer_t(mlp::soft_max, layer_sizes[number_of_layers-1]);
        }

        void forward_pass(const vector_t& input)
        {
            const vector_t* current_v = &input;

            layers[0].activation_f(*current_v, layers[0].activations);
            layers[0].activation_f.derivative(*current_v, layers[0].d_activations);

            for(int i=1; i<layers.size(); i++)
            {
                weights_and_biases[i-1].forward_pass(*current_v, weights_and_biases[i-1].out_temp);
                current_v = &weights_and_biases[i-1].out_temp;
                layers[i].activation_f(*current_v, layers[i].activations);
                if(i<layers.size()-1)
                    layers[i].activation_f.derivative(*current_v, layers[i].d_activations);
            }
        }
        void back_propagate(const vector_t& input){
            const vector_t* visible;
            for(int i=this->layers.size()-1; i>=0; i--)
            {
                auto& layer = layers[i];
                if(i>0)
                {
                    auto& prev_layer = this->layers[i-1];
                    auto& weights = this->weights_and_biases[i - 1].W;
                    auto& weights_transposed = this->weights_and_biases[i - 1].W_transposed;
                    visible = &prev_layer.activations;

                    transpose_o(weights, weights_transposed);
                    vec_mat_mul_o(layer.deltas, weights_transposed, prev_layer.deltas);
                    dot_mul_o(prev_layer.deltas, prev_layer.d_activations, prev_layer.deltas);
                }
                else
                {
                    visible = &input;
                }

                if (i > 0)
                {
                    mul_to_mat_o(*visible, layer.deltas, weights_and_biases[i-1].gradient_temp);
                    mat_mul_o(weights_and_biases[i-1].gradient_temp, LEARNING_CONSTANT, weights_and_biases[i-1].gradient_temp);
                    mat_sub_o(weights_and_biases[i-1].W, weights_and_biases[i-1].gradient_temp, weights_and_biases[i-1].W);

                    splat_mul_o(layer.deltas, LEARNING_CONSTANT, layer.deltas_temp);
                    dot_sub_o(weights_and_biases[i-1].B, layer.deltas_temp, weights_and_biases[i-1].B);
                }
            }

        }

    };
}

auto train(mlp_t& network,
           const matrix_t& X,
           const matrix_t& Y,
           int batch_size=100, int epochs = 100)
{
    for(int epoch_number=0; epoch_number<epochs; epoch_number++){
        int how_many_correct = 0;
        for(int batch_element_iterator=0; batch_element_iterator<batch_size; batch_element_iterator++)
        {
            int i = rand() % X.size();
            network.forward_pass(X[i]);
            auto& y = network.layers.back().activations;
            
            if (are_equal(y, Y[i])) how_many_correct++;
            dot_sub_o(y, Y[i], network.layers.back().deltas);

            network.back_propagate(X[i]);
        }
        std::cout << "EPOCH NUMBER: " << epoch_number << " ACCURACY: "<< how_many_correct*1.0/batch_size <<std::endl;
    }
}



int main() {

    {
        srand(time(NULL));

        auto mnist_dataset=mlp::load_mnist();

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
    }


    cout << counting_allocator<number_t>::allocations/1e6 << "MB" << endl;
    cout << counting_allocator<number_t>::deallocations/1e6 << "MB" << endl;

	return 0;
}
