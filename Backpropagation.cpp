#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <numeric>
#include "matrix_vector_algebra.h"
#include "types.h"
#include "dataset.h"
#include "operations.h"
#include "activation.hpp"
#include "dataset.hpp"
#include <cbor.h>
#include <stdio.h>


using namespace std;


// HYPERPARAMETERS
const float LEARNING_CONSTANT = 0.1f;


// ARCHITECTURE
constexpr int layers_sizes[] = {784, 256, 10};
constexpr int how_many_layers = size(layers_sizes);
constexpr int max_layer_size = max_const(begin(layers_sizes), end(layers_sizes));


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
};

struct layer_connection_t
{ 
    matrix_t W;
    matrix_t deltaW;
    std::vector<number_t> B;

    layer_connection_t(int neuron_num_next, int neuron_num_current) {

        int W_size = neuron_num_next*(neuron_num_current+1);


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


};

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
                mlp::dot_mul_o(prev_layer.deltas, prev_layer.d_activations, prev_layer.deltas);
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

#define BUFFER_SIZE 8
unsigned char buffer[BUFFER_SIZE];
FILE* out;

void flush(size_t bytes) {
  if (bytes == 0) exit(1);  // All items should be successfully encoded
  if (fwrite(buffer, sizeof(unsigned char), bytes, out) != bytes) exit(1);
  if (fflush(out)) exit(1);
}

auto read_network(){
    out = std::fopen("W.cb", "r+");
    long n = 10;

    // Start an indefinite-length array
    flush(cbor_encode_indef_array_start(buffer, BUFFER_SIZE));
    // Write the array items one by one
    for (size_t i = 0; i < n; i++) {
        flush(cbor_encode_uint32(i, buffer, BUFFER_SIZE));
    }
    // Close the array
    flush(cbor_encode_break(buffer, BUFFER_SIZE));

    fclose(out);
}

auto save_network(mlp_t& network){
    cbor_item_t * root = cbor_new_definite_map(1);

    std::FILE* W_file = std::fopen("W.cb", "w+");

    std::vector<layer_t> layers = network.layers;
    std::vector<layer_connection_t> weights_and_biases = network.weights_and_biases;

    int number_of_connections = weights_and_biases.size();

    for (int i = 0; i < number_of_connections; i++){
        layer_connection_t connection = weights_and_biases[i];

        matrix_t W = connection.W;
        cbor_item_t * W_cbor = cbor_new_definite_array(size(W));

        //SERIALIZE W
        for (int i = 0; i<size(W); i++){
            vector_t vector = W[i];
            cbor_item_t * vector_item = cbor_new_definite_array(size(vector));

            for (int j = 0; j<size(W); j++){
                cbor_array_push(vector_item, cbor_build_uint8(vector[j]));
            }
            cbor_array_push(W_cbor, vector_item);
        }

        unsigned char * W_buffer;
        size_t W_buffer_size, W_length = cbor_serialize_alloc(root, &W_buffer, &W_buffer_size);
        fwrite(W_buffer, 1, W_length, W_file);
        fclose(W_file);
    }
}


auto train(mlp_t& network,
           const std::vector<std::vector<number_t>>& X,
           const std::vector<std::vector<number_t>>& Y,
           int batch_size=100, int epochs = 1000)
{
    for(int epoch_number=0; epoch_number<epochs; epoch_number++){
        int how_many_correct = 0;
        for(int batch_element_iterator=0; batch_element_iterator<batch_size; batch_element_iterator++)
        {
            int i = rand() % X.size();
            network.forward_pass(X[i]);
            auto y = network.layers.back().activations;
            
            if (are_equal(y, Y[i])) how_many_correct++;
            network.layers.back().deltas = vector_vector_subtract(y, Y[i]);

            network.back_propagate(X[i]);
        }
        std::cout << "EPOCH NUMBER: " << epoch_number << " ACCURACY: "<< how_many_correct*1.0/batch_size <<std::endl;

    }
}


int main() {

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
    save_network(network);
    read_network();

    
	return 0;
}
