#include <multi_layer_perceptron.hpp>

#include <linalg.hpp>

#include <iostream>

namespace tinyML
{
    void multi_layer_perceptron_t::forward_pass(const vector_t& input) noexcept
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

    void multi_layer_perceptron_t::back_propagate(const vector_t& input) noexcept
    {
        const vector_t* visible;
        for(size_t index=this->layers.size(); index>=1;index--)
        {
            size_t i = index - 1;
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
                mat_mul_o(weights_and_biases[i-1].gradient_temp, learning_constant, weights_and_biases[i-1].gradient_temp);
                mat_sub_o(weights_and_biases[i-1].W, weights_and_biases[i-1].gradient_temp, weights_and_biases[i-1].W);

                splat_mul_o(layer.deltas, learning_constant, layer.deltas_temp);
                dot_sub_o(weights_and_biases[i-1].B, layer.deltas_temp, weights_and_biases[i-1].B);
            }
        }
    }

    void multi_layer_perceptron_t::train(
        const matrix_t& X,
        const matrix_t& Y,
        size_t batch_size,
        size_t epochs,
        bool verbose
        ) noexcept
    {
        for(size_t epoch_number=0; epoch_number<epochs; epoch_number++){
            size_t how_many_correct = 0;

            for(size_t batch_element_iterator=0; batch_element_iterator<batch_size; batch_element_iterator++)
            {
                size_t i = rand() % X.size();
                forward_pass(X[i]);
                auto& y = layers.back().activations;

                if (are_equal(y, Y[i])) how_many_correct++;
                dot_sub_o(y, Y[i], layers.back().deltas);

                back_propagate(X[i]);
            }

            if(verbose)
                std::cout << "EPOCH NUMBER: " << epoch_number << " ACCURACY: "<< how_many_correct*1.0/batch_size <<std::endl;
        }
    }
}