#pragma once

#include "activation.hpp"

namespace mlp
{
    struct layer_t
    {
        activation_f_t activation_f = relu;

        void run_activation_function(const vector_t& in, vector_t& out) const
        {
            activation_f(in, out);
        }
    };

    vector_t calculate_output_later_gradient(const layer_t& layer, const vector_t& pred_values, const vector_t& target_values)
    {
        std::vector<number_t> gradients(pred_values.size(), {});
        auto delta = d_cross_entropy(pred_values, target_values);

        for(size_t i=0; i<pred_values.size(); i++)
        {
            gradients[i] = delta[i] * relu_d(pred_values[i]);
        }
        return gradients;
    }
}

class layer_t
{
public:
    void run_activation_fs(std::vector<number_t>& in_out)
    {
        std::transform(
                std::begin(in_out), std::end(in_out), std::begin(in_out),
                [](auto v){return relu(v);});
    }

    std::vector<number_t> calculate_output_layer_gradient()
    {

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
        for(size_t i=0; i<values.size(); i++)
        {
            auto dow = sum_dow(weights, i, next_layer_gradients);
            gradients[i] = dow * relu_d(values[i]);
        }
        return gradients;
    }
};
