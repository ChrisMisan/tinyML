#include <layers/dense/trainer.hpp>
#include <linalg.hpp>

namespace tinyML
{
    dense_layer_connection_trainer_t::dense_layer_connection_trainer_t(dense_layer_connection_t& trained_layer, const number_t& learning_constant)
            : base_layer_connection_trainer_t(learning_constant)
            , _trained_layer(trained_layer)
            , weights_transposed(trained_layer.output_size(), vector_t(trained_layer.input_size(), 0))
            , activation_derivative_values(trained_layer.output_size(), 0)
            , weights_gradient(trained_layer.input_size(), vector_t(trained_layer.output_size(), 0))
            , biases_gradient(trained_layer.output_size(), 0)
    { }

    void dense_layer_connection_trainer_t::reinitialize()
    {
        std::generate(std::begin(_trained_layer.biases), std::end(_trained_layer.biases), []()
        {
            return get_random_number<number_t>(0.0, 1.0);
        });
        std::for_each(std::begin(_trained_layer.weights), std::end(_trained_layer.weights), [](auto &&wrow)
        {
            std::generate(std::begin(wrow), std::end(wrow), []()
            {
                return get_random_number<number_t>(0.0, 1.0);
            });
        });
    }

    const vector_t& dense_layer_connection_trainer_t::activation_derivatives() const noexcept
    {
        return activation_derivative_values;
    }

    void dense_layer_connection_trainer_t::training_forward_pass(const vector_t& input, vector_t& output)
    {
        vec_mat_mul_o(input, _trained_layer.weights, output);
        dot_add_o(output, _trained_layer.biases, output);

        _trained_layer.activation_f.derivative(output, activation_derivative_values);

        _trained_layer.activation_f.invoke(output, output);
    }

    void dense_layer_connection_trainer_t::training_backward_pass(const vector_t& input, const vector_t& input_deltas, vector_t& output_deltas)
    {
        transpose_o(_trained_layer.weights, weights_transposed);
        vec_mat_mul_o(input_deltas, weights_transposed, output_deltas);

        // weights backprop
        mul_to_mat_o(input, input_deltas, weights_gradient);
        mat_mul_o(weights_gradient, learning_constant, weights_gradient);
        mat_sub_o(_trained_layer.weights, weights_gradient, _trained_layer.weights);

        // biases backprop
        splat_mul_o(input_deltas, learning_constant, biases_gradient);
        dot_sub_o(_trained_layer.biases, biases_gradient, _trained_layer.biases);
    }
}
