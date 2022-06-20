#include <layers/input/trainer.hpp>

namespace tinyML
{
    input_layer_connection_trainer_t::input_layer_connection_trainer_t(
        input_layer_connection_t& trained_layer,
        const number_t& learning_constant
    )
        : base_layer_connection_trainer_t(learning_constant), _trained_layer(trained_layer)
        , activation_derivative_values(trained_layer.output_size())
    {}

    void input_layer_connection_trainer_t::reinitialize()
    { }

    const vector_t& input_layer_connection_trainer_t::activation_derivatives() const noexcept
    {
        return activation_derivative_values;
    }

    void input_layer_connection_trainer_t::training_forward_pass(const vector_t& input, vector_t& output)
    {
        _trained_layer.activation_f.derivative(input, activation_derivative_values);
        _trained_layer.activation_f.invoke(input, output);
    }

    void input_layer_connection_trainer_t::training_backward_pass(
            const vector_t &input, const vector_t &input_deltas, vector_t &output_deltas)
    {
        assert(false);
    }
}
