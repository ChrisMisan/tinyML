#include <layers/input/connection.hpp>
#include <layers/input/trainer.hpp>
#include <default_activation_functions.hpp>

namespace tinyML
{
    void input_layer_connection_t::forward_pass(const vector_t &input, vector_t &output) const
    {
        std::copy(std::begin(input), std::end(input), std::begin(output));
    }

    [[nodiscard]] std::unique_ptr<base_layer_connection_trainer_t> input_layer_connection_t::build_trainer(const number_t &learning_constant)
    {
        return std::unique_ptr<base_layer_connection_trainer_t>(
            new input_layer_connection_trainer_t(*this, learning_constant)
        );
    }

    [[nodiscard]] nlohmann::json input_layer_connection_t::serialize() const
    {
        nlohmann::json j{};
        j["layer_type"] = "input";
        j["output_size"] = output_size();
        std::string activation_func = "unknown";
        if(activation_f == relu) activation_func = "relu";
        else if(activation_f == soft_max) activation_func = "soft_max";
        j["activation_f"] = activation_func;
        return j;
    }
    void input_layer_connection_t::deserialize(const nlohmann::json& data)
    {
    }
}