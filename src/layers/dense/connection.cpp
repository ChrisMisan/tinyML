#include <layers/dense/connection.hpp>
#include <layers/dense/trainer.hpp>
#include <default_activation_functions.hpp>
#include <linalg.hpp>

namespace tinyML
{
    dense_layer_connection_t::dense_layer_connection_t(size_t input_size, size_t output_size, const activation_f_t& activation_f)
        : base_layer_connection_t(input_size, output_size)
        , weights(input_size, vector_t(output_size, 0))
        , biases(output_size, 0)
        , activation_f(activation_f)
    {
    }

    void dense_layer_connection_t::forward_pass(const vector_t& input, vector_t& output) const
    {
        vec_mat_mul_o(input, weights, output);
        dot_add_o(output, biases, output);
        activation_f.invoke(output, output);
    }

    std::unique_ptr<base_layer_connection_trainer_t> dense_layer_connection_t::build_trainer(const number_t& learning_constant)
    {
        return std::unique_ptr<base_layer_connection_trainer_t>(
                new dense_layer_connection_trainer_t(*this, learning_constant)
                );
    }

    [[nodiscard]] nlohmann::json dense_layer_connection_t::serialize() const
    {
        nlohmann::json j{};
        j["layer_type"] = "dense";
        j["output_size"] = output_size();
        std::string activation_func = "unknown";
        if(activation_f == relu) activation_func = "relu";
        else if(activation_f == soft_max) activation_func = "soft_max";
        j["activation_f"] = activation_func;
        auto weights_array = nlohmann::json::array();
        for(auto& wr: weights)
        {
            auto weights_row_array = nlohmann::json::array();
            for(auto& v : wr)
            {
                weights_row_array.push_back(v);
            }
            weights_array.push_back(std::move(weights_row_array));
        }
        auto biases_array = nlohmann::json::array();
        for(auto& v : biases)
        {
            biases_array.push_back(v);
        }
        j["weights"] = weights_array;
        j["biases"] = biases_array;
        return j;
    }

    void dense_layer_connection_t::deserialize(const nlohmann::json& data)
    {
        for(size_t i=0;i<weights.size();i++)
        {
            weights[i] = data["weights"][i].get<vector_t>();
        }
        biases = data["biases"].get<vector_t>();
    }
}
