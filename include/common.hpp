#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <memory>
#include <nlohmann/json.hpp>
#include <cassert>

namespace tinyML
{
    using number_t = float;
    using allocator_t = std::allocator<number_t>;
    using vector_t = std::vector<number_t, allocator_t>;
    using matrix_t = std::vector<std::vector<number_t, allocator_t>>;

    struct activation_f_t;

    class serializable;

    class base_layer_definition_t;
    class base_layer_connection_t;
    class base_layer_connection_trainer_t;

    class multi_layer_perceptron_t;

    class dataset_t;
}
