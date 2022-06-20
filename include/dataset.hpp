#pragma once

#include <common.hpp>

namespace tinyML
{
    class dataset_t
    {
    public:
        matrix_t training_images;
        matrix_t test_images;

        matrix_t hot_encoded_training_labels;
        matrix_t hot_encoded_test_labels;
    public:
        static matrix_t hot_encode(const vector_t& labels, size_t vector_size = 10);
    };

    dataset_t load_mnist();
    dataset_t load_xor();
}
