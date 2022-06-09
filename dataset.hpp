#pragma once

#include "common.hpp"
#include "mnist_reader_less.h"
#include <vector>


namespace mlp {
  struct Dataset {
        matrix_t training_images;
        matrix_t test_images;

        matrix_t  hot_encoded_training_labels;
        matrix_t hot_encoded_test_labels;

        static matrix_t hot_encode(const vector_t& labels, int vector_size = 10);
  };

       
  Dataset load_mnist();
  Dataset load_xor();
    
  std::pair<vector_t, vector_t> get_xor(bool x, bool y);

}
