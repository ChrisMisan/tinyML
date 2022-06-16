#include <dataset.hpp>
#include <mnist/mnist_reader_less.h>
#include <iostream>

namespace tinyML
{
    matrix_t dataset_t::hot_encode(const vector_t& labels, size_t vector_size)
    {
        matrix_t hot_encoded(labels.size(), vector_t(vector_size, 0));
        for (unsigned int i = 0; i < labels.size(); i++)
        {
            int label = static_cast<int>(labels[i]);
            hot_encoded[i][label] = 1.0;
        }
        return hot_encoded;
    }

    vector_t to_vector_t(const std::vector<number_t>& mnist_vector)
    {
        vector_t vec(mnist_vector.size());
        copy(begin(mnist_vector), end(mnist_vector), begin(vec));
        return vec;
    }

    matrix_t to_matrix_t(const std::vector<std::vector<number_t>>& mnist_matrix)
    {
        matrix_t mat(mnist_matrix.size(), vector_t(mnist_matrix.front().size(), 0));
        for(size_t r=0;r<mnist_matrix.size();r++)
        {
            for(size_t c=0;c<mnist_matrix[r].size();c++)
            {
                mat[r][c] = mnist_matrix[r][c];
            }
        }
        return mat;
    }

    dataset_t load_mnist() {
        auto mnist_dataset = mnist::read_dataset<number_t, number_t>();
        dataset_t dataset;
        dataset.training_images = to_matrix_t(mnist_dataset.training_images);
        dataset.test_images = to_matrix_t(mnist_dataset.test_images);
        dataset.hot_encoded_training_labels = dataset_t::hot_encode(to_vector_t(mnist_dataset.training_labels));
        dataset.hot_encoded_test_labels = dataset_t::hot_encode(to_vector_t(mnist_dataset.test_labels));
        return dataset;
    }

    std::pair<vector_t, vector_t> get_xor(bool x1, bool x2) {
        vector_t X(2, 0.0);
        if (x1)
            X[0] = 0.0;
        else
            X[0] = 1.0;

        if (x2)
            X[1] = 0.0;
        else
            X[1] = 1.0;


        vector_t Y(2, 0.0);

        if (x1 ^ x2)
            Y[0] = 1.0;
        else
            Y[1] = 1.0;

        return std::make_pair(X, Y);
    }

    dataset_t load_xor(unsigned int train_size = 100, unsigned int test_size = 20)
    {
        dataset_t dataset;

        matrix_t X;
        matrix_t Y;
        bool x1, x2;
        for (size_t i = 0; i < train_size; i++) {
            x1 = (rand() % 2) == 0;
            x2 = (rand() % 2) == 0;
            auto xor_element = get_xor(x1, x2);
            X.push_back(xor_element.first);
            Y.push_back(xor_element.second);
        }
        dataset.training_images = X;
        dataset.hot_encoded_training_labels = Y;

        X.clear();
        Y.clear();

        for (size_t i = 0; i < test_size; i++) {
            x1 = (rand() % 2) == 0;
            x2 = (rand() % 2) == 0;
            auto xor_element = get_xor(x1, x2);
            X.push_back(xor_element.first);
            Y.push_back(xor_element.second);
        }
        dataset.test_images = X;
        dataset.hot_encoded_test_labels = Y;

        return dataset;
    }
}
