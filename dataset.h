#include "dataset.hpp"

namespace mlp
{


    matrix_t Dataset::hot_encode(const vector_t& labels, int vector_size)
    {
        std::vector<std::vector<number_t>> hot_encoded(labels.size(), std::vector<number_t>(vector_size, 0));
        for (unsigned int i = 0; i < labels.size(); i++)
        {
            int label = labels[i];
            hot_encoded[i][label] = 1.0;
        }
        return hot_encoded;
    }


    Dataset load_mnist() {
        auto mnist_dataset = mnist::read_dataset<number_t, number_t>();
        Dataset dataset;
        dataset.training_images = mnist_dataset.training_images;
        dataset.test_images = mnist_dataset.test_images;
        dataset.hot_encoded_training_labels = Dataset::hot_encode(mnist_dataset.training_labels);
        dataset.hot_encoded_test_labels = Dataset::hot_encode(mnist_dataset.test_labels);
        return dataset;
    }

    Dataset load_xor(unsigned int train_size = 100, unsigned int test_size = 20) {
        Dataset dataset;

        matrix_t X;
        matrix_t Y;
        bool x1, x2;
        for (int i = 0; i < train_size; i++)
        {
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

        for (int i = 0; i < test_size; i++)
        {
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
}