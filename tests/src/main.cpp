#include <numeric>

#include <dataset.hpp>
#include <multi_layer_perceptron.hpp>

using namespace std;
using namespace tinyML;

// HYPERPARAMETERS
constexpr number_t LEARNING_CONSTANT = 0.01f;

// ARCHITECTURE
constexpr size_t layers_sizes[] = {784, 256, 10};
constexpr size_t how_many_layers = size(layers_sizes);
//constexpr size_t max_layer_size = max_const(begin(layers_sizes), end(layers_sizes));


int main() {
    {
        auto mnist_dataset = load_mnist();

        multi_layer_perceptron_t network = multi_layer_perceptron_t(LEARNING_CONSTANT, begin(layers_sizes), end(layers_sizes));
        auto X = mnist_dataset.training_images;
        for(int i=0; i<X.size(); i++)
        {
            for (int j = 0; j < X[i].size(); j++) X[i][j] /= 255.0;
        }
        auto Xt = mnist_dataset.test_images;
        auto y = mnist_dataset.hot_encoded_training_labels;
        auto yt = mnist_dataset.hot_encoded_training_labels;
        network.train(X, y);
    }

	return 0;
}
