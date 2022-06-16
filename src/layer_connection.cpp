#include <layer_connection.hpp>
#include <linalg.hpp>

namespace tinyML
{
    layer_connection_t::layer_connection_t(size_t neuron_num_prev, size_t neuron_num_current)
        : W(neuron_num_prev, vector_t(neuron_num_current, 0))
        , gradient_temp(neuron_num_prev, vector_t(neuron_num_current, 0))
        , W_transposed(neuron_num_current, vector_t(neuron_num_prev, 0))
        , B(neuron_num_current, 0)
        , out_temp(neuron_num_current, 0)
        , out_temp_internal(neuron_num_current, 0)
    {
        for(size_t i=0;i<height(W);i++)
        {
            for(size_t j=0;j<width(W);j++)
            {
                W[i][j] = get_random_number(0.0f, 1.0f);
            }
        }

        for(size_t i=0;i<neuron_num_current;i++)
        {
            B[i]= get_random_number(0.0f, 1.0f);
        }
    }

    void layer_connection_t::forward_pass(const vector_t& input, vector_t& out) noexcept
    {
        vec_mat_mul_o(input, W, out_temp_internal);
        dot_add_o(out_temp_internal, B, out);
    }
}
