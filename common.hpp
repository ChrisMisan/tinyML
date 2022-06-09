#pragma once

#include <vector>
#include <algorithm>
#include <random>

template<typename It>
constexpr auto max_const(It b, It e)
{
    return *std::max_element(b, e);
}

template <typename T>
T get_random_number(T min, T max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}



namespace mlp
{
    using number_t = float;
    using vector_t = std::vector<number_t>;
    using matrix_t = std::vector<std::vector<number_t>>;

    struct activation_f_t;
    struct layer_t;
    struct layer_connection_t;
    struct network_t;

    size_t length(const vector_t& v);
    size_t width(const matrix_t& m);
    size_t height(const matrix_t& m);

    number_t accumulate(const vector_t& v);
    number_t dot_product(const vector_t& v0, const vector_t& v1);

    void dot_mul_o(const vector_t& a, const vector_t& b, vector_t& out);
    void dot_div_o(const vector_t& a, const vector_t& b, vector_t& out);
    void dot_add_o(const vector_t& a, const vector_t& b, vector_t& out);
    void dot_sub_o(const vector_t& a, const vector_t& b, vector_t& out);

    void splat_add_o(const vector_t& a, number_t b, vector_t& out);
    void splat_sub_o(const vector_t& a, number_t b, vector_t& out);
    void splat_mul_o(const vector_t& a, number_t b, vector_t& out);
    void splat_div_o(const vector_t& a, number_t b, vector_t& out);

    vector_t dot_mul(const vector_t& a, const vector_t& b);
    vector_t dot_div(const vector_t& a, const vector_t& b);
    vector_t dot_add(const vector_t& a, const vector_t& b);
    vector_t dot_sub(const vector_t& a, const vector_t& b);

    vector_t splat_add(const vector_t& a, number_t b);
    vector_t splat_sub(const vector_t& a, number_t b);
    vector_t splat_mul(const vector_t& a, number_t b);
    vector_t splat_div(const vector_t& a, number_t b);

    void mat_add_o(const matrix_t& l, const matrix_t& r, matrix_t& out);
    void mat_sub_o(const matrix_t& l, const matrix_t& r, matrix_t& out);
    void mat_mul_o(const matrix_t& l, const matrix_t& r, matrix_t& out);

    matrix_t mat_mul(const matrix_t& l, const matrix_t& r);
    matrix_t mat_add(const matrix_t& l, const matrix_t& r);
    matrix_t mat_sub(const matrix_t& l, const matrix_t& r);

    void mat_vec_mul_o(const matrix_t& l, const vector_t& r, vector_t& out);
    vector_t mat_vec_mul(const matrix_t& l, const vector_t& r);

    void print_vec(const vector_t& vec);
    void print_mat(const matrix_t& mat);

}
