#include "common.hpp"
#include <algorithm>
#include <numeric>

namespace mlp
{
    size_t length(const vector_t& v)
    {
        return v.size();
    }
    size_t width(const matrix_t& m)
    {
        return !m.empty() ? m.front().size() : 0;
    }
    size_t height(const matrix_t& m)
    {
        return m.size();
    }

    number_t accumulate(const vector_t& v)
    {
        return std::accumulate(std::begin(v), std::end(v), number_t(0), std::plus<>{});
    }
    number_t dot_product(const vector_t& v0, const vector_t& v1)
    {
        return std::transform_reduce(std::begin(v0), std::end(v0), std::begin(v1), number_t(0), std::plus<>(), std::multiplies<>{});
    }

    void dot_add_o(const vector_t& a, const vector_t& b, vector_t& out)
    {
        std::transform(std::begin(a), std::end(a), std::begin(b), std::begin(out), std::plus<>{});
    }
    void dot_sub_o(const vector_t& a, const vector_t& b, vector_t& out)
    {
        std::transform(std::begin(a), std::end(a), std::begin(b), std::begin(out), std::minus<>{});
    }
    void dot_mul_o(const vector_t& a, const vector_t& b, vector_t& out)
    {
        std::transform(std::begin(a), std::end(a), std::begin(b), std::begin(out), std::multiplies<>{});
    }
    void dot_div_o(const vector_t& a, const vector_t& b, vector_t& out)
    {
        std::transform(std::begin(a), std::end(a), std::begin(b), std::begin(out), [](auto&& a, auto&& b){ return a/b; });
    }

    void splat_add_o(const vector_t& a, number_t b, vector_t& out)
    {
        std::transform(std::begin(a), std::end(a), std::begin(out), [&b](auto&& v){return v+b;});
    }
    void splat_sub_o(const vector_t& a, number_t b, vector_t& out)
    {
        std::transform(std::begin(a), std::end(a), std::begin(out), [&b](auto&& v){return v-b;});
    }
    void splat_mul_o(const vector_t& a, number_t b, vector_t& out)
    {
        std::transform(std::begin(a), std::end(a), std::begin(out), [&b](auto&& v){return v*b;});
    }
    void splat_div_o(const vector_t& a, number_t b, vector_t& out)
    {
        std::transform(std::begin(a), std::end(a), std::begin(out), [&b](auto&& v){return v/b;});
    }

    vector_t dot_add(const vector_t& a, const vector_t& b)
    {
        vector_t out(a.size());
        dot_add_o(a, b, out);
        return out;
    }
    vector_t dot_sub(const vector_t& a, const vector_t& b)
    {
        vector_t out(a.size());
        dot_sub_o(a, b, out);
        return out;
    }
    vector_t dot_mul(const vector_t& a, const vector_t& b)
    {
        vector_t out(a.size());
        dot_mul_o(a, b, out);
        return out;
    }
    vector_t dot_div(const vector_t& a, const vector_t& b)
    {
        vector_t out(a.size());
        dot_div_o(a, b, out);
        return out;
    }

    vector_t splat_add(const vector_t& a, number_t b)
    {
        vector_t out(a.size());
        splat_add_o(a, b, out);
        return out;
    }
    vector_t splat_sub(const vector_t& a, number_t b)
    {
        vector_t out(a.size());
        splat_sub_o(a, b, out);
        return out;
    }
    vector_t splat_mul(const vector_t& a, number_t b)
    {
        vector_t out(a.size());
        splat_mul_o(a, b, out);
        return out;
    }
    vector_t splat_div(const vector_t& a, number_t b)
    {
        vector_t out(a.size());
        splat_div_o(a, b, out);
        return out;
    }

    void mat_add_o(const matrix_t& l, const matrix_t& r, matrix_t& out)
    {
        for(size_t i=0;i<height(l);i++)
        {
            dot_add_o(l[i], r[i], out[i]);
        }
    }
    void mat_sub_o(const matrix_t& l, const matrix_t& r, matrix_t& out)
    {
        for(size_t i=0;i<height(l);i++)
        {
            dot_sub_o(l[i], r[i], out[i]);
        }
    }
    void mat_mul_o(const matrix_t& l, const matrix_t& r, matrix_t& out)
    {
        for(size_t i=0;i<height(l);i++)
        {
            for(size_t j=0;j<width(r);j++)
            {
                number_t sum = 0;
                for(size_t k=0;k<width(l);k++)
                {
                    sum += l[i][k] * r[k][j];
                }
                out[i][j] = sum;
            }
        }
    }

    matrix_t mat_add(const matrix_t& l, const matrix_t& r)
    {
        matrix_t out(height(l), vector_t(width(l)));
        mat_add_o(l, r, out);
        return out;
    }
    matrix_t mat_sub(const matrix_t& l, const matrix_t& r)
    {
        matrix_t out(height(l), vector_t(width(l)));
        mat_sub_o(l, r, out);
        return out;
    }
    matrix_t mat_mul(const matrix_t& l, const matrix_t& r)
    {
        matrix_t out(height(l), vector_t(width(r)));
        mat_mul_o(l, r, out);
        return out;
    }

    void mat_vec_mul_o(const matrix_t& l, const vector_t& r, vector_t& out)
    {
        for(size_t i=0;i<height(l);i++)
        {
            out[i] = dot_product(l[i], r);
        }
    }
    vector_t mat_vec_mul(const matrix_t& l, const vector_t& r)
    {
        vector_t out(r.size());
        mat_vec_mul_o(l, r, out);
        return out;
    }
}