#pragma once
#include "types.h"
#include <algorithm>
#include <vector>


number_t dotproduct(const vector_t& v, const vector_t& u)
{
    if (v.size() != u.size()) throw std::runtime_error("Vector sizes do not match. Cannot calculate dot product.");

    number_t sum = 0;
    for (unsigned int i = 0; i < v.size(); i++)
    {
        sum += v[i] * u[i];
    }
    return sum;
}

matrix_t convert_vector_to_matrix(const vector_t& v)
{
    matrix_t result = matrix_t(1, v);
    return result;
}

matrix_t transpose(const matrix_t& A)
{
    matrix_t result = matrix_t(A[0].size(), vector_t(A.size(), 0));
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A[i].size(); j++)
        {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

matrix_t transpose(const vector_t& v)
{
    matrix_t A = convert_vector_to_matrix(v);
    return transpose(A);
}



matrix_t matrix_multiply(const matrix_t& A, const matrix_t& B)
{
    matrix_t result = matrix_t(A.size(), vector_t(B[0].size(), 0));
    auto B_transposed = transpose(B);
    if (A[0].size() != B.size()) throw std::runtime_error("Matrix sizes are not compatible");
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < B_transposed.size(); j++)
        {
            result[i][j] = dotproduct(A[i], B_transposed[j]);
        }
    }
    return result;
}

matrix_t matrix_multiply(const matrix_t& A, const vector_t& B)
{
    return matrix_multiply(A, convert_vector_to_matrix(B));
}

matrix_t matrix_multiply(const vector_t& A, const matrix_t& B)
{
    return matrix_multiply(convert_vector_to_matrix(A), B);
}

matrix_t matrix_multiply(const vector_t& A, const vector_t& B)
{
    return matrix_multiply(convert_vector_to_matrix(A), convert_vector_to_matrix(B));
}




vector_t vector_vector_add(const vector_t& v, const vector_t& u)
{
    if(v.size()!=u.size()) throw std::runtime_error("Vector sizes do not match. Cannot add vectors.");
    vector_t result; 
    for(unsigned int i=0; i<v.size(); i++)
    {
        result.push_back(v[i]+u[i]);
    }
    return result;
}

bool are_equal(const vector_t& a, const vector_t& b)
{
    if (a.size() != b.size()) return false;
    for (int i = 0; i < a.size(); i++)
    {
        if (a[i] != b[i]) return false;
    }
    return true;
}
