#pragma once
#include "types.h"
#include <algorithm>
#include <vector>

std::vector<number_t> matrix_vector_multiply(const matrix_t& A, const vector_t& v) 
{
    std::vector<number_t> result;
    std::transform(A.begin(), A.end(), std::back_inserter(result), [&](auto& row) {return vector_vector_dotproduct(row, v);});

    return std::vector<number_t>(result.begin(), result.end());
    
}

number_t vector_vector_dotproduct(const vector_t& v, const vector_t& u)
{
    if(v.size()!=u.size()) throw std::runtime_error("Vector sizes do not match. Cannot calculate dot product.");

    number_t sum = 0;
    for(unsigned int i=0; i<v.size(); i++)
    {
        sum+=v[i]*u[i];
    }
    return sum;
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


