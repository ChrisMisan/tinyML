#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <variant>

namespace tinyML
{
    template<typename T>
    class counting_allocator
    {
    public:
        using value_type = typename std::allocator<T>::value_type ;
        using size_type = typename std::allocator<T>::size_type;
        using difference_type = typename std::allocator<T>::difference_type;
        using propagate_on_container_move_assignment = typename std::allocator<T>::propagate_on_container_move_assignment;
    private:
        std::allocator<T> _alloc = {};
    public:
        static size_t allocations;
        static size_t deallocations;
    public:
        [[nodiscard]] constexpr T* allocate(size_type n)
        {
            auto res = _alloc.allocate(n);
            allocations += n;
            return res;
        }
        constexpr void deallocate(T* p, size_type n)
        {
            _alloc.deallocate(p, n);
            deallocations += n;
        }
    };

    template<typename T>
    size_t counting_allocator<T>::allocations = 0;
    template<typename T>
    size_t counting_allocator<T>::deallocations = 0;

    using number_t = float;
    using allocator_t = std::allocator<number_t>;
    using vector_t = std::vector<number_t, allocator_t>;
    using matrix_t = std::vector<std::vector<number_t, allocator_t>>;

    struct activation_f_t;

    class layer_t;
    class layer_connection_t;
    class multi_layer_perceptron_t;

    class dataset_t;
}
