#pragma once

#include <cstddef>
#include <span>
#include <iterator>
#include <type_traits>
#include <assert.h>

namespace tinyML::math
{
    enum class storage_order
    {
        row_major,
        column_major
    };

    namespace detail
    {
        template<
            typename DataT
        >
        class matrix_view_base
        {
        public:
            using element_type = DataT;
            using value_type = std::remove_cv_t<DataT>;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using pointer = DataT*;
            using const_pointer = const DataT*;
            using reference = DataT&;
            using const_reference = const DataT&;
            using iterator = pointer;
            using reverse_iterator = std::reverse_iterator<iterator>;
        protected:
            pointer _first = nullptr;
        public:
            constexpr matrix_view_base() noexcept = default;
            constexpr explicit matrix_view_base(pointer first) noexcept
                : _first(first)
            {}
            constexpr matrix_view_base(const matrix_view_base& other) noexcept = default;
            constexpr matrix_view_base(matrix_view_base&& other) noexcept = default;
        public:
            constexpr matrix_view_base& operator=(const matrix_view_base& other) noexcept = default;
            constexpr matrix_view_base& operator=(matrix_view_base&& other) noexcept = default;
        public:
            constexpr pointer data() const noexcept
            {
                return _first;
            }
        };
    }


    template<
        typename DataT,
        std::size_t ColumnsExtent = std::dynamic_extent,
        std::size_t RowsExtent = std::dynamic_extent,
        storage_order StorageOrder = storage_order::row_major
    >
    class matrix_view
    {

    };

    template<
        typename DataT,
        storage_order StorageOrder
    >
    class matrix_view<DataT, std::dynamic_extent, std::dynamic_extent, StorageOrder>
    {

    };
}