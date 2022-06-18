#pragma once

#include <math/matrix_view.hpp>

namespace tinyML::math
{
    enum class vector_type
    {
        column,
        row
    };

    template<
        typename DataT,
        std::size_t Extent,
        storage_order StorageOrder
    >
    class matrix_view<DataT, 1, Extent, StorageOrder>
        : public detail::matrix_view_base<DataT>
    {
        friend class matrix_view<DataT, Extent, 1, StorageOrder>;
    private:
        using base = detail::matrix_view_base<DataT>;
    public:
        using size_type = typename base::size_type;
        using pointer = typename base::pointer;
    protected:
        constexpr explicit matrix_view(pointer first) noexcept
            : base(first)
        {}
    public:
        constexpr matrix_view() noexcept requires(Extent == 0) = default;
        template<typename It>
        constexpr explicit matrix_view(It first, size_type count = Extent)
            : base(std::to_address(first))
        { assert(count == Extent); }
        template<typename It, class End>
        constexpr explicit matrix_view(It first, End last)
            : base(std::to_address(first))
        { assert(last - first == Extent); }
        constexpr matrix_view(const matrix_view& other) noexcept = default;
        constexpr matrix_view(matrix_view&& other) noexcept = default;
    public:
        constexpr matrix_view& operator=(const matrix_view& other) noexcept = default;
        constexpr matrix_view& operator=(matrix_view&& other) noexcept = default;
    public:
        
    public:
        constexpr size_type rows_count() const noexcept { return Extent; }
        constexpr size_type columns_count() const noexcept { return 1; }
        constexpr size_type size() const noexcept { return rows_count() * columns_count(); }
    public:
        constexpr matrix_view<DataT, Extent, 1, StorageOrder> transposed() noexcept;
    };

    template<
        typename DataT,
        std::size_t Extent,
        storage_order StorageOrder
    >
    class matrix_view<DataT, Extent, 1, StorageOrder>
        : public detail::matrix_view_base<DataT>
    {
        friend class matrix_view<DataT, 1, Extent, StorageOrder>;
    private:
        using base = detail::matrix_view_base<DataT>;
    public:
        using size_type = typename base::size_type;
        using pointer = typename base::pointer;
    protected:
        constexpr explicit matrix_view(pointer first) noexcept
            : base(first)
        {}
    public:
        constexpr matrix_view() noexcept requires(Extent == 0) = default;
        template<typename It>
        constexpr explicit matrix_view(It first, size_type count = Extent)
            : base(std::to_address(first))
        { assert(count == Extent); }
        template<typename It, class End>
        constexpr explicit matrix_view(It first, End last)
            : base(std::to_address(first))
        { assert(last - first == Extent); }
        constexpr matrix_view(const matrix_view& other) noexcept = default;
        constexpr matrix_view(matrix_view&& other) noexcept = default;
    public:
        constexpr matrix_view& operator=(const matrix_view& other) noexcept = default;
        constexpr matrix_view& operator=(matrix_view&& other) noexcept = default;
    public:
        constexpr size_type rows_count() const noexcept { return 1; }
        constexpr size_type columns_count() const noexcept { return Extent; }
        constexpr size_type size() const noexcept { return rows_count() * columns_count(); }
    public:
        constexpr matrix_view<DataT, 1, Extent, StorageOrder> transposed() noexcept;
    };

    template<typename DataT, std::size_t Extent, storage_order StorageOrder>
    constexpr matrix_view<DataT, Extent, 1, StorageOrder>
    matrix_view<DataT, 1, Extent, StorageOrder>::transposed() noexcept
    {
        return matrix_view<DataT, Extent, 1, StorageOrder>(base::data());
    }

    template<typename DataT, std::size_t Extent, storage_order StorageOrder>
    constexpr matrix_view<DataT, 1, Extent, StorageOrder>
    matrix_view<DataT, Extent, 1, StorageOrder>::transposed() noexcept
    {
        return matrix_view<DataT, 1, Extent, StorageOrder>(base::data());
    }


    template<
        typename DataT,
        storage_order StorageOrder
    >
    class matrix_view<DataT, 1, std::dynamic_extent, StorageOrder>
    {

    public:

    };

    template<
        typename DataT,
        storage_order StorageOrder
    >
    class matrix_view<DataT, std::dynamic_extent, 1, StorageOrder>
    { };



    template<
        typename DataT,
        std::size_t Extent = std::dynamic_extent,
        vector_type VectorType = vector_type::row
    >
    class vector_view {};

    template<
        typename DataT
    >
    class vector_view<DataT, std::dynamic_extent, vector_type::row>
        : public matrix_view<DataT, 1, std::dynamic_extent>
    {

    };

}