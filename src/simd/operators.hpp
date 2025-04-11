#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_operator_plus(const tensor& other) const {
    if constexpr (!has_plus_operator_v<value_type>)
    {
        throw operator_error("Value type must have a plus operator");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Cannot add two tensors with different shapes");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    index_type simd_end = data_.size() - (data_.size() % simd_width);
    data_t     d(data_.size());

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec1   = neon_load<value_type>(&data_[i]);
        neon_type<value_type> vec2   = neon_load<value_type>(&other[i]);
        neon_type<value_type> result = neon_add<value_type>(vec1, vec2);
        neon_store<value_type>(&d[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        d[i] = data_[i] + other[i];
    }

    return self(shape_, d);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_operator_plus(const value_type val) const {
    if constexpr (!has_plus_operator_v<value_type>)
    {
        throw operator_error("Value type must have a plus operator");
    }

    data_t                d(data_.size());
    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type            i       = 0;
    neon_type<value_type> val_vec = neon_dup<value_type>(&val);
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec1   = neon_load<value_type>(&data_[i]);
        neon_type<value_type> result = neon_add<value_type>(vec1, val_vec);
        neon_store<value_type>(&d[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        d[i] = data_[i] + val;
    }

    return self(d, shape_);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_operator_plus_eq(const_reference val) const {
    if constexpr (!has_equal_operator_v<value_type>)
    {
        throw operator_error("Value type must have a plus operator");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type            i       = 0;
    neon_type<value_type> val_vec = neon_dup<value_type>(&val);
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec1   = neon_load<value_type>(&data_[i]);
        neon_type<value_type> result = neon_add<value_type>(vec1, val_vec);
        neon_store<value_type>(&data_[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = data_[i] + val;
    }

    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_operator_minus(const tensor& other) const {
    if constexpr (!has_minus_operator_v<value_type>)
    {
        throw operator_error("Value type must have a minus operator");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Cannot add two tensors with different shapes");
    }

    data_t                d(data_.size());
    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec1   = neon_load<value_type>(&data_[i]);
        neon_type<value_type> vec2   = neon_load<value_type>(&other[i]);
        neon_type<value_type> result = neon_sub<value_type>(vec1, vec2);
        neon_store<value_type>(&d[i], result);
    }

    for (; i < data_[i]; ++i)
    {
        d[i] = data_[i] - other[i];
    }

    return self(shape_, d);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_operator_minus(const value_type val) const {
    if constexpr (!has_minus_operator_v<value_type>)
    {
        throw operator_error("Value type must have a minus operator");
    }

    data_t d(data_.size());

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type            i       = 0;
    neon_type<value_type> val_vec = neon_dup<value_type>(&val);
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec1   = neon_load<value_type>(&data_[i]);
        neon_type<value_type> result = neon_sub<value_type>(vec1, val_vec);
        neon_store<value_type>(&d[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        d[i] = data_[i] - val;
    }

    return self(*this);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_operator_minus_eq(const tensor& other) const {
    if constexpr (!has_minus_operator_v<value_type>)
    {
        throw operator_error("Value type must have a minus operator");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec1   = neon_load<value_type>(&data_[i]);
        neon_type<value_type> vec2   = neon_load<value_type>(&other[i]);
        neon_type<value_type> result = neon_sub<value_type>(vec1, vec2);
        neon_store<value_type>(&data_[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] -= other[i];
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_operator_times_eq(const tensor& other) const {
    if constexpr (!has_minus_operator_v<value_type>)
    {
        throw operator_error("Value type must have a minus operator");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec1   = neon_load<value_type>(&data_[i]);
        neon_type<value_type> vec2   = neon_load<value_type>(&other[i]);
        neon_type<value_type> result = neon_mul<value_type>(vec1, vec2);
        neon_store<value_type>(&data_[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] *= other[i];
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_operator_minus_eq(const_reference val) const {
    if constexpr (!has_minus_operator_v<value_type>)
    {
        throw operator_error("Value type must have a minus operator");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type            i       = 0;
    neon_type<value_type> val_vec = neon_dup<value_type>(&val);
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec1   = neon_load<value_type>(&data_[i]);
        neon_type<value_type> result = neon_mul<value_type>(vec1, val_vec);
        neon_store<value_type>(&data_[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] -= val;
    }

    return *this;
}
