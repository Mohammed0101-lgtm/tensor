#pragma once

#include "../alias.hpp"
#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_dist_(const tensor& other) {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        value_type dummy;

        neon_type<decltype(dummy)> a;
        neon_type<decltype(dummy)> b;
        neon_type<decltype(dummy)> diff;

        neon_load<value_type>(&data_[i], &a);
        neon_load<value_type>(&other[i], &b);
        neon_vabdq<value_type>(&a, &b, &diff);
        neon_store<value_type>(&data_[i], &diff);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::abs(static_cast<_f64>(data_[i] - other.data_[i])));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_dist_(const value_type val) {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> a    = neon_load<value_type>(&data_[i]);
        neon_type<value_type> b    = neon_dup<value_type>(&val);
        neon_type<value_type> diff = neon_vabdq<value_type>(a, b);
        neon_store<value_type>(&data_[i], diff);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::abs(static_cast<_f64>(data_[i] - val)));
    }

    return *this;
}