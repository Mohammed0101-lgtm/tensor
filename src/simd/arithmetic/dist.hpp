#pragma once

#include "../alias.hpp"
#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& internal::neon::dist_(tensor<_Tp>& t, const tensor<_Tp>& other) {
    if (!std::is_arithmetic_v<_Tp>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    if (!t.shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        _Tp dummy;

        neon_type<decltype(dummy)> a;
        neon_type<decltype(dummy)> b;
        neon_type<decltype(dummy)> diff;

        a = neon_load<_Tp>(&data_[i]);
        b = neon_load<_Tp>(&other[i]);
        diff = neon_vabdq<_Tp>(a, b);
        neon_store<_Tp>(&data_[i], diff);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<_Tp>(std::abs(static_cast<_f64>(data_[i] - other[i])));
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::dist_(tensor<_Tp>& t, const _Tp value) {
    if (!std::is_arithmetic_v<_Tp>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> a    = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> b    = neon_dup<_Tp>(value);
        neon_type<_Tp> diff = neon_vabdq<_Tp>(a, b);
        neon_store<_Tp>(&data_[i], diff);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<_Tp>(std::abs(static_cast<_f64>(data_[i] - value)));
    }

    return t;
}