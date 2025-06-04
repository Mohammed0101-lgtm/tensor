#pragma once

#include "../alias.hpp"
#include "tensorbase.hpp"


template<class _Tp>
tensor<_Tp>& internal::neon::cos_(tensor<_Tp>& t) {
    if (!std::is_arithmetic_v<_Tp>)
        throw error::type_error("Type must be arithmetic");

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp>  vec = neon_load<_Tp>(&data_[i]);
        alignas(16) _Tp vals[t.simd_width];
        neon_store<_Tp>(&vals, vec);

        for (std::size_t j = 0; j < t.simd_width; j++)
            vals[j] = static_cast<_Tp>(std::cos(vals[j]));

        neon_type<_Tp> cos_vec = neon_load(&vals);
        neon_store<_Tp>(&data_[i], cos_vec);
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<_Tp>(std::cos(data_[i]));

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::acos_(tensor<_Tp>& t) {
    if (!std::is_arithmetic_v<_Tp>)
        throw error::type_error("Type must be arithmetic");

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp>  vec = neon_load<_Tp>(&data_[i]);
        alignas(16) _Tp vals[t.simd_width];
        neon_store<_Tp>(&vals, vec);

        for (std::size_t j = 0; j < t.simd_width; j++)
            vals[j] = static_cast<_Tp>(std::acos(vals[j]));

        neon_type<_Tp> acos_vec = neon_load(&vals);
        neon_store<_Tp>(&data_[i], acos_vec);
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<_Tp>(std::acos(data_[i]));

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::cosh_(tensor<_Tp>& t) {
    if (!std::is_arithmetic_v<_Tp>)
        throw error::type_error("Type must be arithmetic");

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp>  vec = neon_load<_Tp>(&data_[i]);
        alignas(16) _Tp vals[t.simd_width];
        neon_store<_Tp>(&vals, vec);

        for (std::size_t j = 0; j < t.simd_width; j++)
            vals[j] = static_cast<_Tp>(std::cosh(vals[j]));

        neon_type<_Tp> cosh_vec = neon_load(&vals);
        neon_store<_Tp>(&data_[i], cosh_vec);
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<_Tp>(std::cosh(data_[i]));

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::acosh_(tensor<_Tp>& t) {
    if (!std::is_arithmetic_v<_Tp>)
        throw error::type_error("Type must be arithmetic");

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp>  vec = neon_load<_Tp>(&data_[i]);
        alignas(16) _Tp vals[t.simd_width];
        neon_store<_Tp>(&vals, vec);

        for (std::size_t j = 0; j < t.simd_width; j++)
            vals[j] = static_cast<_Tp>(std::acosh(vals[j]));

        neon_type<_Tp> acosh_vec = neon_load(&vals);
        neon_store<_Tp>(&data_[i], acosh_vec);
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<_Tp>(std::acosh(data_[i]));

    return t;
}