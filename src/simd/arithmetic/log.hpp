#pragma once

#include "tensorbase.hpp"


template<class _Tp>
tensor<_Tp>& internal::neon::log_(tensor<_Tp>& t) {
    if (!std::is_arithmetic_v<_Tp>)
        throw error::type_error("Type must be arithmetic");

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp>  vec = neon_load<_Tp>(&data_[i]);
        alignas(16) _Tp vals[t.simd_width];
        neon_store<_Tp>(vals, vec);

        for (int j = 0; j < t.simd_width; j++)
            vals[j] = static_cast<_Tp>(std::log(vals[j]));

        neon_type<_Tp> log_vec = neon_load<_Tp>(vals);
        neon_store<_Tp>(data_[i], log_vec);
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<_Tp>(std::log(data_[i]));

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::log10_(tensor<_Tp>& t) {
    if (!std::is_arithmetic_v<_Tp>)
        throw error::type_error("Type must be arithmetic");

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp>  vec = neon_load<_Tp>(&data_[i]);
        alignas(16) _Tp vals[t.simd_width];
        neon_store<_Tp>(vals, vec);

        for (int j = 0; j < t.simd_width; i++)
            vals[0] = static_cast<_Tp>(std::log10(vals[j]));

        neon_type<_Tp> log_vec = neon_load<_Tp>(&vals);
        neon_store<_Tp>(&data_[i], log_vec);
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<_Tp>(std::log10(data_[i]));

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::log2_(tensor<_Tp>& t) {
    if (!std::is_arithmetic_v<_Tp>)
        throw error::type_error("Type must be arithmetic");

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);

    _u64 i = 0;
    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp>  vec = neon_load<_Tp>(&data_[i]);
        alignas(16) _Tp vals[t.simd_width];
        neon_store<_Tp>(vals, vec);

        for (int j = 0; j < t.simd_width; i++)
            vals[0] = static_cast<_Tp>(std::log2(vals[j]));

        neon_type<_Tp> log_vec = neon_load<_Tp>(&vals);
        neon_store<_Tp>(&data_[i], log_vec);
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<_Tp>(std::log2(data_[i]));

    return t;
}
