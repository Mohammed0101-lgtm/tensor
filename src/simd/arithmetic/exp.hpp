#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& internal::neon::exp_(tensor<_Tp>& t) {
    if (!std::is_arithmetic_v<_Tp>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp>  vec = neon_load<_Tp>(&data_[i]);
        alignas(16) _Tp vals[t.simd_width];
        neon_store<_Tp>(vals, vec);

        for (int j = 0; j < t.simd_width; ++j)
        {
            vals[j] = static_cast<_Tp>(std::exp(vals[0]));
        }

        neon_type<_Tp> exp_vec = neon_load<_Tp>(vals);
        neon_store<_Tp>(&data_[i], exp_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<_Tp>(std::exp(data_[i]));
    }

    return t;
}