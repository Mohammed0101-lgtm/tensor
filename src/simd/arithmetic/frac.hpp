#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& internal::neon::frac_(tensor<_Tp>& t) {
    if (!std::is_floating_point_v<_Tp>)
        throw error::type_error("Type must be floating point");

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp>  vec = neon_load<_Tp>(&data_[i]);
        alignas(16) _Tp vals[t.simd_width];
        neon_store<_Tp>(vals, vec);

        for (int j = 0; j < t.simd_width; j++)
            vals[j] = frac(vals[j]);

        neon_type<_Tp> atan_vec = neon_load<_Tp>(vals);
        neon_store<_Tp>(&data_[i], atan_vec);
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<_Tp>(frac(data_[i]));

    return t;
}