#pragma once

#include "../alias.hpp"
#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& internal::neon::abs_(tensor<_Tp>& t) {
    if (!std::is_arithmetic_v<_Tp>)
        throw error::type_error("Type must be arithmetic");

    if (std::is_unsigned_v<_Tp>)
        return t;

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> vec     = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> abs_vec = neon_abs<_Tp>(vec);
        neon_store<_Tp>(&data_[i], abs_vec);
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<_Tp>(std::abs(data_[i]));

    return t;
}