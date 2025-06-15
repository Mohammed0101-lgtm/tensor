#pragma once

#include "tensorbase.hpp"


template<class _Tp>
tensor<_Tp>& internal::neon::sqrt_(tensor<_Tp>& t) {
    if (!std::is_arithmetic_v<_Tp>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp>  data_vec = neon_load<_Tp>(&data_[i]);
        alignas(16) _Tp vals[t.simd_width];
        neon_load<_Tp>(vals, data_vec);

        for (int j = 0; j < t.simd_width; ++j)
        {
            if (vals[j] < static_cast<_Tp>(0.0))
            {
                throw std::domain_error("Cannot get the square root of a negative number");
            }

            vals[j] = static_cast<_Tp>(std::sqrt(vals[j]));
        }

        neon_type<_Tp> sqrt_vec = neon_load<_Tp>(vals);
        neon_store<_Tp>(&data_[i], sqrt_vec);
    }

    for (; i < data_.size(); ++i)
    {
        if (data_[i] < static_cast<_Tp>(0.0))
        {
            throw std::domain_error("Cannot get the square root of a negative number");
        }

        data_[i] = static_cast<_Tp>(std::sqrt(data_[i]));
    }

    return t;
}