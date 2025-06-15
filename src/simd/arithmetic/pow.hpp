#pragma once

#include "tensorbase.hpp"


template<class _Tp>
tensor<_Tp>& internal::neon::pow_(tensor<_Tp>& t, const _Tp value) {
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
        neon_store<_Tp>(vals, data_vec);

        for (int j = 0; j < t.simd_width; ++j)
        {
            vals[j] = static_cast<_Tp>(std::pow(vals[j], value));
        }

        neon_type<_Tp> pow_vec = neon_load<_Tp>(vals);
        neon_store<_Tp>(&data_[i], pow_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<_Tp>(std::pow(data_[i], value));
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::pow_(tensor<_Tp>& t, const tensor<_Tp>& other) {
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
        neon_type<_Tp> base_vec   = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> exp_vec    = neon_load<_Tp>(&other[i]);
        neon_type<_Tp> result_vec = {
          static_cast<_Tp>(std::pow(neon_get_lane<_Tp>(base_vec, 0), neon_get_lane<_Tp>(exp_vec, 0))),
          static_cast<_Tp>(std::pow(neon_get_lane<_Tp>(base_vec, 1), neon_get_lane<_Tp>(exp_vec, 1))),
          static_cast<_Tp>(std::pow(neon_get_lane<_Tp>(base_vec, 2), neon_get_lane<_Tp>(exp_vec, 2))),
          static_cast<_Tp>(std::pow(neon_get_lane<_Tp>(base_vec, 3), neon_get_lane<_Tp>(exp_vec, 3))),
        };
        neon_store<_Tp>(&data_[i], result_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<_Tp>(std::pow((_f32) data_[i], (_f32) other[i]));
    }

    return t;
}