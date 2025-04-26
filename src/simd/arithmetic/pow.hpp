#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_pow_(const value_type val) {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type>  data_vec = neon_load<value_type>(&data_[i]);
        alignas(16) value_type vals[simd_width];
        neon_load<value_type>(vals, data_vec);

        for (int j = 0; j < simd_width; ++j)
        {
            vals[j] = static_cast<value_type>(std::pow(vals[j], val));
        }

        neon_type<value_type> pow_vec = neon_load<value_type>(vals);
        neon_store<value_type>(&data_[i], pow_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::pow(data_[i], val));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_pow_(const tensor& other) {
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
        neon_type<value_type> base_vec   = neon_load<value_type>(&data_[i]);
        neon_type<value_type> exp_vec    = neon_load<value_type>(&other[i]);
        neon_type<value_type> result_vec = {
          static_cast<value_type>(
            std::pow(neon_get_lane<value_type>(base_vec, 0), neon_get_lane<value_type>(exp_vec, 0))),
          static_cast<value_type>(
            std::pow(neon_get_lane<value_type>(base_vec, 1), neon_get_lane<value_type>(exp_vec, 1))),
          static_cast<value_type>(
            std::pow(neon_get_lane<value_type>(base_vec, 2), neon_get_lane<value_type>(exp_vec, 2))),
          static_cast<value_type>(
            std::pow(neon_get_lane<value_type>(base_vec, 3), neon_get_lane<value_type>(exp_vec, 3))),
        };
        neon_store<value_type>(&data_[i], result_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::pow(static_cast<_f32>(data_[i]), static_cast<_f32>(other[i])));
    }

    return *this;
}