#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_frac_() {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type>  vec = neon_load<value_type>(&data_[i]);
        alignas(16) value_type vals[simd_width];
        neon_store<value_type>(vals, vec);

        for (int j = 0; j < simd_width; j++)
        {
            vals[j] = frac(vals[j]);
        }

        neon_type<value_type> atan_vec = neon_load<value_type>(vals);
        neon_store<value_type>(&data_[i], atan_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(frac(data_[i]));
    }

    return *this;
}