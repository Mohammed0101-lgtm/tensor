#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sqrt_() {
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
            if (vals[j] < static_cast<value_type>(0))
            {
                throw std::domain_error("Cannot get the square root of a negative number");
            }

            vals[j] = static_cast<value_type>(std::sqrt(vals[j]));
        }

        neon_type<value_type> sqrt_vec = neon_load<value_type>(vals);
        neon_store(&data_[i], sqrt_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::sqrt(data_[i]));
    }

    return *this;
}