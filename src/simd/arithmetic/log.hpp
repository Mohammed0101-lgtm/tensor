#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_log_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
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
            vals[j] = static_cast<value_type>(std::log(vals[j]));
        }

        neon_type<value_type> log_vec = neon_load<value_type>(vals);
        neon_store<value_type>(data_[i], log_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::log(data_[i]));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_log10_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type>  vec = neon_load<value_type>(&data_[i]);
        alignas(16) value_type vals[simd_width];
        neon_store<value_type>(vals, vec);

        for (int j = 0; j < simd_width; i++)
        {
            vals[0] = static_cast<value_type>(std::log10(vals[j]));
        }

        neon_type<value_type> log_vec = neon_load<value_type>(&vals);
        neon_store<value_type>(&data_[i], log_vec);
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<value_type>(std::log10(data_[i]));

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_log2_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type>  vec = neon_load<value_type>(&data_[i]);
        alignas(16) value_type vals[simd_width];
        neon_store<value_type>(vals, vec);

        for (int j = 0; j < simd_width; i++)
        {
            vals[0] = static_cast<value_type>(std::log2(vals[j]));
        }

        neon_type<value_type> log_vec = neon_load<value_type>(&vals);
        neon_store<value_type>(&data_[i], log_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::log2(data_[i]));
    }

    return *this;
}
