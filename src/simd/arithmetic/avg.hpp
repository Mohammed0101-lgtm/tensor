#pragma once

#include "tensorbase.hpp"

template<class _Tp>
double tensor<_Tp>::neon_mean() const {
    double m = 0.0;

    if (empty())
    {
        return m;
    }

    const index_type      simd_end = data_.size() - (data_.size() % simd_width);
    neon_type<value_type> sum_vec  = neon_dup<value_type>(value_type(0.0f));

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> data_vec = neon_load<value_type>(&data_[i]);
        sum_vec                        = neon_add<value_type>(sum_vec, data_vec);
    }

    alignas(16) value_type partial_sum[simd_width];
    neon_store<value_type>(partial_sum, sum_vec);

    for (std::size_t j = 0; j < simd_width; ++j)
    {
        m += partial_sum[j];
    }

    for (; i < data_.size(); ++i)
    {
        m += data_[i];
    }

    return m / static_cast<double>(data_.size());
}
