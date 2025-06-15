#pragma once

#include "tensorbase.hpp"


template<class _Tp>
double internal::neon::mean(const tensor<_Tp>& t) {
    double m = 0.0;

    if (t.empty())
    {
        return m;
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    neon_type<_Tp>    sum_vec  = neon_dup<_Tp>(_Tp(0.0f));
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> data_vec = neon_load<_Tp>(&data_[i]);
        sum_vec                 = neon_add<_Tp>(sum_vec, data_vec);
    }

    alignas(16) _Tp partial_sum[t.simd_width];
    neon_store<_Tp>(partial_sum, sum_vec);

    for (std::size_t j = 0; j < t.simd_width; ++j)
    {
        m += partial_sum[j];
    }

    for (; i < data_.size(); ++i)
    {
        m += data_[i];
    }

    return m / static_cast<double>(data_.size());
}