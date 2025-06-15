#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& internal::neon::tan_(tensor<_Tp>& t) {
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
            vals[j] = static_cast<_Tp>(std::tan(vals[j]));
        }

        neon_type<_Tp> tan_vec = neon_load<_Tp>(vals);
        neon_store<_Tp>(&data_[i], tan_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<_Tp>(std::tan(data_[i]));
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::tanh_(tensor<_Tp>& t) {
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
            vals[j] = static_cast<_Tp>(std::tanh(vals[j]));
        }

        neon_type<_Tp> tanh_vec = neon_load<_Tp>(vals);
        neon_store(&data_[i], tanh_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<_Tp>(std::tanh(static_cast<_f32>(data_[i])));
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::atan_(tensor<_Tp>& t) {
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
            if (vals[j] < static_cast<_Tp>(-1.0) || vals[j] > static_cast<_Tp>(1.0))
            {
                throw std::domain_error("Input value is out of domain for atan()");
            }

            vals[j] = static_cast<_Tp>(std::atan(vals[j]));
        }

        neon_type<_Tp> atan_vec = neon_load<_Tp>(vals);
        neon_store(&data_[i], atan_vec);
    }

    for (; i < data_.size(); ++i)
    {
        if (data_[i] < static_cast<_Tp>(-1.0) || data_[i] > static_cast<_Tp>(1.0))
        {
            throw std::domain_error("Input value is out of domain for atan()");
        }

        data_[i] = static_cast<_Tp>(std::atan(data_[i]));
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::atanh_(tensor<_Tp>& t) {
    if (!std::is_arithmetic_v<_Tp>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp>                      data_vec = neon_load<_Tp>(&data_[i]);
        alignas(sizeof(neon_type<_Tp>)) _Tp vals[t.simd_width];
        neon_store<_Tp>(vals, data_vec);

        for (int j = 0; j < t.simd_width; ++j)
        {
            if (vals[j] <= static_cast<_Tp>(-1.0) || vals[j] >= static_cast<_Tp>(1.0))
            {
                throw std::domain_error("Input value is out of domain for atanh()");
            }

            vals[j] = static_cast<_Tp>(std::atanh(vals[j]));
        }

        neon_type<_Tp> atanh_vec = neon_load<_Tp>(vals);
        neon_store<_Tp>(&data_[i], atanh_vec);
    }

    for (; i < data_.size(); ++i)
    {
        if (data_[i] <= static_cast<_Tp>(-1.0) || data_[i] >= static_cast<_Tp>(1.0))
        {
            throw std::domain_error("Input value is out of domain for atanh()");
        }

        data_[i] = static_cast<_Tp>(std::atanh(data_[i]));
    }

    return t;
}