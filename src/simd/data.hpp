#pragma once

#include "alias.hpp"
#include "tensorbase.hpp"

template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::neon_count_nonzero(index_type dimension) const {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    index_type       c           = 0;
    index_type       local_count = 0;
    index_type       i           = 0;
    const index_type simd_end    = data_.size() - (data_.size() % simd_width);

    if (dimension == 0)
    {
        for (; i < simd_end; i += simd_width)
        {
            neon_type<value_type> vec          = neon_load<value_type>(&data_[i]);
            neon_type<value_type> zero_vec     = neon_dup<value_type>(value_type(0.0f));
            neon_u32              nonzero_mask = neon_cgt<value_type>(vec, zero_vec);
            local_count += neon_addv<value_type>(neon_and<value_type>(
              reinterpret_cast<neon_type<value_type>>(nonzero_mask), neon_dup<value_type>(value_type(1.0f))));
        }

        for (index_type j = i; j < data_.size(); ++j)
            if (data_[j] != 0)
                ++local_count;

        c += local_count;
    }
    else
    {
        if (dimension < 0 || dimension >= static_cast<index_type>(shape_.size()))
            throw index_error("Invalid dimension provided.");
        throw std::runtime_error("Dimension-specific non-zero counting is not implemented yet.");
    }

    return c;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_zeros_(shape_type shape_) {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    if (shape_.empty())
    {
        shape_ = shape_;
    }
    else
    {
        shape_ = shape_;
    }

    std::size_t s = computeSize(shape_);
    data_.resize(s);
    compute_strides();

    const index_type      simd_end = s - (s % simd_width);
    neon_type<value_type> zero_vec = neon_dup<value_type>(value_type(0.0f));

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_store<value_type>(&data_[i], zero_vec);
    }

    for (; i < s; ++i)
    {
        data_[i] = value_type(0.0);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_ones_(shape_type shape_) {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    if (shape_.empty())
    {
        shape_ = shape_;
    }
    else
    {
        shape_ = shape_;
    }

    std::size_t s = computeSize(shape_);
    data_.resize(s);
    compute_strides();

    const index_type      simd_end = s - (s % simd_width);
    neon_type<value_type> one_vec  = neon_dup<value_type>(1.0f);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_store<value_type>(&data_[i], one_vec);
    }

    for (; i < s; ++i)
    {
        data_[i] = value_type(1.0);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_randomize_(const shape_type& shape_, bool bounded) {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    if (bounded && !std::is_floating_point_v<value_type>)
    {
        throw type_error("Cannot bound non floating point data type");
    }

    if (shape_.empty() && shape_.empty())
    {
        throw shape_error("randomize_ : Shape must be initialized");
    }

    if (shape_.empty() || shape_ != shape_)
    {
        shape_ = shape_;
    }

    index_type s = computeSize(shape_);
    data_.resize(s);
    compute_strides();

    std::random_device                   rd;
    std::mt19937                         gen(rd());
    std::uniform_real_distribution<_f32> unbounded_dist(1.0f, static_cast<_f32>(RAND_MAX));
    std::uniform_real_distribution<_f32> bounded_dist(0.0f, 1.0f);
    index_type                           i = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        const neon_f32 scale = vdupq_n_f32(bounded ? static_cast<_f32>(RAND_MAX) : 1.0f);
        for (; i + _ARM64_REG_WIDTH <= static_cast<index_type>(s); i += _ARM64_REG_WIDTH)
        {
            neon_f32 random_values;

            if (bounded)
            {
                random_values = {bounded_dist(gen), bounded_dist(gen), bounded_dist(gen), bounded_dist(gen)};
            }
            else
            {
                random_values = {unbounded_dist(gen), unbounded_dist(gen), unbounded_dist(gen), unbounded_dist(gen)};
            }

            if (!bounded)
            {
                random_values = vmulq_f32(random_values, vrecpeq_f32(scale));
            }

            vst1q_f32(&data_[i], random_values);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        const neon_f32 scale = vdupq_n_f32(static_cast<_f32>(RAND_MAX));
        for (; i + _ARM64_REG_WIDTH <= static_cast<index_type>(s); i += _ARM64_REG_WIDTH)
        {
            neon_f32 rand_vals = {static_cast<_f32>(unbounded_dist(gen)), static_cast<_f32>(unbounded_dist(gen)),
                                  static_cast<_f32>(unbounded_dist(gen)), static_cast<_f32>(unbounded_dist(gen))};
            rand_vals          = vmulq_f32(rand_vals, vrecpeq_f32(scale));
            neon_u32 int_vals  = vcvtq_u32_f32(rand_vals);

            vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), int_vals);
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        const neon_f32 scale = vdupq_n_f32(static_cast<_f32>(RAND_MAX));
        for (; i + _ARM64_REG_WIDTH <= static_cast<index_type>(s); i += _ARM64_REG_WIDTH)
        {
            neon_f32 rand_vals = {static_cast<_f32>(unbounded_dist(gen)), static_cast<_f32>(unbounded_dist(gen)),
                                  static_cast<_f32>(unbounded_dist(gen)), static_cast<_f32>(unbounded_dist(gen))};
            rand_vals          = vmulq_f32(rand_vals, vrecpeq_f32(scale));
            neon_s32 int_vals  = vcvtq_s32_f32(rand_vals);

            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), int_vals);
        }
    }

    for (; i < static_cast<index_type>(s); ++i)
    {
        data_[i] = value_type(bounded ? bounded_dist(gen) : unbounded_dist(gen));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_negative_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type       i        = 0;

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        neon_f32 neg_multiplier = vdupq_n_f32(-1);
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 v   = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 neg = vmulq_f32(v, neg_multiplier);
            vst1q_f32(reinterpret_cast<_f32*>(&data_[i]), neg);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        neon_s32 neg_multiplier = vdupq_n_s32(-1);
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 v   = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 neg = vmulq_s32(v, neg_multiplier);
            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), neg);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        neon_s32 neg_multiplier = vdupq_n_s32(-1);
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 v   = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 neg = vmulq_u32(v, neg_multiplier);
            vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), neg);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = -data_[i];
    }

    return *this;
}