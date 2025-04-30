#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_cross_product(const tensor& other) const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    if (empty() || other.empty())
    {
        throw std::invalid_argument("Cannot cross product an empty vector");
    }

    if (!equal_shape(shape(), shape_type({3})) || !equal_shape(other.shape(), shape_type({3})))
    {
        throw shape_error("Cross product can only be performed on 3-element vectors");
    }

    tensor ret({3});

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    neon_type<value_type> a      = neon_load<value_type>(data_.data());
    neon_type<value_type> b      = neon_load<value_type>(other.storage().data());
    neon_type<value_type> a_yzx  = neon_ext<value_type>(a, a, 1);
    neon_type<value_type> b_yzx  = neon_ext<value_type>(b, b, 1);
    neon_type<value_type> result = neon_sub<value_type>(neon_mul<value_type>(a_yzx, b), neon_mul<value_type>(a, b_yzx));
    result                       = neon_ext(result, result, 3);

    neon_store<value_type>(ret.storage().data(), result);

    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_dot(const tensor& other) const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    if (empty() || other.empty())
    {
        throw std::invalid_argument("Cannot dot product an empty vector");
    }

    if (equal_shape(shape_, shape_type({1})) && equal_shape(other.shape(), shape_type({1})))
    {
        if (shape_[0] != other.shape()[0])
        {
            throw shape_error("Vectors must have the same size for dot product");
        }
    }

    const_pointer     this_data  = data_.data();
    const_pointer     other_data = other.storage().data();
    const std::size_t size       = data_.size();
    value_type        ret        = 0;

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    if constexpr (std::is_floating_point_v<value_type>)
    {
        std::size_t i       = 0;
        neon_f32    sum_vec = vdupq_n_f32(0.0f);

        for (; i + simd_width <= size; i += simd_width)
        {
            neon_f32 a_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this_data[i]));
            neon_f32 b_vec = vld1q_f32(reinterpret_cast<const _f32*>(&other_data[i]));
            sum_vec        = vmlaq_f32(sum_vec, a_vec, b_vec);  // Perform multiply-accumulate
        }

        float32x2_t sum_half = vadd_f32(vget_high_f32(sum_vec), vget_low_f32(sum_vec));
        ret                  = vget_lane_f32(vpadd_f32(sum_half, sum_half), 0);

        for (; i < size; ++i)
        {
            ret += static_cast<value_type>(this_data[i]) * static_cast<value_type>(other_data[i]);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        std::size_t i       = 0;
        neon_u32    sum_vec = vdupq_n_u32(0.0f);

        for (; i + simd_width <= size; i += simd_width)
        {
            neon_u32 a_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this_data[i]));
            neon_u32 b_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other_data[i]));
            sum_vec        = vmlaq_u32(sum_vec, a_vec, b_vec);
        }

        uint32x2_t sum_half = vadd_u32(vget_high_u32(sum_vec), vget_low_u32(sum_vec));
        ret                 = vget_lane_u32(vpadd_u32(sum_half, sum_half), 0);

        for (; i < size; ++i)
        {
            ret += static_cast<value_type>(this_data[i]) * static_cast<value_type>(other_data[i]);
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        std::size_t i       = 0;
        neon_s32    sum_vec = vdupq_n_f32(0.0f);

        for (; i + simd_width <= size; i += simd_width)
        {
            neon_s32 a_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this_data[i]));
            neon_s32 b_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other_data[i]));
            sum_vec        = vmlaq_s32(sum_vec, a_vec, b_vec);  // Perform multiply-accumulate
        }

        int32x2_t sum_half = vadd_s32(vget_high_s32(sum_vec), vget_low_s32(sum_vec));
        ret                = vget_lane_s32(vpadd_s32(sum_half, sum_half), 0);

        for (; i < size; ++i)
        {
            ret += static_cast<value_type>(this_data[i]) * static_cast<value_type>(other_data[i]);
        }
    }

    return self({ret}, {1});
}