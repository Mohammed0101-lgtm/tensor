#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> internal::neon::cross_product(tensor<_Tp>& t, const tensor<_Tp>& other) {
    if (!std::is_arithmetic_v<_Tp>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    if (t.empty() || other.empty())
    {
        throw std::invalid_argument("Cannot cross product an empty vector");
    }

    if (!t.shape().equal(shape::Shape({3})) || !other.shape().equal(shape::Shape({3})))
    {
        throw error::shape_error("Cross product can only be performed on 3-element vectors");
    }

    tensor<_Tp>       ret({3});
    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    neon_type<_Tp>    a        = neon_load<_Tp>(data_.data());
    neon_type<_Tp>    b        = neon_load<_Tp>(other.storage().data());
    neon_type<_Tp>    a_yzx    = neon_ext<_Tp>(a, a, 1);
    neon_type<_Tp>    b_yzx    = neon_ext<_Tp>(b, b, 1);
    neon_type<_Tp>    result   = neon_sub<_Tp>(neon_mul<_Tp>(a_yzx, b), neon_mul<_Tp>(a, b_yzx));
    result                     = neon_ext(result, result, 3);
    neon_store<_Tp>(ret.storage().data(), result);

    return ret;
}

template<class _Tp>
tensor<_Tp> internal::neon::dot(tensor<_Tp>& t, const tensor<_Tp>& other) {
    if (!std::is_arithmetic_v<_Tp>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    if (t.empty() || other.empty())
    {
        throw std::invalid_argument("Cannot dot product an empty vector");
    }

    if (t.shape().equal(shape::Shape({1})) && other.shape().equal(shape::Shape({1})))
    {
        if (t.shape()[0] != other.shape()[0])
        {
            throw error::shape_error("Vectors must have the same size for dot product");
        }
    }

    std::vector<_Tp>& data_      = t.storage_();
    const _Tp*        this_data  = data_.data();
    const _Tp*        other_data = other.storage().data();
    const std::size_t size       = data_.size();
    _Tp               ret        = 0;
    const _u64        simd_end   = data_.size() - (data_.size() % t.simd_width);

    if constexpr (std::is_floating_point_v<_Tp>)
    {
        std::size_t i       = 0;
        neon_f32    sum_vec = vdupq_n_f32(0.0f);

        for (; i + t.simd_width <= size; i += t.simd_width)
        {
            neon_f32 a_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this_data[i]));
            neon_f32 b_vec = vld1q_f32(reinterpret_cast<const _f32*>(&other_data[i]));
            sum_vec        = vmlaq_f32(sum_vec, a_vec, b_vec);  // Perform multiply-accumulate
        }

        float32x2_t sum_half = vadd_f32(vget_high_f32(sum_vec), vget_low_f32(sum_vec));
        ret                  = vget_lane_f32(vpadd_f32(sum_half, sum_half), 0);

        for (; i < size; ++i)
        {
            ret += static_cast<_Tp>(this_data[i]) * static_cast<_Tp>(other_data[i]);
        }
    }
    else if constexpr (std::is_unsigned_v<_Tp>)
    {
        std::size_t i       = 0;
        neon_u32    sum_vec = vdupq_n_u32(0.0f);

        for (; i + t.simd_width <= size; i += t.simd_width)
        {
            neon_u32 a_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this_data[i]));
            neon_u32 b_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other_data[i]));
            sum_vec        = vmlaq_u32(sum_vec, a_vec, b_vec);
        }

        uint32x2_t sum_half = vadd_u32(vget_high_u32(sum_vec), vget_low_u32(sum_vec));
        ret                 = vget_lane_u32(vpadd_u32(sum_half, sum_half), 0);

        for (; i < size; ++i)
        {
            ret += static_cast<_Tp>(this_data[i]) * static_cast<_Tp>(other_data[i]);
        }
    }
    else if constexpr (std::is_signed_v<_Tp>)
    {
        std::size_t i       = 0;
        neon_s32    sum_vec = vdupq_n_f32(0.0f);

        for (; i + t.simd_width <= size; i += t.simd_width)
        {
            neon_s32 a_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this_data[i]));
            neon_s32 b_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other_data[i]));
            sum_vec        = vmlaq_s32(sum_vec, a_vec, b_vec);  // Perform multiply-accumulate
        }

        int32x2_t sum_half = vadd_s32(vget_high_s32(sum_vec), vget_low_s32(sum_vec));
        ret                = vget_lane_s32(vpadd_s32(sum_half, sum_half), 0);

        for (; i < size; ++i)
        {
            ret += static_cast<_Tp>(this_data[i]) * static_cast<_Tp>(other_data[i]);
        }
    }

    return self({ret}, {1});
}