#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<bool> tensor<_Tp>::neon_equal(const tensor& other) const {
    if constexpr (!has_equal_operator_v<value_type>)
    {
        throw operator_error("Value type must have equal to operator");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    std::vector<bool> ret(data_.size());
    const index_type  simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type        i        = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec1  = vld1q_f32(reinterpret_cast<const float*>(&data_[i]));
            neon_f32 data_vec2  = vld1q_f32(reinterpret_cast<const float*>(&other.data_[i]));
            neon_u32 cmp_result = vceqq_f32(data_vec1, data_vec2);
            neon_u8  mask       = vreinterpretq_u8_u32(cmp_result);

            ret[i]     = vgetq_lane_u8(mask, 0);
            ret[i + 1] = vgetq_lane_u8(mask, 4);
            ret[i + 2] = vgetq_lane_u8(mask, 8);
            ret[i + 3] = vgetq_lane_u8(mask, 12);
        }
    }
    else
    {  // Handles both signed and unsigned integers
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec1  = vld1q_u32(reinterpret_cast<const uint32_t*>(&data_[i]));
            neon_u32 data_vec2  = vld1q_u32(reinterpret_cast<const uint32_t*>(&other.data_[i]));
            neon_u32 cmp_result = vceqq_u32(data_vec1, data_vec2);
            neon_u8  mask       = vreinterpretq_u8_u32(cmp_result);

            ret[i]     = vgetq_lane_u8(mask, 0);
            ret[i + 1] = vgetq_lane_u8(mask, 4);
            ret[i + 2] = vgetq_lane_u8(mask, 8);
            ret[i + 3] = vgetq_lane_u8(mask, 12);
        }
    }

    // Handle remaining elements
    for (; i < data_.size(); ++i)
    {
        ret[i] = (data_[i] == other.data_[i]);
    }

    return tensor<bool>(shape_, ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::neon_equal(const value_type val) const {
    if constexpr (!has_equal_operator_v<value_type>)
    {
        throw operator_error("Value type must have equal to operator");
    }

    std::vector<bool> ret(data_.size());
    index_type        i        = 0;
    const index_type  simd_end = data_.size() - (data_.size() % 4);

    if constexpr (std::is_floating_point_v<value_type>)
    {
        float32x4_t val_vec = vdupq_n_f32(val);

        for (; i < simd_end; i += 4)
        {
            float32x4_t data_vec   = vld1q_f32(reinterpret_cast<const float*>(&data_[i]));
            uint32x4_t  cmp_result = vceqq_f32(data_vec, val_vec);

            uint32_t results[4];
            vst1q_u32(results, cmp_result);  // Store results into an array
            for (int j = 0; j < 4; ++j)
                ret[i + j] = results[j] != 0;
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        int32x4_t val_vec = vdupq_n_s32(val);

        for (; i < simd_end; i += 4)
        {
            int32x4_t  data_vec   = vld1q_s32(reinterpret_cast<const int32_t*>(&data_[i]));
            uint32x4_t cmp_result = vceqq_s32(data_vec, val_vec);

            uint32_t results[4];
            vst1q_u32(results, cmp_result);
            for (int j = 0; j < 4; ++j)
                ret[i + j] = results[j] != 0;
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        uint32x4_t val_vec = vdupq_n_u32(val);

        for (; i < simd_end; i += 4)
        {
            uint32x4_t data_vec   = vld1q_u32(reinterpret_cast<const uint32_t*>(&data_[i]));
            uint32x4_t cmp_result = vceqq_u32(data_vec, val_vec);

            uint32_t results[4];
            vst1q_u32(results, cmp_result);
            for (int j = 0; j < 4; ++j)
                ret[i + j] = results[j] != 0;
        }
    }

    // Handle the remaining elements that don't fit in a SIMD register
    for (; i < data_.size(); ++i)
    {
        ret[i] = (data_[i] == val);
    }

    return tensor<bool>(shape_, ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::neon_less_equal(const tensor& other) const {
    if constexpr (!has_less_operator_v<value_type>)
    {
        throw operator_error("Value type must have less than operator");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    std::vector<_u32> ret(data_.size());
    std::size_t       vs = data_.size() / _ARM64_REG_WIDTH * _ARM64_REG_WIDTH;
    index_type        i  = 0;

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        for (; i < vs; i += _ARM64_REG_WIDTH)
        {
            neon_f32 va       = vld1q_f32(data_.data() + i);
            neon_f32 vb       = vld1q_f32(other.data_.data() + i);
            neon_u32 leq_mask = vcleq_f32(va, vb);
            vst1q_u32(&ret[i], leq_mask);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        for (; i < vs; i += _ARM64_REG_WIDTH)
        {
            neon_s32 va       = vld1q_s32(data_.data() + i);
            neon_s32 vb       = vld1q_s32(other.data_.data() + i);
            neon_u32 leq_mask = vcleq_s32(va, vb);
            vst1q_u32(&ret[i], leq_mask);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        for (; i < vs; i += _ARM64_REG_WIDTH)
        {
            neon_u32 va       = vld1q_u32(data_.data() + i);
            neon_u32 vb       = vld1q_u32(other.data_.data() + i);
            neon_u32 leq_mask = vcleq_u32(va, vb);
            vst1q_u32(&ret[i], leq_mask);
        }
    }

    // Convert `ret` (integer masks) to boolean
    std::vector<bool> d(data_.size());

    for (std::size_t j = 0; j < i; ++j)
    {
        d[j] = ret[j] != 0;
    }

    for (; i < d.size(); ++i)
    {
        d[i] = (data_[i] <= other[i]);
    }

    return tensor<bool>(shape_, d);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::neon_less_equal(const value_type val) const {
    if constexpr (!has_less_equal_operator_v<value_type>)
    {
        throw operator_error("Value type must have less than or equal to operator");
    }

    std::vector<_u32> ret(data_.size());
    index_type        i = 0;

    std::size_t vector_size = data_.size() / _ARM64_REG_WIDTH * _ARM64_REG_WIDTH;

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        for (; i < vector_size; i += _ARM64_REG_WIDTH)
        {
            neon_f32 vec_a    = vld1q_f32(data_.data() + i);
            neon_f32 vec_b    = vdupq_n_f32(val);
            neon_u32 leq_mask = vcleq_f32(vec_a, vec_b);

            vst1q_u32(&ret[i], leq_mask);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        for (; i < vector_size; i += _ARM64_REG_WIDTH)
        {
            neon_s32 vec_a    = vld1q_s32(data_.data() + i);
            neon_s32 vec_b    = vdupq_n_s32(val);
            neon_u32 leq_mask = vcleq_s32(vec_a, vec_b);

            vst1q_u32(&ret[i], leq_mask);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        for (; i < vector_size; i += _ARM64_REG_WIDTH)
        {
            neon_u32 vec_a    = vld1q_u32(data_.data() + i);
            neon_u32 vec_b    = vdupq_n_u32(val);
            neon_u32 leq_mask = vcleq_u32(vec_a, vec_b);

            vst1q_u32(&ret[i], leq_mask);
        }
    }

    for (; i < data_.size(); ++i)
    {
        ret[i] = (data_[i] <= val) ? 1 : 0;
    }

    std::vector<bool> to_bool(ret.size());
    i = 0;

    for (int i = i; i >= 0; i--)
    {
        to_bool[i] = ret[i] == 1 ? true : false;
    }

    return tensor<bool>(to_bool, shape_);
}

template<class _Tp>
inline tensor<bool> tensor<_Tp>::neon_less(const value_type val) const {
    if constexpr (!has_less_operator_v<value_type>)
    {
        throw operator_error("Value type must have less than operator");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(_ARM64_REG_WIDTH % sizeof(value_type) == 0,
                  "size of value_type must divide evenly into the register width. "
                  "Ensure value_type fits into a whole number of registers without leaving unused bits.");
}