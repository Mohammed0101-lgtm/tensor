#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_cross_product(const tensor& __other) const {
  if (this->empty() || __other.empty())
    throw std::invalid_argument("Cannot cross product an empty vector");

  if (this->shape() != std::vector<int>{3} || __other.shape() != std::vector<int>{3})
    throw std::invalid_argument("Cross product can only be performed on 3-element vectors");

  tensor __ret({3});

  if constexpr (std::is_floating_point_v<value_type>) {
    neon_f32 __a      = vld1q_f32(reinterpret_cast<const _f32*>(this->__data_.data()));
    neon_f32 __b      = vld1q_f32(reinterpret_cast<const _f32*>(__other.storage().data()));
    neon_f32 __a_yzx  = vextq_f32(__a, __a, 1);
    neon_f32 __b_yzx  = vextq_f32(__b, __b, 1);
    neon_f32 __result = vsubq_f32(vmulq_f32(__a_yzx, __b), vmulq_f32(__a, __b_yzx));
    __result          = vextq_f32(__result, __result, 3);

    vst1q_f32(reinterpret_cast<_f32*>(__ret.storage().data()), __result);
  } else if constexpr (std::is_signed_v<value_type>) {
    neon_s32 __a      = vld1q_s32(reinterpret_cast<const _s32*>(this->__data_.data()));
    neon_s32 __b      = vld1q_s32(reinterpret_cast<const _s32*>(__other.storage().data()));
    neon_s32 __a_yzx  = vextq_s32(__a, __a, 1);
    neon_s32 __b_yzx  = vextq_s32(__b, __b, 1);
    neon_s32 __result = vsubq_s32(vmulq_s32(__a_yzx, __b), vmulq_s32(__a, __b_yzx));
    __result          = vextq_s32(__result, __result, 3);

    vst1q_s32(reinterpret_cast<_s32*>(__ret.storage().data()), __result);
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __a      = vld1q_u32(reinterpret_cast<const _u32*>(this->__data_.data()));
    neon_u32 __b      = vld1q_u32(reinterpret_cast<const _u32*>(__other.storage().data()));
    neon_u32 __a_yzx  = vextq_u32(__a, __a, 1);
    neon_u32 __b_yzx  = vextq_u32(__b, __b, 1);
    neon_u32 __result = vsubq_u32(vmulq_u32(__a_yzx, __b), vmulq_u32(__a, __b_yzx));
    __result          = vextq_u32(__result, __result, 3);

    vst1q_u32(reinterpret_cast<_u32*>(__ret.storage().data()), __result);
  }

  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_dot(const tensor& __other) const {
  if (this->empty() || __other.empty())
    throw std::invalid_argument("Cannot dot product an empty vector");

  if (this->__shape_.size() == 1 && __other.shape().size() == 1) {
    if (this->__shape_[0] != __other.shape()[0])
      throw std::invalid_argument("Vectors must have the same size for dot product");

    const_pointer __this_data  = this->__data_.data();
    const_pointer __other_data = __other.storage().data();
    const size_t  __size       = this->__data_.size();
    value_type    __ret        = 0;

    if constexpr (std::is_floating_point_v<value_type>) {
      size_t   __i     = 0;
      neon_f32 sum_vec = vdupq_n_f32(0.0f);

      for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH) {
        neon_f32 a_vec = vld1q_f32(reinterpret_cast<const _f32*>(&__this_data[__i]));
        neon_f32 b_vec = vld1q_f32(reinterpret_cast<const _f32*>(&__other_data[__i]));
        sum_vec        = vmlaq_f32(sum_vec, a_vec, b_vec);  // Perform multiply-accumulate
      }

      float32x2_t sum_half = vadd_f32(vget_high_f32(sum_vec), vget_low_f32(sum_vec));
      __ret                = vget_lane_f32(vpadd_f32(sum_half, sum_half), 0);

      for (; __i < __size; ++__i)
        __ret +=
            static_cast<value_type>(__this_data[__i]) * static_cast<value_type>(__other_data[__i]);
    } else if constexpr (std::is_unsigned_v<value_type>) {
      size_t   __i     = 0;
      neon_u32 sum_vec = vdupq_n_u32(0.0f);

      for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH) {
        neon_u32 a_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__this_data[__i]));
        neon_u32 b_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other_data[__i]));
        sum_vec        = vmlaq_u32(sum_vec, a_vec, b_vec);
      }

      uint32x2_t sum_half = vadd_u32(vget_high_u32(sum_vec), vget_low_u32(sum_vec));
      __ret               = vget_lane_u32(vpadd_u32(sum_half, sum_half), 0);

      for (; __i < __size; ++__i)
        __ret +=
            static_cast<value_type>(__this_data[__i]) * static_cast<value_type>(__other_data[__i]);
    } else if constexpr (std::is_signed_v<value_type>) {
      size_t   __i     = 0;
      neon_s32 sum_vec = vdupq_n_f32(0.0f);

      for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH) {
        neon_s32 a_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__this_data[__i]));
        neon_s32 b_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other_data[__i]));
        sum_vec        = vmlaq_s32(sum_vec, a_vec, b_vec);  // Perform multiply-accumulate
      }

      int32x2_t sum_half = vadd_s32(vget_high_s32(sum_vec), vget_low_s32(sum_vec));
      __ret              = vget_lane_s32(vpadd_s32(sum_half, sum_half), 0);

      for (; __i < __size; ++__i)
        __ret +=
            static_cast<value_type>(__this_data[__i]) * static_cast<value_type>(__other_data[__i]);
    }

    return __self({__ret}, {1});
  }

  if (this->__shape_.size() == 2 && __other.shape().size() == 2) return this->matmul(__other);

  if (this->__shape_.size() == 3 && __other.shape().size() == 3)
    return this->cross_product(__other);

  return __self();
}