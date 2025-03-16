#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_dist_(const tensor& __other) {
  assert(__equal_shape(this->shape(), __other.shape()));
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a    = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b    = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __diff = vabdq_f32(__a, __b);
      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __diff);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __a    = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __b    = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __diff = vabdq_s32(__a, __b);
      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __diff);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __a    = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __b    = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __diff = vabdq_u32(__a, __b);
      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __diff);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::abs(static_cast<_f64>(this->__data_[__i] - __other.__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_dist_(const value_type __val) {
  index_type       __i        = 0;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a    = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b    = vdupq_n_f32(&__val);
      neon_f32 __diff = vabdq_f32(__a, __b);
      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __diff);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __a    = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __b    = vdupq_n_s32(&__val);
      neon_s32 __diff = vabdq_s32(__a, __b);
      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __diff);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __a    = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __b    = vdupq_n_u32(&__val);
      neon_u32 __diff = vabdq_u32(__a, __b);
      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __diff);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] =
        static_cast<value_type>(std::abs(static_cast<_f64>(this->__data_[__i] - __val)));

  return *this;
}