#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_relu_() {
  if constexpr (std::is_unsigned_v<value_type>) return *this;

  index_type __s = this->__data_.size();
  index_type __i = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    const neon_f32 __vZero = vdupq_n_f32(0.0f);

    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH) {
      neon_f32 __v = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      __v          = vmaxq_f32(__v, __vZero);

      vst1q_f32(&this->__data_[__i], __v);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    const neon_s32 __vZero = vdupq_n_s32(0);

    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH) {
      neon_s32 __v = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      __v          = vmaxq_s32(__v, __vZero);

      vst1q_s32(&this->__data_[__i], __v);
    }
  }

  for (__i = 0; __i < __s; ++__i) this->__data_[__i] = std::max(this->__data_[__i], value_type(0));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_clipped_relu_(const value_type __clip_limit) {
  if constexpr (std::is_unsigned_v<value_type>) return *this;

  index_type __s = this->__data_.size();
  index_type __i = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    const neon_f32 __vZero = vdupq_n_f32(0.0f);
    const neon_f32 __vClip = vdupq_n_f32(__clip_limit);

    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH) {
      neon_f32 __v = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      __v          = vminq_f32(vmaxq_f32(__v, __vZero), __vClip);

      vst1q_f32(&this->__data_[__i], __v);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    const neon_s32 __vZero = vdupq_n_s32(0);
    const neon_s32 __vClip = vdupq_n_s32(__clip_limit);

    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH) {
      neon_s32 __v = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      __v          = vminq_s32(vmaxq_s32(__v, __vZero), __vClip);

      vst1q_s32(&this->__data_[__i], __v);
    }
  }
#pragma omp parallel
  for (; __i < __s; ++__i)
    this->__data_[__i] = std::min(std::max(this->__data_[__i], value_type(0)), __clip_limit);

  return *this;
}
