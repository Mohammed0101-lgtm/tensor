#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_abs_() {
  index_type __i = 0;

  if (std::is_unsigned_v<value_type>) return *this;

  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::abs(__vals[0]));
      __vals[1] = static_cast<_f32>(std::abs(__vals[1]));
      __vals[2] = static_cast<_f32>(std::abs(__vals[2]));
      __vals[3] = static_cast<_f32>(std::abs(__vals[3]));

      neon_f32 __abs_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __abs_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::abs(__vals[0]));
      __vals[1] = static_cast<_s32>(std::abs(__vals[1]));
      __vals[2] = static_cast<_s32>(std::abs(__vals[2]));
      __vals[3] = static_cast<_s32>(std::abs(__vals[3]));

      neon_s32 __abs_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __abs_vec);
    }
  }
  // don't need to check for unsigned values because they are already positive
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::abs(this->__data_[__i]));

  return *this;
}