#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_frac_() {
  index_type __i = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = this->__frac(__vals[0]);
      __vals[1] = this->__frac(__vals[1]);
      __vals[2] = this->__frac(__vals[2]);
      __vals[3] = this->__frac(__vals[3]);

      neon_f32 __atan_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __atan_vec);
    }
  }
  if constexpr (std::is_same_v<value_type, _f64>) {
    index_type __simd_end = this->__data_.size() - (this->__data_.size() % (_ARM64_REG_WIDTH / 2));

    for (; __i < __simd_end; __i += (_ARM64_REG_WIDTH / 2)) {
      neon_f64 __data_vec = vld1q_f64(reinterpret_cast<const _f64*>(&this->__data_[__i]));
      _f64     __vals[_ARM64_REG_WIDTH];
      vst1q_f64(__vals, __data_vec);

      __vals[0] = static_cast<_f64>(this->__frac(__vals[0]));
      __vals[1] = static_cast<_f64>(this->__frac(__vals[1]));
      __vals[2] = static_cast<_f64>(this->__frac(__vals[2]));
      __vals[3] = static_cast<_f64>(this->__frac(__vals[3]));

      neon_f64 __atan_vec = vld1q_f64(__vals);
      vst1q_f64(&this->__data_[__i], __atan_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));

  return *this;
}