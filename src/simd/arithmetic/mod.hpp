#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmod_(const value_type __val) {
  assert(std::is_floating_point_v<value_type> &&
         "fmod : template class must be a floating point type");
  index_type __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() - _ARM64_REG_WIDTH);
    neon_f32         __b        = vdupq_n_f32(reinterpret_cast<_f32>(__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a         = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __div       = vdivq_f32(__a, __b);
      neon_f32 __floor_div = vrndq_f32(__div);
      neon_f32 __mult      = vmulq_f32(__floor_div, __b);
      neon_f32 __mod       = vsubq_f32(__a, __mult);

      vst1q_f32(&this->__data_[__i], __mod);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__val)));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmod_(const tensor& __other) {
  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");

  index_type __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a         = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b         = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __div       = vdivq_f32(__a, __b);
      neon_f32 __floor_div = vrndq_f32(__div);
      neon_f32 __mult      = vmulq_f32(__floor_div, __b);
      neon_f32 __mod       = vsubq_f32(__a, __mult);

      vst1q_f32(&this->__data_[__i], __mod);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}