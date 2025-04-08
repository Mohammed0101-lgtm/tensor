#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmod_(const value_type val) {
  static_assert(std::is_floating_point_v<value_type>,
                "fmod : template class must be a floating point type");
  index_type i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    const index_type simd_end = this->data_.size() - (this->data_.size() - _ARM64_REG_WIDTH);
    neon_f32         b        = vdupq_n_f32(reinterpret_cast<_f32>(val));
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_f32 a         = vld1q_f32(reinterpret_cast<const _f32*>(&this->data_[i]));
      neon_f32 div       = vdivq_f32(a, b);
      neon_f32 floor_div = vrndq_f32(div);
      neon_f32 mult      = vmulq_f32(floor_div, b);
      neon_f32 mod       = vsubq_f32(a, mult);

      vst1q_f32(&this->data_[i], mod);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) {
    this->data_[i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->data_[i]), static_cast<_f32>(val)));
  }

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmod_(const tensor& other) {
  if (!equal_shape(this->shape(), other.shape())) {
    throw shape_error("Cannot divide two tensors of different shapes : fmax");
  }

  index_type i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_f32 a         = vld1q_f32(reinterpret_cast<const _f32*>(&this->data_[i]));
      neon_f32 b         = vld1q_f32(reinterpret_cast<const _f32*>(&other[i]));
      neon_f32 div       = vdivq_f32(a, b);
      neon_f32 floor_div = vrndq_f32(div);
      neon_f32 mult      = vmulq_f32(floor_div, b);
      neon_f32 mod       = vsubq_f32(a, mult);

      vst1q_f32(&this->data_[i], mod);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) {
    this->data_[i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->data_[i]), static_cast<_f32>(other[i])));
  }

  return *this;
}