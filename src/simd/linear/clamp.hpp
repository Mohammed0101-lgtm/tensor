#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_clamp_(const_reference __min_val, const_reference __max_val) {
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    neon_f32 __min_vec = vdupq_n_f32(__min_val);
    neon_f32 __max_vec = vdupq_n_f32(__max_val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __clamped  = vminq_f32(vmaxq_f32(__data_vec, __min_vec), __max_vec);

      vst1q_f32(&this->__data_[__i], __clamped);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    neon_s32 __min_vec = vdupq_n_s32(__min_val);
    neon_s32 __max_vec = vdupq_n_s32(__max_val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __clamped  = vminq_s32(vmaxq_s32(__data_vec, __min_vec), __max_vec);

      vst1q_s32(&this->__data_[__i], __clamped);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __min_vec = vdupq_n_u32(__min_val);
    neon_u32 __max_vec = vdupq_n_u32(__max_val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __clamped  = vminq_u32(vmaxq_u32(__data_vec, __min_vec), __max_vec);

      vst1q_u32(&this->__data_[__i], __clamped);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) {
    this->__data_[__i] = std::max(__min_val, this->__data_[__i]);
    this->__data_[__i] = std::min(__max_val, this->__data_[__i]);
  }

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_ceil_() {
  if (!std::is_floating_point_v<value_type>) throw __type_error__("Type must be floating point");

  index_type __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    for (; __i + _ARM64_REG_WIDTH <= this->__data_.size(); __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __ceil_vec = vrndpq_f32(__data_vec);

      vst1q_f32(&this->__data_[__i], __ceil_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::ceil(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_floor_() {
  if (!std::is_floating_point_v<value_type>) throw __type_error__("Type must be floating point");

  index_type __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    for (; __i < this->__data_.size(); __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec  = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __floor_vec = vrndmq_f32(__data_vec);

      vst1q_f32(&this->__data_[__i], __floor_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::floor(static_cast<_f32>(this->__data_[__i])));

  return *this;
}