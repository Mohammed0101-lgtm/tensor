#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_cos_() {
  index_type __i = 0;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::cos(__vals[0]));
      __vals[1] = static_cast<_f32>(std::cos(__vals[1]));
      __vals[2] = static_cast<_f32>(std::cos(__vals[2]));
      __vals[3] = static_cast<_f32>(std::cos(__vals[3]));

      neon_f32 __cos_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __cos_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::cos(__vals[0]));
      __vals[1] = static_cast<_s32>(std::cos(__vals[1]));
      __vals[2] = static_cast<_s32>(std::cos(__vals[2]));
      __vals[3] = static_cast<_s32>(std::cos(__vals[3]));

      neon_s32 __cos_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __cos_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::cos(__vals[0]));
      __vals[1] = static_cast<_u32>(std::cos(__vals[1]));
      __vals[2] = static_cast<_u32>(std::cos(__vals[2]));
      __vals[3] = static_cast<_u32>(std::cos(__vals[3]));

      neon_u32 __cos_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __cos_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::cos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_acos_() {
  index_type __i = 0;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::acos(__vals[0]));
      __vals[1] = static_cast<_f32>(std::acos(__vals[1]));
      __vals[2] = static_cast<_f32>(std::acos(__vals[2]));
      __vals[3] = static_cast<_f32>(std::acos(__vals[3]));

      neon_f32 __cos_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __cos_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::acos(__vals[0]));
      __vals[1] = static_cast<_s32>(std::acos(__vals[1]));
      __vals[2] = static_cast<_s32>(std::acos(__vals[2]));
      __vals[3] = static_cast<_s32>(std::acos(__vals[3]));

      neon_s32 __cos_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __cos_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::acos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_cosh_() {
  index_type       __i        = 0;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::cosh(__vals[0]));
      __vals[1] = static_cast<_f32>(std::cosh(__vals[1]));
      __vals[2] = static_cast<_f32>(std::cosh(__vals[2]));
      __vals[3] = static_cast<_f32>(std::cosh(__vals[3]));

      neon_f32 __cosh_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __cosh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::cosh(__vals[0]));
      __vals[1] = static_cast<_s32>(std::cosh(__vals[1]));
      __vals[2] = static_cast<_s32>(std::cosh(__vals[2]));
      __vals[3] = static_cast<_s32>(std::cosh(__vals[3]));

      neon_s32 __cosh_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __cosh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::cosh(__vals[0]));
      __vals[1] = static_cast<_u32>(std::cosh(__vals[1]));
      __vals[2] = static_cast<_u32>(std::cosh(__vals[2]));
      __vals[3] = static_cast<_u32>(std::cosh(__vals[3]));

      neon_u32 __cosh_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __cosh_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::cosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_acosh_() {
  index_type       __i        = 0;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::acosh(__vals[0]));
      __vals[1] = static_cast<_f32>(std::acosh(__vals[1]));
      __vals[2] = static_cast<_f32>(std::acosh(__vals[2]));
      __vals[3] = static_cast<_f32>(std::acosh(__vals[3]));

      neon_f32 __acosh_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __acosh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::acosh(__vals[0]));
      __vals[1] = static_cast<_s32>(std::acosh(__vals[1]));
      __vals[2] = static_cast<_s32>(std::acosh(__vals[2]));
      __vals[3] = static_cast<_s32>(std::acosh(__vals[3]));

      neon_s32 __acosh_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __acosh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::acosh(__vals[0]));
      __vals[1] = static_cast<_u32>(std::acosh(__vals[1]));
      __vals[2] = static_cast<_u32>(std::acosh(__vals[2]));
      __vals[3] = static_cast<_u32>(std::acosh(__vals[3]));

      neon_u32 __acosh_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __acosh_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::acosh(this->__data_[__i]));

  return *this;
}
