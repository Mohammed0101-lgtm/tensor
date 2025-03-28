#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_tan_() {
  index_type __i        = 0;
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::tan(__vals[0]));
      __vals[1] = static_cast<_f32>(std::tan(__vals[1]));
      __vals[2] = static_cast<_f32>(std::tan(__vals[2]));
      __vals[3] = static_cast<_f32>(std::tan(__vals[3]));

      neon_f32 __tan_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __tan_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::tan(__vals[0]));
      __vals[1] = static_cast<_s32>(std::tan(__vals[1]));
      __vals[2] = static_cast<_s32>(std::tan(__vals[2]));
      __vals[3] = static_cast<_s32>(std::tan(__vals[3]));

      neon_s32 __tan_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __tan_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::tan(__vals[0]));
      __vals[1] = static_cast<_u32>(std::tan(__vals[1]));
      __vals[2] = static_cast<_u32>(std::tan(__vals[2]));
      __vals[3] = static_cast<_u32>(std::tan(__vals[3]));

      neon_u32 __tan_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __tan_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::tan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_tanh_() {
  index_type __i = 0;

  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::tanh(__vals[0]));
      __vals[1] = static_cast<_f32>(std::tanh(__vals[1]));
      __vals[2] = static_cast<_f32>(std::tanh(__vals[2]));
      __vals[3] = static_cast<_f32>(std::tanh(__vals[3]));

      neon_f32 __tanh_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __tanh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::tanh(__vals[0]));
      __vals[1] = static_cast<_s32>(std::tanh(__vals[1]));
      __vals[2] = static_cast<_s32>(std::tanh(__vals[2]));
      __vals[3] = static_cast<_s32>(std::tanh(__vals[3]));

      neon_s32 __tanh_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __tanh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::tanh(__vals[0]));
      __vals[1] = static_cast<_u32>(std::tanh(__vals[1]));
      __vals[2] = static_cast<_u32>(std::tanh(__vals[2]));
      __vals[3] = static_cast<_u32>(std::tanh(__vals[3]));

      neon_u32 __tanh_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __tanh_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::tanh(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_atan_() {
  index_type __i        = 0;
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::atan(static_cast<_f32>(__vals[0])));
      __vals[1] = static_cast<_f32>(std::atan(static_cast<_f32>(__vals[1])));
      __vals[2] = static_cast<_f32>(std::atan(static_cast<_f32>(__vals[2])));
      __vals[3] = static_cast<_f32>(std::atan(static_cast<_f32>(__vals[3])));

      neon_f32 __atan_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __atan_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::atan(static_cast<_u32>(__vals[0])));
      __vals[1] = static_cast<_u32>(std::atan(static_cast<_u32>(__vals[1])));
      __vals[2] = static_cast<_u32>(std::atan(static_cast<_u32>(__vals[2])));
      __vals[3] = static_cast<_u32>(std::atan(static_cast<_u32>(__vals[3])));

      neon_u32 __atan_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __atan_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::atan(static_cast<_s32>(__vals[0])));
      __vals[1] = static_cast<_s32>(std::atan(static_cast<_s32>(__vals[1])));
      __vals[2] = static_cast<_s32>(std::atan(static_cast<_s32>(__vals[2])));
      __vals[3] = static_cast<_s32>(std::atan(static_cast<_s32>(__vals[3])));

      neon_s32 __atan_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __atan_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_atanh_() {
  index_type __i        = 0;
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::atanh(__vals[0]));
      __vals[1] = static_cast<_f32>(std::atanh(__vals[1]));
      __vals[2] = static_cast<_f32>(std::atanh(__vals[2]));
      __vals[3] = static_cast<_f32>(std::atanh(__vals[3]));

      neon_f32 __atan_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __atan_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::atanh(__vals[0]));
      __vals[1] = static_cast<_u32>(std::atanh(__vals[1]));
      __vals[2] = static_cast<_u32>(std::atanh(__vals[2]));
      __vals[3] = static_cast<_u32>(std::atanh(__vals[3]));

      neon_u32 __atan_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __atan_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::atanh(__vals[0]));
      __vals[1] = static_cast<_s32>(std::atanh(__vals[1]));
      __vals[2] = static_cast<_s32>(std::atanh(__vals[2]));
      __vals[3] = static_cast<_s32>(std::atanh(__vals[3]));

      neon_s32 __atan_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __atan_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}