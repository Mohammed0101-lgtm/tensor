#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_logical_or_(const value_type __val) {
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform logical OR on non-integral values");

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_signed_v<value_type> || std::is_same_v<value_type, bool>) {
    neon_s32 __val_vec = vdupq_n_s32(static_cast<_s32>(__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __or       = vorrq_s32(__data_vec, __val_vec);

      vst1q_s32(&this->__data_[__i], __or);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __val_vec = vdupq_n_u32(static_cast<_u32>(__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __or       = vorrq_u32(__data_vec, __val_vec);

      vst1q_u32(&this->__data_[__i], __or);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] || __val);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_logical_xor_(const value_type __val) {
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot get the element wise xor of non-integral and non-boolean value");

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_signed_v<value_type> || std::is_same_v<value_type, bool>) {
    neon_s32 __val_vec = vdupq_n_s32(static_cast<_s32>(__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __xor      = veorq_s32(__data_vec, __val_vec);

      vst1q_s32(&this->__data_[__i], __xor);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __val_vec = vdupq_n_u32(static_cast<_u32>(__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __xor      = veorq_u32(__data_vec, __val_vec);

      vst1q_u32(&this->__data_[__i], __xor);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] ^ __val);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_logical_and_(const value_type __val) {
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot get the element wise and of non-integral and non-boolean value");

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_signed_v<value_type>) {
    neon_s32 __vals = vdupq_n_s32(reinterpret_cast<_s32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __and = vandq_s32(__vec, __vals);

      vst1q_s32(&this->__data_[__i], __and);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __vals = vdupq_n_u32(reinterpret_cast<_u32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __and = vandq_u32(__vec, __vals);

      vst1q_u32(&this->__data_[__i], __and);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] && __val);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_logical_or_(const tensor& __other) {
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot get the element wise not of non-integral values");

  assert(__equal_shape(this->shape(), __other.shape()));
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_unsigned_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __or_vec    = vornq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __or_vec);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __or_vec    = vornq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __or_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (this->__data_[__i] || __other[__i]);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_logical_xor_(const tensor& __other) {
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw __type_error__("Cannot get the element wise xor of non-integral and non-boolean value");

  assert(__equal_shape(this->shape(), __other.shape()));
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_unsigned_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __xor_vec   = veorq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __xor_vec);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __xor_vec   = veorq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __xor_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (this->__data_[__i] ^ __other[__i]);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_logical_and_(const tensor& __other) {
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw __type_error__("Cannot get the element-wise and of non-integral and non-boolean value");

  assert(__equal_shape(this->shape(), __other.shape()));
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_unsigned_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __and_vec   = vandq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __and_vec);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __and_vec   = vandq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __and_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (this->__data_[__i] && __other[__i]);

  return *this;
}