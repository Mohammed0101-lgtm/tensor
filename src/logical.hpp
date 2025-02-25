#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_or_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform logical OR on non-integral and non-boolean values");

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value || std::is_same<value_type, bool>::value) {
    neon_s32 __val_vec = vdupq_n_s32(static_cast<_s32>(__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __or       = vorrq_s32(__data_vec, __val_vec);

      vst1q_s32(&this->__data_[__i], __or);
    }
  } else if constexpr (std::is_same<value_type, _u32>::value) {
    neon_u32 __val_vec = vdupq_n_u32(static_cast<_u32>(__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __or       = vorrq_u32(__data_vec, __val_vec);

      vst1q_u32(&this->__data_[__i], __or);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] || __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_or_(const value_type __val) const {
  return this->logical_or_(__val);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_xor_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error(
        "Cannot get the element wise xor of non-integral and non-boolean value");

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value || std::is_same<value_type, bool>::value) {
    neon_s32 __val_vec = vdupq_n_s32(static_cast<_s32>(__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __xor      = veorq_s32(__data_vec, __val_vec);

      vst1q_s32(&this->__data_[__i], __xor);
    }
  } else if constexpr (std::is_same<value_type, _u32>::value) {
    neon_u32 __val_vec = vdupq_n_u32(static_cast<_u32>(__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __xor      = veorq_u32(__data_vec, __val_vec);

      vst1q_u32(&this->__data_[__i], __xor);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] ^ __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_xor_(const value_type __val) const {
  return this->logical_xor(__val);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_and_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error(
        "Cannot get the element wise and of non-integral and non-boolean value");

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value) {
    neon_s32 __vals = vdupq_n_s32(reinterpret_cast<_s32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __and = vandq_s32(__vec, __vals);

      vst1q_s32(&this->__data_[__i], __and);
    }
  } else if constexpr (std::is_same<value_type, _u32>::value) {
    neon_u32 __vals = vdupq_n_u32(reinterpret_cast<_u32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __and = vandq_u32(__vec, __vals);

      vst1q_u32(&this->__data_[__i], __and);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] && __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_and_(const value_type __val) const {
  return this->logical_and_(__val);
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::logical_not_() {
  this->bitwise_not_();
  this->bool_();
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_not_() const {
  return this->logical_not_();
}

template <class _Tp>
tensor<bool> tensor<_Tp>::logical_not() const {
  tensor<bool> __ret = this->bool_();
  __ret.logical_not_();
  return __ret;
}

template <class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const value_type __val) const {
  tensor<bool> __ret = this->clone().bool_();
  __ret.logical_or_(__val);
  return __ret;
}

template <class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const tensor& __other) const {
  tensor<bool> __ret = this->clone().bool_();
  __ret.logical_or_(__other);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.logical_xor_(__other);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const value_type __val) const {
  __self __ret = this->clone();
  __ret.logical_xor(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.logical_and_(__other);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const value_type __val) const {
  __self __ret = this->clone();
  __ret.logical_and_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_or_(const tensor& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error(
        "Cannot get the element wise not of non-integral and non-boolean value");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __or_vec    = vornq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __or_vec);
    }
  } else if constexpr (std::is_signed<value_type>::value) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __or_vec    = vornq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __or_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = (this->__data_[__i] || __other[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_or_(const tensor& __other) const {
  return this->logical_or_(__other);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_xor_(const tensor& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error(
        "Cannot get the element wise xor of non-integral and non-boolean value");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __xor_vec   = veorq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __xor_vec);
    }
  } else if constexpr (std::is_signed<value_type>::value) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __xor_vec   = veorq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __xor_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = (this->__data_[__i] ^ __other[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_xor_(const tensor& __other) const {
  return this->logical_xor_(__other);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_and_(const tensor& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error(
        "Cannot get the element-wise and of non-integral and non-boolean value");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __and_vec   = vandq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __and_vec);
    }
  } else if constexpr (std::is_signed<value_type>::value) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __and_vec   = vandq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __and_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = (this->__data_[__i] && __other[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_and_(const tensor& __other) const {
  return this->logical_and_(__other);
}