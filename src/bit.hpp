#pragma once

#include "tensorbase.hpp"


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_right_shift_(const int __amount) {
  this->__check_is_integral_type("Cannot perform a bitwise right shift on non-integral values");

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    const neon_s32 __shift_amount_vec = vdupq_n_s32(-__amount);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec    = vld1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]));
      neon_s32 __shifted_vec = vshlq_s32(__data_vec, __shift_amount_vec);

      vst1q_s32(&this->__data_[__i], __shifted_vec);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    const neon_s32 __shift_amount_vec = vdupq_n_s32(-__amount);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec    = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __shifted_vec = vshlq_u32(__data_vec, __shift_amount_vec);

      vst1q_u32(&this->__data_[__i], __shifted_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] >>= __amount;

  return *this;
}


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_left_shift_(const int __amount) {
  this->__check_is_integral_type("Cannot perform a bitwise left shift on non-integral values");

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    const neon_s32 __shift_amount_vec = vdupq_n_s32(__amount);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec    = vld1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]));
      neon_s32 __shifted_vec = vshlq_s32(__data_vec, __shift_amount_vec);

      vst1q_s32(&this->__data_[__i], __shifted_vec);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    const neon_s32 __shift_amount_vec = vdupq_n_s32(__amount);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec    = vld1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]));
      neon_u32 __shifted_vec = vshlq_u32(__data_vec, __shift_amount_vec);

      vst1q_u32(&this->__data_[__i], __shifted_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] <<= __amount;

  return *this;
}


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_or_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");
  }

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec   = vld1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]));
      neon_s32 __result_vec = vorrq_s32(__data_vec, __val_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __result_vec);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    neon_u32 __val_vec = vdupq_n_u32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec   = vld1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]));
      neon_u32 __result_vec = vorrq_u32(__data_vec, __val_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __result_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] |= __val;

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec   = vld1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]));
      neon_s32 __result_vec = veorq_s32(__data_vec, __val_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __result_vec);
    }
  }
  else
  {
    neon_u32 __val_vec = vdupq_n_u32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec   = vld1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]));
      neon_u32 __result_vec = veorq_u32(__data_vec, __val_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __result_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] ^= __val;

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_xor(const value_type __val) const {
  __self __ret = this->clone();
  __ret.bitwise_xor_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_not() const {
  __self __ret = this->clone();
  __ret.bitwise_not_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_not_() {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise not on non integral or boolean value");

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __result_vec = vmvnq_s32(__data_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __result_vec);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __result_vec = vmvnq_u32(__data_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __result_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = ~this->__data_[__i];

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_and_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(reinterpret_cast<_s32>(&__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __result_vec = vandq_s32(__data_vec, __val_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __result_vec);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    neon_u32 __val_vec = vdupq_n_u32(reinterpret_cast<_u32>(&__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __result_vec = vandq_u32(__data_vec, __val_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __result_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] &= __val;

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_and(const value_type __val) const {
  __self __ret = this->clone();
  __ret.bitwise_and_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_and(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.bitwise_and_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_left_shift(const int __amount) const {
  __self __ret = this->clone();
  __ret.bitwise_left_shift_(__amount);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_xor(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.bitwise_xor_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_right_shift(const int __amount) const {
  __self __ret = this->clone();
  __ret.bitwise_right_shift_(__amount);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_and_(const tensor& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __xor_vec   = vandq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __xor_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __xor_vec   = vandq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __xor_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] &= __other[__i];

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_or(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.bitwise_or_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_or(const value_type __val) const {
  __self __ret = this->clone();
  __ret.bitwise_or_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_or_(const tensor& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __xor_vec   = vornq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __xor_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __xor_vec   = vornq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __xor_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] |= __other[__i];

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const tensor& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __xor_vec   = veorq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __xor_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __xor_vec   = veorq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __xor_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] ^= __other[__i];

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fill_(const value_type __val) const {
  this->__data_(this->__data_.size(), __val);
  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fill_(const tensor& __other) const {
  assert(this->__shape_ == __other.shape());

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = __other[__i];

  return *this;
}
