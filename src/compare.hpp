#pragma once

#include "tensorbase.hpp"


template<class _Tp>
tensor<bool> tensor<_Tp>::not_equal(const tensor& __other) const {}

template<class _Tp>
tensor<bool> tensor<_Tp>::not_equal(const value_type __val) const {}

template<class _Tp>
tensor<bool> tensor<_Tp>::less(const tensor& __other) const {}

template<class _Tp>
tensor<bool> tensor<_Tp>::less(const value_type __val) const {}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater(const tensor& __other) const {}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater(const value_type __val) const {}

template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  assert(this->__shape_ == __other.shape() && "equal : tensor shapes");
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec1  = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __data_vec2  = vld1q_f32(reinterpret_cast<const _f32*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vceqq_f32(__data_vec1, __data_vec2);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec1  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __data_vec2  = vld1q_s32(reinterpret_cast<const _s32*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vceqq_s32(__data_vec1, __data_vec2);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec1  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __data_vec2  = vld1q_u32(reinterpret_cast<const _u32*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vceqq_u32(__data_vec1, __data_vec2);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] == __other[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    neon_f32         __val_vec  = vdupq_n_f32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_u32 __cmp_result = vceqq_f32(__data_vec, __val_vec);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    neon_s32         __val_vec  = vdupq_n_s32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_u32 __cmp_result = vceqq_s32(__data_vec, __val_vec);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    neon_u32         __val_vec  = vdupq_n_u32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __cmp_result = vceqq_u32(__data_vec, __val_vec);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] == __val);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

#if defined(__ARM_NEON)
  using neon_type    = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;
  size_t vector_size = this->__data_.size() / _ARM64_REG_WIDTH * _ARM64_REG_WIDTH;

  for (; __i < vector_size; __i += _ARM64_REG_WIDTH)
  {
    neon_type vec_a    = vld1q(this->__data_.data() + __i);
    neon_type vec_b    = vld1q(__other.__data_.data() + __i);
    neon_u32  leq_mask = std::is_same_v<value_type, _f32> ? vcleq_f32(vec_a, vec_b) : vcleq_s32(vec_a, vec_b);
    vst1q_u32(reinterpret_cast<_u32*>(&__ret[__i]), leq_mask);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] <= __other[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

#if defined(__ARM_NEON)
  using neon_type    = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;
  size_t vector_size = this->__data_.size() / _ARM64_REG_WIDTH * _ARM64_REG_WIDTH;

  for (; __i < vector_size; __i += _ARM64_REG_WIDTH)
  {
    neon_type vec_a    = vld1q(this->__data_.data() + __i);
    neon_type vec_b    = std::is_same_v<value_type, _f32> ? vdupq_n_f32(__val) : vdupq_n_s32(__val);
    neon_u32  leq_mask = std::is_same_v<value_type, _f32> ? vcleq_f32(vec_a, vec_b) : vcleq_s32(vec_a, vec_b);

    vst1q_u32(reinterpret_cast<_u32*>(__ret.data() + __i), leq_mask);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] <= __val);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % (_ARM64_REG_WIDTH / 2));

    for (; __i < __simd_end; __i += (_ARM64_REG_WIDTH / 2))
    {
      neon_f32 __data_vec1  = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __data_vec2  = vld1q_f32(reinterpret_cast<const _f32*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vcgeq_f32(__data_vec1, __data_vec2);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 1) & 1;
      __ret[__i + 2] = (__mask >> 2) & 1;
      __ret[__i + 3] = (__mask >> 3) & 1;
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec1  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __data_vec2  = vld1q_s32(reinterpret_cast<const _s32*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vcgeq_s32(__data_vec1, __data_vec2);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 1) & 1;
      __ret[__i + 2] = (__mask >> 2) & 1;
      __ret[__i + 3] = (__mask >> 3) & 1;
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec1  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __data_vec2  = vld1q_u32(reinterpret_cast<const _u32*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vcgeq_u32(__data_vec1, __data_vec2);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 1) & 1;
      __ret[__i + 2] = (__mask >> 2) & 1;
      __ret[__i + 3] = (__mask >> 3) & 1;
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] >= __other[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __val_vec = vdupq_n_f32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_u32 __cmp_result = vcgeq_f32(__data_vec, __val_vec);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_u32 __cmp_result = vcgeq_s32(__data_vec, __val_vec);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __val_vec = vdupq_n_u32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __cmp_result = vcgeq_u32(__data_vec, __val_vec);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] >= __val);
  }

  return tensor<bool>(__ret, this->__shape_);
}
