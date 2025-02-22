#pragma once

#include "tensorbase.hpp"

template<class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::operator()(std::initializer_list<index_type> __index_list) {
  return this->__data_[this->__compute_index(shape_type(__index_list))];
}

template<class _Tp>
typename tensor<_Tp>::const_reference tensor<_Tp>::operator()(std::initializer_list<index_type> __index_list) const {
  return this->__data_[this->__compute_index(shape_type(__index_list))];
}

template<class _Tp>
bool tensor<_Tp>::operator!=(const tensor& __other) const {
  return !(*this == __other);
}

template<class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::operator[](const index_type __in) {
  if (__in >= this->__data_.size() || __in < 0)
    throw std::out_of_range("Access index is out of range");

  return this->__data_[__in];
}

template<class _Tp>
typename tensor<_Tp>::const_reference tensor<_Tp>::operator[](const index_type __in) const {
  if (__in >= this->__data_.size() || __in < 0)
    throw std::out_of_range("Access index is out of range");

  return this->__data_[__in];
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const tensor& __other) const {
  static_assert(has_plus_operator_v<value_type>);

  if (__other.shape() != this->__shape_)
    throw std::invalid_argument("Cannot add two tensors with different shapes");

  data_t     __d(this->__data_.size());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec1   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __vec2   = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __result = vaddq_f32(__vec1, __vec2);

      vst1q_f32(reinterpret_cast<_f32*>(&__d[__i]), __result);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec1   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __vec2   = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __result = vaddq_s32(__vec1, __vec2);

      vst1q_s32(reinterpret_cast<_s32*>(&__d[__i]), __result);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec1   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __vec2   = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __result = vaddq_u32(__vec1, __vec2);

      vst1q_u32(reinterpret_cast<_u32*>(&__d[__i]), __result);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    __d[__i] = this->__data_[__i] + __other[__i];

  return __self(this->__shape_, __d);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const value_type __val) const {
  static_assert(has_plus_operator_v<value_type>);

  data_t     __d(this->__data_.size());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __val_vec = vdupq_n_f32(reinterpret_cast<_f32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec = vld1q_f32(reinterpret_castr<const _f32*>(&this->__data_[__i]));
      neon_f32 __res = vaddq_f32(__vec, __val_vec);

      vst1q_f32(&__d[__i], __res);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(reinterpret_cast<_s32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_castr<const _s32*>(&this->__data_[__i]));
      neon_s32 __res = vaddq_s32(__vec, __val_vec);

      vst1q_s32(&__d[__i], __res);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __val_vec = vdupq_n_u32(reinterpret_cast<_u32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_castr<const _u32*>(&this->__data_[__i]));
      neon_u32 __res = vaddq_u32(__vec, __val_vec);

      vst1q_u32(&__d[__i], __res);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    __d[__i] = this->__data_[__i] + __val;

  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator*(const value_type __val) const {
  static_assert(has_times_operator_v<value_type>);
  data_t     __d(this->__data_.size());
  index_type __i = 0;

  for (; __i < this->__data_.size(); __i++)
    __d[__i] = this->__data_[__i] + __val;

  return __self(this->__shape_, __d);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator*(const tensor& __other) const {
  static_assert(has_times_operator_v<value_type>);
  assert(this->__shape_ == __other.shape());
  data_t     __d(this->__data_.size());
  index_type __i = 0;

  for (; __i < this->__data_.size(); __i++)
    __d[__i] = this->__data_[__i] * __other[__i];

  return __self(this->__shape_, __d);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator+=(const tensor& __other) const {
  static_assert(has_plus_operator_v<value_type>);

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)

#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] += __other[__i];

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator+=(const_reference __val) const {
  static_assert(has_plus_operator_v<value_type>);
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __val_vec = vdupq_n_f32(reinterpret_cast<_f32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __add_vec  = vaddq_f32(__data_vec, __val_vec);

      vst1q_f32(&this->__data_[__i], __add_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(reinterpret_cast<_s32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __add_vec  = vaddq_s32(__data_vec, __val_vec);

      vst1q_s32(&this->__data_[__i], __add_vec);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __val_vec = vdupq_n_u32(reinterpret_cast<_u32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __add_vec  = vaddq_u32(__data_vec, __val_vec);

      vst1q_u32(&this->__data_[__i], __add_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = this->__data_[__i] + __val;

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const tensor& __other) const {
  static_assert(has_minus_operator_v<value_type>);

  if (__other.shape() != this->__shape_)
    throw std::invalid_argument("Cannot add two tensors with different shapes");

  data_t     __d(this->__data_.size());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __oth = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __sub = vsubq_f32(__vec, __oth);

      vst1q_f32(&__d[__i], __sub);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __oth = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __sub = vsubq_s32(__vec, __oth);

      vst1q_s32(&__d[__i], __sub);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __oth = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __sub = vsubq_u32(__vec, __oth);

      vst1q_u32(&__d[__i], __sub);
    }
  }
#endif

  for (; __i < this->__data_[__i]; __i++)
    __d[__i] = this->__data_[__i] - __other[__i];

  return __self(this->__shape_, __d);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const value_type __val) const {
  static_assert(has_minus_operator_v<value_type>);
  data_t     __d(this->__data_.size());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __vals = vdupq_n_f32(reinterpret_cast<_f32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __sub = vsubq_f32(__vec, __vals);

      vst1q_f32(&__d[__i], __sub);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __vals = vdupq_n_s32(reinterpret_cast<_s32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __sub = vsubq_s32(__vec, __vals);

      vst1q_s32(&__d[__i], __sub);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __vals = vdupq_n_u32(reinterpret_cast<_u32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __sub = vsubq_u32(__vec, __vals);

      vst1q_u32(&__d[__i], __sub);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    __d[__i] = this->__data_[__i] - __val;

  return __self(*this);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator-=(const tensor& __other) const {
  static_assert(has_minus_operator_v<value_type>);
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __oth = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __sub = vsubq_f32(__vec, __oth);

      vst1q_f32(&this->__data_[__i], __sub);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __oth = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __sub = vsubq_s32(__vec, __oth);

      vst1q_s32(&this->__data_[__i], __sub);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __oth = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __sub = vsubq_u32(__vec, __oth);

      vst1q_u32(&this->__data_[__i], __sub);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] -= __other[__i];

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator*=(const tensor& __other) const {
  static_assert(has_times_operator_v<value_type>);
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __oth = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __mul = vmulq_f16(__vec, __oth);

      vst1q_f32(&this->__data_[__i], __mul);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __oth = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __mul = vmulq_s16(__vec, __oth);

      vst1q_s32(&this->__data_[__i], __mul);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __oth = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __mul = vmulq_u16(__vec, __oth);

      vst1q_u32(&this->__data_[__i], __mul);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] *= __other[__i];

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator/(const_reference __val) const {
  static_assert(has_divide_operator_v<value_type>);

  if (__val == value_type(0))
    throw std::invalid_argument("Cannot divide by zero : undefined operation");

  data_t     __d(this->__data_.size());
  index_type __i = 0;

  for (; __i < this->__data_.size(); __i++)
    __d[__i] = this->__data_[__i] / __val;

  return __self(this->__shape_, __d);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator*=(const_reference __val) const {
  static_assert(has_times_operator_v<value_type>);
  index_type __i = 0;

#if defined(__ARM_NEON)

#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] *= __val;

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator/=(const tensor& __other) const {
  static_assert(has_divide_operator_v<value_type>);
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] /= __other[__i];

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator/=(const_reference __val) const {
  static_assert(has_divide_operator_v<value_type>);
  index_type __i = 0;

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] /= __val;

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator/(const tensor& __other) const {
  static_assert(has_divide_operator_v<value_type>);
  
  if (__other.count_nonzero() != __other.size(0))
    throw std::invalid_argument("Cannot divide by zero : undefined operation");

  assert(this->__shape_ == __other.shape());

  data_t     __d(this->__data_.size());
  index_type __i = 0;

  for (; __i < this->__data_.size(); __i++)
    __d[__i] = this->__data_[__i] / __other[__i];

  return __self(this->__shape_, __d);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator-=(const_reference __val) const {
  static_assert(has_minus_operator_v<value_type>);
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __val = vld1q_f32(reinterpret_cast<const _f32*>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __sub = vsubq_f32(__vec, __val);

      vst1q_f32(&this->__data_[__i], __sub);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __val = vld1q_u32(reinterpret_cast<const _u32*>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __sub = vsubq_u32(__vec, __val);

      vst1q_u32(&this->__data_[__i], __sub);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __val = vld1q_s32(reinterpret_cast<const _s32*>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __sub = vsubq_s32(__vec, __val);

      vst1q_s32(&this->__data_[__i], __sub);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] -= __val;

  return *this;
}

template<class _Tp>
bool tensor<_Tp>::operator==(const tensor& __other) const {
  if ((this->__shape_ != __other.shape()) && (this->__strides_ != __other.strides()))
    return false;

  return this->__data_ == __other.storage();
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator=(tensor&& __other) const noexcept {
  if (this != &__other)
  {
    this->__data_    = std::move(__other.storage());
    this->__shape_   = std::move(__other.shape());
    this->__strides_ = std::move(__other.strides());
  }
  return *this;
}

template<class _Tp>
tensor<bool>& tensor<_Tp>::operator!() const {
  static_assert(std::is_same<value_type, bool>::value);

#if defined(__ARM_NEON)
#endif
  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = !(this->__data_[__i]);

  return *this;
}