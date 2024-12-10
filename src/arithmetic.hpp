#pragma once

#include "tensorbase.hpp"


template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmax(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fmax_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmax(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fmax_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const value_type __val) {

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const index_type __simd_end   = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    neon_f32         __scalar_val = vdupq_n_f32(__val);

    for (index_type __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __a       = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __max_val = vmaxq_f32(__a, __scalar_val);

      vst1q_f32(&this->__data_[__i], __max_val);
    }

    for (index_type __i = __simd_end; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = std::fmax(this->__data_[__i], __val);
    }
  }
#else
  std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                 [&__val](const_reference __v) { return std::fmax(__v, __val); });
#endif

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const tensor<value_type>& __other) {
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __a       = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b       = vld1q_f32(reinterpret_cast<const _f32*>(&(__other[__i])));
      neon_f32 __max_val = vmaxq_f32(__a, __b);

      vst1q_f32(&this->__data_[__i], __max_val);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = std::fmax(this->__data_[__i], __other[__i]);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmod(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fmod_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmod(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fmod_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmod_(const value_type __val) const {
  assert(std::is_floating_point<value_type>::value && "fmod : template class must be a floating point type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() - _ARM64_REG_WIDTH);
    neon_f32         __b        = vdupq_n_f32(reinterpret_cast<_f32>(__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __a         = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __div       = vdivq_f32(__a, __b);
      neon_f32 __floor_div = vrndq_f32(__div);
      neon_f32 __mult      = vmulq_f32(__floor_div, __b);
      neon_f32 __mod       = vsubq_f32(__a, __mult);

      vst1q_f32(&this->__data_[__i], __mod);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] =
      static_cast<value_type>(std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__val)));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmod_(const tensor& __other) const {
  this->__check_is_scalar_type("Cannot divide non scalar values");

  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
  {
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");
  }

  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __a         = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b         = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __div       = vdivq_f32(__a, __b);
      neon_f32 __floor_div = vrndq_f32(__div);
      neon_f32 __mult      = vmulq_f32(__floor_div, __b);
      neon_f32 __mod       = vsubq_f32(__a, __mult);

      vst1q_f32(&this->__data_[__i], __mod);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] =
      static_cast<value_type>(std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::frac_() const {
  this->__check_is_scalar_type("Cannot get the fraction of a non-scalar type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_same_v<value_type, _f32>)
  {
    index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = this->__frac(__vals[0]);
      __vals[1] = this->__frac(__vals[1]);
      __vals[2] = this->__frac(__vals[2]);
      __vals[3] = this->__frac(__vals[3]);

      neon_f32 __atan_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __atan_vec);
    }
  }
  if constexpr (std::is_same_v<value_type, _f64>)
  {
    index_type __simd_end = this->__data_.size() - (this->__data_.size() % (_ARM64_REG_WIDTH / 2));

    for (; __i < __simd_end; __i += (_ARM64_REG_WIDTH / 2))
    {
      neon_f64 __data_vec = vld1q_f64(reinterpret_cast<const _f64*>(&this->__data_[__i]));
      _f64     __vals[_ARM64_REG_WIDTH];
      vst1q_f64(__vals, __data_vec);

      __vals[0] = static_cast<_f64>(this->__frac(__vals[0]));
      __vals[1] = static_cast<_f64>(this->__frac(__vals[1]));
      __vals[2] = static_cast<_f64>(this->__frac(__vals[2]));
      __vals[3] = static_cast<_f64>(this->__frac(__vals[3]));

      neon_f64 __atan_vec = vld1q_f64(__vals);
      vst1q_f64(&this->__data_[__i], __atan_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::frac() const {
  __self __ret = this->clone();
  __ret.frac_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::log_() const {
  this->__check_is_integral_type("Given data type must be an integral");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::log(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::log(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::log(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::log(static_cast<_f32>(__vals[3])));

    neon_type __log_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __log_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::log(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log() const {
  __self __ret = this->clone();
  __ret.log_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::log10_() const {
  this->__check_is_integral_type("Given data type must be an integral");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::log10(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::log10(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::log10(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::log10(static_cast<_f32>(__vals[3])));

    neon_type __log_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __log_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::log10(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log10() const {
  __self __ret = this->clone();
  __ret.log10_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::log2_() const {
  this->__check_is_integral_type("Given data type must be an integral");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::log2(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::log2(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::log2(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::log2(static_cast<_f32>(__vals[3])));

    neon_type __log2_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __log2_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::log2(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log2() const {
  __self __ret = this->clone();
  __ret.log2_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::exp_() const {
  this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::exp(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::exp(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::exp(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::exp(static_cast<_f32>(__vals[3])));

    neon_type __exp_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __exp_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::exp(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::exp() const {
  __self __ret = this->clone();
  __ret.exp_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sqrt_() const {
  this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::sqrt(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::sqrt(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::sqrt(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::sqrt(static_cast<_f32>(__vals[3])));

    neon_type __sqrt_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __sqrt_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::sqrt(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sqrt() const {
  __self __ret = this->clone();
  __ret.sqrt_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::square() const {
  __self __ret = this->clone();
  __ret.square_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::cos_() const {
  this->__check_is_scalar_type("Cannot perform a cosine on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::cos(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::cos(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::cos(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::cos(static_cast<_f32>(__vals[3])));

    neon_type __cos_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __cos_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::cos(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::acos_() const {
  this->__check_is_scalar_type("Cannot perform a acos on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::acos(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::acos(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::acos(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::acos(static_cast<_f32>(__vals[3])));

    neon_type __cos_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __cos_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::acos(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::acos() const {
  __self __ret = this->clone();
  __ret.acos_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cos() const {
  __self __ret = this->clone();
  __ret.cos_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sin_() const {
  this->__check_is_integral_type("Cannot perform a sin on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::sin(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::sin(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::sin(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::sin(static_cast<_f32>(__vals[3])));

    neon_type __sin_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __sin_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::sin(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sin() const {
  __self __ret = this->clone();
  __ret.sin_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::tan_() const {
  this->__check_is_integral_type("template class must be integral type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<_f32, value_type>, neon_f32, neon_s32>::type;

  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::tan(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::tan(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::tan(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::tan(static_cast<_f32>(__vals[3])));

    neon_type __tan_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __tan_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::tan(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::tanh_() const {
  this->__check_is_integral_type("template class must be integral type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<_f32, value_type>, neon_f32, neon_s32>::type;

  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::tanh(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::tanh(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::tanh(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::tanh(static_cast<_f32>(__vals[3])));

    neon_type __tanh_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __tanh_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::tanh(static_cast<_f32>(this->__data_[__i])));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::asin() const {
  __self __ret = this->clone();
  __ret.asin_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cosh() const {
  __self __ret = this->clone();
  __ret.cosh_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::atan() const {
  __self __ret = this->clone();
  __ret.atan_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sinc_() const {
  this->__check_is_arithmetic_type("sinc_: template type must be an arithmetic type");

  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_same<value_type, _f32>::value)
  {
    using neon_type             = neon_f32;
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_type __v      = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_type __pi_v   = vmulq_f32(__v, vdupq_n_f32(M_PI));                        // pi * x
      neon_type __sinc_v = vbslq_f32(vcgeq_f32(vabsq_f32(__v), vdupq_n_f32(1e-6f)),  // Check |x| > epsilon
                                     vdivq_f32(vsinq_f32(__pi_v), __pi_v),           // sinc(x) = sin(pi * x) / (pi * x)
                                     vdupq_n_f32(1.0f));                             // sinc(0) = 1

      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __sinc_v);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    value_type x = this->__data_[__i];
    this->__data_[__i] =
      (std::abs(x) < 1e-6) ? static_cast<value_type>(1.0) : static_cast<value_type>(std::sin(M_PI * x) / (M_PI * x));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::atan_() const {
  this->__check_is_integral_type("template class must be integral type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type       = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::atan(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::atan(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::atan(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::atan(static_cast<_f32>(__vals[3])));

    neon_type __atan_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __atan_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::atanh_() const {
  this->__check_is_integral_type("template class must be integral type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type       = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::atanh(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::atanh(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::atanh(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::atanh(static_cast<_f32>(__vals[3])));

    neon_type __atan_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __atan_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::atanh() const {
  __self __ret = this->clone();
  __ret.atanh_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sinc() const {
  __self __ret = this->clone();
  __ret.sinc_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sinh_() const {
  this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type             = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::sinh(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::sinh(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::sinh(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::sinh(static_cast<_f32>(__vals[3])));

    neon_type __sinh_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __sinh_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::sinh(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sinh() const {
  __self __ret = this->clone();
  __ret.sinh_();
  return __ret;
}


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::asinh_() const {
  this->__check_is_scalar_type("Cannot perform asinh on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  using neon_type       = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::asinh(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::asinh(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::asinh(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::asinh(static_cast<_f32>(__vals[3])));

    neon_type __asinh_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __asinh_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::asinh(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::asinh() const {
  __self __ret = this->clone();
  __ret.asinh_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::asin_() const {
  this->__check_is_scalar_type("Cannot perform asin on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type             = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::asin(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::asin(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::asin(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::asin(static_cast<_f32>(__vals[3])));

    neon_type __asin_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __asin_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::asin(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::cosh_() const {
  this->__check_is_scalar_type("Cannot perform a cosh on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type             = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::cosh(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::cosh(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::cosh(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::cosh(static_cast<_f32>(__vals[3])));

    neon_type __cosh_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __cosh_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::cosh(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::acosh_() const {
  this->__check_is_scalar_type("Cannot perform a acosh on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type             = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::acosh(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::acosh(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::acosh(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::acosh(static_cast<_f32>(__vals[3])));

    neon_type __acosh_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __acosh_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::acosh(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::acosh() const {
  __self __ret = this->clone();
  __ret.acosh_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::pow_(const value_type __val) {
  this->__check_is_integral_type("cannot get the power of a non-integral value");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type             = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::pow(static_cast<_f32>(__vals[0]), static_cast<_f32>(__val)));
    __vals[1] = static_cast<value_type>(std::pow(static_cast<_f32>(__vals[1]), static_cast<_f32>(__val)));
    __vals[2] = static_cast<value_type>(std::pow(static_cast<_f32>(__vals[2]), static_cast<_f32>(__val)));
    __vals[3] = static_cast<value_type>(std::pow(static_cast<_f32>(__vals[3]), static_cast<_f32>(__val)));

    neon_type __pow_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __pow_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::pow(this->__data_[__i], __val));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::pow(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.pow_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::pow(const value_type __val) const {
  __self __ret = this->clone();
  __ret.pow_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::pow_(const tensor& __other) {
  this->__check_is_integral_type("cannot get the power of a non integral value");
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  neon_type __base_vec   = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
  neon_type __exp_vec    = vld1q(reinterpret_cast<const value_type*>(&__other[__i]));
  neon_type __result_vec = {static_cast<value_type>(std::pow(static_cast<_f32>(vget_lane(__base_vec, 0)),
                                                             static_cast<_f32>(vget_lane(__exp_vec, 0)))),
                            static_cast<value_type>(std::pow(static_cast<_f32>(vget_lane(__base_vec, 1)),
                                                             static_cast<_f32>(vget_lane(__exp_vec, 1)))),
                            static_cast<value_type>(std::pow(static_cast<_f32>(vget_lane(__base_vec, 2)),
                                                             static_cast<_f32>(vget_lane(__exp_vec, 2)))),
                            static_cast<value_type>(std::pow(static_cast<_f32>(vget_lane(__base_vec, 3)),
                                                             static_cast<_f32>(vget_lane(__exp_vec, 3))))};

  vst1q(&this->__data_[__i], __result_vec);
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] =
      static_cast<value_type>(std::pow(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::abs() const {
  __self __ret = this->clone();
  __ret.abs_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::abs_() const {
  this->__check_is_integral_type("template class must be integral type");
  index_type __i = 0;

  if (std::is_unsigned<value_type>::value)
  {
    return *this;
  }

#if defined(__ARM_NEON)
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  using neon_type       = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<neon_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::abs(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::abs(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::abs(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::abs(static_cast<_f32>(__vals[3])));

    neon_type __abs_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __abs_vec);
  }

#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::abs(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log_softmax(const index_type __dim) const {
  __self __ret = this->clone();
  __ret.log_softmax_(__dim);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::dist(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.dist_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::dist(const value_type __val) const {
  __self __ret = this->clone();
  __ret.dist_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::dist_(const tensor& __other) const {
  this->__check_is_arithmetic_type("dist: template type must be an arithmetic type");

  assert(this->__shape_ == __other.shape());

  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type __a    = vld1q(reinterpret_cast<const neon_type*>(&this->__data_[__i]));
    neon_type __b    = vld1q(reinterpret_cast<const neon_type*>(&__other.__data_[__i]));
    neon_type __diff = vabdq(__a, __b);
    vst1q(reinterpret_cast<neon_type*>(&this->__data_[__i]), __diff);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] =
      static_cast<value_type>(std::abs(static_cast<_f64>(this->__data_[__i] - __other.__data_[__i])));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::dist_(const value_type __val) const {
  this->__check_is_arithmetic_type("dist: template type must be an arithmetic type");

  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type __a    = vld1q(reinterpret_cast<const neon_type*>(&this->__data_[__i]));
    neon_type __b    = vdupq(reinterpret_cast<const neon_type*>(__val));
    neon_type __diff = vabdq(__a, __b);
    vst1q(reinterpret_cast<neon_type*>(&this->__data_[__i]), __diff);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::abs(static_cast<_f64>(this->__data_[__i] - __val)));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::remainder(const value_type __val) const {
  __self __ret = this->clone();
  __ret.remainder_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::remainder(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.remainder_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::remainder_(const value_type __val) const {
  this->__check_is_arithmetic_type("remainder_: template type must be an arithmetic type");
  assert(__val != 0 && "Remainder by zero is undefined");

  for (index_type __i = 0; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] %= __val;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::remainder_(const tensor& __other) const {
  this->__check_is_arithmetic_type("remainder_: template type must be an arithmetic type");

  assert(this->__shape_ == __other.shape());

  for (index_type __i = 0; __i < this->__data_.size(); __i++)
  {
    assert(__other.__data_[__i] != 0 && "Remainder by zero is undefined");
    this->__data_[__i] %= __other.__data_[__i];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::maximum(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.maximum_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::maximum(const value_type& __val) const {
  __self __ret = this->clone();
  __ret.maximum_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::maximum_(const tensor& __other) const {
  this->__check_is_arithmetic_type("maximum_: template type must be an arithmetic type");

  assert(this->__shape_ == __other.shape());

  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type __a   = vld1q(reinterpret_cast<const neon_type*>(&this->__data_[__i]));
    neon_type __b   = vld1q(reinterpret_cast<const neon_type*>(&__other.__data_[__i]));
    neon_type __max = vmaxq(__a, __b);
    vst1q(reinterpret_cast<neon_type*>(&this->__data_[__i]), __max);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = std::max(this->__data_[__i], __other.__data_[__i]);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::maximum_(const value_type __val) const {
  this->__check_is_arithmetic_type("maximum_: template type must be an arithmetic type");

  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  neon_type        __val_vec  = vdupq_n(__val);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type __a   = vld1q(reinterpret_cast<const neon_type*>(&this->__data_[__i]));
    neon_type __max = vmaxq(__a, __val_vec);
    vst1q(reinterpret_cast<neon_type*>(&this->__data_[__i]), __max);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = std::max(this->__data_[__i], __val);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::square_() const {
  return this->pow_(static_cast<value_type>(2.0f));
}


template<class _Tp>
double tensor<_Tp>::mean() const {
  this->__check_is_integral_type("Input must be of integral type to calculate the mean.");

  double __m = 0.0;

  if (this->empty())
  {
    return __m;
  }

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  neon_s32         __sum_vec  = vdupq_n_s32(0);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
    __sum_vec           = vaddq_s32(__sum_vec, __data_vec);
  }

  _s32 __partial_sum[4];
  vst1q_s32(__partial_sum, __sum_vec);
  __m += __partial_sum[0] + __partial_sum[1] + __partial_sum[2] + __partial_sum[3];
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __m += this->__data_[__i];
  }

  return static_cast<double>(__m) / static_cast<double>(this->__data_.size());
}


// used as a helper function
int64_t __lcm(const int64_t __a, const int64_t __b) { return (__a * __b) / std::gcd(__a, __b); }


template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::lcm() const {
  this->__check_is_scalar_type("Given template type must be an int");

  index_type __ret = static_cast<index_type>(this->__data_[0]);
  index_type __i   = 1;

  for (; __i < this->__data_.size(); __i++)
  {
    __ret = __lcm(static_cast<index_type>(this->__data_[__i]), __ret);
  }

  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::tanh() const {
  __self __ret = this->clone();
  __ret.tanh_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::tan() const {
  __self __ret = this->clone();
  __ret.tan_();
  return __ret;
}
