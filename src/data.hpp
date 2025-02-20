#pragma once

#include "tensorbase.hpp"


template<class _Tp>
size_t tensor<_Tp>::n_dims() const noexcept {
  return this->__shape_.size();
}

template<class _Tp>
typename tensor<_Tp>::shape_type tensor<_Tp>::shape() const noexcept {
  return this->__shape_;
}

template<class _Tp>
typename tensor<_Tp>::data_t tensor<_Tp>::storage() const noexcept {
  return this->__data_;
}

template<class _Tp>
typename tensor<_Tp>::shape_type tensor<_Tp>::strides() const noexcept {
  return this->__strides_;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::reshape_as(const tensor& __other) const {
  return this->reshape(__other.__shape_);
}

template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::capacity() const noexcept {
  return this->__data_.capacity();
}

template<class _Tp>
bool tensor<_Tp>::empty() const {
  return this->__data_.empty();
}

template<class _Tp>
typename tensor<_Tp>::iterator tensor<_Tp>::begin() noexcept {
  return this->__data_.begin();
}

template<class _Tp>
typename tensor<_Tp>::const_iterator tensor<_Tp>::begin() const noexcept {
  return this->__data_.begin();
}

template<class _Tp>
typename tensor<_Tp>::iterator tensor<_Tp>::end() noexcept {
  return this->__data_.end();
}

template<class _Tp>
typename tensor<_Tp>::const_iterator tensor<_Tp>::end() const noexcept {
  return this->__data_.end();
}

template<class _Tp>
typename tensor<_Tp>::reverse_iterator tensor<_Tp>::rbegin() noexcept {
  return this->__data_.rbegin();
}

template<class _Tp>
typename tensor<_Tp>::const_reverse_iterator tensor<_Tp>::rbegin() const noexcept {
  return this->__data_.rbegin();
}

template<class _Tp>
typename tensor<_Tp>::reverse_iterator tensor<_Tp>::rend() noexcept {
  return this->__data_.rend();
}

template<class _Tp>
typename tensor<_Tp>::const_reverse_iterator tensor<_Tp>::rend() const noexcept {
  return this->__data_.rend();
}

template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::size(const index_type __dim) const {
  if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
    throw std::invalid_argument("dimension input is out of range");

  if (this->__data_.empty())
    return 0;

  if (__dim == 0)
    return this->__data_.size();
    
  return this->__shape_[__dim - 1];
}

template<class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::at(tensor<_Tp>::shape_type __idx) {
  if (__idx.empty())
    throw std::invalid_argument("Passing an empty vector as indices for a tensor");

  index_type __i = this->__compute_index(__idx);
  if (__i < 0 || __i >= this->__data_.size())
    throw std::invalid_argument("input indices are out of bounds");

  return this->__data_[__i];
}

template<class _Tp>
typename tensor<_Tp>::const_reference tensor<_Tp>::at(const tensor<_Tp>::shape_type __idx) const {
  if (__idx.empty())
    throw std::invalid_argument("Passing an empty vector as indices for a tensor");

  index_type __i = this->__compute_index(__idx);

  if (__i < 0 || __i >= this->__data_.size())
    throw std::invalid_argument("input indices are out of bounds");

  return this->__data_[__i];
}

template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::count_nonzero(index_type __dim) const {
  this->__check_is_scalar_type("Cannot compare a non-scalar value to zero");
  index_type __c = 0;

  if (__dim == -1)
  {

#pragma omp parallel
    {
      index_type __local_count = 0;

#ifdef __AVX__
      if constexpr (std::is_same_v<value_type, _f32>)
      {
        index_type __size = this->__data_.size();
        index_type __i    = 0;

        for (; __i + _AVX_REG_WIDTH <= __size; __i += _AVX_REG_WIDTH)
        {
          __m256 __vec          = _mm256_loadu_ps(&this->__data_[__i]);
          __m256 __nonzero_mask = _mm256_cmp_ps(__vec, _mm256_setzero_ps(), _CMP_NEQ_OQ);
          __local_count += _mm256_movemask_ps(__nonzero_mask);
        }
      }

#endif
      index_type __i = 0;

#if defined(__ARM_NEON)
      if constexpr (std::is_floating_point<value_type>::value)
      {
        index_type __size = this->__data_.size();

        for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
        {
          neon_f32 __vec          = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
          neon_u32 __nonzero_mask = vcgtq_f32(__vec, vdupq_n_f32(0.0f));
          __local_count += vaddvq_u32(__nonzero_mask);
        }
      }
      else if constexpr (std::is_unsigned<value_type>::value)
      {
        index_type __size = this->__data_.size();

        for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
        {
          neon_u32 __vec          = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
          neon_u32 __nonzero_mask = vcgtq_u32(__vec, vdupq_n_u32(0));
          __local_count += vaddvq_u32(__nonzero_mask);
        }
      }
      else if constexpr (std::is_signed<value_type>::value)
      {
        index_type __size = this->__data_.size();

        for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
        {
          neon_s32 __vec          = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
          neon_s32 __nonzero_mask = vcgtq_s32(__vec, vdupq_n_s32(0));
          __local_count += vaddvq_s32(__nonzero_mask);
        }
      }

#endif
      for (index_type __j = __i; __j < this->__data_.size(); __j++)
      {
        if (this->__data_[__j] != 0)
          __local_count++;
      }

#pragma omp atomic
      __c += __local_count;
    }
  }
  else
  {
    if (__dim < 0 || __dim >= static_cast<index_type>(__shape_.size()))
      throw std::invalid_argument("Invalid dimension provided.");

    throw std::runtime_error("Dimension-specific non-zero counting is not implemented yet.");
  }

  return __c;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::zeros(const shape_type& __sh) {
  __self __ret = this->clone();
  __ret.zeros_(__sh);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::zeros_(shape_type __sh) {
  if (__sh.empty())
    __sh = this->__shape_;
  else
    this->__shape_ = __sh;

  size_t __s = this->__computeSize(this->__shape_);

  this->__data_.resize(__s);
  this->__compute_strides();

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = __s - (__s % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __zero_vec = vdupq_n_f32(0.0f);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
      vst1q_f32(&this->__data_[__i], __zero_vec);
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __zero_vec = vdupq_n_s32(0);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
      vst1q_s32(&this->__data_[__i], __zero_vec);
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __zero_vec = vdupq_n_u32(0);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
      vst1q_u32(&this->__data_[__i], __zero_vec);
  }
#endif

  for (; __i < __s; __i++)
    this->__data_[__i] = value_type(0.0);

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::ones_(shape_type __sh) {
  if (__sh.empty())
    __sh = this->__shape_;
  else
    this->__shape_ = __sh;

  size_t __s = this->__computeSize(this->__shape_);

  this->__data_.resize(__s);
  this->__compute_strides();
  this->__check_is_scalar_type("template type must be a scalar : tensor.ones()");

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = __s - (__s % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __one_vec = vdupq_n_f32(1.0f);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __one_vec);
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __one_vec = vdupq_n_s32(1);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __one_vec);
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __one_vec = vdupq_n_u32(1);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __one_vec);
  }
#endif

  for (; __i < __s; __i++)
    this->__data_[__i] = value_type(1.0);

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::ones(const shape_type& __sh) {
  __self __ret = this->clone();
  __ret.ones_(__sh);
  return __ret;
}

template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::hash() const {
  index_type            __hash_val = 0;
  std::hash<value_type> __hasher;

  index_type __i = 0;

  for (; __i < this->__data_.size(); __i++)
    __hash_val ^= __hasher(this->__data_[__i]) + 0x9e3779b9 + (__hash_val << 6) + (__hash_val >> 2);

  return __hash_val;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::row(const index_type __index) const {
  if (this->__shape_.size() != 2)
    throw std::runtime_error("Cannot get a row from a non two dimensional tensor");

  if (this->__shape_[0] <= __index || __index < 0)
    throw std::invalid_argument("Index input is out of range");

  data_t     __r;
  index_type __start = this->__shape_[1] * __index;
  index_type __end   = this->__shape_[1] * __index + this->__shape_[1];
  index_type __i     = __start;

  for (; __i < __end; __i++)
    __r.push_back(this->__data_[__i]);

  return __self(__r, {this->__shape_[1]});
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::col(const index_type __index) const {
  if (this->__shape_.size() != 2)
    throw std::runtime_error("Cannot get a column from a non two dimensional tensor");

  if (this->__shape_[1] <= __index || __index < 0)
    throw std::invalid_argument("Index input out of range");

  data_t     __c;
  index_type __i = 0;

  for (; __i < this->__shape_[0]; __i++)
    __c.push_back(this->__data_[this->__compute_index({__i, __index})]);

  return __self(__c, {this->__shape_[0]});
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::view(std::initializer_list<index_type> __sh) {
  index_type __s = this->__computeSize(__sh);

  if (__s != this->__data_.size())
    throw std::invalid_argument("Total elements do not match for new shape");

  this->__shape_ = __sh;
  this->__compute_strides();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::randomize(const shape_type& __sh, bool __bounded) {
  __self __ret = this->clone();
  __ret.randomize_(__sh, __bounded);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::randomize_(const shape_type& __sh, bool __bounded) {
  if (__bounded)
    assert(std::is_floating_point<value_type>::value && "Cannot bound non floating point data type");

  if (__sh.empty() && this->__shape_.empty())
    throw std::invalid_argument("randomize_ : Shape must be initialized");

  if (this->__shape_.empty() || this->__shape_ != __sh)
    this->__shape_ = __sh;

  index_type __s = this->__computeSize(this->__shape_);
  this->__data_.resize(__s);
  this->__compute_strides();

  std::random_device                   __rd;
  std::mt19937                         __gen(__rd());
  std::uniform_real_distribution<_f32> __unbounded_dist(1.0f, static_cast<_f32>(RAND_MAX));
  std::uniform_real_distribution<_f32> __bounded_dist(0.0f, 1.0f);
  index_type                           __i = 0;

#if defined(__AVX__)
  const __m256 __scale = _mm256_set1_ps(__bounded ? static_cast<_f32>(RAND_MAX) : 1.0f);
  for (; __i + _AVX_REG_WIDTH <= static_cast<index_type>(__s); __i += _AVX_REG_WIDTH)
  {
    __m256 __random_values =
      _mm256_setr_ps(__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen),
                     __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen));

    if (!__bounded)
      __random_values = _mm256_div_ps(__random_values, __scale);

    _mm256_storeu_ps(&this->__data_[__i], __random_values);
  }

#elif defined(__SSE__)
  const __m128 __scale = _mm_set1_ps(__bounded ? static_cast<_f32>(RAND_MAX) : 1.0f);
  for (; __i + 4 <= static_cast<index_type>(__s); __i += 4)
  {
    __m128 __random_values =
      _mm_setr_ps(__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen));

    if (!__bounded)
      __random_values = _mm_div_ps(__random_values, __scale);

    _mm_storeu_ps(&this->__data_[__i], __random_values);
  }

#elif defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const neon_f32 __scale = vdupq_n_f32(__bounded ? static_cast<_f32>(RAND_MAX) : 1.0f);
    for (; __i + _ARM64_REG_WIDTH <= static_cast<index_type>(__s); __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __random_values;

      if (__bounded)
        __random_values = {__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen)};
      else
        __random_values = {__unbounded_dist(__gen), __unbounded_dist(__gen), __unbounded_dist(__gen),
                           __unbounded_dist(__gen)};

      if (!__bounded)
        __random_values = vmulq_f32(__random_values, vrecpeq_f32(__scale));

      vst1q_f32(&this->__data_[__i], __random_values);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    const neon_f32 __scale = vdupq_n_f32(static_cast<_f32>(RAND_MAX));
    for (; __i + _ARM64_REG_WIDTH <= static_cast<index_type>(__s); __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __rand_vals = {static_cast<_f32>(__unbounded_dist(__gen)), static_cast<_f32>(__unbounded_dist(__gen)),
                              static_cast<_f32>(__unbounded_dist(__gen)), static_cast<_f32>(__unbounded_dist(__gen))};
      __rand_vals          = vmulq_f32(__rand_vals, vrecpeq_f32(__scale));
      neon_u32 __int_vals  = vcvtq_u32_f32(__rand_vals);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __int_vals);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    const neon_f32 __scale = vdupq_n_f32(static_cast<_f32>(RAND_MAX));
    for (; __i + _ARM64_REG_WIDTH <= static_cast<index_type>(__s); __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __rand_vals = {static_cast<_f32>(__unbounded_dist(__gen)), static_cast<_f32>(__unbounded_dist(__gen)),
                              static_cast<_f32>(__unbounded_dist(__gen)), static_cast<_f32>(__unbounded_dist(__gen))};
      __rand_vals          = vmulq_f32(__rand_vals, vrecpeq_f32(__scale));
      neon_s32 __int_vals  = vcvtq_s32_f32(__rand_vals);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __int_vals);
    }
  }
#endif

  for (; __i < static_cast<index_type>(__s); __i++)
    this->__data_[__i] = value_type(__bounded ? __bounded_dist(__gen) : __unbounded_dist(__gen));

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::clone() const {
  data_t     __d = this->__data_;
  shape_type __s = this->__shape_;
  return __self(__d, __s);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::negative_() const {
  this->__check_is_arithmetic_type("negative_: template type must be an arithmetic type");

  index_type __i = 0;

#if defined(__ARM_NEON)

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>)
  {
    neon_f32 __neg_multiplier = vdupq_n_f32(-1);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __v   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __neg = vmulq_f32(__v, __neg_multiplier);
      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __neg);
    }
  }
  else if constexpr (std::is_same_v<value_type, _s32>)
  {
    neon_s32 __neg_multiplier = vdupq_n_s32(-1);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __v   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __neg = vmulq_s32(__v, __neg_multiplier);
      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __neg);
    }
  }
  else if constexpr (std::is_same_v<value_type, _u32>)
  {
    neon_s32 __neg_multiplier = vdupq_n_s32(-1);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __v   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __neg = vmulq_u32(__v, __neg_multiplier);
      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __neg);
    }
  }

#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = -this->__data_[__i];

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::negative() const {
  __self __ret = this->clone();
  __ret.negative_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::permute_(const index_type __dim) const {
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::permute(const index_type __dim) const {}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::repeat_(const data_t& __d) const {
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::repeat(const data_t& __d) const {
  __self __ret = this->clone();
  __ret.repeat_(__d);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sort(index_type __dim, bool __descending) const {}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::fill(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fill_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fill(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fill_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::resize_as(const shape_type __sh) const {
  __self __ret = this->clone();
  __ret.resize_as_(__sh);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::all() const {
  this->__check_is_arithmetic_type("all: template type must be an arithmetic type");

  bool       __result = true;
  index_type __i      = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    if (this->__data_[__i] == static_cast<value_type>(0))
    {
      __result = false;
      break;
    }
  }

  tensor __output;
  __output.__data_ = {__result ? static_cast<value_type>(1) : static_cast<value_type>(0)};

  return __output;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::any() const {
  this->__check_is_arithmetic_type("any: template type must be an arithmetic type");

  bool result = false;

  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
  {
    if (this->__data_[__i] != static_cast<value_type>(0))
    {
      result = true;
      break;
    }
  }

  tensor output;
  output.__data_ = {result ? static_cast<value_type>(1) : static_cast<value_type>(0)};

  return output;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::gcd(const tensor& __other) const {
  assert(this->__shape_ == __other.shape());

  tensor     __ret = this->clone();
  index_type __i   = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    index_type __gcd__ = static_cast<index_type>(this->__data_[__i] * __other[__i]);
    index_type __lcm__ = __lcm(static_cast<index_type>(this->__data_[__i]), static_cast<index_type>(__other[__i]));
    __gcd__ /= __lcm__;
    __ret[__i] = __gcd__;
  }

  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::gcd(const value_type __val) const {
  tensor     __ret = this->clone();
  index_type __i   = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    index_type __gcd__ = static_cast<index_type>(this->__data_[__i] * __val);
    index_type __lcm__ = __lcm(static_cast<index_type>(this->__data_[__i]), static_cast<index_type>(__val));
    __gcd__ /= __lcm__;
    __ret[__i] = __gcd__;
  }

  return __ret;
}