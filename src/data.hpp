#pragma once

#include "tensorbase.hpp"

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::reshape_as(const tensor& __other) const {
  return this->reshape(__other.shape());
}

template <class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::size(const index_type __dim) const {
  if (__dim < 0 || __dim > static_cast<index_type>(this->__shape_.size()))
    throw std::invalid_argument("dimension input is out of range");

  if (__dim == 0) return this->__data_.size();

  return this->__shape_[__dim - 1];
}

template <class _Tp>
inline typename tensor<_Tp>::reference tensor<_Tp>::at(tensor<_Tp>::shape_type __idx) {
  if (__idx.empty()) throw std::invalid_argument("Passing an empty vector as indices for a tensor");

  index_type __i = this->__compute_index(__idx);
  if (__i < 0 || __i >= this->__data_.size())
    throw std::invalid_argument("input indices are out of bounds");

  return this->__data_[__i];
}

template <class _Tp>
inline typename tensor<_Tp>::const_reference tensor<_Tp>::at(
    const tensor<_Tp>::shape_type __idx) const {
  if (__idx.empty()) throw std::invalid_argument("Passing an empty vector as indices for a tensor");

  index_type __i = this->__compute_index(__idx);

  if (__i < 0 || __i >= this->__data_.size())
    throw std::invalid_argument("input indices are out of bounds");

  return this->__data_[__i];
}

template <class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::count_nonzero(index_type __dim) const {
#if defined(__ARM_NEON)
  return this->neon_count_nonzero(__dim);
#endif
  index_type __c           = 0;
  index_type __local_count = 0;
  index_type __i           = 0;
  if (__dim == 0) {
#ifdef __AVX__
    if constexpr (std::is_same_v<value_type, _f32>) {
      index_type __size = this->__data_.size();
      index_type __i    = 0;

      for (; __i + _AVX_REG_WIDTH <= __size; __i += _AVX_REG_WIDTH) {
        __m256 __vec          = _mm256_loadu_ps(&this->__data_[__i]);
        __m256 __nonzero_mask = _mm256_cmp_ps(__vec, _mm256_setzero_ps(), _CMP_NEQ_OQ);
        __local_count += _mm256_movemask_ps(__nonzero_mask);
      }
    }

#endif

#pragma omp parallel for reduction(+ : __local_count)
    for (index_type __j = __i; __j < this->__data_.size(); ++__j)
      if (this->__data_[__j] != 0) ++__local_count;

    __c += __local_count;
  } else {
    if (__dim < 0 || __dim >= static_cast<index_type>(__shape_.size()))
      throw std::invalid_argument("Invalid dimension provided.");

    throw std::runtime_error("Dimension-specific non-zero counting is not implemented yet.");
  }

  return __c;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::push_back(value_type __v) const {
  if (this->__shape_.size() != 1)
    throw std::range_error("push_back is only supported for one dimensional tensors");

  this->__data_.push_back(__v);
  ++this->__shape_[0];
  this->__compute_strides();
  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::zeros(const shape_type& __sh) {
  __self __ret = this->clone();
  __ret.zeros_(__sh);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::zeros_(shape_type __sh) {
#if defined(__ARM_NEON)
  return this->neon_zeros_(__sh);
#endif
  if (__sh.empty())
    __sh = this->__shape_;
  else
    this->__shape_ = __sh;

  size_t __s = this->__computeSize(this->__shape_);
  this->__data_.resize(__s);
  this->__compute_strides();
  index_type __i = 0;
#pragma omp parallel
  for (; __i < __s; ++__i) this->__data_[__i] = value_type(0.0);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::zeros_(shape_type __sh) const {
#if defined(__ARM_NEON)
  return this->neon_zeros_(__sh);
#endif
  if (__sh.empty())
    __sh = this->__shape_;
  else
    this->__shape_ = __sh;

  size_t __s = this->__computeSize(this->__shape_);
  this->__data_.resize(__s);
  this->__compute_strides();
  index_type __i = 0;
#pragma omp parallel
  for (; __i < __s; ++__i) this->__data_[__i] = value_type(0.0);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::ones_(shape_type __sh) {
#if defined(__ARM_NEON)
  return this->neon_ones_(__sh);
#endif
  if (__sh.empty())
    __sh = this->__shape_;
  else
    this->__shape_ = __sh;

  size_t __s = this->__computeSize(this->__shape_);
  this->__data_.resize(__s);
  this->__compute_strides();
  index_type __i = 0;
#pragma omp parallel
  for (; __i < __s; ++__i) this->__data_[__i] = value_type(1.0);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::ones_(shape_type __sh) const {
#if defined(__ARM_NEON)
  return this->neon_ones_(__sh);
#endif
  if (__sh.empty())
    __sh = this->__shape_;
  else
    this->__shape_ = __sh;

  size_t __s = this->__computeSize(this->__shape_);
  this->__data_.resize(__s);
  this->__compute_strides();
  index_type __i = 0;
#pragma omp parallel
  for (; __i < __s; ++__i) this->__data_[__i] = value_type(1.0);

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::ones(const shape_type& __sh) {
  __self __ret = this->clone();
  __ret.ones_(__sh);
  return __ret;
}

template <class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::hash() const {
  index_type            __hash_val = 0;
  std::hash<value_type> __hasher;

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    __hash_val ^= __hasher(this->__data_[__i]) + 0x9e3779b9 + (__hash_val << 6) + (__hash_val >> 2);

  return __hash_val;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::row(const index_type __index) const {
  if (this->__shape_.size() != 2)
    throw std::runtime_error("Cannot get a row from a non two dimensional tensor");

  if (this->__shape_[0] <= __index || __index < 0)
    throw std::invalid_argument("Index input is out of range");

  data_t     __r;
  index_type __start = this->__shape_[1] * __index;
  index_type __end   = this->__shape_[1] * __index + this->__shape_[1];
  index_type __i     = __start;
  for (; __i < __end; ++__i) __r.push_back(this->__data_[__i]);

  return __self({this->__shape_[1]}, __r);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::col(const index_type __index) const {
  if (this->__shape_.size() != 2)
    throw std::runtime_error("Cannot get a column from a non two dimensional tensor");

  if (this->__shape_[1] <= __index || __index < 0)
    throw std::invalid_argument("Index input out of range");

  data_t     __c;
  index_type __i = 0;
  for (; __i < this->__shape_[0]; ++__i)
    __c.push_back(this->__data_[this->__compute_index({__i, __index})]);

  return __self({this->__shape_[0]}, __c);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::view(std::initializer_list<index_type> __sh) {
  index_type __s = this->__computeSize(__sh);

  if (__s != this->__data_.size())
    throw std::invalid_argument("Total elements do not match for new shape");

  this->__shape_ = __sh;
  this->__compute_strides();
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::view(std::initializer_list<index_type> __new_sh) const {
  return this->view(__new_sh);
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::randomize(const shape_type& __sh, bool __bounded) {
  __self __ret = this->clone();
  __ret.randomize_(__sh, __bounded);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::get_minor(index_type __a, index_type __b) const {}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::randomize_(const shape_type& __sh, bool __bounded) {
#if defined(__ARM_NEON)
  return this->neon_randomize_(__sh, __bounded);
#endif
  if (__bounded)
    assert(std::is_floating_point_v<value_type> && "Cannot bound non floating point data type");

  if (__sh.empty() && this->__shape_.empty())
    throw std::invalid_argument("randomize_ : Shape must be initialized");

  if (this->__shape_.empty() || this->__shape_ != __sh) this->__shape_ = __sh;

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
  for (; __i + _AVX_REG_WIDTH <= static_cast<index_type>(__s); __i += _AVX_REG_WIDTH) {
    __m256 __random_values = _mm256_setr_ps(
        __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen),
        __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen));

    if (!__bounded) __random_values = _mm256_div_ps(__random_values, __scale);

    _mm256_storeu_ps(&this->__data_[__i], __random_values);
  }

#elif defined(__SSE__)
  const __m128 __scale = _mm_set1_ps(__bounded ? static_cast<_f32>(RAND_MAX) : 1.0f);
  for (; __i + 4 <= static_cast<index_type>(__s); __i += 4) {
    __m128 __random_values = _mm_setr_ps(__bounded_dist(__gen), __bounded_dist(__gen),
                                         __bounded_dist(__gen), __bounded_dist(__gen));

    if (!__bounded) __random_values = _mm_div_ps(__random_values, __scale);

    _mm_storeu_ps(&this->__data_[__i], __random_values);
  }
#endif
#pragma omp parallel
  for (; __i < static_cast<index_type>(__s); ++__i)
    this->__data_[__i] = value_type(__bounded ? __bounded_dist(__gen) : __unbounded_dist(__gen));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::randomize_(const shape_type& __sh, bool __bounded) const {
#if defined(__ARM_NEON)
  return this->neon_randomize_(__sh, __bounded);
#endif
  if (__bounded)
    assert(std::is_floating_point_v<value_type> && "Cannot bound non floating point data type");

  if (__sh.empty() && this->__shape_.empty())
    throw std::invalid_argument("randomize_ : Shape must be initialized");

  if (this->__shape_.empty() || this->__shape_ != __sh) this->__shape_ = __sh;

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
  for (; __i + _AVX_REG_WIDTH <= static_cast<index_type>(__s); __i += _AVX_REG_WIDTH) {
    __m256 __random_values = _mm256_setr_ps(
        __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen),
        __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen));

    if (!__bounded) __random_values = _mm256_div_ps(__random_values, __scale);

    _mm256_storeu_ps(&this->__data_[__i], __random_values);
  }

#elif defined(__SSE__)
  const __m128 __scale = _mm_set1_ps(__bounded ? static_cast<_f32>(RAND_MAX) : 1.0f);
  for (; __i + 4 <= static_cast<index_type>(__s); __i += 4) {
    __m128 __random_values = _mm_setr_ps(__bounded_dist(__gen), __bounded_dist(__gen),
                                         __bounded_dist(__gen), __bounded_dist(__gen));

    if (!__bounded) __random_values = _mm_div_ps(__random_values, __scale);

    _mm_storeu_ps(&this->__data_[__i], __random_values);
  }
#endif
#pragma omp parallel
  for (; __i < static_cast<index_type>(__s); ++__i)
    this->__data_[__i] = value_type(__bounded ? __bounded_dist(__gen) : __unbounded_dist(__gen));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::clone() const {
  data_t     __d = this->__data_;
  shape_type __s = this->__shape_;
  return __self(__s, __d);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::negative_() {
#if defined(__ARM_NEON)
  return this->neon_negative_();
#endif
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = -this->__data_[__i];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::negative_() const {
#if defined(__ARM_NEON)
  return this->neon_negative_();
#endif
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = -this->__data_[__i];

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::negative() const {
  __self __ret = this->clone();
  __ret.negative_();
  return __ret;
}

void _permutations(std::vector<std::vector<int>>& __res, std::vector<int>& __arr, int __idx) {
  if (__idx == __arr.size() - 1) {
    __res.push_back(__arr);
    return;
  }

  for (int __i = __idx; __i < __arr.size(); ++__i) {
    std::swap(__arr[__idx], __arr[__i]);
    _permutations(__res, __arr, __idx + 1);
    std::swap(__arr[__idx], __arr[__i]);
  }
}

void _nextPermutation(std::vector<int>& __arr) {
  std::vector<std::vector<int>> __ret;
  _permutations(__ret, __arr, 0);
  std::sort(__ret.begin(), __ret.end());

  for (int __i = 0; __i < __ret.size(); ++__i) {
    if (__ret[__i] == __arr) {
      if (__i < __ret.size() - 1) __arr = __ret[__i + 1];
      if (__i == __ret.size() - 1) __arr = __ret[0];
      break;
    }
  }
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::permute_(const index_type __dim) {
  if (__dim < 0 || __dim > this->n_dims())
    throw std::invalid_argument("Dimension index is out of range");

  if (__dim == 0) {
    _nextPermutation(this->__data_);
    return *this;
  }

  index_type __start = this->__strides_[__dim - 1];
  index_type __end   = this->__strides_[__dim];

  data_t __p(this->__data_.begin() + __start, this->__data_.begin() + __end);
  _nextPermutation(__p);
#pragma omp parallel
  for (index_type __i = __start, __pi = 0; __i < __end && __pi < __p.size(); ++__i, ++__pi)
    this->__data_[__i] = __p[__pi];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::permute_(const index_type __dim) const {
  if (__dim < 0 || __dim > this->n_dims())
    throw std::invalid_argument("Dimension index is out of range");

  if (__dim == 0) {
    _nextPermutation(this->__data_);
    return *this;
  }

  index_type __start = this->__strides_[__dim - 1];
  index_type __end   = this->__strides_[__dim];

  data_t __p(this->__data_.begin() + __start, this->__data_.begin() + __end);
  _nextPermutation(__p);
#pragma omp parallel
  for (index_type __i = __start, __pi = 0; __i < __end && __pi < __p.size(); ++__i, ++__pi)
    this->__data_[__i] = __p[__pi];

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::permute(const index_type __dim) const {
  __self __ret = this->clone();
  __ret.permute_(__dim);
  return __ret;
}

template <class _Tp>
const tensor<_Tp>& tensor<_Tp>::repeat_(const data_t& __d, int __dim) const {
  if (__d.empty()) {
    std::cerr << "Error: Cannot repeat an empty data tensor." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (this->size(0) < __d.size()) this->__data_ = data_t(__d.begin(), __d.end());

  size_t __start      = 0;
  size_t __end        = __d.size();
  size_t __total_size = this->size(0);

  if (__total_size < __d.size()) return *this;

  unsigned int __nbatches  = __total_size / __d.size();
  size_t       __remainder = __total_size % __d.size();

  for (unsigned int __i = 0; __i < __nbatches; ++__i) {
    for (size_t __j = __start, __k = 0; __k < __d.size(); ++__j, ++__k)
      this->__data_[__j] = __d[__k];

    __start += __d.size();
  }
#pragma omp parallel
  for (size_t __j = __start, __k = 0; __j < __total_size && __k < __remainder; ++__j, ++__k)
    this->__data_[__j] = __d[__k];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::repeat_(const data_t& __d, int __dim) {
  if (__d.empty()) {
    std::cerr << "Error: Cannot repeat an empty data tensor." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (this->size(0) < __d.size()) this->__data_ = data_t(__d.begin(), __d.end());

  size_t __start      = 0;
  size_t __end        = __d.size();
  size_t __total_size = this->size(0);

  if (__total_size < __d.size()) return *this;

  unsigned int __nbatches  = __total_size / __d.size();
  size_t       __remainder = __total_size % __d.size();

  for (unsigned int __i = 0; __i < __nbatches; ++__i) {
    for (size_t __j = __start, __k = 0; __k < __d.size(); ++__j, ++__k)
      this->__data_[__j] = __d[__k];

    __start += __d.size();
  }
#pragma omp parallel
  for (size_t __j = __start, __k = 0; __j < __total_size && __k < __remainder; ++__j, ++__k)
    this->__data_[__j] = __d[__k];

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::repeat(const data_t& __d, int __dim) const {
  __self __ret = this->clone();
  __ret.repeat_(__d, __dim);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::sort(index_type __dim, bool __descending) const {}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::fill(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fill_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::fill(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fill_(__other);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::resize_as(const shape_type __sh) const {
  __self __ret = this->clone();
  __ret.resize_as_(__sh);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::all() const {
  bool       __result = true;
  index_type __i      = 0;

  for (; __i < this->__data_.size(); ++__i) {
    if (this->__data_[__i] == static_cast<value_type>(0)) {
      __result = false;
      break;
    }
  }

  tensor __ret;
  __ret.__data_ = {__result ? static_cast<value_type>(1) : static_cast<value_type>(0)};

  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::any() const {
  bool result = false;

  for (index_type __i = 0; __i < this->__data_.size(); ++__i) {
    if (this->__data_[__i] != static_cast<value_type>(0)) {
      result = true;
      break;
    }
  }

  tensor ret;
  ret.__data_ = {result ? static_cast<value_type>(1) : static_cast<value_type>(0)};

  return ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::gcd(const tensor& __other) const {
  assert(this->__shape_ == __other.shape());

  tensor     __ret = this->clone();
  index_type __i   = 0;

  for (; __i < this->__data_.size(); ++__i) {
    index_type __gcd__ = static_cast<index_type>(this->__data_[__i] * __other[__i]);
    index_type __lcm__ =
        __lcm(static_cast<index_type>(this->__data_[__i]), static_cast<index_type>(__other[__i]));
    __gcd__ /= __lcm__;
    __ret[__i] = __gcd__;
  }

  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::gcd(const value_type __val) const {
  tensor     __ret = this->clone();
  index_type __i   = 0;

  for (; __i < this->__data_.size(); ++__i) {
    index_type __gcd__ = static_cast<index_type>(this->__data_[__i] * __val);
    index_type __lcm__ =
        __lcm(static_cast<index_type>(this->__data_[__i]), static_cast<index_type>(__val));
    __gcd__ /= __lcm__;
    __ret[__i] = __gcd__;
  }

  return __ret;
}