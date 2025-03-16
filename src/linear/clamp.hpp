#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::clamp_(const_reference __min_val, const_reference __max_val) {
  index_type __i = 0;

#if defined(__AVX2__)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _AVX_REG_WIDTH);
  __m256           __min_vec  = _mm256_set1_ps(__min_val);
  __m256           __max_vec  = _mm256_set1_ps(__max_val);

  for (; __i < __simd_end; __i += _AVX_REG_WIDTH) {
    __m256 __data_vec = _mm256_loadu_ps(&this->__data_[__i]);
    __m256 __clamped  = _mm256_min_ps(_mm256_max_ps(data_vec, __min_vec), __max_vec);

    _mm256_storeu_ps(&this->__data_[__i], __clamped);
  }

#elif defined(__ARM_NEON)
  return this->neon_clamp_(__min_val, __max_val);
#endif
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) {
    this->__data_[__i] = std::max(__min_val, this->__data_[__i]);
    this->__data_[__i] = std::min(__max_val, this->__data_[__i]);
  }

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::clamp_(const_reference __min_val,
                                              const_reference __max_val) const {
  index_type __i = 0;

#if defined(__AVX2__)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _AVX_REG_WIDTH);
  __m256           __min_vec  = _mm256_set1_ps(__min_val);
  __m256           __max_vec  = _mm256_set1_ps(__max_val);

  for (; __i < __simd_end; __i += _AVX_REG_WIDTH) {
    __m256 __data_vec = _mm256_loadu_ps(&this->__data_[__i]);
    __m256 __clamped  = _mm256_min_ps(_mm256_max_ps(data_vec, __min_vec), __max_vec);

    _mm256_storeu_ps(&this->__data_[__i], __clamped);
  }

#elif defined(__ARM_NEON)
  return this->neon_clamp_(__min_val, __max_val);
#endif
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) {
    this->__data_[__i] = std::max(__min_val, this->__data_[__i]);
    this->__data_[__i] = std::min(__max_val, this->__data_[__i]);
  }

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::clamp(const_reference __min_val, const_reference __max_val) const {
  __self __ret = this->clone();
  __ret.clamp_(__min_val, __max_val);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::floor() const {
  __self __ret = this->clone();
  __ret.floor_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::floor_() {
#if defined(__ARM_NEON)
  return this->neon_floor_();
#endif
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::floor(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::floor_() const {
#if defined(__ARM_NEON)
  return this->neon_floor_();
#endif
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::floor(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::ceil_() {
#if defined(__ARM_NEON)
  return this->neon_ceil_();
#endif
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::ceil(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::ceil_() const {
#if defined(__ARM_NEON)
  return this->neon_ceil_();
#endif
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::ceil(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::ceil() const {
  __self __ret = this->clone();
  __ret.ceil_();
  return __ret;
}