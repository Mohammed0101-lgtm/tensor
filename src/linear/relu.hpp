#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::relu() const {
  return clamp_min(value_type(0));
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::relu_() {
#if defined(__ARM_NEON)
  return this->neon_relu_();
#endif
  return clamp_min_(value_type(0));
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::relu_() const {
#if defined(__ARM_NEON)
  return this->neon_relu_();
#endif
  return clamp_min_(value_type(0));
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::clipped_relu_(const value_type __clip_limit) {
#if defined(__ARM_NEON)
  return this->neon_clipped_relu_(__clip_limit);
#endif
  if constexpr (std::is_unsigned_v<value_type>) return *this;

  index_type __s = this->__data_.size();
  index_type __i = 0;

#pragma omp parallel

#ifdef __CUDACC__
  if (this->__is_cuda_tensor) {
    pointer __d_data = thrust::raw_pointer_cast(this->__data_.data());
    thrust::transform(
        thrust::device, __d_data, __d_data + __s, __d_data,
        [] __device__(value_type __x) { return min(max(__x, value_type(0)), __clip_limit); });
    return *this;
  }

#elif defined(__SSE__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    __m128 __zero = _mm_setzero_ps();
    __m128 __clip = _mm_set1_ps(__clip_limit);

    for (; __i + 4 <= __s; __i += 4) {
      __m128 __x      = _mm_loadu_ps(&this->__data_[__i]);
      __m128 __result = _mm_min_ps(_mm_max_ps(__x, __zero), __clip);

      _mm_storeu_ps(&this->__data_[__i], __result);
    }
  }

#elif defined(__AVX__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    __m256 __zero = _mm256_setzero_ps();
    __m256 __clip = _mm256_set1_ps(__clip_limit);

    for (; __i + _AVX_REG_WIDTH <= __s; __i += _AVX_REG_WIDTH) {
      __m256 __x      = _mm256_loadu_ps(&this->__data_[__i]);
      __m256 __result = _mm256_min_ps(_mm256_max_ps(__x, __zero), __clip);

      _mm256_storeu_ps(&this->__data_[__i], __result);
    }
  }
#endif

  for (; __i < __s; ++__i)
    this->__data_[__i] = std::min(std::max(this->__data_[__i], value_type(0)), __clip_limit);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::clipped_relu_(const value_type __clip_limit) const {
#if defined(__ARM_NEON)
  return this->neon_clipped_relu_(__clip_limit);
#endif
  if constexpr (std::is_unsigned_v<value_type>) return *this;

  index_type __s = this->__data_.size();
  index_type __i = 0;

#pragma omp parallel

#ifdef __CUDACC__
  if (this->__is_cuda_tensor) {
    pointer __d_data = thrust::raw_pointer_cast(this->__data_.data());
    thrust::transform(
        thrust::device, __d_data, __d_data + __s, __d_data,
        [] __device__(value_type __x) { return min(max(__x, value_type(0)), __clip_limit); });
    return *this;
  }

#elif defined(__SSE__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    __m128 __zero = _mm_setzero_ps();
    __m128 __clip = _mm_set1_ps(__clip_limit);

    for (; __i + 4 <= __s; __i += 4) {
      __m128 __x      = _mm_loadu_ps(&this->__data_[__i]);
      __m128 __result = _mm_min_ps(_mm_max_ps(__x, __zero), __clip);

      _mm_storeu_ps(&this->__data_[__i], __result);
    }
  }

#elif defined(__AVX__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    __m256 __zero = _mm256_setzero_ps();
    __m256 __clip = _mm256_set1_ps(__clip_limit);

    for (; __i + _AVX_REG_WIDTH <= __s; __i += _AVX_REG_WIDTH) {
      __m256 __x      = _mm256_loadu_ps(&this->__data_[__i]);
      __m256 __result = _mm256_min_ps(_mm256_max_ps(__x, __zero), __clip);

      _mm256_storeu_ps(&this->__data_[__i], __result);
    }
  }
#endif

  for (; __i < __s; ++__i)
    this->__data_[__i] = std::min(std::max(this->__data_[__i], value_type(0)), __clip_limit);

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::clipped_relu() const {
  __self __ret = this->clone();
  __ret.clipped_relu_();
  return __ret;
}