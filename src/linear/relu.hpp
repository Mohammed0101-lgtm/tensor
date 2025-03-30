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

/*
template <class _Tp>
tensor<_Tp>& tensor<_Tp>::clipped_relu_(const value_type __clip_limit) {
#if defined(__ARM_NEON)
  return this->neon_clipped_relu_(__clip_limit);
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

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
#endif

  for (; __i < __s; ++__i)
    this->__data_[__i] = std::min(std::max(this->__data_[__i], value_type(0)), __clip_limit);

  return *this;
}
*/

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::clipped_relu_(const value_type __clip_limit) {
#if defined(__ARM_NEON)
  return this->neon_clipped_relu_(__clip_limit);
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

  this->clamp_(value_type(0), std::numeric_limits<value_type>::max());
  this->clamp_(std::numeric_limits<value_type>::lowest(), __clip_limit);

  return *this;
}

/*
template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::clipped_relu_(const value_type __clip_limit) const {
#if defined(__ARM_NEON)
  return this->neon_clipped_relu_(__clip_limit);
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

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
#endif

  for (; __i < __s; ++__i)
    this->__data_[__i] = std::min(std::max(this->__data_[__i], value_type(0)), __clip_limit);

  return *this;
}
*/

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::clipped_relu_(const value_type __clip_limit) const {
#if defined(__ARM_NEON)
  return this->neon_clipped_relu_(__clip_limit);
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

  this->clamp_(value_type(0), std::numeric_limits<value_type>::max());
  this->clamp_(std::numeric_limits<value_type>::lowest(), __clip_limit);

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::clipped_relu() const {
  __self __ret = this->clone();
  __ret.clipped_relu_();
  return __ret;
}