#pragma once

#include "tensorbase.hpp"

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmod(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fmod_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmod(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fmod_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::fmod_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_fmod_(__val);
#endif
  if (!std::is_floating_point_v<value_type>)
    throw std::logic_error("Cannot perform fmod on non-floating point values");

  if (__val == value_type(0)) throw std::runtime_error("Cannot perform fmod with zero");
#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__val)));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmod_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_fmod_(__val);
#endif
  if (!std::is_floating_point_v<value_type>)
    throw std::logic_error("Cannot perform fmod on non-floating point values");

  if (__val == value_type(0)) throw std::runtime_error("Cannot perform fmod with zero");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__val)));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::fmod_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_fmod_(__other);
#endif
  if (!__equal_shape(this->shape(), __other.shape()) || this->__data_.size() != __other.size(0))
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmod");

  if (!std::is_floating_point_v<value_type>)
    throw std::logic_error("Cannot perform fmod on non-floating point values");

  if (__other.count_nonzero(0) != __other.size(0))
    throw std::invalid_argument("Cannot divide by zero : undefined operation");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmod_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_fmod_(__other);
#endif
  if (__equal_shape(this->shape(), __other.shape()) || this->__data_.size() != __other.size(0))
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmod");

  if (!std::is_floating_point_v<value_type>)
    throw std::logic_error("Cannot perform fmod on non-floating point values");

  if (__other.count_nonzero(0) != __other.size(0))
    throw std::invalid_argument("Cannot divide by zero : undefined operation");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}
