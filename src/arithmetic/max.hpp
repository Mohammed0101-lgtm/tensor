#pragma once

#include "tensorbase.hpp"

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmax(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fmax_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmax(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fmax_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_fmax(__val);
#endif
  if (!std::is_floating_point_v<value_type>) throw __type_error__("Type must be floating point");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::fmax(this->__data_[__i], __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmax_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_fmax_(__val);
#endif
  if (!std::is_floating_point_v<value_type>) throw __type_error__("Type must be floating point");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::fmax(this->__data_[__i], __val);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_fmax_(__other);
#endif
  if (!std::is_floating_point_v<value_type>) throw __type_error__("Type must be floating point");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::fmax(this->__data_[__i], __other[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmax_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_fmax_(__other);
#endif
  if (!std::is_floating_point_v<value_type>) throw __type_error__("Type must be floating point");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::fmax(this->__data_[__i], __other[__i]);

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::maximum(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.maximum_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::maximum(const_reference __val) const {
  __self __ret = this->clone();
  __ret.maximum_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::maximum_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_maximum_(__other);
#endif
  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::max(this->__data_[__i], __other[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::maximum_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_maximum_(__other);
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::max(this->__data_[__i], __other[__i]);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::maximum_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_maximum_(__val);
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::max(this->__data_[__i], __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::maximum_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_maximum_(__val);
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::max(this->__data_[__i], __val);

  return *this;
}
