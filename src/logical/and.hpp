#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_and_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_logical_and_(__val);
#endif

  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot get the element wise and of non-integral and non-boolean value");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] && __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_and_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_logical_and_(__val);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot get the element wise and of non-integral and non-boolean value");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] && __val);

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.logical_and_(__other);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const value_type __val) const {
  __self __ret = this->clone();
  __ret.logical_and_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_and_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_logical_and_(__other);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot get the element-wise and of non-integral and non-boolean value");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (this->__data_[__i] && __other[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_and_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_logical_and_(__other);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot get the element-wise and of non-integral and non-boolean value");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (this->__data_[__i] && __other[__i]);

  return *this;
}