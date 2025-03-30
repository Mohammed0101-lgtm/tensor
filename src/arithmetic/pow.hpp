#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::pow_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_pow_(__val);
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::pow(this->__data_[__i], __val));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::pow_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_pow_(__val);
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::pow(this->__data_[__i], __val));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::pow(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.pow_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::pow(const value_type __val) const {
  __self __ret = this->clone();
  __ret.pow_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::pow_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_pow_(__other);
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::pow(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::pow_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_pow_(__other);
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::pow(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::square_() {
  return this->pow_(static_cast<value_type>(2.0f));
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::square_() const {
  return this->pow_(static_cast<value_type>(2.0f));
}
