#pragma once

#include "tensorbase.hpp"

template <class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::operator()(
    std::initializer_list<index_type> __index_list) {
  return this->__data_[this->__compute_index(shape_type(__index_list))];
}

template <class _Tp>
typename tensor<_Tp>::const_reference tensor<_Tp>::operator()(
    std::initializer_list<index_type> __index_list) const {
  return this->__data_[this->__compute_index(shape_type(__index_list))];
}

template <class _Tp>
bool tensor<_Tp>::operator!=(const tensor& __other) const {
  return !(*this == __other);
}

template <class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::operator[](const index_type __idx) {
  if (__idx >= this->__data_.size() || __idx < 0)
    throw __index_error__("Access index is out of range");

  return this->__data_[__idx];
}

template <class _Tp>
typename tensor<_Tp>::const_reference tensor<_Tp>::operator[](const index_type __idx) const {
  if (__idx >= this->__data_.size() || __idx < 0)
    throw __index_error__("Access index is out of range");

  return this->__data_[__idx];
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_operator_plus(__other);
#endif
  static_assert(has_plus_operator_v<value_type>, "Value type must have a plus operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

  data_t __d(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    __d[__i] = this->__data_[__i] + __other[__i];

  return __self(this->__shape_, __d);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_operator_plus(__val);
#endif
  static_assert(has_plus_operator_v<value_type>, "Value type must have a plus operator");
  data_t __d(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) __d[__i] = this->__data_[__i] + __val;

  return __self(__d, this->__shape_);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::operator*(const value_type __val) const {
  static_assert(has_times_operator_v<value_type>, "Value type must have a times operator");
  data_t __d(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) __d[__i] = this->__data_[__i] + __val;

  return __self(this->__shape_, __d);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::operator*(const tensor& __other) const {
  static_assert(has_times_operator_v<value_type>, "Value type must have a times operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

  data_t __d(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    __d[__i] = this->__data_[__i] * __other[__i];

  return __self(this->__shape_, __d);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::operator+=(const tensor& __other) const {
  static_assert(has_plus_operator_v<value_type>, "Value type must have a plus operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] += __other[__i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::operator+=(const_reference __val) const {
#if defined(__ARM_NEON)
  return this->neon_operator_plus_eq(__val);
#endif
  static_assert(has_plus_operator_v<value_type>, "Value type must have a plus operator");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = this->__data_[__i] + __val;

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_operator_minus(__other);
#endif
  static_assert(has_minus_operator_v<value_type>, "Value type must have a minus operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

  data_t __d(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_[__i]; ++__i)
    __d[__i] = this->__data_[__i] - __other[__i];

  return __self(this->__shape_, __d);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_operator_minus(__val);
#endif
  static_assert(has_minus_operator_v<value_type>, "Value type must have a minus operator");
  data_t __d(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) __d[__i] = this->__data_[__i] - __val;

  return __self(*this);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::operator-=(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_operator_minus_eq(__other);
#endif
  static_assert(has_minus_operator_v<value_type>, "Value type must have a minus operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] -= __other[__i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::operator*=(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_operator_times_eq(__other);
#endif
  static_assert(has_times_operator_v<value_type>, "Value type must have a times operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] *= __other[__i];

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::operator/(const_reference __val) const {
  static_assert(has_divide_operator_v<value_type>, "Value type must have a divide operator");

  if (__val == value_type(0)) throw std::logic_error("Cannot divide by zero : undefined operation");

  data_t __d(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) __d[__i] = this->__data_[__i] / __val;

  return __self(this->__shape_, __d);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::operator*=(const_reference __val) const {
  static_assert(has_times_operator_v<value_type>, "Value type must have a times operator");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] *= __val;

  return *this;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::operator=(const tensor& __other) const {
  this->__shape_ = __other.shape();
  this->__data_  = __other.data();
  this->__compute_strides();
  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::operator/=(const tensor& __other) const {
  static_assert(has_divide_operator_v<value_type>, "Value type must have a divide operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

  if (__other.count_nonzero(0) != __other.size(0))
    throw std::logic_error("Cannot divide by zero : undefined operation");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] /= __other[__i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::operator/=(const_reference __val) const {
  static_assert(has_divide_operator_v<value_type>, "Value type must have a divide operator");

  if (__val == value_type(0))
    throw std::invalid_argument("Cannot divide by zero : undefined operation");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] /= __val;

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::operator/(const tensor& __other) const {
  static_assert(has_divide_operator_v<value_type>, "Value type must have a divide operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

  if (__other.count_nonzero(0) != __other.size(0))
    throw std::logic_error("Cannot divide by zero : undefined operation");

  data_t __d(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    __d[__i] = this->__data_[__i] / __other[__i];

  return __self(this->__shape_, __d);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::operator-=(const_reference __val) const {
#if defined(__ARM_NEON)
  return this->neon_operator_minus_eq(__val);
#endif
  static_assert(has_minus_operator_v<value_type>, "Value type must have a minus operator");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] -= __val;

  return *this;
}

template <class _Tp>
bool tensor<_Tp>::operator==(const tensor& __other) const {
  if (this->__equal_shape(this->shape(), __other.shape()) &&
      this->__strides_ == __other.strides() && this->__data_ == __other.storage())
    return true;
  return false;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::operator=(tensor&& __other) const noexcept {
  if (this != &__other) {
    this->__data_    = std::move(__other.storage());
    this->__shape_   = std::move(__other.shape());
    this->__strides_ = std::move(__other.strides());
  }
  return *this;
}

template <class _Tp>
const tensor<bool>& tensor<_Tp>::operator!() const {
  return this->logical_not_();
}

template <class _Tp>
tensor<bool>& tensor<_Tp>::operator!() {
  return this->logical_not_();
}