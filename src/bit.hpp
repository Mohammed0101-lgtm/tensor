#pragma once

#include "tensorbase.hpp"

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::bitwise_right_shift_(const int __amount) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_right_shift_(__amount);
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] >>= __amount;

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_right_shift_(const int __amount) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_right_shift_(__amount);
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] >>= __amount;

  return *this;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::bitwise_left_shift_(const int __amount) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_left_shift_(__amount);
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] <<= __amount;

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_left_shift_(const int __amount) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_left_shift_(__amount);
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] <<= __amount;

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_or_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_or_(__val);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise OR on non-integral values");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] |= __val;

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_or_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_or_(__val);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise OR on non-integral values");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] |= __val;

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_xor_(__val);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise XOR on non-integral values");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] ^= __val;

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_xor_(__val);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise XOR on non-integral");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] ^= __val;

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_xor(const value_type __val) const {
  __self __ret = this->clone();
  __ret.bitwise_xor_(__val);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_not() const {
  __self __ret = this->clone();
  __ret.bitwise_not_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_not_() {
#if defined(__ARM_NEON)
  return this->neon_bitwise_not_();
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise NOT on non integral values");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = ~this->__data_[__i];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_not_() const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_not_();
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise NOT on non integral values");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = ~this->__data_[__i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_and_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_and_(__val);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise AND on non-integral values");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] &= __val;

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_and_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_and_(__val);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise AND on non-integral values");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] &= __val;

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_and(const value_type __val) const {
  __self __ret = this->clone();
  __ret.bitwise_and_(__val);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_and(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.bitwise_and_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_left_shift(const int __amount) const {
  __self __ret = this->clone();
  __ret.bitwise_left_shift_(__amount);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_xor(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.bitwise_xor_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_right_shift(const int __amount) const {
  __self __ret = this->clone();
  __ret.bitwise_right_shift_(__amount);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_and_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_and_(__other);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise AND on non-integral values");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensor shapes must be equal");
#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] &= __other[__i];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_and_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_and_(__other);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise AND on non-integral values");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");
#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] &= __other[__i];

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_or(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.bitwise_or_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_or(const value_type __val) const {
  __self __ret = this->clone();
  __ret.bitwise_or_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_or_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_or_(__other);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise OR on non-integral values");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");
#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] |= __other[__i];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_or_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_or_(__other);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise OR on non-integral values");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");
#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] |= __other[__i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_xor_(__other);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise XOR on non-integral values");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");
#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] ^= __other[__i];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_xor_(__other);
#endif
  if (!std::is_integral_v<value_type>)
    throw __type_error__("Cannot perform a bitwise XOR on non-integral values");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");
#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] ^= __other[__i];

  return *this;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::fill_(const value_type __val) {
  this->__data_(this->__data_.size(), __val);
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fill_(const value_type __val) const {
  return this->fill_(__val);
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::fill_(const tensor& __other) {
  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] = __other[__i];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fill_(const tensor& __other) const {
  return this->fill_(__other);
}