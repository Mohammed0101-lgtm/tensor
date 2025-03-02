#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_or_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_logical_or_(__val);
#endif
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw std::runtime_error("Cannot perform logical OR on non-integral and non-boolean values");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] || __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_or_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_logical_or_(__val);
#endif
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw std::runtime_error("Cannot perform logical OR on non-integral and non-boolean values");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] || __val);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_xor_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_logical_xor_(__val);
#endif
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw std::runtime_error(
        "Cannot get the element wise xor of non-integral and non-boolean value");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] ^ __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_xor_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_logical_xor_(__val);
#endif
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw std::runtime_error(
        "Cannot get the element wise xor of non-integral and non-boolean value");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] ^ __val);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_and_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_logical_and_(__val);
#endif
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw std::runtime_error(
        "Cannot get the element wise and of non-integral and non-boolean value");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] && __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_and_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_logical_and_(__val);
#endif
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw std::runtime_error(
        "Cannot get the element wise and of non-integral and non-boolean value");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] && __val);

  return *this;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::logical_not_() {
  this->bitwise_not_();
  this->bool_();
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_not_() const {
  this->bitwise_not_();
  this->bool_();
  return *this;
}

template <class _Tp>
tensor<bool> tensor<_Tp>::logical_not() const {
  tensor<bool> __ret = this->bool_();
  __ret.logical_not_();
  return __ret;
}

template <class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const value_type __val) const {
  tensor<bool> __ret = this->clone().bool_();
  __ret.logical_or_(__val);
  return __ret;
}

template <class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const tensor& __other) const {
  tensor<bool> __ret = this->clone().bool_();
  __ret.logical_or_(__other);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.logical_xor_(__other);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const value_type __val) const {
  __self __ret = this->clone();
  __ret.logical_xor(__val);
  return __ret;
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
tensor<_Tp>& tensor<_Tp>::logical_or_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_logical_or_(__other);
#endif
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw std::runtime_error(
        "Cannot get the element wise not of non-integral and non-boolean value");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (this->__data_[__i] || __other[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_or_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_logical_or_(__other);
#endif
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw std::runtime_error(
        "Cannot get the element wise not of non-integral and non-boolean value");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (this->__data_[__i] || __other[__i]);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_xor_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_logical_xor_(__other);
#endif
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw std::runtime_error(
        "Cannot get the element wise xor of non-integral and non-boolean value");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (this->__data_[__i] ^ __other[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_xor_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_logical_xor_(__other);
#endif
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw std::runtime_error(
        "Cannot get the element wise xor of non-integral and non-boolean value");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (this->__data_[__i] ^ __other[__i]);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_and_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_logical_and_(__other);
#endif
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw std::runtime_error(
        "Cannot get the element-wise and of non-integral and non-boolean value");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (this->__data_[__i] && __other[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_and_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_logical_and_(__other);
#endif
  if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    throw std::runtime_error(
        "Cannot get the element-wise and of non-integral and non-boolean value");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (this->__data_[__i] && __other[__i]);

  return *this;
}