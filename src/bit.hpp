#pragma once

#include "tensorbase.hpp"

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::bitwise_right_shift_(const int __amount) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_right_shift_(__amount);
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] >>= __amount;

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_right_shift_(const int __amount) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_right_shift_(__amount);
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] >>= __amount;

  return *this;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::bitwise_left_shift_(const int __amount) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_left_shift_(__amount);
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] <<= __amount;

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_left_shift_(const int __amount) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_left_shift_(__amount);
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] <<= __amount;

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_or_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_or_(__val);
#endif
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] |= __val;

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_or_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_or_(__val);
#endif
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] |= __val;

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_xor_(__val);
#endif
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] ^= __val;

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_xor_(__val);
#endif
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] ^= __val;

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
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise not on non integral or boolean value");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = ~this->__data_[__i];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_not_() const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_not_();
#endif
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise not on non integral or boolean value");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = ~this->__data_[__i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_and_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_and_(__val);
#endif
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] &= __val;

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_and_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_and_(__val);
#endif
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] &= __val;

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
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] &= __other[__i];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_and_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_and_(__other);
#endif
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] &= __other[__i];

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
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] |= __other[__i];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_or_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_or_(__other);
#endif
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] |= __other[__i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_bitwise_xor_(__other);
#endif
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] ^= __other[__i];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_bitwise_xor_(__other);
#endif
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] ^= __other[__i];

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
  assert(this->__shape_ == __other.shape());

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = __other[__i];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fill_(const tensor& __other) const {
  return this->fill_(__other);
}