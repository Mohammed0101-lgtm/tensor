#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<bool> tensor<_Tp>::not_equal(const tensor& __other) const {
  if (!std::is_integral_v<value_type> && !std::is_scalar_v<value_type>)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  assert(this->__shape_ == __other.shape() && "not_equal : tensor shapes");
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < this->__data_.size(); ++__i) __ret[__i] = (this->__data_[__i] != __other[__i]);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::not_equal(const value_type __val) const {
  if (!std::is_integral_v<value_type> && !std::is_scalar_v<value_type>)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < this->__data_.size(); ++__i) __ret[__i] = (this->__data_[__i] != __val);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::less(const tensor& __other) const {
  if (!std::is_integral_v<value_type> && !std::is_scalar_v<value_type>)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < this->__data_.size(); ++__i) __ret[__i] = (this->__data_[__i] < __other[__i]);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::less(const value_type __val) const {
  if (!std::is_integral_v<value_type> && !std::is_scalar_v<value_type>)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < this->__data_.size(); ++__i) __ret[__i] = (this->__data_[__i] < __val);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::greater(const tensor& __other) const {
  if (!std::is_integral_v<value_type> && !std::is_scalar_v<value_type>)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < this->__data_.size(); ++__i) __ret[__i] = (this->__data_[__i] > __other[__i]);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::greater(const value_type __val) const {
  if (!std::is_integral_v<value_type> && !std::is_scalar_v<value_type>)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < this->__data_.size(); ++__i) __ret[__i] = (this->__data_[__i] > __val);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::equal(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_equal(__other);
#endif
  if (!std::is_integral_v<value_type> && !std::is_scalar_v<value_type>)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  assert(this->__shape_ == __other.shape() && "equal : tensor shapes");
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;
  for (; __i < this->__data_.size(); ++__i) __ret[__i] = (this->__data_[__i] == __other[__i]);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::equal(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_equal(__val);
#endif
  if (!std::is_integral_v<value_type> && !std::is_scalar_v<value_type>)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;
  for (; __i < this->__data_.size(); ++__i) __ret[__i] = (this->__data_[__i] == __val);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_less_equal(__other);
#endif
  if (!std::is_arithmetic_v<value_type>)
    throw std::runtime_error("Cannot compare non-numeric values");

  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < __ret.size(); ++__i) __ret[__i] = (this->__data_[__i] <= __other.__data_[__i]);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_less_equal(__val);
#endif
  throw std::runtime_error("Cannot compare non-integral or scalar value");

  std::vector<_u32> __ret(this->__data_.size());
  if (!std::is_integral_v<value_type> && !std::is_scalar_v<value_type>) index_type __i = 0;

  for (; __i < this->__data_.size(); ++__i) __ret[__i] = (this->__data_[__i] <= __val) ? 1 : 0;

  std::vector<bool> __to_bool(__ret.size());
  for (int i = __i; i >= 0; i--) __to_bool[i] = __ret[i] == 1 ? true : false;

  return tensor<bool>(__to_bool, this->__shape_);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const tensor& __other) const {
  return __other.less_equal(*this);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const value_type __val) const {
  return !(this->less(__val));
}
