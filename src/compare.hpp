#pragma once

#include "tensorbase.hpp"
#include "types.hpp"

template <class _Tp>
tensor<bool> tensor<_Tp>::not_equal(const tensor& __other) const {
  static_assert(has_not_equal_operator_v<value_type>, "Value type must have an equal operator");

  assert(__equal_shape(this->shape(), __other.shape()) && "not_equal : tensor shapes");
  std::vector<bool> __ret(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    __ret[__i] = (this->__data_[__i] != __other[__i]);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::not_equal(const value_type __val) const {
  static_assert(has_equal_operator_v<value_type>, "Value type must have an equal operator");
  std::vector<bool> __ret(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    __ret[__i] = (this->__data_[__i] != __val);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::less(const tensor& __other) const {
  static_assert(has_less_operator_v<value_type>, "Value type must have a less operator");
  assert(__equal_shape(this->shape(), __other.shape()));
  std::vector<bool> __ret(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    __ret[__i] = (this->__data_[__i] < __other[__i]);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::less(const value_type __val) const {
  static_assert(has_less_operator_v<value_type>, "Value type must have a less operator");
  std::vector<bool> __ret(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    __ret[__i] = (this->__data_[__i] < __val);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::greater(const tensor& __other) const {
  return __other.less(*this);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::greater(const value_type __val) const {
  static_assert(has_greater_operator_v<value_type>, "Value type must have a less operator");
  std::vector<bool> __ret(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    __ret[__i] = (this->__data_[__i] > __val);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::equal(const tensor& __other) const {
  tensor<bool> __ret = this->not_equal(__other);
  __ret.logical_not_();
  return __ret;
}

template <class _Tp>
tensor<bool> tensor<_Tp>::equal(const value_type __val) const {
  return !(this->not_equal(__val));
}

template <class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_less_equal(__other);
#endif
  static_assert(has_less_equal_operator_v<value_type>,
                "Value type must have a less equal operator");
  assert(__equal_shape(this->shape(), __other.shape()));
  std::vector<bool> __ret(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < __ret.size(); ++__i)
    __ret[__i] = (this->__data_[__i] <= __other.__data_[__i]);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_less_equal(__val);
#endif
  static_assert(has_less_equal_operator_v<value_type>,
                "Value type must have a less equal operator");
  std::vector<bool> __ret(this->__data_.size());

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    __ret[__i] = this->__data_[__i] <= __val;

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const tensor& __other) const {
  return __other.less_equal(*this);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const value_type __val) const {
  return !(this->less(__val));
}
