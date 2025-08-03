#pragma once

#include "internal/simd/neon/bit/or.hpp"
#include "tensor.hpp"


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_or_(const value_type value) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::bitwise_or_(*this, value);
  }

  if constexpr (!std::is_integral_v<value_type>)
  {
    throw error::type_error("Cannot perform a bitwise OR on non-integral values");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem | value;
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_or(const value_type value) const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.bitwise_or_(value);
  return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_or_(const tensor& other) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::bitwise_or_(*this, other);
  }

  if constexpr (!std::is_integral_v<value_type>)
  {
    throw error::type_error("Cannot perform a bitwise OR on non-integral values");
  }

  if (!this->shape_().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  container_type& a = this->storage_();
  index_type      i = 0;

  for (auto& elem : a)
  {
    elem = elem | other[i++];
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_or(const tensor& other) const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.bitwise_or_(other);
  return ret;
}