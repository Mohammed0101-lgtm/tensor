#pragma once

#include "internal/simd/neon/logical/and.hpp"
#include "tensor.hpp"


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_and_(const value_type value) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::logical_and_(*this, value);
  }

  if (!std::is_integral_v<value_type>)
  {
    throw error::type_error("Cannot get the element wise and of non-integral and non-boolean value");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = (elem && value);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const tensor& other) const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.logical_and_(other);
  return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const value_type value) const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.logical_and_(value);
  return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_and_(const tensor& other) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::logical_and_(*this, other);
  }

  if (!std::is_integral_v<value_type>)
  {
    throw error::type_error("Cannot get the element-wise and of non-integral and non-boolean value");
  }

  if (!this->shape_().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  index_type i = 0;

  container_type& a = this->storage_();

  for (auto& elem : a)

  {
    elem = (elem && other[i++]);
  }

  return *this;
}