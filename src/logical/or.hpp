#pragma once

#include "internal/simd/neon/logical/or.hpp"
#include "tensor.hpp"


template<class _Tp>
arch::tensor<_Tp>& arch::tensor<_Tp>::logical_or_(const value_type value)
{
  if constexpr (!std::is_integral_v<value_type>)
  {
    throw error::type_error("Cannot perform logical OR on non-integral and non-boolean values");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = (elem || value);
  }

  return *this;
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::logical_or(const value_type value) const
{
  if (this->empty())
  {
    return arch::tensor<bool>({0});
  }

  arch::tensor<bool> ret = clone().bool_();
  ret.logical_or_(value);
  return ret;
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::logical_or(const tensor& other) const
{
  if (this->empty())
  {
    return arch::tensor<bool>({0});
  }

  arch::tensor<bool> ret = clone().bool_();
  ret.logical_or_(other);
  return ret;
}

template<class _Tp>
arch::tensor<_Tp>& arch::tensor<_Tp>::logical_or_(const tensor& other)
{
  if constexpr (!std::is_integral_v<value_type>)
  {
    throw error::type_error("Cannot get the element wise not of non-integral and non-boolean value");
  }

  if (!this->shape_().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  index_type      i = 0;
  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = (elem || other[i++]);
  }

  return *this;
}
