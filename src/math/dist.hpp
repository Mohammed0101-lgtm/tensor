#pragma once

#include "tensor.hpp"
#include "types.hpp"


template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::dist(const tensor& other) const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.dist_(other);
  return ret;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::dist(const value_type value) const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.dist_(value);
  return ret;
}

template<class _Tp>
arch::tensor<_Tp>& arch::tensor<_Tp>::dist_(const tensor& other)
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::dist_(*this, other);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  if (!this->shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  if constexpr (!internal::concepts::has_minus_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a minus operator");
  }

  container_type& a = this->storage_();
  index_type      i = 0;

  for (auto& elem : a)
  {
    elem = std::abs(elem - other[i++]);
  }

  return *this;
}

template<class _Tp>
arch::tensor<_Tp>& arch::tensor<_Tp>::dist_(const value_type value)
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::dist_(*this, value);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  if constexpr (!internal::concepts::has_minus_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a minus operator");
  }

  const container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::abs(elem - value);
  }

  return *this;
}