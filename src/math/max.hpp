#pragma once

#include "tensor.hpp"

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::fmax(const tensor& other) const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.fmax_(other);
  return ret;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::fmax(const value_type value) const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.fmax_(value);
  return ret;
}

template<class _Tp>
inline arch::tensor<_Tp>& arch::tensor<_Tp>::fmax_(const value_type value)
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::fmax_(*this, value);
  }

  if (!std::is_floating_point_v<value_type>)
  {
    throw error::type_error("Type must be floating point");
  }

  const container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::fmax(elem, value);
  }

  return *this;
}

template<class _Tp>
arch::tensor<_Tp>& arch::tensor<_Tp>::fmax_(const tensor& other)
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::fmax_(*this, other);
  }

  if (!std::is_floating_point_v<value_type>)
  {
    throw error::type_error("Type must be floating point");
  }

  if (!this->shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  const container_type& a = this->storage_();
  index_type            i = 0;

  for (auto& elem : a)
  {
    elem = std::fmax(elem, other[i++]);
  }

  return *this;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::maximum(const tensor& other) const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.maximum_(other);
  return ret;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::maximum(const_reference value) const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.maximum_(value);
  return ret;
}

template<class _Tp>
inline arch::tensor<_Tp>& arch::tensor<_Tp>::maximum_(const tensor& other)
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::maximum_(*this, other);
  }

  if (!this->shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  container_type& a = this->storage_();
  index_type      i = 0;

  for (auto& elem : a)
  {
    elem = std::max(elem, other[i++]);
  }

  return *this;
}

template<class _Tp>
arch::tensor<_Tp>& arch::tensor<_Tp>::maximum_(const value_type value)
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::maximum_(*this, value);
  }

  const container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::max(elem, value);
  }

  return *this;
}
