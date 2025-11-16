#pragma once


#include "internal/simd/neon/bit/xor.hpp"
#include "tensor.hpp"


template<class _Tp>
arch::tensor<_Tp>& arch::tensor<_Tp>::bitwise_xor_(const tensor& other)
{
  if (using_neon())
  {
    return internal::simd::neon::bitwise_xor_(*this, other);
  }

  if constexpr (!std::is_integral_v<value_type>)
  {
    throw error::type_error("Cannot perform a bitwise XOR on non-integral values");
  }

  if (!this->shape_().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  container_type& a = this->storage_();
  index_type      i = 0;

  for (auto& elem : a)
  {
    elem = elem ^ other[i++];
  }

  return *this;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::bitwise_xor(const tensor& other) const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.bitwise_xor_(other);
  return ret;
}

template<class _Tp>
inline arch::tensor<_Tp>& arch::tensor<_Tp>::bitwise_xor_(const value_type value)
{
  if (using_neon())
  {
    return internal::simd::neon::bitwise_xor_(*this, value);
  }

  if constexpr (!std::is_integral_v<value_type>)
  {
    throw error::type_error("Cannot perform a bitwise XOR on non-integral values");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem ^ value;
  }

  return *this;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::bitwise_xor(const value_type value) const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.bitwise_xor_(value);
  return ret;
}