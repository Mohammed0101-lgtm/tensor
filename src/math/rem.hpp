#pragma once

#include "tensor.hpp"


template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::remainder(const value_type value) const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.remainder_(value);
  return ret;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::remainder(const tensor& other) const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.remainder_(other);
  return ret;
}

template<class _Tp>
inline arch::tensor<_Tp>& arch::tensor<_Tp>::remainder_(const value_type value)
{
  if (this->empty())
  {
    return *this;
  }
  /*
    if (using_neon())
    {
        return internal::simd::neon::remainder_(*this, value);
    }
    */
  if constexpr (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithemtic");
  }

  if (!value)
  {
    throw std::invalid_argument("Remainder by zero is undefined");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem % value;
  }

  return *this;
}

template<class _Tp>
inline arch::tensor<_Tp>& arch::tensor<_Tp>::remainder_(const tensor& other)
{
  if (this->empty())
  {
    return *this;
  }
  /*
    if (using_neon())
    {
        return internal::simd::neon::remainder_(*this, other);
    }
*/
  if constexpr (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  assert(other.count_nonzero() == other.size(0) && "Remainder by zero is undefined");

  if (!this->shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  const container_type& a = this->storage_();
  index_type            i = 0;

  for (auto& elem : a)
  {
    elem %= other[i++];
  }

  return *this;
}