#pragma once


#include "internal/simd/neon/compare.hpp"
#include "tensor.hpp"
#include "types.hpp"


template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::not_equal(const tensor& other) const
{
  if (using_neon())
  {
    return internal::simd::neon::not_equal(*this, other);
  }

  if constexpr (!internal::concepts::has_not_equal_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have an equal to operator");
  }

  if (!this->shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  std::vector<bool> ret(this->size(0));

  for (index_type i = 0, n = this->size(0); i < n; ++i)
  {
    ret[i] = ((*this)[i] != other[i]);
  }

  return arch::tensor<bool>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::not_equal(const value_type value) const
{
  if (using_neon())
  {
    return internal::simd::neon::not_equal(*this, value);
  }

  if constexpr (!internal::concepts::has_equal_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have an equal to operator");
  }

  std::vector<bool>     ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = (elem != value);
  }

  return arch::tensor<bool>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::less(const tensor& other) const
{
  if (using_neon())
  {
    return internal::simd::neon::less(*this, other);
  }

  if constexpr (!internal::concepts::has_less_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a less than operator");
  }

  if (!this->shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  std::vector<bool> ret(this->size(0));

  for (index_type i = 0, n = this->size(0); i < n; ++i)
  {
    ret[i] = ((*this)[i] < other[i]);
  }

  return arch::tensor<bool>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::less(const value_type value) const
{
  if (using_neon())
  {
    return internal::simd::neon::less(*this, value);
  }

  if constexpr (!internal::concepts::has_less_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a less  than operator");
  }

  std::vector<bool>     ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (const auto& elem : a)
  {
    ret[i] = (elem < value);
    ++i;
  }

  return arch::tensor<bool>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::greater(const tensor& other) const
{
  return other.less(*this);
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::greater(const value_type value) const
{
  if (using_neon())
  {
    return internal::simd::neon::greater(*this, value);
  }

  if constexpr (!internal::concepts::has_greater_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a greater than operator");
  }

  std::vector<bool>     ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = (elem > value);
  }

  return arch::tensor<bool>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::equal(const tensor& other) const
{
  if (using_neon())
  {
    return internal::simd::neon::equal(*this, other);
  }

  if constexpr (!internal::concepts::has_equal_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have equal operator");
  }

  if (!this->shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors must have the same shape");
  }

  std::vector<bool>     ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (auto& elem : a)
  {
    ret[i] = (elem == other[i]);
    ++i;
  }

  return arch::tensor<bool>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::equal(const value_type value) const
{
  if constexpr (!internal::concepts::has_equal_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have equal operator");
  }

  std::vector<bool>     ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (auto& elem : a)
  {
    ret[i++] = (elem == value);
  }

  return arch::tensor<bool>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::less_equal(const tensor& other) const
{
  if (using_neon())
  {
    return internal::simd::neon::less_equal(*this, other);
  }

  if constexpr (!internal::concepts::has_less_equal_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a less than or equal to operator");
  }

  if (!this->shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  std::vector<bool> ret(this->size(0));

  for (index_type i = 0; i < ret.size(); ++i)
  {
    ret[i] = ((*this)[i] <= other[i]);
  }

  return arch::tensor<bool>(std::move(this->shape()), std::move(ret), std::move(this->device()));
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::less_equal(const value_type value) const
{
  if (using_neon())
  {
    return internal::simd::neon::less_equal(value);
  }

  if constexpr (!internal::concepts::has_less_equal_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a less than or equal to operator");
  }

  std::vector<bool>     ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = (elem <= value);
  }

  return arch::tensor<bool>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::greater_equal(const tensor& other) const
{
  return other.less_equal(*this);
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::greater_equal(const value_type value) const
{
  if (using_neon())
  {
    return internal::simd::neon::greater_equal(*this, value);
  }

  if constexpr (!internal::concepts::has_greater_equal_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a greater than or equal to operator");
  }

  std::vector<bool>     ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = (elem >= value);
  }

  return arch::tensor<bool>(std::move(this->shape()), std::move(ret));
}
