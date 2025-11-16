#pragma once

#include "internal/simd/neon/func.hpp"
#include "internal/simd/neon/types.hpp"
#include "tensor.hpp"


static bool using_neon()
{
/*
#ifdef __ARM_NEON
return true;
#endif
*/
  return false;
}

template<class _Tp>
arch::tensor<_s64> arch::tensor<_Tp>::int64_() const
{
  if (using_neon())
  {
    return internal::simd::neon::int64_(*this);
  }

  if (!std::is_convertible_v<value_type, _s64>)
  {
    throw error::type_error("Type must be convertible to 64 bit signed int");
  }

  if (this->empty())
  {
    return arch::tensor<_s64>(std::move(this->shape()));
  }

  std::vector<_s64>     ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = static_cast<_s64>(elem);
  }

  return arch::tensor<_s64>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<_s32> arch::tensor<_Tp>::int32_() const
{
  if (using_neon())
  {
    return internal::simd::neon::int32_(*this);
  }

  if (!std::is_convertible_v<value_type, int>)
  {
    throw error::type_error("Type must be convertible to 32 bit signed int");
  }

  if (this->empty())
  {
    return arch::tensor<int>(std::move(this->shape()));
  }

  std::vector<int>      ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = static_cast<_s32>(elem);
  }

  return arch::tensor<_s32>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<_u32> arch::tensor<_Tp>::uint32_() const
{
  if (using_neon())
  {
    return internal::simd::neon::uint32_(*this);
  }

  if (!std::is_convertible_v<value_type, unsigned int>)
  {
    throw error::type_error("Type must be convertible to 32 bit unsigned int");
  }

  if (this->empty())
  {
    return arch::tensor<unsigned int>(std::move(this->shape()));
  }

  std::vector<unsigned int> ret(this->size(0));
  const container_type&     a = this->storage_();
  index_type                i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = static_cast<unsigned int>(elem);
  }

  return arch::tensor<unsigned int>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<_f32> arch::tensor<_Tp>::float32_() const
{
  if (using_neon())
  {
    return internal::simd::neon::float32_(*this);
  }

  if (!std::is_convertible_v<value_type, _f32>)
  {
    throw error::type_error("Type must be convertible to 32 bit float");
  }

  if (this->empty())
  {
    return arch::tensor<_f32>(std::move(this->shape()));
  }

  std::vector<_f32>     ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = static_cast<_f32>(elem);
  }

  return arch::tensor<_f32>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<double> arch::tensor<_Tp>::float64_() const
{
  if (using_neon())
  {
    return internal::simd::neon::float64_(*this);
  }

  if (!std::is_convertible_v<value_type, double>)
  {
    throw error::type_error("Type must be convertible to 64 bit float");
  }

  if (this->empty())
  {
    return arch::tensor<double>(std::move(this->shape()));
  }

  std::vector<double>   ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = double(elem);
  }

  return arch::tensor<double>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<_u64> arch::tensor<_Tp>::uint64_() const
{
  if (using_neon())
  {
    return internal::simd::neon::uint64_(*this);
  }

  if (!std::is_convertible_v<value_type, _u64>)
  {
    throw error::type_error("Type must be convertible to unsigned 64 bit int");
  }

  if (this->empty())
  {
    return arch::tensor<_u64>(std::move(this->shape()));
  }

  std::vector<_u64>     ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = static_cast<_u64>(elem);
  }

  return arch::tensor<_u64>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<short> arch::tensor<_Tp>::int16_() const
{
  if (using_neon())
  {
    return internal::simd::neon::int16_(*this);
  }

  if (!std::is_convertible_v<value_type, short>)
  {
    throw error::type_error("Type must be convertible to short (aka 16 bit int)");
  }

  if (this->empty())
  {
    return arch::tensor<short>(std::move(this->shape()));
  }

  std::vector<short>    ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = short(elem);
  }

  return arch::tensor<short>(std::move(this->shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::bool_() const
{
  if (!std::is_convertible_v<value_type, bool>)
  {
    throw error::type_error("Type must be convertible to bool");
  }

  std::vector<bool>     ret(this->size(0));
  const container_type& a = this->storage_();
  index_type            i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = bool(elem);
  }

  return arch::tensor<bool>(std::move(this->shape()), std::move(ret));
}