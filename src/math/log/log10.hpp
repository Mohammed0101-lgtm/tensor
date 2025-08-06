#pragma once

#include "internal/simd/neon/math/log/log10.hpp"
#include "tensor.hpp"


template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::log10_() {
  if (this->empty())
  {
    return *this;
  }

  if (internal::types::using_neon())
  {
    return internal::simd::neon::log10_(*this);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::log10(elem);
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::log10() const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.log10_();
  return ret;
}