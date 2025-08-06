#pragma once

#include "internal/simd/neon/math/trig/cos/cos.hpp"
#include "tensor.hpp"


template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::cos_() {
  if (this->empty())
  {
    return *this;
  }

  if (internal::types::using_neon())
  {
    return internal::simd::neon::cos_(*this);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::cos(elem);
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::cos() const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.cos_();
  return ret;
}