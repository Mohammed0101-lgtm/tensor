#pragma once

#include "internal/simd/neon/math/trig/sin/asin.hpp"
#include "tensor.hpp"


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::asin_() {
  if (this->empty())
  {
    return *this;
  }

  if (internal::types::using_neon())
  {
    return internal::simd::neon::asin_(*this);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    if (elem < _Tp(-1.0) || elem > _Tp(1.0))
    {
      throw std::domain_error("Input data is out of domain for asin()");
    }

    elem = std::asin(elem);
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::asin() const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.asin_();
  return ret;
}