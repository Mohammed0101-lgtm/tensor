#pragma once

#include "tensor.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sigmoid() const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.sigmoid_();
  return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sigmoid_() {
  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = 1.0 / (1.0 + std::exp(-elem));
  }

  return *this;
}
