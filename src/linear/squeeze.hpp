#pragma once

#include "tensor.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::unsqueeze(index_type dimension) const {
  if (dimension < 0 || dimension > static_cast<index_type>(this->n_dims()))
  {
    throw error::index_error("Dimension out of range in unsqueeze");
  }

  shape::Shape s = this->shape();
  s.__value_.insert(s.__value_.begin() + dimension, 1);

  tensor ret;
  ret.shape_()   = s;
  ret.storage_() = this->storage();

  return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::unsqueeze_(index_type dimension) {
  if (dimension < 0 || dimension > static_cast<index_type>(this->n_dims()))
  {
    throw error::index_error("Dimension out of range in unsqueeze");
  }

  this->shape_().value_.insert(this->shape_().value_.begin() + dimension, 1);

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::squeeze_(index_type dimension) {
  return *this;
}