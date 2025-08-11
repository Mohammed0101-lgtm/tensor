#pragma once

#include "tensor.hpp"


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::transpose_() {
  if (this->n_dims() != 2)
    throw error::shape_error("Transpose operation is only valid for 2D tensors");

  const index_type rows = this->shape_()[0];
  const index_type cols = this->shape_()[1];

  if (rows != cols)
    throw error::shape_error("In-place transpose is only supported for square tensors");

  for (index_type i = 0; i < rows; ++i)
    for (index_type j = i + 1; j < cols; ++j)
      std::swap((*this)[i * cols + j], (*this)[j * cols + i]);

  return *this;
}