#pragma once

#include "tensor.hpp"

template<class _Tp>
arch::tensor<_Tp> arch::tensor<_Tp>::det() const
{
  if (this->n_dims() < 2)
  {
    throw error::shape_error("det: tensor must be at least 2D");
  }

  index_type h, w;

  if (this->n_dims() == 2)
  {
    h = this->shape()[0];
    w = this->shape()[1];
  }
  else
  {
    index_type last        = this->n_dims() - 1;
    index_type second_last = this->n_dims() - 2;

    if (this->shape_()[last] == 1)
    {
      h = this->shape_()[second_last - 1];
      w = this->shape_()[second_last];
    }
    else if (this->shape_()[second_last] == 1)
    {
      h = this->shape_()[last - 1];
      w = this->shape_()[last];
    }
    else
    {
      h = this->shape_()[second_last];
      w = this->shape_()[last];
    }
  }

  if (h != w)
  {
    throw error::shape_error("det: tensor must be a square matrix (n x n)");
  }

  index_type n = h;

  if (n == 2)
  {
    return arch::tensor<_Tp>({1}, {(*this)({0, 0}) * (*this)({1, 1}) - (*this)({0, 1}) * (*this)({1, 0})});
  }

  value_type determinant = 0;

  for (index_type col = 0; col < n; ++col)
  {
    arch::tensor<_Tp> minor = get_minor(0, col);
    value_type        sign  = (col % 2 == 0) ? 1 : -1;
    determinant += sign * (*this)({0, col}) * minor.det()[0];
  }

  return arch::tensor<_Tp>({1}, {determinant});
}