#pragma once

#include "tensor.hpp"

template<class _Tp>
arch::tensor<_Tp> arch::tensor<_Tp>::cross_product(const tensor& other) const
{
  if (this->empty() || other.empty())
  {
    throw std::invalid_argument("Cannot cross product an empty vector");
  }

  if (this->shape() != shape_type{3} || other.shape() != shape_type{3})
  {
    throw std::logic_error("Cross product can only be performed on 3-element vectors");
  }

  tensor          ret({3});
  const_reference a1 = (*this)[0];
  const_reference a2 = (*this)[1];
  const_reference a3 = (*this)[2];

  const_reference b1 = other[0];
  const_reference b2 = other[1];
  const_reference b3 = other[2];

  ret[0] = a2 * b3 - a3 * b2;
  ret[1] = a3 * b1 - a1 * b3;
  ret[2] = a1 * b2 - a2 * b1;

  return ret;
}

template<class _Tp>
arch::tensor<_Tp> arch::tensor<_Tp>::dot(const tensor& other) const
{
  if (this->empty() || other.empty())
  {
    throw std::invalid_argument("Cannot dot product an empty vector");
  }

  if (this->n_dims() == 1 && other.n_dims() == 1)
  {
    if (this->shape_()[0] != other.shape_()[0])
    {
      throw error::shape_error("Vectors must have the same size for dot product");
    }

    const_pointer     this_data     = this->storage_().data();
    container_type    other_storage = other.storage();
    const_pointer     other_data    = other_storage.data();
    const std::size_t size          = this->size(0);
    value_type        ret           = std::inner_product(this_data, this_data + size, other_data, value_type(0));

    return self({1}, {ret});
  }

  if (this->n_dims() == 2 && other.shape().size() == 2)
  {
    return matmul(other);
  }

  if (this->n_dims() == 3 && other.shape().size() == 3)
  {
    return cross_product(other);
  }

  return self();
}

template<class _Tp>
arch::tensor<_Tp> arch::tensor<_Tp>::cumprod(index_type dimension) const
{
  if (dimension == -1)
  {
    container_type flat = this->storage_();
    container_type ret(flat.size());
    ret[0]       = flat[0];
    index_type i = 1;

    for (; i < flat.size(); ++i)
    {
      ret[i] = ret[i - 1] * flat[i];
    }

    return self(ret, {flat.size()});
  }
  else
  {
    if (dimension < 0 || dimension >= static_cast<index_type>(this->n_dims()))
    {
      throw error::index_error("Invalid dimension provided for cumprod.");
    }

    container_type ret(this->storage_());
    // TODO : compute_outer_size() implementation
    index_type outer_size = compute_outer_size(dimension);
    index_type inner_size = this->shape_()[dimension];
    index_type st         = this->shape_().strides()[dimension];
    index_type i          = 0;

    for (; i < outer_size; ++i)
    {
      index_type base = i * st;
      ret[base]       = (*this)[base];
      index_type j    = 1;

      for (; j < inner_size; ++j)
      {
        index_type curr = base + j;
        ret[curr]       = ret[base + j - 1] * (*this)[curr];
      }
    }

    return self(std::move(this->shape()), std::move(ret));
  }
}
