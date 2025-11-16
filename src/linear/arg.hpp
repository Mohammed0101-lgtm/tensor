#pragma once

#include "tensor.hpp"

template<class _Tp>
arch::tensor<typename arch::tensor<_Tp>::index_type> arch::tensor<_Tp>::argmax_(index_type dimension) const
{
  if (dimension < 0 || dimension >= this->n_dims())
  {
    throw error::index_error("Dimension out of range in argmax");
  }

  shape::Shape ret_sh = this->shape();
  ret_sh.__value_.erase(ret_sh.__value_.begin() + dimension);
  arch::tensor<index_type> ret(ret_sh);
  ret.storage_().resize(ret_sh.flatten_size());

  index_type outer_size = 1;
  index_type inner_size = 1;
  index_type i          = 0;

  for (; i < dimension; ++i)
  {
    outer_size *= this->shape_()[i];
  }

  for (i = dimension + 1; i < this->n_dims(); ++i)
  {
    inner_size *= this->shape_()[i];
  }

  for (i = 0; i < outer_size; ++i)
  {
    index_type j = 0;

    for (; j < inner_size; ++j)
    {
      index_type max_index = 0;
      value_type max_value = (*this)[i * this->shape_()[dimension] * inner_size + j];
      index_type k         = 1;

      for (; k < this->shape_()[dimension]; ++k)
      {
        value_type v = (*this)[(i * this->shape_()[dimension] + k) * inner_size + j];

        if (v > max_value)
        {
          max_value = v;
          max_index = k;
        }
      }

      ret[i * inner_size + j] = max_index;
    }
  }

  return ret;
}

template<class _Tp>
arch::tensor<_Tp> arch::tensor<_Tp>::argmax(index_type dimension) const
{
  if (dimension < 0 || dimension >= this->n_dims())
  {
    throw error::index_error("Dimension out of range in argmax");
  }

  tensor       ret;
  shape::Shape ret_sh = this->shape();

  ret_sh.__value_.erase(ret_sh.__value_.begin() + dimension);
  ret.shape_ = ret_sh;
  ret.data_.resize(ret_sh.flatten_size(), value_type(0));

  index_type outer_size = 1;
  index_type inner_size = 1;
  index_type i          = 0;

  for (; i < dimension; ++i)
  {
    outer_size *= this->shape_()[i];
  }

  for (i = dimension + 1; i < static_cast<index_type>(this->n_dims()); ++i)
  {
    inner_size *= this->shape_()[i];
  }

  for (i = 0; i < outer_size; ++i)
  {
    index_type j = 0;

    for (; j < inner_size; ++j)
    {
      value_type max_value = (*this)[i * this->shape_()[dimension] * inner_size + j];
      index_type k         = 1;

      for (; k < this->shape_()[dimension]; ++k)
      {
        value_type v = (*this)[(i * this->shape_()[dimension] + k) * inner_size + j];

        if (v > max_value)
        {
          max_value = v;
        }
      }

      ret.data_[i * inner_size + j] = max_value;
    }
  }

  return ret;
}

template<class _Tp>
arch::tensor<typename arch::tensor<_Tp>::index_type> arch::tensor<_Tp>::argsort(index_type d, bool ascending) const
{
  index_type adjusted = (d < 0) ? d + this->size(0) : d;

  if (adjusted != 0)
  {
    throw error::index_error("Invalid dimension for argsort: only 1D tensors are supported");
  }

  index_type size = static_cast<index_type>(this->size(0));
  shape_type indices(size);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&](index_type a, index_type b) { return ascending ? (*this)[a]<(*this)[b] : (*this)[a]>(*this)[b]; });

  return arch::tensor<index_type>(std::move(indices));
}