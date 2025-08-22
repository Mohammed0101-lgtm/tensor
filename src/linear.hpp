#pragma once

#include "tensor.hpp"
#include "types.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::reshape(const shape::Shape sh) const {
  container_type d = this->storage();
  index_type     s = sh.flatten_size();

  if (s != this->size(0))
  {
    throw error::shape_error(
      "input shape must have size of elements equal to the current number of elements in the tensor data");
  }

  return tensor<_Tp>(std::move(sh), std::move(d), std::move(this->device()));
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::absolute(const tensor& other) const {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::absolute(*this, other);
  }

  index_type     s = other.storage().size();
  container_type a(s);
  index_type     i = 0;

  for (; i < s; ++i)
  {
    a[i] = static_cast<value_type>(std::fabs(static_cast<_f32>(other[i])));
  }

  return self(other.shape(), a);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::pop_back() const {
  if (this->shape().equal(shape::Shape({1, this->shape()[0]})))
  {
    throw error::index_error("push_back is only supported for one dimensional tensors");
  }

  this->storage_().pop_back();
  --(this->shape()[0]);
  this->shape_().compute_strides();
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cat(const std::vector<tensor<_Tp>>& others, index_type dimension) const {
  for (const tensor& t : others)
  {
    index_type i = 0;

    for (; i < this->n_dims(); ++i)
    {
      if (i != dimension && this->shape_()[i] != t.shape_()[i])
      {
        throw error::shape_error("Cannot concatenate tensors with different shapes along non-concatenation "
                                 "dimensions");
      }
    }
  }

  shape::Shape ret_sh = this->shape();

  for (const tensor& t : others)
  {
    ret_sh[dimension] += t.shape_[dimension];
  }

  container_type c;
  c.reserve(this->size(0));
  c.insert(c.end(), this->storage_().begin(), this->storage_().end());

  for (const tensor& t : others)
  {
    c.insert(c.end(), t.storage_().begin(), t.storage_().end());
  }

  return self(ret_sh, c);
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::resize_as_(const shape::Shape sh) {
  // TODO: implement in place resize as here
  this->shape_() = sh;
  this->shape_().compute_strides();
  return *this;
}
