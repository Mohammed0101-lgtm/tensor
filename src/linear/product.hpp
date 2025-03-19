#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::cross_product(const tensor& __other) const {
  if (this->empty() || __other.empty())
    throw std::invalid_argument("Cannot cross product an empty vector");

  if (this->shape() != std::vector<int>{3} || __other.shape() != std::vector<int>{3})
    throw std::invalid_argument("Cross product can only be performed on 3-element vectors");

  tensor __ret({3});

#if defined(__ARM_NEON) && defined(__aarch64__)
  return this->neon_cross_product(__other);
#elif defined(__CUDACC__)
  pointer __d_a;
  pointer __d_b;
  pointer __d_c;

  cudaMalloc(&__d_a, 3 * sizeof(value_type));
  cudaMalloc(&__d_b, 3 * sizeof(value_type));
  cudaMalloc(&__d_c, 3 * sizeof(value_type));

  cudaMemcpy(__d_a, this->__data_.data(), 3 * sizeof(value_type), cudaMemcpyHostToDevice);
  cudaMemcpy(__d_b, __other.storage().data(), 3 * sizeof(value_type), cudaMemcpyHostToDevice);

  dim3 block(1);
  dim3 grid(1);
  cross_product_kernel<<<grid, block>>>(__d_a, __d_b, __d_c);

  cudaMemcpy(__ret.storage().data(), __d_c, 3 * sizeof(value_type), cudaMemcpyDeviceToHost);

  cudaFree(__d_a);
  cudaFree(__d_b);
  cudaFree(__d_c);

#else
  const_reference __a1 = this->__data_[0];
  const_reference __a2 = this->__data_[1];
  const_reference __a3 = this->__data_[2];

  const_reference __b1 = __other[0];
  const_reference __b2 = __other[1];
  const_reference __b3 = __other[2];

  __ret[0] = __a2 * __b3 - __a3 * __b2;
  __ret[1] = __a3 * __b1 - __a1 * __b3;
  __ret[2] = __a1 * __b2 - __a2 * __b1;
#endif

  return __ret;
}

#ifdef __CUDACC__
template <class _Tp>
__global__ void cross_product_kernel(_Tp* __a, _Tp* __b, _Tp* __c) {
  __c[0] = __a[1] * __b[2] - __a[2] * __b[1];
  __c[1] = __a[2] * __b[0] - __a[0] * __b[2];
  __c[2] = __a[0] * __b[1] - __a[1] * __b[0];
}
#endif


template <class _Tp>
tensor<_Tp> tensor<_Tp>::dot(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_dot(__other);
#endif
  if (this->empty() || __other.empty())
    throw std::invalid_argument("Cannot dot product an empty vector");

  if (this->__shape_.size() == 1 && __other.shape().size() == 1) {
    if (this->__shape_[0] != __other.shape()[0])
      throw std::invalid_argument("Vectors must have the same size for dot product");

    const_pointer __this_data  = this->__data_.data();
    const_pointer __other_data = __other.storage().data();
    const size_t  __size       = this->__data_.size();
    value_type    __ret        = 0;

    __ret = std::inner_product(__this_data, __this_data + __size, __other_data, value_type(0));
    return __self({__ret}, {1});
  }

  if (this->__shape_.size() == 2 && __other.shape().size() == 2) return this->matmul(__other);

  if (this->__shape_.size() == 3 && __other.shape().size() == 3)
    return this->cross_product(__other);

  return __self();
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::cumprod(index_type __dim) const {
  if (__dim == -1) {
    data_t __flat = this->__data_;
    data_t __ret(__flat.size());
    __ret[0] = __flat[0];

    index_type __i = 1;

    for (; __i < __flat.size(); ++__i) __ret[__i] = __ret[__i - 1] * __flat[__i];

    return __self(__ret, {__flat.size()});
  } else {
    if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
      throw std::invalid_argument("Invalid dimension provided.");

    data_t __ret(this->__data_);
    // TODO : compute_outer_size() implementation
    index_type __outer_size = this->__compute_outer_size(__dim);
    index_type __inner_size = this->__shape_[__dim];
    index_type __st         = this->__strides_[__dim];

    index_type __i = 0;

    for (; __i < __outer_size; ++__i) {
      index_type __base = __i * __st;
      __ret[__base]     = __data_[__base];
      index_type __j    = 1;

      for (; __j < __inner_size; ++__j) {
        index_type __curr = __base + __j;
        __ret[__curr]     = __ret[__base + __j - 1] * __data_[__curr];
      }
    }
#endif

    return __self(__ret, this->__shape_);
  }
}
