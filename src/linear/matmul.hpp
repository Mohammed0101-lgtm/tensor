#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::matmul(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_matmul(__other);
#endif

  static_assert(has_times_operator_v<value_type>, "Value type must have a times operator");
  static_assert(has_plus_operator_v<value_type>, "Value type must have a plus operator");

  if (this->n_dims() < 2 || __other.n_dims() < 2)
    throw __shape_error__("matmul is only supported for 2D tensors");

  // Get shape of 'this' tensor
  index_type __h, __w;
  if (this->n_dims() == 2) {
    __h = this->__shape_[0];
    __w = this->__shape_[1];
  } else {
    index_type __last        = this->n_dims() - 1;
    index_type __second_last = this->n_dims() - 2;

    if (__second_last > 0 && this->__shape_[__last] == 1) {
      __h = this->__shape_[__second_last - 1];
      __w = this->__shape_[__second_last];
    } else if (this->__shape_[__second_last] == 1) {
      __h = this->__shape_[__last - 1];
      __w = this->__shape_[__last];
    } else {
      __h = this->__shape_[__second_last];
      __w = this->__shape_[__last];
    }
  }

  // Get shape of 'other' tensor
  index_type __h1, __w1;
  if (__other.n_dims() == 2) {
    __h1 = __other.shape()[0];
    __w1 = __other.shape()[1];
  } else {
    index_type __last        = __other.n_dims() - 1;
    index_type __second_last = __other.n_dims() - 2;

    if (__second_last > 0 && __other.shape()[__last] == 1) {
      __h1 = __other.shape()[__second_last - 1];
      __w1 = __other.shape()[__second_last];
    } else if (__other.shape()[__second_last] == 1) {
      __h1 = __other.shape()[__last - 1];
      __w1 = __other.shape()[__last];
    } else {
      __h1 = __other.shape()[__second_last];
      __w1 = __other.shape()[__last];
    }
  }

  // Validate shapes for matrix multiplication
  if (__w != __h1) {
    throw __shape_error__(
        "Shape mismatch for matrix multiplication: "
        "this shape: [" +
        std::to_string(__h) + ", " + std::to_string(__w) +
        "] "
        "other shape: [" +
        std::to_string(__h1) + ", " + std::to_string(__w1) + "]");
  }

  // Define output shape
  shape_type __ret_sh = {__h, __w1};
  data_t     __ret_d(__h * __w1, value_type(0));

#pragma omp parallel for collapse(2)
  for (int64_t __i = 0; __i < __h; ++__i) {
    for (int64_t __j = 0; __j < __w1; ++__j) {
      value_type __sum = value_type(0);
      for (int64_t __k = 0; __k < __w; ++__k)
        __sum = __sum + (this->__data_[__i * __w + __k] * __other.__data_[__k * __w1 + __j]);

      __ret_d[__i * __w1 + __j] = __sum;
    }
  }

  return tensor<_Tp>(__ret_sh, __ret_d);
}

#ifdef __CUDACC__
template <class _Tp>
__global__ void matmul_kernel(_Tp* __a, _Tp* __b, _Tp* __c, int __m, int __n, int __k) {
  int __row = blockIdx.y * blockDim.y + threadIdx.y;
  int __col = blockIdx.x * blockDim.x + threadIdx.x;

  if (__row < __m && __col < __k) {
    _Tp __sum = 0;
    for (int __i = 0; __i < __n; ++__i) __sum += __a[__row * __n + __i] * __b[__i * __k + __col];

    __c[__row * __k + __col] = __sum;
  }
}
#endif