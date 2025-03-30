#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::transpose() const {
  if (this->__shape_.size() != 2)
    throw __shape_error__("Matrix transposition can only be done on 2D tensors");

  tensor           __ret({this->__shape_[1], this->__shape_[0]});
  const index_type __rows = this->__shape_[0];
  const index_type __cols = this->__shape_[1];

#ifdef __CUDACC__
  if (this->__is_cuda_tensor) {
    dim3 blockDim(16, 16);
    dim3 gridDim((__cols + blockDim.x - 1) / blockDim.x, (__rows + blockDim.y - 1) / blockDim.y);
    transpose_kernel<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(this->__data_.data()),
                                            thrust::raw_pointer_cast(__ret.__data_.data()), __rows,
                                            __cols);
    cudaDeviceSynchronize();
    return __ret;
  }
#endif

#if defined(__ARM_NEON)
  return this->neon_transpose();
#endif

  index_type __i = 0;

  for (; __i < __rows; ++__i) {
    index_type __j = 0;

    for (; __j < __cols; ++__j) __ret.at({__j, __i}) = this->at({__i, __j});
  }

  return __ret;
}

#ifdef __CUDACC__
template <class _Tp>
__global__ void transpose_kernel(_Tp* __input, _Tp* __output, int __rows, int __cols) {
  int __i = blockIdx.y * blockDim.y + threadIdx.y;
  int __j = blockIdx.x * blockDim.x + threadIdx.x;

  if (__i < __rows && __j < __cols) output[__j * __rows + __i] = input[__i * __cols + __j];
}
#endif

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::transpose_() {
  if (this->__shape_.size() != 2)
    throw __shape_error__("Transpose operation is only valid for 2D tensors");

  const index_type __rows = this->__shape_[0];
  const index_type __cols = this->__shape_[1];

  if (__rows != __cols)
    throw __shape_error__("In-place transpose is only supported for square tensors");

  for (index_type __i = 0; __i < __rows; ++__i)
    for (index_type __j = __i + 1; __j < __cols; ++__j)
      std::swap(this->__data_[__i * __cols + __j], this->__data_[__j * __cols + __i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::transpose_() const {
  return this->transpose_();
}
