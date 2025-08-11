#pragma once

#include "tensor.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::matmul(const tensor& other) const {
  static_assert(internal::types::has_times_operator_v<value_type>,
                "Error: value_type must support multiplication (*) for matmul");
  static_assert(internal::types::has_plus_operator_v<value_type>,
                "Error: value_type must support addition (+) for matmul");

  if (this->n_dims() < 2 || other.n_dims() < 2)
  {
    throw error::shape_error("matmul is only supported for tensors with at least 2 dimensions");
  }

  // Helper lambda to extract effective matrix dimensions
  auto extract_hw = [](const tensor& t) -> std::pair<index_type, index_type> {
    const auto& shape = t.shape();
    index_type  ndims = t.n_dims();

    if (ndims == 2)
    {
      return {shape[0], shape[1]};
    }

    index_type last        = ndims - 1;
    index_type second_last = ndims - 2;

    if (second_last > 0 && shape[last] == 1)
    {
      return {shape[second_last - 1], shape[second_last]};
    }

    if (shape[second_last] == 1)
    {
      return {shape[last - 1], shape[last]};
    }

    return {shape[second_last], shape[last]};
  };

  // Get shapes
  auto [h, w]   = extract_hw(*this);
  auto [h1, w1] = extract_hw(other);

  if (w != h1)
  {
    throw error::shape_error("Shape mismatch in matmul: "
                             "lhs = ["
                             + std::to_string(h) + ", " + std::to_string(w)
                             + "], "
                               "rhs = ["
                             + std::to_string(h1) + ", " + std::to_string(w1) + "]");
  }

  shape_type     result_shape = {h, w1};
  container_type result_data(h * w1, value_type(0));

  for (int64_t i = 0; i < h; ++i)
  {
    for (int64_t j = 0; j < w1; ++j)
    {
      value_type sum = value_type(0);

      for (int64_t k = 0; k < w; ++k)
      {
        sum += (*this)[i * w + k] * other[k * w1 + j];
      }

      result_data[i * w1 + j] = sum;
    }
  }

  return tensor<_Tp>(result_shape, std::move(result_data));
}

#ifdef CUDACC
template<class _Tp>
global void matmul_kernel(_Tp* a, _Tp* b, _Tp* c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < k)
  {
    _Tp sum = 0;

    for (int i = 0; i < n; ++i)
    {
      sum += a[row * n + i] * b[i * k + col];
    }

    c[row * k + col] = sum;
  }
}
#endif