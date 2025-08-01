#pragma once

#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
tensor<_Tp> transpose(const tensor<_Tp>& t) {
  if (!t.shape().equal(shape::Shape({t.shape()[0], t.shape()[1]})))
  {
    throw error::shape_error("Matrix transposition can only be done on 2D tensors");
  }

  std::vector<_Tp>& data_ = t.storage_();
  tensor            ret({t.shape()[1], t.shape()[0]});
  const _u64        rows     = t.shape()[0];
  const _u64        cols     = t.shape()[1];
  const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64              i        = 0;

  for (; i < rows; i += t.simd_width)
  {
    for (_u64 j = 0; j < cols; j += t.simd_width)
    {
      if (i + t.simd_width <= rows && j + t.simd_width <= cols)
      {
        wide_neon_type<_Tp> input;

        for (_u64 k = 0; k < t.simd_width; ++k)
        {
          input[k] = neon_load<_Tp>(&data_[(i + k) * cols + j]);
        }

        wide_neon_type<_Tp> output = wide_neon_load<_Tp>(&input);

        for (_u64 k = 0; k < t.simd_width; ++k)
        {
          neon_store<_Tp>(&ret[(j + k) * rows + i], output[k]);
        }
      }
      else
      {
        for (_u64 ii = i; ii < std::min(static_cast<_u64>(i + t.simd_width), rows); ++ii)
        {
          for (_u64 jj = j; jj < std::min(static_cast<_u64>(j + t.simd_width), cols); ++jj)
          {
            ret.at({jj, ii}) = t.at({ii, jj});
          }
        }
      }
    }
  }

  for (; i < rows; ++i)
  {
    _u64 j = 0;

    for (; j < cols; ++j)
    {
      ret.at({j, i}) = t.at({i, j});
    }
  }

  return ret;
}

}