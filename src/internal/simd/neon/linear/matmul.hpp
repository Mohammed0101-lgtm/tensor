#pragma once

#include "tensor.hpp"
#include <vector>


namespace internal::simd::neon {

template<class _Tp>
arch::tensor<_Tp> matmul(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  if constexpr (!internal::concepts::has_plus_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a plus operator");
  }

  if constexpr (!internal::concepts::has_times_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have times operator");
  }

  if (t.shape().size() < 2 || other.shape().size() < 2)
  {
    throw error::shape_error("matmul is only supported for 2D tensors");
  }

  if (!t.shape().equal(shape::Shape({t.shape()[0], t.shape()[1]}))
      || !other.shape().equal(shape::Shape({other.shape()[0], other.shape()[1]})))
  {
    throw error::shape_error("matmul is only supported for 2D tensors");
  }

  if (t.shape()[1] != other.shape()[0])
  {
    if (t.shape()[0] == other.shape()[1])
    {
      return other.matmul(t);
    }

    throw error::shape_error("Shape mismatch for matrix multiplication: "
                             "this shape: ["
                             + std::to_string(t.shape()[0]) + ", " + std::to_string(t.shape()[1])
                             + "] "
                               "other shape: ["
                             + std::to_string(other.shape()[0]) + ", " + std::to_string(other.shape()[1]) + "]");
  }


  typename arch::tensor<_Tp>::container_type& data_  = t.storage_();
  shape::Shape                                ret_sh = {t.shape()[0], other.shape()[1]};
  typename arch::tensor<_Tp>::container_type  ret_d(ret_sh[0] * ret_sh[1], _Tp(0));
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);

  if constexpr (std::is_same_v<_Tp, _f32>)
  {
    for (_u64 i = 0; i < ret_sh[0]; i += t.simd_width)
    {
      for (_u64 j = 0; j < ret_sh[1]; j += t.simd_width)
      {
        for (_u64 k = 0; k < t.shape()[1]; k += t.simd_width)
        {
          for (_u64 ii = i; ii < std::min(static_cast<_u64>(i + t.simd_width), ret_sh[0]); ++ii)
          {
            for (_u64 jj = j; jj < std::min(static_cast<_u64>(j + t.simd_width), ret_sh[1]); ++jj)
            {
              neon_f32 sum_vec = vdupq_n_f32(0);

              for (_u64 kk = k; kk < std::min(static_cast<_u64>(k + t.simd_width), t.shape()[1]); kk += t.simd_width)
              {
                neon_f32 a_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[ii * t.shape()[1] + kk]));
                neon_f32 b_vec = vld1q_f32(reinterpret_cast<const _f32*>(&other.data_[kk * other.shape()[1] + jj]));
                sum_vec        = vmlaq_f32(sum_vec, a_vec, b_vec);
              }

              float32x2_t sum_low  = vget_low_f32(sum_vec);
              float32x2_t sum_high = vget_high_f32(sum_vec);
              sum_low              = vadd_f32(sum_low, sum_high);
              float32x2_t sum_dup  = vpadd_f32(sum_low, sum_low);
              ret_d[ii * ret_sh[1] + jj] += vget_lane_f32(sum_dup, 0);
            }
          }
        }
      }
    }
  }
  else if constexpr (std::is_same_v<_Tp, _s32>)
  {
    for (_u64 i = 0; i < ret_sh[0]; i += t.simd_width)
    {
      for (_u64 j = 0; j < ret_sh[1]; j += t.simd_width)
      {
        for (_u64 k = 0; k < t.shape()[1]; k += t.simd_width)
        {
          for (_u64 ii = i; ii < std::min(static_cast<_u64>(i + t.simd_width), ret_sh[0]); ++ii)
          {
            for (_u64 jj = j; jj < std::min(static_cast<_u64>(j + t.simd_width), ret_sh[1]); ++jj)
            {
              neon_s32 sum_vec = vdupq_n_s32(0);

              for (_u64 kk = k; kk < std::min(static_cast<_u64>(k + t.simd_width), t.shape()[1]); kk += t.simd_width)
              {
                neon_s32 a_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[ii * t.shape()[1] + kk]));
                neon_s32 b_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other.data_[kk * other.shape()[1] + jj]));
                sum_vec        = vmlaq_s32(sum_vec, a_vec, b_vec);
              }

              int32x2_t sum_low  = vget_low_s32(sum_vec);
              int32x2_t sum_high = vget_high_s32(sum_vec);
              sum_low            = vadd_s32(sum_low, sum_high);
              int32x2_t sum_dup  = vpadd_s32(sum_low, sum_low);
              ret_d[ii * ret_sh[1] + jj] += vget_lane_s32(sum_dup, 0);
            }
          }
        }
      }
    }
  }
  else if constexpr (std::is_unsigned_v<_Tp>)
  {
    for (_u64 i = 0; i < ret_sh[0]; i += t.simd_width)
    {
      for (_u64 j = 0; j < ret_sh[1]; j += t.simd_width)
      {
        for (_u64 k = 0; k < t.shape()[1]; k += t.simd_width)
        {
          for (_u64 ii = i; ii < std::min(static_cast<_u64>(i + t.simd_width), ret_sh[0]); ++ii)
          {
            for (_u64 jj = j; jj < std::min(static_cast<_u64>(j + t.simd_width), ret_sh[1]); ++jj)
            {
              neon_u32 sum_vec = vdupq_n_u32(0);

              for (int64_t kk = k; kk < std::min(static_cast<_u64>(k + t.simd_width), t.shape()[1]); kk += t.simd_width)
              {
                neon_u32 a_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[ii * t.shape()[1] + kk]));
                neon_u32 b_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other.data_[kk * other.shape()[1] + jj]));
                sum_vec        = vmlaq_u32(sum_vec, a_vec, b_vec);
              }

              uint32x2_t sum_low  = vget_low_u32(sum_vec);
              uint32x2_t sum_high = vget_high_u32(sum_vec);
              sum_low             = vadd_u32(sum_low, sum_high);
              uint32x2_t sum_dup  = vpadd_u32(sum_low, sum_low);
              ret_d[ii * ret_sh[1] + jj] += vget_lane_u32(sum_dup, 0);
            }
          }
        }
      }
    }
  }

  for (_u64 i = 0; i < ret_sh[0]; ++i)
  {
    for (_u64 j = 0; j < ret_sh[1]; ++j)
    {
      _Tp sum = _Tp(0);

      for (_u64 k = 0; k < t.shape()[1]; ++k)
      {
        sum = sum + (data_[i * t.shape()[1] + k] * other[k * other.shape()[1] + j]);
      }

      ret_d[i * ret_sh[1] + j] = sum;
    }
  }

  return self(std::move(ret_sh), std::move(ret_d));
}

}