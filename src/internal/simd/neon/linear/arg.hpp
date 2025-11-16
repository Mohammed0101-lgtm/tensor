#pragma once

#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
arch::tensor<_u64> argmax_(const arch::tensor<_Tp>& t, const _u64 dimension)
{
  if (dimension < 0 || dimension >= t.shape().size())
  {
    throw error::index_error("Dimension out of range in argmax");
  }

  typename arch::tensor<_Tp>::container_type& data_ = t.storage_();
  arch::tensor<_u64>                          ret;
  shape::Shape                                ret_sh = t.shape();
  ret_sh.__value_.erase(ret_sh.__value_.begin() + dimension);
  ret.shape() = ret_sh;
  ret.storage_().resize(ret_sh.flatten_size());

  _u64 outer_size = 1;
  _u64 inner_size = 1;
  _u64 i          = 0;

  for (; i < dimension; ++i)
  {
    outer_size *= t.shape()[i];
  }

  for (i = dimension + 1; i < t.n_dims(); ++i)
  {
    inner_size *= t.shape_[i];
  }

  const _u64 simd_end = data_.size() - (data_.size() % t.simd_width);

  for (i = 0; i < outer_size; ++i)
  {
    _u64 j = 0;
    for (; j < inner_size; ++j)
    {
      neon_type<_Tp> max_vec       = neon_dup<_Tp>(-std::numeric_limits<_Tp>::infinity());
      neon_type<_Tp> index_vec     = neon_dup<_Tp>(_Tp(0.0f));
      neon_type<_Tp> increment     = neon_dup<_Tp>(_Tp(1.0f));
      neon_type<_Tp> current_index = {_Tp(0), _Tp(1), _Tp(2), _Tp(3)};
      _u64           k             = 0;

      for (; k + t.simd_width <= t.shape()[dimension]; k += t.simd_width)
      {
        neon_type<_Tp> data_vec = neon_load<_Tp>(&data_[(i * t.shape()[dimension] + k) * inner_size + j]);
        neon_type<_Tp> mask     = neon_vcgtq<_Tp>(data_vec, max_vec);
        max_vec                 = neon_vbslq<_Tp>(mask, data_vec, max_vec);
        index_vec               = neon_vbslq<_Tp>(mask, current_index, index_vec);
        current_index           = neon_add<_Tp>(current_index, increment);
      }

      alignas(16) _Tp  max_vals[t.simd_width];
      alignas(16) _u32 indices[t.simd_width];

      neon_store<_Tp>(max_vals, max_vec);
      neon_store<_Tp>(indices, index_vec);

      _Tp  max_val   = max_vals[0];
      _u32 max_index = indices[0];

      for (int k = 1; k < t.simd_width; ++k)
      {
        if (max_vals[k] > max_val)
        {
          max_val   = max_vals[k];
          max_index = indices[k];
        }

        for (; k < t.shape()[dimension]; ++k)
        {
          _Tp v = data_[(i * t.shape_[dimension] + k) * inner_size + j];

          if (v > max_val)
          {
            max_val   = v;
            max_index = k;
          }
        }
        ret[i * inner_size + j] = max_index;
      }
    }
  }

  return ret;
}

template<class _Tp>
arch::tensor<_Tp> argmax(const arch::tensor<_Tp>& t, _u64 dimension)
{
  if (dimension < 0 || dimension >= t.shape().size())
  {
    throw error::index_error("Dimension out of range in argmax");
  }

  typename arch::tensor<_Tp>::container_type& data_ = t.storage_();
  arch::tensor<_Tp>                           ret;
  shape::Shape                                ret_sh = t.shape();

  ret_sh.__value_.erase(ret_sh.__value_.begin() + dimension);
  ret.shape_ = ret_sh;
  ret.storage_().resize(ret_sh.flatten_size(), _Tp(0));

  _u64 outer_size = 1;
  _u64 inner_size = 1;
  _u64 i          = 0;

  for (; i < dimension; ++i)
  {
    outer_size *= t.shape()[i];
  }

  for (i = dimension + 1; i < static_cast<_u64>(t.shape().size()); ++i)
  {
    inner_size *= t.shape()[i];
  }

  const _u64 simd_end = data_.size() - (data_.size() % t.simd_width);

  for (i = 0; i < outer_size; ++i)
  {
    for (_u64 j = 0; j < inner_size; ++j)
    {
      neon_type<_Tp> max_vec = neon_dup<_Tp>(-std::numeric_limits<_Tp>::infinity());
      _u64           k       = 0;

      for (; k + t.simd_width <= t.shape()[dimension]; k += t.simd_width)
      {
        neon_type<_Tp> data_vec = neon_load<_Tp>(&data_[(i * t.shape()[dimension] + k) * inner_size + j]);
        max_vec                 = neon_max<_Tp>(max_vec, data_vec);
      }

      _Tp max_value = neon_maxv<_Tp>(max_vec);

      for (; k < t.shape()[dimension]; ++k)
      {
        _Tp v     = data_[(i * t.shape()[dimension] + k) * inner_size + j];
        max_value = std::max(max_value, v);
      }

      ret[i * inner_size + j] = max_value;
    }
  }

  return ret;
}

template<class _Tp>
arch::tensor<_u64> argsort(const arch::tensor<_Tp>& t, _u64 d, bool ascending)
{
  typename arch::tensor<_Tp>::container_type& data_    = t.storage_();
  _u64                                        adjusted = (d < 0) ? d + data_.size() : d;

  if (adjusted != 0)
  {
    throw error::index_error("Invalid dimension for argsort: only 1D tensors are supported");
  }

  _u64         size = static_cast<_u64>(data_.size());
  shape::Shape indices(size);
  std::iota(indices.__value_.begin(), indices.__value_.end(), 0);

  const _u64 simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64       i        = 0;

  if constexpr (std::is_floating_point_v<_Tp>)
  {
    for (; i + t.simd_width <= size; i += t.simd_width)
    {
      neon_f32    data_vec   = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
      float32x2_t min1       = vpmin_f32(vget_low_f32(data_vec), vget_high_f32(data_vec));
      float32x2_t min2       = vpmin_f32(min1, min1);
      neon_f32    cmp_vec    = vdupq_lane_f32(min2, 0);
      neon_u32    cmp_result = ascending ? vcltq_f32(data_vec, cmp_vec) : vcgtq_f32(data_vec, cmp_vec);

      for (int j = 0; j < t.simd_width; ++j)
        indices[i + j] = (cmp_result[j] ? i + j : i + j + 1);
    }
  }
  else if constexpr (std::is_signed_v<_Tp>)
  {
    for (; i + t.simd_width <= size; i += t.simd_width)
    {
      neon_s32  data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
      int32x2_t min1       = vpmin_s32(vget_low_s32(data_vec), vget_high_s32(data_vec));
      int32x2_t min2       = vpmin_s32(min1, min1);
      neon_s32  cmp_vec    = vdupq_lane_s32(min2, 0);
      neon_u32  cmp_result = ascending ? vcltq_s32(data_vec, cmp_vec) : vcgtq_s32(data_vec, cmp_vec);

      for (int j = 0; j < t.simd_width; ++j)
        indices[i + j] = (cmp_result[j] ? i + j : i + j + 1);
    }
  }
  else if constexpr (std::is_unsigned_v<_Tp>)
  {
    for (; i + t.simd_width <= size; i += t.simd_width)
    {
      neon_u32   data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
      uint32x2_t min1       = vpmin_u32(vget_low_u32(data_vec), vget_high_u32(data_vec));
      uint32x2_t min2       = vpmin_u32(min1, min1);
      neon_u32   cmp_vec    = vdupq_lane_u32(min2, 0);
      neon_u32   cmp_result = ascending ? vcltq_u32(data_vec, cmp_vec) : vcgtq_u32(data_vec, cmp_vec);

      for (int j = 0; j < t.simd_width; ++j)
        indices[i + j] = (cmp_result[j] ? i + j : i + j + 1);
    }
  }

  for (; i < size; ++i)
  {
    indices[i] = i;
  }

  std::sort(indices.__value_.begin(), indices.__value_.end(),
            [&](_u64 a, _u64 b) { return ascending ? data_[a] < data_[b] : data_[a] > data_[b]; });

  return arch::tensor<_u64>(std::move(indices));
}

}