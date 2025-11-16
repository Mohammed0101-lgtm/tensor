#pragma once

#include "internal/simd/neon/alias.hpp"
#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
arch::tensor<_Tp>& logical_or_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  if (!std::is_integral_v<_Tp>)
  {
    throw error::type_error("Cannot get the element wise not of non-integral values");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  typename arch::tensor<_Tp>::container_type& data_    = t.storage_();
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  if constexpr (std::is_unsigned_v<_Tp>)
  {
    for (; i < simd_end; i += t.simd_width)
    {
      neon_u32 data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
      neon_u32 other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other[i]));
      neon_u32 or_vec    = vornq_u32(data_vec, other_vec);
      vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), or_vec);
    }
  }
  else if constexpr (std::is_signed_v<_Tp>)
  {
    for (; i < simd_end; i += t.simd_width)
    {
      neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
      neon_s32 other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other[i]));
      neon_s32 or_vec    = vornq_s32(data_vec, other_vec);
      vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), or_vec);
    }
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = (data_[i] || other[i]);
  }

  return t;
}

template<class _Tp>
arch::tensor<_Tp>& logical_or_(arch::tensor<_Tp>& t, const _Tp value)
{
  if (!std::is_integral_v<_Tp>)
  {
    throw error::type_error("Cannot perform logical OR on non-integral values");
  }

  typename arch::tensor<_Tp>::container_type& data_    = t.storage_();
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  if constexpr (std::is_signed_v<_Tp> || std::is_same_v<_Tp, bool>)
  {
    neon_s32 val_vec = vdupq_n_s32(static_cast<_s32>(value));

    for (; i < simd_end; i += t.simd_width)
    {
      neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
      neon_s32 or_vec   = vorrq_s32(data_vec, val_vec);
      vst1q_s32(&data_[i], or_vec);
    }
  }
  else if constexpr (std::is_unsigned_v<_Tp>)
  {
    neon_u32 val_vec = vdupq_n_u32(static_cast<_u32>(value));

    for (; i < simd_end; i += t.simd_width)
    {
      neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
      neon_u32 or_vec   = vorrq_u32(data_vec, val_vec);
      vst1q_u32(&data_[i], or_vec);
    }
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = static_cast<_Tp>(data_[i] || value);
  }

  return t;
}

}