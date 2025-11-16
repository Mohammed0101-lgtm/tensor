#pragma once

#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
arch::tensor<_Tp>& fmax_(arch::tensor<_Tp>& t, const _Tp v)
{
  if (!std::is_floating_point_v<_Tp>)
  {
    throw error::type_error("Type must be floating point");
  }

  typename arch::tensor<_Tp>::container_type& data_    = t.storage_();
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  if constexpr (std::is_floating_point_v<_Tp>)
  {
    neon_f32 scalar_val = vdupq_n_f32(v);

    for (; i < t.simd_end; i += t.t.simd_width)
    {
      neon_f32 a       = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
      neon_f32 max_val = vmaxq_f32(a, scalar_val);
      vst1q_f32(&data_[i], max_val);
    }
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = std::fmax(data_[i], v);
  }

  return t;
}

template<class _Tp>
arch::tensor<_Tp>& fmax_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  if (!std::is_floating_point_v<_Tp>)
  {
    throw error::type_error("Type must be floating point");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  typename arch::tensor<_Tp>::container_type& data_    = t.storage_();
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  if constexpr (std::is_floating_point_v<_Tp>)
  {
    for (; i < simd_end; i += _ARM64_REG_WIDTH)
    {
      neon_f32 a       = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
      neon_f32 b       = vld1q_f32(reinterpret_cast<const _f32*>(&(other[i])));
      neon_f32 max_val = vmaxq_f32(a, b);
      vst1q_f32(&data_[i], max_val);
    }
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = std::fmax(data_[i], other[i]);
  }

  return t;
}

template<class _Tp>
arch::tensor<_Tp>& maximum_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  typename arch::tensor<_Tp>::container_type& data_    = t.storage_();
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> a   = neon_load<_Tp>(&data_[i]);
    neon_type<_Tp> b   = neon_load<_Tp>(&other[i]);
    neon_type<_Tp> max = neon_max<_Tp>(a, b);
    neon_store<_Tp>(&data_[i], max);
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = std::max(data_[i], other[i]);
  }

  return t;
}

template<class _Tp>
arch::tensor<_Tp>& maximum_(arch::tensor<_Tp>& t, const _Tp value)
{
  if constexpr (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Value type must be arithmetic");
  }

  typename arch::tensor<_Tp>::container_type& data_    = t.storage_();
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  neon_type<_Tp>                              val_vec  = neon_dup<_Tp>(value);
  _u64                                        i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> a   = neon_load<_Tp>(&data_[i]);
    neon_type<_Tp> max = neon_max<_Tp>(a, val_vec);
    neon_store<_Tp>(&data_[i], max);
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = std::max(data_[i], value);
  }

  return t;
}
}
