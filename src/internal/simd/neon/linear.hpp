#pragma once

#include "tensor.hpp"
#include "types.hpp"


namespace internal::simd::neon {

template<class _Tp>
arch::tensor<_Tp> absolute(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  if (std::is_unsigned_v<_Tp>)
  {
    return t;
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors must have the same shape");
  }

  typename arch::tensor<_Tp>::container_type& data_ = t.storage_();
  _u64                                        s     = t.storage().size();
  typename arch::tensor<_Tp>::container_type  a(s);
  _u64                                        i = 0;

  if constexpr (std::is_floating_point_v<_Tp>)
  {
    for (; i < data_.size(); i += _ARM64_REG_WIDTH)
    {
      neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
      neon_f32 abs_vec  = vabsq_f32(data_vec);
      vst1q_f32(reinterpret_cast<_f32*>(&a[i]), abs_vec);
    }
  }
  else if constexpr (std::is_signed_v<_Tp>)
  {
    for (; i < data_.size(); i += _ARM64_REG_WIDTH)
    {
      neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
      neon_s32 abs_vec  = vabsq_s32(data_vec);
      vst1q_s32(reinterpret_cast<_s32*>(&a[i]), abs_vec);
    }
  }

  for (; i < s; ++i)
  {
    a[i] = static_cast<_Tp>(std::abs(data_[i]));
  }

  return arch::tensor<_Tp>(std::move(t.shape()), std::move(a));
}

template<class _Tp>
arch::tensor<_Tp> absolute_(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  if (std::is_unsigned_v<_Tp>)
  {
    return t;
  }

  typename arch::tensor<_Tp>::container_type& data_ = t.storage_();
  _u64                                        s     = t.storage().size();
  typename arch::tensor<_Tp>::container_type  a(s);
  _u64                                        i = 0;

  if constexpr (std::is_floating_point_v<_Tp>)
  {
    for (; i < data_.size(); i += _ARM64_REG_WIDTH)
    {
      neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
      neon_f32 abs_vec  = vabsq_f32(data_vec);
      vst1q_f32(reinterpret_cast<_f32*>(&data_[i]), abs_vec);
    }
  }
  else if constexpr (std::is_signed_v<_Tp>)
  {
    for (; i < data_.size(); i += _ARM64_REG_WIDTH)
    {
      neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
      neon_s32 abs_vec  = vabsq_s32(data_vec);
      vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), abs_vec);
    }
  }

  for (; i < s; ++i)
  {
    data_[i] = static_cast<_Tp>(std::abs(data_[i]));
  }

  return self(std::move(t.shape()), std::move(a));
}

}