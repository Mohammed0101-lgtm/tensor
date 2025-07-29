#pragma once

#include "internal/simd/neon/alias.hpp"
#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
tensor<_Tp>& logical_xor_(tensor<_Tp>& t, const tensor<_Tp>& other) {
  if (!std::is_integral_v<_Tp>)
  {
    throw error::type_error("Cannot get the element wise xor of non-integral value");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  std::vector<_Tp>& data_    = t.storage_();
  const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64              i        = 0;

  if constexpr (std::is_unsigned_v<_Tp>)
  {
    for (; i < simd_end; i += t.simd_width)
    {
      neon_u32 data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
      neon_u32 other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other[i]));
      neon_u32 xor_vec   = veorq_u32(data_vec, other_vec);
      vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), xor_vec);
    }
  }
  else if constexpr (std::is_signed_v<_Tp>)
  {
    for (; i < simd_end; i += t.simd_width)
    {
      neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
      neon_s32 other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other[i]));
      neon_s32 xor_vec   = veorq_s32(data_vec, other_vec);
      vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), xor_vec);
    }
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = (data_[i] xor other[i]);
  }

  return t;
}

template<class _Tp>
tensor<_Tp>& logical_xor_(tensor<_Tp>& t, const _Tp value) {
  if constexpr (!std::is_integral_v<_Tp>)
  {
    throw error::type_error("Cannot get the element wise xor of non-integral value");
  }

  std::vector<_Tp>& data_    = t.storage_();
  const std::size_t size     = data_.size();
  const _u64        simd_end = size - (size % t.simd_width);

  neon_type<_Tp> v_vec = neon_dup<_Tp>(value);

  _Tp* __restrict a_ptr = data_.data();

  for (std::size_t i = 0; i < size; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vr = neon_xor<_Tp>(va, v_vec);
    neon_store<_Tp>(a_ptr + i, vr);
  }

  for (std::size_t i = simd_end; i < data_.size(); ++i)
  {
    a_ptr[i] = a_ptr[i] ^ value;
  }

  return t;
}

}