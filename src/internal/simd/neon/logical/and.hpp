#pragma once

#include "internal/simd/neon/alias.hpp"
#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
tensor<_Tp>& logical_and_(tensor<_Tp>& t, const _Tp value) {
  if (!std::is_integral_v<_Tp>)
  {
    throw error::type_error("Cannot get the element wise and of non-integral value");
  }

  std::vector<_Tp>& data_    = t.storage_();
  const std::size_t size     = data_.size();
  const _u64        simd_end = size - (size % t.simd_width);
  neon_type<_Tp>    v_vec    = neon_dup<_Tp>(value);

  _Tp* __restrict a_ptr = data_.data();

  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vr = neon_and<_Tp>(va, v_vec);
    neon_store<_Tp>(a_ptr + i, vr);
  }


  for (std::size_t i = simd_end; i < size; ++i)
  {
    a_ptr[i] = a_ptr[i] && value;
  }

  return t;
}

template<class _Tp>
tensor<_Tp>& logical_and_(tensor<_Tp>& t, const tensor<_Tp>& other) {
  if constexpr (!std::is_integral_v<_Tp>)
  {
    throw error::type_error("Cannot get the element wise and of non-integral value");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  std::vector<_Tp>&       a        = t.storage_();
  const std::vector<_Tp>& b        = other.storage_();
  const std::size_t       size     = a.size();
  const _u64              simd_end = size - (size % t.simd_width);

  _Tp* __restrict a_ptr       = a.data();
  const _Tp* __restrict b_ptr = b.data();

  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vb = neon_load<_Tp>(b_ptr + i);
    neon_type<_Tp> vr = neon_and<_Tp>(va, vb);
    neon_store<_Tp>(a_ptr + i, vr);
  }

  for (std::size_t i = simd_end; i < size; ++i)
  {
    a_ptr[i] = a_ptr[i] && b_ptr[i];
  }

  return t;
}

}