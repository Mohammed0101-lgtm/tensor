#pragma once

#include "internal/simd/neon/alias.hpp"
#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
arch::tensor<_Tp>& bitwise_and_(arch::tensor<_Tp>& t, const _Tp value)
{
  if (!std::is_integral_v<_Tp>)
  {
    throw error::type_error("Cannot perform a bitwise AND on non-integral values");
  }

  typename arch::tensor<_Tp>::container_type& data_    = t.storage_();
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  neon_type<_Tp>                              val_vec  = neon_dup<_Tp>(value);
  _u64                                        i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> data_vec = neon_load<_Tp>(&data_[i]);
    neon_type<_Tp> res_vec  = neon_and<_Tp>(data_vec, val_vec);
    neon_store<_Tp>(&data_[i], res_vec);
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] &= value;
  }

  return t;
}

template<class _Tp>
arch::tensor<_Tp>& bitwise_and_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  if (!std::is_integral_v<_Tp>)
  {
    throw error::type_error("Cannot perform a bitwise AND on non-integral values");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  typename arch::tensor<_Tp>::container_type& data_    = t.storage_();
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> data_vec  = neon_load<_Tp>(&data_[i]);
    neon_type<_Tp> other_vec = neon_load<_Tp>(&other[i]);
    neon_type<_Tp> xor_vec   = neon_and<_Tp>(data_vec, other_vec);
    neon_store<_Tp>(&data_[i], xor_vec);
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] &= other[i];
  }

  return t;
}

}