#pragma once

#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
arch::tensor<_Tp>& pow_(arch::tensor<_Tp>& t, const _Tp value)
{
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  typename arch::tensor<_Tp>::container_type& data_    = t.storage_();
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp>  data_vec = neon_load<_Tp>(&data_[i]);
    alignas(16) _Tp vals[t.simd_width];
    neon_store<_Tp>(vals, data_vec);

    for (int j = 0; j < t.simd_width; ++j)
    {
      vals[j] = static_cast<_Tp>(std::pow(vals[j], value));
    }

    neon_type<_Tp> pow_vec = neon_load<_Tp>(vals);
    neon_store<_Tp>(&data_[i], pow_vec);
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = static_cast<_Tp>(std::pow(data_[i], value));
  }

  return t;
}

template<class _Tp>
arch::tensor<_Tp>& pow_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  typename arch::tensor<_Tp>::container_type&       a        = t.storage_();
  const typename arch::tensor<_Tp>::container_type& b        = other.storage_();
  const std::size_t                                 size     = a.size();
  const std::size_t                                 simd_end = size - (size % t.simd_width);

  _Tp* __restrict aptr       = a.data();
  const _Tp* __restrict bptr = b.data();

  std::size_t i = 0;
  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(aptr + i);
    neon_type<_Tp> vb = neon_load<_Tp>(bptr + i);
    neon_type<_Tp> vr = {
      static_cast<_Tp>(std::pow(neon_get_lane<_Tp, 0>(va), neon_get_lane<_Tp, 0>(vb))),
      static_cast<_Tp>(std::pow(neon_get_lane<_Tp, 1>(va), neon_get_lane<_Tp, 1>(vb))),
      static_cast<_Tp>(std::pow(neon_get_lane<_Tp, 2>(va), neon_get_lane<_Tp, 2>(vb))),
      static_cast<_Tp>(std::pow(neon_get_lane<_Tp, 3>(va), neon_get_lane<_Tp, 3>(vb))),
    };
    neon_store<_Tp>(aptr + i, vr);
  }

  for (; i < size; ++i)
  {
    aptr[i] = static_cast<_Tp>(std::pow(static_cast<_f32>(aptr[i]), static_cast<_f32>(bptr[i])));
  }

  return t;
}
}
