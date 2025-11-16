#pragma once

#include "concepts.hpp"
#include "tensor.hpp"
#include "types.hpp"


namespace internal::simd::neon {

template<class _Tp>
arch::tensor<bool> equal(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  if constexpr (!internal::concepts::has_equal_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have equal to operator");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  typename arch::tensor<_Tp>::container_type& data_    = t.storage_();
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  std::vector<bool>                           ret(data_.size());
  _u64                                        i = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    auto                 data_vec  = neon_load<_Tp>(&data_[i]);
    auto                 other_vec = neon_load<_Tp>(&other.storage_()[i]);
    auto                 res_vec   = neon_ceq<_Tp>(data_vec, other_vec);
    alignas(16) uint32_t buffer[t.simd_width];
    vst1q_u32(buffer, res_vec);

    for (int j = 0; j < t.simd_width; ++j)
    {
      ret[i + j] = buffer[j] == 0xFFFFFFFF;
    }
  }

  for (; i < data_.size(); ++i)
  {
    ret[i] = (data_[i] == other[i]);
  }

  return arch::tensor<bool>(t.shape(), std::move(ret));
}

template<class _Tp>
arch::tensor<bool> equal(const arch::tensor<_Tp>& t, const _Tp value)
{
  if constexpr (!internal::concepts::has_equal_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have equal to operator");
  }

  typename arch::tensor<_Tp>::container_type& data_ = t.storage_();
  std::vector<bool>                           ret(data_.size());
  const _u64                                  simd_end = data_.size() - (data_.size() % 4);
  _u64                                        i        = 0;
  neon_type<_Tp>                              val_vec  = neon_dup<_Tp>(value);

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp>   data_vec   = neon_load<_Tp>(&data_[i]);
    neon_u32         cmp_result = neon_ceq<_Tp>(data_vec, val_vec);
    alignas(16) _u32 results[t.simd_width];
    neon_store<_Tp>(results, cmp_result);

    for (int j = 0; j < t.simd_width; ++j)
    {
      ret[i + j] = results[j] != 0;
    }
  }

  for (; i < data_.size(); ++i)
  {
    ret[i] = (data_[i] == value);
  }

  return arch::tensor<bool>(std::move(t.shape()), std::move(ret));
}

template<class _Tp>
arch::tensor<bool> less_equal(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  if constexpr (!internal::concepts::has_less_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have less than operator");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  typename arch::tensor<_Tp>::container_type& data_ = t.storage_();
  std::vector<_u32>                           ret(data_.size());
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp>  vec_a     = neon_load<_Tp>(&data_[i]);
    neon_type<_Tp>  vec_b     = neon_load<_Tp>(&other[i]);
    neon_type<_Tp>  leq_mask  = neon_cleq<_Tp>(vec_a, vec_b);
    neon_type<_u32> leq_mask_ = reinterpret_cast<neon_type<_u32>>(leq_mask);
    neon_store<_u32>(&ret[i], leq_mask_);
  }

  // Convert `ret` (integer masks) to boolean
  std::vector<bool> d(data_.size());

  for (std::size_t j = 0; j < i; ++j)
  {
    d[j] = ret[j] != 0;
  }

  for (; i < d.size(); ++i)
  {
    d[i] = (data_[i] <= other[i]);
  }

  return arch::tensor<bool>(std::move(t.shape()), std::move(d));
}

template<class _Tp>
arch::tensor<bool> less_equal(const arch::tensor<_Tp>& t, const _Tp value)
{
  if constexpr (!internal::concepts::has_less_equal_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have less than or equal to operator");
  }

  typename arch::tensor<_Tp>::container_type& data_ = t.storage_();
  std::vector<_u32>                           ret(data_.size());
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> vec_a    = neon_load<_Tp>(&data_[i]);
    neon_type<_Tp> vec_b    = neon_dup<_Tp>(value);
    neon_type<_Tp> leq_mask = neon_cleq<_Tp>(vec_a, vec_b);
    neon_store<_Tp>(&ret[i], leq_mask);
  }

  for (; i < data_.size(); ++i)
  {
    ret[i] = (data_[i] <= value) ? 1 : 0;
  }

  std::vector<bool> to_bool(ret.size());
  i = 0;

  for (int i = i; i >= 0; --i)
  {
    to_bool[i] = ret[i] == 1 ? true : false;
  }

  return arch::tensor<bool>(std::move(t.shape()), std::move(to_bool));
}

template<class _Tp>
arch::tensor<bool> greater(const arch::tensor<_Tp>& t, const _Tp value)
{
  if constexpr (!internal::concepts::has_greater_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have greater than operator");
  }

  typename arch::tensor<_Tp>::container_type& data_ = t.storage_();
  std::vector<_u32>                           ret(data_.size());
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp>  vec_a   = neon_load<_Tp>(&data_[i]);
    neon_type<_Tp>  vec_b   = neon_dup<_Tp>(value);
    neon_type<_u32> gr_mask = neon_cgt<_Tp>(vec_a, vec_b);
    neon_store<_u32>(&ret[i], gr_mask);
  }

  for (; i < data_.size(); ++i)
  {
    ret[i] = (data_[i] > value) ? 1 : 0;
  }

  std::vector<bool> to_bool(ret.size());

  for (int j = i; j >= 0; --j)
  {
    to_bool[j] = ret[j] == 1 ? true : false;
  }

  return arch::tensor<bool>(std::move(t.shape()), std::move(to_bool));
}

template<class _Tp>
arch::tensor<bool> greater(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  if constexpr (!internal::concepts::has_greater_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have greater than operator");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors must have the same shape");
  }

  typename arch::tensor<_Tp>::container_type& data_ = t.storage_();
  std::vector<_u32>                           ret(data_.size());
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> vec_a   = neon_load<_Tp>(&data_[i]);
    neon_type<_Tp> vec_b   = neon_load<_Tp>(&other[i]);
    neon_type<_Tp> gr_mask = neon_cgt(vec_a, vec_b);
    neon_store<_Tp>(&ret[i], gr_mask);
  }

  for (; i < data_.size(); ++i)
  {
    ret[i] = (data_[i] > other[i]) ? 1 : 0;
  }

  std::vector<bool> to_bool(ret.size());
  i = 0;

  for (int i = i; i >= 0; --i)
  {
    to_bool[i] = ret[i] == 1 ? true : false;
  }

  return arch::tensor<bool>(std::move(t.shape()), std::move(to_bool));
}

template<class _Tp>
arch::tensor<bool> greater_equal(const arch::tensor<_Tp>& t, const _Tp value)
{
  if constexpr (!internal::concepts::has_greater_equal_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have greater than or equal to operator");
  }

  typename arch::tensor<_Tp>::container_type& data_ = t.storage_();
  std::vector<_u32>                           ret(data_.size());
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp>  vec_a   = neon_load<_Tp>(&data_[i]);
    neon_type<_Tp>  vec_b   = neon_dup<_Tp>(value);
    neon_type<_u32> ge_mask = neon_cgeq<_Tp>(vec_a, vec_b);
    neon_store<_u32>(&ret[i], ge_mask);
  }

  for (; i < data_.size(); ++i)
  {
    ret[i] = (data_[i] >= value) ? 1 : 0;
  }

  std::vector<bool> to_bool(ret.size());

  for (int j = i; j >= 0; --j)
  {
    to_bool[j] = ret[j] == 1 ? true : false;
  }

  return arch::tensor<bool>(std::move(t.shape()), std::move(to_bool));
}

template<class _Tp>
arch::tensor<bool> greater_equal(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  if constexpr (!internal::concepts::has_greater_equal_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have greater than or equal to operator");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  typename arch::tensor<_Tp>::container_type& data_ = t.storage_();
  std::vector<_u32>                           ret(data_.size());
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp>  vec_a   = neon_load<_Tp>(&data_[i]);
    neon_type<_Tp>  vec_b   = neon_load<_Tp>(&other[i]);
    neon_type<_u32> ge_mask = neon_cgeq<_Tp>(vec_a, vec_b);
    neon_store<_u32>(&ret[i], ge_mask);
  }

  for (; i < data_.size(); ++i)
  {
    ret[i] = (data_[i] >= other[i]) ? 1 : 0;
  }

  std::vector<bool> to_bool(ret.size());

  for (int j = i; j >= 0; --j)
  {
    to_bool[j] = ret[j] == 1 ? true : false;
  }

  return arch::tensor<bool>(std::move(t.shape()), std::move(to_bool));
}


template<class _Tp>
arch::tensor<bool> not_equal(const arch::tensor<_Tp>& t, const _Tp value)
{
  arch::tensor<bool> a = t.equal(value);
  a                    = a.logical_not_();
  return a;
}

template<class _Tp>
arch::tensor<bool> not_equal(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  arch::tensor<bool> a = t.equal(other);
  a                    = a.logical_not_();
  return a;
}

template<class _Tp>
inline arch::tensor<bool> less(const arch::tensor<_Tp>& t, const _Tp value)
{
  arch::tensor<bool> a = t.greater_equal(value);
  a                    = a.logical_not_();
  return a;
}

template<class _Tp>
inline arch::tensor<bool> less(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other)
{
  arch::tensor<bool> a = t.greater_equal(other);
  a                    = a.logical_not_();
  return a;
}

}