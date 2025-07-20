#pragma once

#include "tensor.hpp"


using namespace internal;

template<class _Tp>
tensor<_Tp> neon::operator_plus(const tensor<_Tp>& t, const tensor<_Tp>& other) {
  if constexpr (!types::has_plus_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a plus operator");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Cannot add two tensors with different shapes");
  }

  const std::vector<_Tp>& a        = t.storage_();
  const std::vector<_Tp>& b        = other.storage_();
  const std::size_t       size     = a.size();
  const _u64              simd_end = size - (size % t.simd_width);
  std::vector<_Tp>        ret(size);

  _Tp* __restrict r_ptr       = ret.data();
  const _Tp* __restrict a_ptr = a.data();
  const _Tp* __restrict b_ptr = b.data();

#pragma omp parallel for
  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vb = neon_load<_Tp>(b_ptr + i);
    neon_type<_Tp> vr = neon_add<_Tp>(va, vb);
    neon_store<_Tp>(r_ptr + i, vr);
  }

#pragma omp parallel for
  for (std::size_t i = simd_end; i < size; ++i)
  {
    r_ptr[i] = a_ptr[i] + b_ptr[i];
  }

  return tensor<_Tp>(t.shape(), std::move(ret));
}

template<class _Tp>
tensor<_Tp> neon::operator_plus(const tensor<_Tp>& t, const _Tp value) {
  if constexpr (!types::has_plus_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a plus operator");
  }

  const std::vector<_Tp>& a        = t.storage_();
  const std::size_t       size     = a.size();
  const _u64              simd_end = size - (size % t.simd_width);
  std::vector<_Tp>        ret(size);
  neon_type<_Tp>          v_vec = neon_dup<_Tp>(value);

  _Tp* __restrict r_ptr       = ret.data();
  const _Tp* __restrict a_ptr = a.data();

#pragma omp parallel for
  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vr = neon_add<_Tp>(va, v_vec);
    neon_store<_Tp>(r_ptr + i, vr);
  }

#pragma omp parallel for
  for (std::size_t i = simd_end; i < size; ++i)
  {
    r_ptr[i] = a_ptr[i] + value;
  }

  return tensor<_Tp>(t.shape_, std::move(ret));
}

template<class _Tp>
tensor<_Tp>& neon::operator_plus_eq(tensor<_Tp>& t, const _Tp& value) {
  if constexpr (!types::has_equal_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a plus operator");
  }

  std::vector<_Tp>& a        = t.storage_();
  std::size_t       size     = a.size();
  const _u64        simd_end = size - (size % t.simd_width);
  neon_type<_Tp>    v_vec    = neon_dup<_Tp>(value);

  _Tp* __restrict a_ptr = a.data();

#pragma omp parallel for
  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vr = neon_add<_Tp>(va, v_vec);
    neon_store<_Tp>(a_ptr + i, vr);
  }

#pragma omp parallel for
  for (std::size_t i = simd_end; i < size; ++i)
  {
    a_ptr[i] = a_ptr[i] + value;
  }

  return t;
}

template<class _Tp>
tensor<_Tp> neon::operator_minus(const tensor<_Tp>& t, const tensor<_Tp>& other) {
  if constexpr (!types::has_minus_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a minus operator");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Cannot add two tensors with different shapes");
  }

  std::vector<_Tp>& a        = t.storage_();
  std::vector<_Tp>& b        = other.storage_();
  std::size_t       size     = a.size();
  const _u64        simd_end = size - (size % t.simd_width);
  std::vector<_Tp>  ret(size);

  _Tp* __restrict r_ptr       = ret.data();
  const _Tp* __restrict a_ptr = a.data();
  const _Tp* __restrict b_ptr = b.data();

#pragma omp parallel for
  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vb = neon_load<_Tp>(b_ptr + i);
    neon_type<_Tp> vr = neon_sub<_Tp>(va, vb);
    neon_store<_Tp>(r_ptr + i, vr);
  }

#pragma omp parallel for
  for (std::size_t i = simd_end; i < size; ++i)
  {
    r_ptr[i] = a_ptr[i] - b_ptr[i];
  }

  return tensor<_Tp>(t.shape(), std::move(ret));
}

template<class _Tp>
tensor<_Tp> neon::operator_minus(const tensor<_Tp>& t, const _Tp value) {
  if constexpr (!types::has_minus_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a minus operator");
  }

  std::vector<_Tp>& data_    = t.storage_();
  const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
  std::vector<_Tp>  d(data_.size());
  neon_type<_Tp>    val_vec = neon_dup<_Tp>(value);

#pragma omp parallel for
  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> vec1   = neon_load<_Tp>(&data_[i]);
    neon_type<_Tp> result = neon_sub<_Tp>(vec1, val_vec);
    neon_store<_Tp>(&d[i], result);
  }

#pragma omp parallel for
  for (std::size_t i = simd_end; i < data_.size(); ++i)
  {
    d[i] = data_[i] - value;
  }

  return tensor<_Tp>(t.shape(), d);
}

template<class _Tp>
tensor<_Tp>& neon::operator_minus_eq(tensor<_Tp>& t, const tensor<_Tp>& other) {
  using namespace internal;

  if constexpr (!types::has_minus_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a minus operator");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  std::vector<_Tp>&       a        = t.storage_();
  const std::vector<_Tp>& b        = other.storage_();
  const std::size_t       size     = a.size();
  const _u64              simd_end = size - (size % t.simd_width);

  _Tp* __restrict a_ptr      = a.data();
  const _Tp __restrict b_ptr = b.data();

#pragma omp parallel for
  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vb = neon_load<_Tp>(b_ptr + i);
    neon_type<_Tp> vr = neon_sub<_Tp>(va, vb);
    neon_store<_Tp>(a_ptr + i, vr);
  }

#pragma omp parallel for
  for (std::size_t i = simd_end; i < size; ++i)
  {
    a_ptr[i] = a_ptr[i] - b_ptr[i];
  }

  return t;
}

template<class _Tp>
tensor<_Tp> neon::operator_times(const tensor<_Tp>& t, const tensor<_Tp>& other) {
  using namespace internal;

  if constexpr (!types::has_times_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a times operator");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  const std::vector<_Tp>& a        = t.storage_();
  const std::vector<_Tp>& b        = other.storage_();
  const std::size_t       size     = a.size();
  const std::size_t       simd_end = size - (size % t.simd_width);

  std::vector<_Tp> result(size);

  _Tp* __restrict r_ptr       = result.data();
  const _Tp* __restrict a_ptr = a.data();
  const _Tp* __restrict b_ptr = b.data();

#pragma omp parallel for
  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vb = neon_load<_Tp>(b_ptr + i);
    neon_type<_Tp> vr = neon_mul<_Tp>(va, vb);
    neon_store<_Tp>(r_ptr + i, vr);
  }

#pragma omp parallel for
  for (std::size_t i = simd_end; i < size; ++i)
  {
    r_ptr[i] = a_ptr[i] * b_ptr[i];
  }

  return tensor<_Tp>(t.shape(), std::move(result));
}

/*
template<class _Tp>
tensor<_Tp> neon::operator_times(const tensor<_Tp>& t, const tensor<_Tp>& other) {
    if constexpr (!types::has_times_operator_v<_Tp>)
    {
        throw error::operator_error("Value type must have a times operator");
    }
    
    if (!t.shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }
    
    const _Tp* __restrict__ a = t.storage_().data();
    const _Tp* __restrict__ b = other.storage_().data();
    
    const size_t size     = t.storage_().size();
    const size_t simd_end = size - (size % t.simd_width);
    
    _Tp* __restrict__ out = static_cast<_Tp*>(std::aligned_alloc(16, size * sizeof(_Tp)));
    
    for (size_t i = 0; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> va = neon_load<_Tp>(a + i);
        neon_type<_Tp> vb = neon_load<_Tp>(b + i);
        neon_type<_Tp> vr = neon_mul<_Tp>(va, vb);
        neon_store<_Tp>(out + i, vr);
    }
    
    for (size_t i = simd_end; i < size; ++i)
    {
        out[i] = a[i] * b[i];
    }
    
    // Move result into tensor without copy
    std::vector<_Tp> vec(out, out + size);
    std::free(out);  // don't forget to free it
    return tensor<_Tp>(t.shape(), std::move(vec));
}
*/

template<class _Tp>
tensor<_Tp>& neon::operator_times_eq(tensor<_Tp>& t, const tensor<_Tp>& other) {
  if constexpr (!types::has_minus_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a minus operator");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  std::vector<_Tp>& a        = t.storage_();
  std::vector<_Tp>& b        = other.storage_();
  std::size_t       size     = a.size();
  const _u64        simd_end = a.size() - (a.size() % t.simd_width);

  _Tp* __restrict aptr       = a.data();
  const _Tp* __restrict bptr = b.data();

#pragma omp parallel for
  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    auto va = neon_load<_Tp>(aptr + i);
    auto vb = neon_load<_Tp>(bptr + i);
    auto vr = neon_mul<_Tp>(va, vb);
    neon_store<_Tp>(aptr + i, vr);
  }

#pragma omp parallel for
  for (std::size_t i = simd_end; i < size; ++i)
  {
    aptr[i] *= bptr[i];
  }

  return t;
}

template<class _Tp>
tensor<_Tp>& neon::operator_minus_eq(tensor<_Tp>& t, const _Tp& value) {
  if constexpr (!types::has_minus_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a minus operator");
  }

  std::vector<_Tp>& a        = t.storage_();
  std::size_t       size     = a.size();
  const _u64        simd_end = size - (size % t.simd_width);
  neon_type<_Tp>    v_vec    = neon_dup<_Tp>(value);

  _Tp* __restrict a_ptr = a.data();

#pragma omp parallel for
  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vr = neon_mul<_Tp>(va, v_vec);
    neon_store<_Tp>(a_ptr + i, vr);
  }

#pragma omp parallel for
  for (std::size_t i = simd_end; i < size; ++i)
  {
    a_ptr[i] = a_ptr - value;
  }

  return t;
}

template<class _Tp>
tensor<_Tp> neon::operator_divide(const tensor<_Tp>& t, const _Tp& value) {
  if (t.empty())
  {
    return tensor<_Tp>(t.shape(), {});
  }

  if (!types::has_divide_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a divide operator");
  }

  if (value == static_cast<_Tp>(0))
  {
    throw std::logic_error("Cannot divide by zero : undefined operation");
  }

  std::vector<_Tp>& a        = t.storage_();
  std::size_t       size     = a.size();
  neon_type<_Tp>    v_vec    = neon_dup<_Tp>(value);
  const _u64        simd_end = size - (size % t.simd_width);
  std::vector<_Tp>  ret(size);

  const _Tp* __restrict a_ptr = a.data();
  _Tp* __restrict r_ptr       = ret.data();

  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vr = neon_div<_Tp>(va, v_vec);
    neon_store<_Tp>(r_ptr + i, vr);
  }

  for (std::size_t i = simd_end; i < size; ++i)
  {
    r_ptr[i] = a_ptr[i] / value;
  }

  return tensor<_Tp>(t.shape(), std::move(ret));
}

template<class _Tp>
tensor<_Tp> neon::operator_divide(const tensor<_Tp>& t, const tensor<_Tp>& other) {
  if (t.empty())
  {
    return tensor<_Tp>(t.shape(), {});
  }

  if (!types::has_divide_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a divide operator");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  if (other.count_nonzero(0) != other.size(0))
  {
    throw std::logic_error("Cannot divide by zero : undefined operation");
  }

  std::vector<_Tp>& a        = t.storage_();
  std::vector<_Tp>& b        = other.storage_();
  std::size_t       size     = a.size();
  const _u64        simd_end = size - (size % t.simd_width);
  std::vector<_Tp>  ret(size);

  _Tp* __restrict r_ptr       = ret.data();
  const _Tp* __restrict a_ptr = a.data();
  const _Tp* __restrict b_ptr = b.data();

#pragma omp parallel for
  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vb = neon_load<_Tp>(b_ptr + i);
    neon_type<_Tp> vr = neon_div<_Tp>(va, vb);
    neon_store<_Tp>(r_ptr + i, vr);
  }

#pragma omp parallel for
  for (std::size_t i = simd_end; i < size; ++i)
  {
    r_ptr[i] = a_ptr[i] / b_ptr[i];
  }

  return tensor<_Tp>(t.shape(), std::move(ret));
}

template<class _Tp>
tensor<_Tp> neon::operator_divide_eq(const tensor<_Tp>& t, const _Tp& value) {
  if (t.empty())
  {
    return tensor<_Tp>(t.shape(), {});
  }

  if (!types::has_divide_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a divide operator");
  }

  if (value == static_cast<_Tp>(0))
  {
    throw std::logic_error("Cannot divide by zero : undefined operation");
  }

  std::vector<_Tp>& a        = t.storage_();
  std::size_t       size     = a.size();
  const _u64        simd_end = size - (size % t.simd_width);
  neon_type<_Tp>    v_vec    = neon_dup<_Tp>(value);
  std::vector<_Tp>  ret(size);

  _Tp* __restrict r_ptr       = ret.data();
  const _Tp* __restrict a_ptr = a.data();

#pragma omp parallel for
  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vr = neon_div<_Tp>(va, v_vec);
    neon_store<_Tp>(a_ptr + i, vr);
  }

#pragma omp parallel for
  for (std::size_t i = simd_end; i < size; ++i)
  {
    r_ptr[i] = a_ptr[i] / value;
  }

  return t;
}

template<class _Tp>
tensor<_Tp> neon::operator_divide_eq(const tensor<_Tp>& t, const tensor<_Tp>& other) {
  if (t.empty())
  {
    return tensor<_Tp>(t.shape(), {});
  }

  if (!types::has_divide_operator_v<_Tp>)
  {
    throw error::operator_error("Value type must have a divide operator");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  if (other.count_nonzero(0) != other.size(0))
  {
    throw std::logic_error("Cannot divide by zero : undefined operation");
  }

  std::vector<_Tp>& a        = t.storage_();
  std::vector<_Tp>& b        = other.storage_();
  std::size_t       size     = a.size();
  const _u64        simd_end = size - (size % t.simd_width);
  std::vector<_Tp>  ret(size);

  _Tp* __restrict a_ptr       = a.data();
  const _Tp* __restrict b_ptr = other.data();

#pragma omp parallel for
  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> va = neon_load<_Tp>(a_ptr + i);
    neon_type<_Tp> vb = neon_load<_Tp>(b_ptr + i);
    neon_type<_Tp> vr = neon_div<_Tp>(va, vb);
    neon_store<_Tp>(a_ptr + i, vr);
  }

#pragma omp parallel for
  for (std::size_t i = simd_end; i < size; ++i)
  {
    a_ptr[i] = a_ptr[i] / b_ptr[i];
  }

  return t;
}