#pragma once

#include "internal/simd/neon/operators.hpp"
#include "tensor.hpp"
#include "types.hpp"


template<class _Tp>
bool tensor<_Tp>::operator!=(const tensor& other) const {
  return !(*this == other);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const tensor& other) const {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::operator_plus(*this, other);
  }

  if constexpr (!internal::types::has_plus_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a plus operator");
  }

  if (!this->shape_().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  std::size_t           size = this->size(0);
  const container_type& b    = other.storage_();

  container_type ret(size);

#pragma omp parallel
  for (std::size_t i = 0; i < size; ++i)
  {
    ret[i] = (*this)[i] + b[i];
  }

  return self(this->shape(), std::move(ret));
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const value_type value) const {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::operator_plus(*this, value);
  }

  if constexpr (!internal::types::has_plus_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a plus operator");
  }

  container_type ret(this->size(0));

  for (index_type i = 0; i < this->size(0); ++i)
  {
    ret[i] = (*this)[i] + value;
  }

  return self(this->shape(), std::move(ret));
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator*(const value_type value) const {
  /*
  if (internal::types::using_neon()) 
  {
    return internal::simd::neon::operator_times(*this, value);
  }
  */

  if constexpr (!internal::types::has_times_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a times operator");
  }

  container_type ret(this->size(0));

  for (index_type i = 0; i < this->size(0); ++i)
  {
    ret[i] = (*this)[i] * value;
  }

  return self(this->shape(), std::move(ret));
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator*(const tensor& other) const {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::operator_times(*this, other);
  }

  if constexpr (!internal::types::has_times_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a times operator");
  }

  if (!this->shape_().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  container_type ret(this->size(0));

  for (index_type i = 0; i < this->size(0); ++i)
  {
    ret[i] = (*this)[i] * other[i];
  }

  return self(this->shape(), std::move(ret));
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator+=(const tensor& other) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::operator_plus_eq(*this, other);
  }

  if constexpr (!internal::types::has_plus_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a plus equal to operator");
  }

  if (!this->shape_().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  index_type i = 0;

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem + other[i++];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator+=(const_reference value) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::operator_plus_eq(*this, value);
  }

  if constexpr (!internal::types::has_plus_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a plus operator");
  }

  for (index_type i = 0; i < this->size(0); ++i)
  {
    (*this)[i] = (*this)[i] + value;
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem + value;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const tensor& other) const {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::operator_minus(*this, other);
  }

  if constexpr (!internal::types::has_minus_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a minus operator");
  }

  if (!this->shape_().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  container_type ret(this->size(0));

  for (index_type i = 0; i < this->size(0); ++i)
  {
    ret[i] = (*this)[i] - other[i];
  }

  return self(this->shape(), std::move(ret));
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const value_type value) const {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::operator_minus(*this, value);
  }

  if constexpr (!internal::types::has_minus_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a minus operator");
  }

  container_type d(this->size(0));

  for (index_type i = 0; i < this->size(0); ++i)
  {
    d[i] = (*this)[i] - value;
  }

  return self(*this);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator-=(const tensor& other) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::operator_minus_eq(*this, other);
  }

  if constexpr (!internal::types::has_minus_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a minus operator");
  }

  if (!this->shape_().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  index_type      i = 0;
  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem - other[i++];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator*=(const tensor& other) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::operator_times_eq(*this, other);
  }

  if constexpr (!internal::types::has_times_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a times operator");
  }

  if (!this->shape_().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  index_type      i = 0;
  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem * other[i++];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator/(const_reference value) const {
  if (internal::types::using_neon() && std::is_floating_point_v<value_type>)
  {
    return internal::simd::neon::operator_divide(*this, value);
  }

  if constexpr (!internal::types::has_divide_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a divide operator");
  }

  if (value == value_type(0))
  {
    throw std::logic_error("Cannot divide by zero : undefined operation");
  }

  container_type  ret(this->size(0));
  index_type      i = 0;
  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    ret[i++] = elem / value;
  }

  return self(this->shape(), std::move(ret));
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator*=(const_reference value) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::operator_times_eq(*this, value);
  }

  if constexpr (!internal::types::has_times_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a times operator");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem * value;
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::operator=(const tensor& other) const {
  this->shape_()   = other.shape();
  this->storage_() = other.storage();
  this->shape_().compute_strides();
  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator/=(const tensor& other) {
  if (internal::types::using_neon() && std::is_floating_point_v<value_type>)
  {
    return internal::simd::neon::operator_divide_eq(*this, other);
  }

  if constexpr (!internal::types::has_divide_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a divide operator");
  }

  if (!this->shape_().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  if (other.count_nonzero(0) != other.size(0))
  {
    throw std::logic_error("Cannot divide by zero : undefined operation");
  }

  index_type      i = 0;
  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem / other[i++];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator/=(const_reference value) {
  if (internal::types::using_neon() && std::is_floating_point_v<value_type>)
  {
    return internal::simd::neon::operator_divide_eq(*this, value);
  }

  if constexpr (!internal::types::has_divide_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a divide operator");
  }

  if (value == value_type(0))
  {
    throw std::invalid_argument("Cannot divide by zero : undefined operation");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem / value;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator/(const tensor& other) const {
  if (internal::types::using_neon() && std::is_floating_point_v<value_type>)
  {
    return internal::simd::neon::operator_divide(*this, other);
  }

  if constexpr (!internal::types::has_divide_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a divide operator");
  }

  if (!this->shape_().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  if (other.count_nonzero(0) != other.size(0))
  {
    throw std::logic_error("Cannot divide by zero : undefined operation");
  }

  container_type ret(this->size(0));

  for (index_type i = 0; i < this->size(0); ++i)
  {
    ret[i] = (*this)[i] / other[i];
  }

  return self(this->shape(), std::move(ret));
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator-=(const_reference value) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::operator_minus_eq(*this, value);
  }

  if constexpr (!internal::types::has_minus_operator_v<value_type>)
  {
    throw error::operator_error("Value type must have a minus operator");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem - value;
  }

  return *this;
}

template<class _Tp>
bool tensor<_Tp>::operator==(const tensor& other) const {
  if (this->shape_().equal(other.shape()) && this->storage_() == other.storage())
  {
    return true;
  }

  return false;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator=(tensor&& other) const noexcept {
  if (this != &other)
  {
    this->storage_() = std::move(other.storage());
    this->shape_()   = std::move(other.shape());
  }

  return *this;
}

template<class _Tp>
const tensor<bool>& tensor<_Tp>::operator!() const {
  return logical_not_();
}

template<class _Tp>
tensor<bool>& tensor<_Tp>::operator!() {
  return logical_not_();
}