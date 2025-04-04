#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_operator_plus(const tensor& other) const {
  static_assert(has_plus_operator_v<value_type>, "Value type must have a plus operator");

  if (!equal_shape(this->shape(), other.shape())) {
    throw shape_error("Cannot add two tensors with different shapes");
  }

  constexpr size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
  static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

  index_type simd_end = this->data_.size() - (this->data_.size() % simd_width);
  data_t     d(this->data_.size());

  index_type i = 0;
  for (; i < simd_end; i += simd_width) {
    neon_type<value_type> vec1   = neon_load<value_type>(&this->data_[i]);
    neon_type<value_type> vec2   = neon_load<value_type>(&other[i]);
    neon_type<value_type> result = neon_add<value_type>(vec1, vec2);
    neon_store<value_type>(&d[i], result);
  }

#pragma omp parallel
  for (; i < this->data_.size(); ++i) d[i] = this->data_[i] + other[i];

  return self(this->shape_, d);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_operator_plus(const value_type val) const {
  static_assert(has_plus_operator_v<value_type>, "Value type must have a plus operator");

  data_t           d(this->data_.size());
  constexpr size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
  static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

  index_type simd_end = this->data_.size() - (this->data_.size() % simd_width);

  index_type            i       = 0;
  neon_type<value_type> val_vec = neon_dup<value_type>(&val);
  for (; i < simd_end; i += simd_width) {
    neon_type<value_type> vec1   = neon_load<value_type>(&this->data_[i]);
    neon_type<value_type> result = neon_add<value_type>(vec1, val_vec);
    neon_store<value_type>(&d[i], result);
  }

#pragma omp parallel
  for (; i < this->data_.size(); ++i) d[i] = this->data_[i] + val;

  return self(d, this->shape_);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_operator_plus_eq(const_reference val) const {
  static_assert(has_plus_operator_v<value_type>, "Value type must have a plus operator");

  constexpr size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
  static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

  index_type simd_end = this->data_.size() - (this->data_.size() % simd_width);

  index_type            i       = 0;
  neon_type<value_type> val_vec = neon_dup<value_type>(&val);
  for (; i < simd_end; i += simd_width) {
    neon_type<value_type> vec1   = neon_load<value_type>(&this->data_[i]);
    neon_type<value_type> result = neon_add<value_type>(vec1, val_vec);
    neon_store<value_type>(&this->data_[i], result);
  }

#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] = this->data_[i] + val;

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_operator_minus(const tensor& other) const {
  static_assert(has_minus_operator_v<value_type>, "Value type must have a minus operator");

  if (!equal_shape(this->shape(), other.shape())) {
    throw shape_error("Cannot add two tensors with different shapes");
  }

  data_t           d(this->data_.size());
  constexpr size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
  static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

  index_type simd_end = this->data_.size() - (this->data_.size() % simd_width);

  index_type i = 0;
  for (; i < simd_end; i += simd_width) {
    neon_type<value_type> vec1   = neon_load<value_type>(&this->data_[i]);
    neon_type<value_type> vec2   = neon_load<value_type>(&other[i]);
    neon_type<value_type> result = neon_sub<value_type>(vec1, vec2);
    neon_store<value_type>(&d[i], result);
  }
#pragma omp parallel
  for (; i < this->data_[i]; ++i) d[i] = this->data_[i] - other[i];

  return self(this->shape_, d);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_operator_minus(const value_type val) const {
  static_assert(has_minus_operator_v<value_type>, "Value type must have a minus operator");
  data_t d(this->data_.size());

  constexpr size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
  static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

  index_type simd_end = this->data_.size() - (this->data_.size() % simd_width);

  index_type            i       = 0;
  neon_type<value_type> val_vec = neon_dup<value_type>(&val);
  for (; i < simd_end; i += simd_width) {
    neon_type<value_type> vec1   = neon_load<value_type>(&this->data_[i]);
    neon_type<value_type> result = neon_sub<value_type>(vec1, val_vec);
    neon_store<value_type>(&d[i], result);
  }

#pragma omp parallel
  for (; i < this->data_.size(); ++i) d[i] = this->data_[i] - val;

  return self(*this);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_operator_minus_eq(const tensor& other) const {
  static_assert(has_minus_operator_v<value_type>, "Value type must have a minus operator");

  if (!equal_shape(this->shape(), other.shape())) {
    throw shape_error("Tensors shapes must be equal");
  }

  constexpr size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
  static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

  index_type simd_end = this->data_.size() - (this->data_.size() % simd_width);

  index_type i = 0;
  for (; i < simd_end; i += simd_width) {
    neon_type<value_type> vec1   = neon_load<value_type>(&this->data_[i]);
    neon_type<value_type> vec2   = neon_load<value_type>(&other[i]);
    neon_type<value_type> result = neon_sub<value_type>(vec1, vec2);
    neon_store<value_type>(&this->data_[i], result);
  }

#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] -= other[i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_operator_times_eq(const tensor& other) const {
  static_assert(has_times_operator_v<value_type>, "Value type must have a times operator");

  if (!equal_shape(this->shape(), other.shape())) {
    throw shape_error("Tensors shapes must be equal");
  }

  constexpr size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
  static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

  index_type simd_end = this->data_.size() - (this->data_.size() % simd_width);

  index_type i = 0;
  for (; i < simd_end; i += simd_width) {
    neon_type<value_type> vec1   = neon_load<value_type>(&this->data_[i]);
    neon_type<value_type> vec2   = neon_load<value_type>(&other[i]);
    neon_type<value_type> result = neon_mul<value_type>(vec1, vec2);
    neon_store<value_type>(&this->data_[i], result);
  }

#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] *= other[i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_operator_minus_eq(const_reference val) const {
  static_assert(has_minus_operator_v<value_type>, "Value type must have a minus operator");

  constexpr size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
  static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

  index_type simd_end = this->data_.size() - (this->data_.size() % simd_width);

  index_type            i       = 0;
  neon_type<value_type> val_vec = neon_dup<value_type>(&val);
  for (; i < simd_end; i += simd_width) {
    neon_type<value_type> vec1   = neon_load<value_type>(&this->data_[i]);
    neon_type<value_type> result = neon_mul<value_type>(vec1, val_vec);
    neon_store<value_type>(&this->data_[i], result);
  }

#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] -= val;

  return *this;
}
