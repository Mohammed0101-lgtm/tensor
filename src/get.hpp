#pragma once

#include "tensorbase.hpp"

template <class _Tp>
inline typename tensor<_Tp>::data_t tensor<_Tp>::storage() const noexcept {
  return this->__data_;
}

template <class _Tp>
inline typename tensor<_Tp>::shape_type tensor<_Tp>::shape() const noexcept {
  return this->__shape_;
}

template <class _Tp>
inline typename tensor<_Tp>::shape_type tensor<_Tp>::strides() const noexcept {
  return this->__strides_;
}

template <class _Tp>
inline tensor<_Tp>::Device tensor<_Tp>::device() const noexcept {
  return this->__device_;
}

template <class _Tp>
inline size_t tensor<_Tp>::n_dims() const noexcept {
  return this->__shape_.size();
}

template <class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::capacity() const noexcept {
  return this->__data_.capacity();
}

template <class _Tp>
bool tensor<_Tp>::empty() const {
  return this->__data_.empty();
}