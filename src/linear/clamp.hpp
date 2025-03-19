#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::clamp_(const_reference __min_val, const_reference __max_val) {
#if defined(__ARM_NEON)
  return this->neon_clamp_(__min_val, __max_val);
#endif
  index_type __i = 0;

#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) {
    this->__data_[__i] = std::max(__min_val, this->__data_[__i]);
    this->__data_[__i] = std::min(__max_val, this->__data_[__i]);
  }

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::clamp_(const_reference __min_val,
                                              const_reference __max_val) const {

#if defined(__ARM_NEON)
  return this->neon_clamp_(__min_val, __max_val);
#endif
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) {
    this->__data_[__i] = std::max(__min_val, this->__data_[__i]);
    this->__data_[__i] = std::min(__max_val, this->__data_[__i]);
  }

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::clamp(const_reference __min_val, const_reference __max_val) const {
  __self __ret = this->clone();
  __ret.clamp_(__min_val, __max_val);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::floor() const {
  __self __ret = this->clone();
  __ret.floor_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::floor_() {
#if defined(__ARM_NEON)
  return this->neon_floor_();
#endif
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::floor(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::floor_() const {
#if defined(__ARM_NEON)
  return this->neon_floor_();
#endif
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::floor(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::ceil_() {
#if defined(__ARM_NEON)
  return this->neon_ceil_();
#endif
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::ceil(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::ceil_() const {
#if defined(__ARM_NEON)
  return this->neon_ceil_();
#endif
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::ceil(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::ceil() const {
  __self __ret = this->clone();
  __ret.ceil_();
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::clamp_min(const_reference __min_val) const {
  return this->clamp(__min_val);
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::clamp_min_(const_reference __min_val) {
  return this->clamp_(__min_val);
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::clamp_min_(const_reference __min_val) const {
  return this->clamp_(__min_val);
}
//
template <class _Tp>
tensor<_Tp> tensor<_Tp>::clamp_max(const_reference __max_val) const {
  return this->clamp(std::numeric_limits<value_type>::lowest(), __max_val);
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::clamp_max_(const_reference __max_val) {
  return this->clamp_(std::numeric_limits<value_type>::lowest(), __max_val);
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::clamp_max_(const_reference __max_val) const {
  return this->clamp_(std::numeric_limits<value_type>::lowest(), __max_val);
}