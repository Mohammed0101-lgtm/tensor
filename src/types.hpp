#pragma once

#include "tensorbase.hpp"

template <class, class = std::void_t<>>
struct has_plus_operator : std::false_type {};

template <class _Tp>
struct has_plus_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>
    : std::true_type {};

template <class _Tp>
constexpr bool has_plus_operator_v = has_plus_operator<_Tp>::value;

template <class, class = std::void_t<>>
struct has_minus_operator : std::false_type {};

template <class _Tp>
struct has_minus_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>
    : std::true_type {};

template <class _Tp>
constexpr bool has_minus_operator_v = has_minus_operator<_Tp>::value;

template <class, class = std::void_t<>>
struct has_times_operator : std::false_type {};

template <class _Tp>
struct has_times_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>
    : std::true_type {};

template <class _Tp>
constexpr bool has_times_operator_v = has_times_operator<_Tp>::value;

template <class, class = std::void_t<>>
struct has_divide_operator : std::false_type {};

template <class _Tp>
struct has_divide_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>
    : std::true_type {};

template <class _Tp>
constexpr bool has_divide_operator_v = has_divide_operator<_Tp>::value;

template <class _Tp>
tensor<_s32> tensor<_Tp>::int32_() const {
#if defined(__ARM_NEON)
  return this->neon_int32_();
#endif
  static_assert(std::is_convertible_v<value_type, _s32>);

  if (this->empty()) return tensor<_s32>(this->__shape_);

  std::vector<_s32> __d;
  index_type        __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) __d.push_back(static_cast<_s32>(this->__data_[__i]));

  return tensor<_s32>(this->__shape_, __d);
}

template <class _Tp>
tensor<_u32> tensor<_Tp>::uint32_() const {
#if defined(__ARM_NEON)
  return this->neon_uint32_();
#endif
  static_assert(std::is_convertible_v<value_type, _u32>);

  if (this->empty()) return tensor<_u32>(this->__shape_);

  std::vector<_u32> __d(this->__data_.size());
  index_type        __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) __d[__i] = static_cast<_u32>(this->__data_[__i]);

  return tensor<_u32>(this->__shape_, __d);
}

template <class _Tp>
tensor<_f32> tensor<_Tp>::float32_() const {
#if defined(__ARM_NEON)
  return this->neon_float32_();
#endif
  static_assert(std::is_convertible_v<value_type, _f32>,
                "Tensor value type must be convertible to _f32.");

  if (this->empty()) return tensor<_f32>(this->__shape_);

  std::vector<_f32> __d(this->__data_.size());
  index_type        __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) __d[__i] = static_cast<_f32>(this->__data_[__i]);

  return tensor<_f32>(this->__shape_, __d);
}

template <class _Tp>
tensor<_f64> tensor<_Tp>::double_() const {
#if defined(__ARM_NEON)
  return this->neon_double_();
#endif
  static_assert(std::is_convertible_v<value_type, _f64>,
                "Tensor value type must be convertible to _f64.");

  if (this->empty()) return tensor<_f64>(this->__shape_);

  std::vector<_f64> __d(this->__data_.size());
  index_type        __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) __d[__i] = static_cast<_f64>(this->__data_[__i]);

  return tensor<_f64>(this->__shape_, __d);
}

template <class _Tp>
tensor<uint64_t> tensor<_Tp>::unsigned_long_() const {
#if defined(__ARM_NEON)
  return this->neon_unsigned_long_();
#endif
  static_assert(std::is_convertible_v<value_type, uint64_t>,
                "Tensor value type must be convertible to uint64_t.");

  if (this->empty()) return tensor<uint64_t>(this->__shape_);

  std::vector<uint64_t> __d(this->__data_.size());
  index_type            __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) __d[__i] = static_cast<uint64_t>(this->__data_[__i]);

  return tensor<uint64_t>(this->__shape_, __d);
}

template <class _Tp>
tensor<int64_t> tensor<_Tp>::long_() const {
#if defined(__ARM_NEON)
  return this->neon_long_();
#endif
  static_assert(std::is_convertible_v<value_type, int64_t>,
                "Tensor value type must be convertible to int64_t.");

  if (this->empty()) return tensor<int64_t>(this->__shape_);

  std::vector<int64_t> __d(this->__data_.size());
  index_type           __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) __d[__i] = static_cast<int64_t>(this->__data_[__i]);

  return tensor<int64_t>(this->__shape_, __d);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::bool_() const {
  std::vector<bool> __d;

  static_assert(std::is_convertible_v<value_type, bool>);
#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    __d.push_back(static_cast<bool>(this->__data_[__i]));

  return tensor<bool>(this->__shape_, __d);
}