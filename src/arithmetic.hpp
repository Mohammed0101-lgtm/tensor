#pragma once

#include "tensorbase.hpp"

#define __builtin_neon_vgetq_lane_f32
#define __builtin_neon_vsetq_lane_f32

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmax(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fmax_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmax(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fmax_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_fmax(__val);
#endif
  size_t __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::fmax(this->__data_[__i], __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmax_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_fmax_(__val);
#endif
  size_t __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::fmax(this->__data_[__i], __val);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_fmax_(__other);
#endif
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::fmax(this->__data_[__i], __other[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmax_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_fmax_(__other);
#endif
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::fmax(this->__data_[__i], __other[__i]);

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmod(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fmod_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmod(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fmod_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::fmod_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_fmod_(__val);
#endif
  assert(std::is_floating_point_v<value_type> &&
         "fmod : template class must be a floating point type");
  index_type __i = 0;

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__val)));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmod_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_fmod_(__val);
#endif
  assert(std::is_floating_point_v<value_type> &&
         "fmod : template class must be a floating point type");
  index_type __i = 0;

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__val)));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::fmod_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_fmod_(__other);
#endif

  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");

  index_type __i = 0;

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmod_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_fmod_(__other);
#endif

  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");

  index_type __i = 0;

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::frac_() {
#if defined(__ARM_NEON)
  return this->neon_frac_();
#endif
  index_type __i = 0;

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::frac_() const {
#if defined(__ARM_NEON)
  return this->neon_frac_();
#endif
  index_type __i = 0;

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::frac() const {
  __self __ret = this->clone();
  __ret.frac_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::log_() {
#if defined(__ARM_NEON)
  return this->neon_log_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log_() const {
#if defined(__ARM_NEON)
  return this->neon_log_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::log() const {
  __self __ret = this->clone();
  __ret.log_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::log10_() {
#if defined(__ARM_NEON)
  return this->neon_log10_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log10(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log10_() const {
#if defined(__ARM_NEON)
  return this->neon_log10_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log10(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::log10() const {
  __self __ret = this->clone();
  __ret.log10_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::log2_() {
#if defined(__ARM_NEON)
  return this->neon_log2_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log2(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log2_() const {
#if defined(__ARM_NEON)
  return this->neon_log2_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log2(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::log2() const {
  __self __ret = this->clone();
  __ret.log2_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::exp_() {
#if defined(__ARM_NEON)
  return this->neon_exp_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::exp(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::exp_() const {
#if defined(__ARM_NEON)
  return this->neon_exp_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::exp(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::exp() const {
  __self __ret = this->clone();
  __ret.exp_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sqrt_() {
#if defined(__ARM_NEON)
  return this->neon_sqrt_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sqrt(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sqrt_() const {
#if defined(__ARM_NEON)
  return this->neon_sqrt_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sqrt(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sqrt() const {
  __self __ret = this->clone();
  __ret.sqrt_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::square() const {
  __self __ret = this->clone();
  __ret.square_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::cos_() {
#if defined(__ARM_NEON)
  return this->neon_cos_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::cos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::cos_() const {
#if defined(__ARM_NEON)
  return this->neon_cos_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::cos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::acos_() {
#if defined(__ARM_NEON)
  return this->neon_acos_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::acos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::acos_() const {
#if defined(__ARM_NEON)
  return this->neon_acos_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::acos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::acos() const {
  __self __ret = this->clone();
  __ret.acos_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::cos() const {
  __self __ret = this->clone();
  __ret.cos_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sin_() {
#if defined(__ARM_NEON)
  return this->neon_sin_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sin(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sin_() const {
#if defined(__ARM_NEON)
  return this->neon_sin_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sin(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sin() const {
  __self __ret = this->clone();
  __ret.sin_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::tan_() {
#if defined(__ARM_NEON)
  return this->neon_tan_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::tan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::tan_() const {
#if defined(__ARM_NEON)
  return this->neon_tan_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::tan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::tanh_() {
#if defined(__ARM_NEON)
  return this->neon_tanh_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::tanh(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::tanh_() const {
#if defined(__ARM_NEON)
  return this->neon_tanh_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::tanh(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::asin() const {
  __self __ret = this->clone();
  __ret.asin_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::cosh() const {
  __self __ret = this->clone();
  __ret.cosh_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::atan() const {
  __self __ret = this->clone();
  __ret.atan_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sinc_() {
#if defined(__ARM_NEON)
  return this->neon_sinc_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (std::abs(this->__data_[__i]) < 1e-6)
                             ? static_cast<value_type>(1.0)
                             : static_cast<value_type>(std::sin(M_PI * this->__data_[__i]) /
                                                       (M_PI * this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sinc_() const {
#if defined(__ARM_NEON)
  return this->neon_sinc_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (std::abs(this->__data_[__i]) < 1e-6)
                             ? static_cast<value_type>(1.0)
                             : static_cast<value_type>(std::sin(M_PI * this->__data_[__i]) /
                                                       (M_PI * this->__data_[__i]));
  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::atan_() {
#if defined(__ARM_NEON)
  return this->neon_atan_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::atan_() const {
#if defined(__ARM_NEON)
  return this->neon_atan_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::atanh_() {
#if defined(__ARM_NEON)
  return this->neon_atanh_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::atanh_() const {
#if defined(__ARM_NEON)
  return this->neon_atanh_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::atanh() const {
  __self __ret = this->clone();
  __ret.atanh_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sinc() const {
  __self __ret = this->clone();
  __ret.sinc_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sinh_() {
#if defined(__ARM_NEON)
  return this->neon_sinh_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sinh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sinh_() const {
#if defined(__ARM_NEON)
  return this->neon_sinh_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sinh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sinh() const {
  __self __ret = this->clone();
  __ret.sinh_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::asinh_() {
#if defined(__ARM_NEON)
  return this->neon_sinh_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::asinh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::asinh_() const {
#if defined(__ARM_NEON)
  return this->neon_sinh_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::asinh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::asinh() const {
  __self __ret = this->clone();
  __ret.asinh_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::asin_() {
#if defined(__ARM_NEON)
  return this->neon_asin_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::asin(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::asin_() const {
#if defined(__ARM_NEON)
  return this->neon_asin_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::asin(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::cosh_() {
#if defined(__ARM_NEON)
  return this->neon_cosh_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::cosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::cosh_() const {
#if defined(__ARM_NEON)
  return this->neon_cosh_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::cosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::acosh_() {
#if defined(__ARM_NEON)
  return this->neon_acosh_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::acosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::acosh_() const {
#if defined(__ARM_NEON)
  return this->neon_acosh_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::acosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::acosh() const {
  __self __ret = this->clone();
  __ret.acosh_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::pow_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_pow_(__val);
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::pow(this->__data_[__i], __val));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::pow_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_pow_(__val);
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::pow(this->__data_[__i], __val));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::pow(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.pow_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::pow(const value_type __val) const {
  __self __ret = this->clone();
  __ret.pow_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::pow_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_pow_(__other);
#endif
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::pow(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::pow_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_pow_(__other);
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::pow(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::abs() const {
  __self __ret = this->clone();
  __ret.abs_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::abs_() {
#if defined(__ARM_NEON)
  return this->neon_abs_();
#endif
  if (std::is_unsigned_v<value_type>) return *this;

  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::abs(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::abs_() const {
#if defined(__ARM_NEON)
  return this->neon_abs_();
#endif
  if (std::is_unsigned_v<value_type>) return *this;

  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::abs(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::log_softmax(const index_type __dim) const {
  __self __ret = this->clone();
  __ret.log_softmax_(__dim);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::dist(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.dist_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::dist(const value_type __val) const {
  __self __ret = this->clone();
  __ret.dist_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::dist_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_dist_(__other);
#endif
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::abs(static_cast<_f64>(this->__data_[__i] - __other.__data_[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::dist_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_dist_(__other);
#endif
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::abs(static_cast<_f64>(this->__data_[__i] - __other.__data_[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::dist_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_dist_(__val);
#endif
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] =
        static_cast<value_type>(std::abs(static_cast<_f64>(this->__data_[__i] - __val)));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::dist_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_dist_(__val);
#endif
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] =
        static_cast<value_type>(std::abs(static_cast<_f64>(this->__data_[__i] - __val)));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::remainder(const value_type __val) const {
  __self __ret = this->clone();
  __ret.remainder_(__val);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::remainder(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.remainder_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::remainder_(const value_type __val) {
  assert(__val != 0 && "Remainder by zero is undefined");
#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] %= __val;
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::remainder_(const value_type __val) const {
  assert(__val != 0 && "Remainder by zero is undefined");
#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] %= __val;
  return *this;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::remainder_(const tensor& __other) {
  assert(__other.count_nonzero() == __other.size(0) && "Remainder by zero is undefined");
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] %= __other.__data_[__i];
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::remainder_(const tensor& __other) const {
  assert(__other.count_nonzero() == __other.size(0) && "Remainder by zero is undefined");
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] %= __other.__data_[__i];
  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::maximum(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.maximum_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::maximum(const_reference __val) const {
  __self __ret = this->clone();
  __ret.maximum_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::maximum_(const tensor& __other) {
#if defined(__ARM_NEON)
  return this->neon_maximum_(__other);
#endif
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::max(this->__data_[__i], __other.__data_[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::maximum_(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_maximum_(__other);
#endif
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::max(this->__data_[__i], __other.__data_[__i]);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::maximum_(const value_type __val) {
#if defined(__ARM_NEON)
  return this->neon_maximum_(__val);
#endif
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::max(this->__data_[__i], __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::maximum_(const value_type __val) const {
#if defined(__ARM_NEON)
  return this->neon_maximum_(__val);
#endif
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::max(this->__data_[__i], __val);

  return *this;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::square_() {
  return this->pow_(static_cast<value_type>(2.0f));
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::square_() const {
  return this->pow_(static_cast<value_type>(2.0f));
}

template <class _Tp>
double tensor<_Tp>::mean() const {
#if defined(__ARM_NEON)
  return this->neon_mean();
#endif
  double __m = 0.0;

  if (this->empty()) return __m;

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i) __m += this->__data_[__i];

  return static_cast<double>(__m) / static_cast<double>(this->__data_.size());
}

// used as a helper function
inline int64_t __lcm(const int64_t __a, const int64_t __b) {
  return (__a * __b) / std::gcd(__a, __b);
}

template <class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::lcm() const {
  index_type __ret = static_cast<index_type>(this->__data_[0]);
  index_type __i   = 1;
  for (; __i < this->__data_.size(); ++__i)
    __ret = __lcm(static_cast<index_type>(this->__data_[__i]), __ret);

  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::expand_as(shape_type __sh, index_type __dim) const {}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::lcm(const tensor& __other) const {
  assert(this->__shape_ == __other.shape());
  tensor     __ret = this->clone();
  index_type __i   = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    __ret[__i] = static_cast<value_type>(this->__lcm(static_cast<index_type>(this->__data_[__i]),
                                                     static_cast<index_type>(__other[__i])));

  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::tanh() const {
  __self __ret = this->clone();
  __ret.tanh_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::tan() const {
  __self __ret = this->clone();
  __ret.tan_();
  return __ret;
}

template <class _Tp>
double tensor<_Tp>::mode(const index_type __dim) const {
  if (__dim >= this->n_dims() || __dim < -1)
    throw std::invalid_argument("given dimension is out of range of the tensor dimensions");

  index_type __stride = (__dim == -1) ? 0 : this->__strides_[__dim];
  index_type __end    = (__dim == -1) ? this->__data_.size() : this->__strides_[__dim];

  if (this->__data_.empty()) return std::numeric_limits<double>::quiet_NaN();

  std::unordered_map<value_type, size_t> __counts;
  for (index_type __i = __stride; __i < __end; ++__i) ++__counts[this->__data_[__i]];

  value_type __ret  = 0;
  size_t     __most = 0;
  for (const std::pair<value_type, size_t>& __pair : __counts) {
    if (__pair.second > __most) {
      __ret  = __pair.first;
      __most = __pair.second;
    }
  }

  return static_cast<double>(__ret);
}

template <class _Tp>
double tensor<_Tp>::median(const index_type __dim) const {
  if (__dim >= this->n_dims() || __dim < -1)
    throw std::invalid_argument("given dimension is out of range of the tensor dimensions");

  index_type __stride = (__dim == -1) ? 0 : this->__strides_[__dim];
  index_type __end    = (__dim == -1) ? this->__data_.size() : this->__strides_[__dim];

  data_t __d(this->__data_.begin() + __stride, this->__data_.begin() + __end);

  if (__d.empty()) return std::numeric_limits<double>::quiet_NaN();

  std::nth_element(__d.begin(), __d.begin() + __d.size() / 2, __d.end());

  if (__d.size() % 2 == 0) {
    std::nth_element(__d.begin(), __d.begin() + __d.size() / 2 - 1, __d.end());
    return (static_cast<double>(__d[__d.size() / 2]) + __d[__d.size() / 2 - 1]) / 2.0;
  }

  return __d[__d.size() / 2];
}