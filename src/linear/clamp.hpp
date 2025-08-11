#pragma once

#include "tensor.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::clamp_(const_reference min_val, const_reference max_val) {
  index_type      i = 0;
  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::max(min_val, elem);
    elem = std::min(max_val, elem);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::clamp(const_reference min_val, const_reference max_val) const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.clamp_(min_val, max_val);
  return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::floor() const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.floor_();
  return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::floor_() {
  if (!std::is_floating_point_v<value_type>)
  {
    throw error::type_error("Type must be floating point");
  }

  index_type      i = 0;
  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::floor(elem);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::ceil_() {
  if (!std::is_floating_point_v<value_type>)
  {
    throw error::type_error("Type must be floating point");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::ceil(elem);
  }

  return *this;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::ceil() const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.ceil_();
  return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::clamp_min(const_reference min_val) const {
  return clamp(min_val);
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::clamp_min_(const_reference min_val) {
  return clamp_(min_val);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::clamp_max(const_reference max_val) const {
  return clamp(std::numeric_limits<value_type>::lowest(), max_val);
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::clamp_max_(const_reference max_val) {
  return clamp_(std::numeric_limits<value_type>::lowest(), max_val);
}