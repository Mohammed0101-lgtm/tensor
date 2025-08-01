#pragma once

#include "internal/simd/neon/func.hpp"
#include "tensor.hpp"


namespace internal {
namespace types {

// this should be probablu moved to a separate file for routing the neon code
bool using_neon() {
  /*
#ifdef __ARM_NEON
return true;
#endif
return false;
*/
  return true;
}

template<class, class = std::void_t<>>
struct has_left_shift_operator: std::false_type 
{
};

template<class _Tp>
struct has_left_shift_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>: std::true_type
{
};

template<class _Tp>
constexpr bool has_left_shift_operator_v = has_left_shift_operator<_Tp>::value;

template<class, class = std::void_t<>>
struct has_right_shift_operator: std::false_type
{
};

template<class _Tp>
struct has_right_shift_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>: std::true_type
{
};

template<class _Tp>
constexpr bool has_right_shift_operator_v = has_right_shift_operator<_Tp>::value;

template<class, class = std::void_t<>>
struct has_plus_operator: std::false_type
{
};

template<class _Tp>
struct has_plus_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>: std::true_type
{
};

template<class _Tp>
constexpr bool has_plus_operator_v = has_plus_operator<_Tp>::value;

template<class, class = std::void_t<>>
struct has_minus_operator: std::false_type
{
};

template<class _Tp>
struct has_minus_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>: std::true_type
{
};

template<class _Tp>
constexpr bool has_minus_operator_v = has_minus_operator<_Tp>::value;

template<class, class = std::void_t<>>
struct has_times_operator: std::false_type
{
};

template<class _Tp>
struct has_times_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>: std::true_type
{
};

template<class _Tp>
constexpr bool has_times_operator_v = has_times_operator<_Tp>::value;

template<class, class = std::void_t<>>
struct has_divide_operator: std::false_type
{
};

template<class _Tp>
struct has_divide_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() + std::declval<_Tp>())>>: std::true_type
{
};

template<class _Tp>
constexpr bool has_divide_operator_v = has_divide_operator<_Tp>::value;

template<typename _Tp, typename = void>
struct has_equal_operator: std::false_type
{
};

template<typename _Tp>
struct has_equal_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() == std::declval<_Tp>())>>: std::true_type
{
};

template<typename _Tp>
constexpr bool has_equal_operator_v = has_equal_operator<_Tp>::value;

template<typename _Tp, typename = void>
struct has_not_equal_operator: std::false_type
{
};

template<typename _Tp>
struct has_not_equal_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() != std::declval<_Tp>())>>: std::true_type
{
};

template<typename _Tp>
constexpr bool has_not_equal_operator_v = has_not_equal_operator<_Tp>::value;

template<typename _Tp, typename = void>
struct has_less_operator: std::false_type
{
};

template<typename _Tp>
struct has_less_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() < std::declval<_Tp>())>>: std::true_type
{
};

template<typename _Tp>
constexpr bool has_less_operator_v = has_less_operator<_Tp>::value;

template<typename _Tp, typename = void>
struct has_less_equal_operator: std::false_type
{
};

template<typename _Tp>
struct has_less_equal_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() <= std::declval<_Tp>())>>: std::true_type
{
};

template<typename _Tp>
constexpr bool has_less_equal_operator_v = has_less_equal_operator<_Tp>::value;

template<typename _Tp, typename = void>
struct has_greater_operator: std::false_type
{
};

template<typename _Tp>
struct has_greater_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() > std::declval<_Tp>())>>: std::true_type
{
};

template<typename _Tp>
constexpr bool has_greater_operator_v = has_greater_operator<_Tp>::value;

template<typename _Tp, typename = void>
struct has_greater_equal_operator: std::false_type
{
};

template<typename _Tp>
struct has_greater_equal_operator<_Tp, std::void_t<decltype(std::declval<_Tp>() >= std::declval<_Tp>())>>
    : std::true_type
{
};

template<typename _Tp>
constexpr bool has_greater_equal_operator_v = has_greater_equal_operator<_Tp>::value;

}  // namespace types
}  // namespace internal

template<class _Tp>
tensor<_s64> tensor<_Tp>::int64_() const {
  if (internal::types::using_neon())
  {
    return internal::neon::int64_(*this);
  }

  if (!std::is_convertible_v<value_type, _s64>)
  {
    throw error::type_error("Type must be convertible to 64 bit signed int");
  }

  if (this->empty())
  {
    return tensor<_s64>(this->shape());
  }

  std::vector<_s64> d(this->size(0));
  const data_t&     a = this->storage_();
  index_type        i = 0;

  for (const auto& elem : a)
  {
    d[i++] = static_cast<_s64>(elem);
  }

  return tensor<_s64>(this->shape(), d);
}

template<class _Tp>
tensor<_s32> tensor<_Tp>::int32_() const {
  if (internal::types::using_neon())
  {
    return internal::neon::int32_(*this);
  }

  if (!std::is_convertible_v<value_type, int>)
  {
    throw error::type_error("Type must be convertible to 32 bit signed int");
  }

  if (this->empty())
  {
    return tensor<int>(this->shape());
  }

  std::vector<int> d(this->size(0));
  const data_t&    a = this->storage_();
  index_type       i = 0;

  for (const auto& elem : a)
  {
    d[i++] = static_cast<_s32>(elem);
  }

  return tensor<_s32>(this->shape(), d);
}

template<class _Tp>
tensor<_u32> tensor<_Tp>::uint32_() const {
  if (internal::types::using_neon())
  {
    return internal::neon::uint32_(*this);
  }

  if (!std::is_convertible_v<value_type, unsigned int>)
  {
    throw error::type_error("Type must be convertible to 32 bit unsigned int");
  }

  if (this->empty())
  {
    return tensor<unsigned int>(this->shape());
  }

  std::vector<unsigned int> ret(this->size(0));
  const data_t&             a = this->storage_();
  index_type                i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = static_cast<unsigned int>(elem);
  }

  return tensor<unsigned int>(this->shape(), std::move(ret));
}

template<class _Tp>
tensor<_f32> tensor<_Tp>::float32_() const {
  if (internal::types::using_neon())
  {
    return internal::neon::float32_(*this);
  }

  if (!std::is_convertible_v<value_type, _f32>)
  {
    throw error::type_error("Type must be convertible to 32 bit float");
  }

  if (this->empty())
  {
    return tensor<_f32>(this->shape());
  }

  std::vector<_f32> d(this->size(0));
  const data_t&     a = this->storage_();
  index_type        i = 0;

  for (const auto& elem : a)
  {
    d[i++] = static_cast<_f32>(elem);
  }

  return tensor<_f32>(this->shape(), d);
}

template<class _Tp>
tensor<double> tensor<_Tp>::float64_() const {
  if (internal::types::using_neon())
  {
    return internal::neon::float64_(*this);
  }

  if (!std::is_convertible_v<value_type, double>)
  {
    throw error::type_error("Type must be convertible to 64 bit float");
  }

  if (this->empty())
  {
    return tensor<double>(this->shape());
  }

  std::vector<double> d(this->size(0));
  const data_t&       a = this->storage_();
  index_type          i = 0;

  for (const auto& elem : a)
  {
    d[i++] = double(elem);
  }

  return tensor<double>(this->shape(), d);
}

template<class _Tp>
tensor<_u64> tensor<_Tp>::uint64_() const {
  if (internal::types::using_neon())
  {
    return internal::neon::uint64_(*this);
  }

  if (!std::is_convertible_v<value_type, _u64>)
  {
    throw error::type_error("Type must be convertible to unsigned 64 bit int");
  }

  if (this->empty())
  {
    return tensor<_u64>(this->shape());
  }

  std::vector<_u64> ret(this->size(0));
  const data_t&     a = this->storage_();
  index_type        i = 0;

  for (const auto& elem : a)
  {
    ret[i++] = static_cast<_u64>(elem);
  }

  return tensor<_u64>(this->shape(), std::move(ret));
}

template<class _Tp>
tensor<short> tensor<_Tp>::int16_() const {
  if (internal::types::using_neon())
  {
    return internal::neon::int16_(*this);
  }

  if (!std::is_convertible_v<value_type, short>)
  {
    throw error::type_error("Type must be convertible to short (aka 16 bit int)");
  }

  if (this->empty())
  {
    return tensor<short>(this->shape());
  }

  std::vector<short> d(this->size(0));
  const data_t&      a = this->storage_();
  index_type         i = 0;

  for (const auto& elem : a)
  {
    d[i++] = short(elem);
  }

  return tensor<short>(this->shape(), d);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::bool_() const {
  if (!std::is_convertible_v<value_type, bool>)
  {
    throw error::type_error("Type must be convertible to bool");
  }

  std::vector<bool> d(this->size(0));
  const data_t&     a = this->storage_();
  index_type        i = 0;

  for (const auto& elem : a)
  {
    d[i++] = bool(elem);
  }

  return tensor<bool>(this->shape(), d);
}