#pragma once

#include "tensorbase.hpp"

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

template<class _Tp>
tensor<_s32> tensor<_Tp>::int32_() const {
    if (!std::is_convertible_v<value_type, _s32>)
    {
        throw type_error("Type must be convertible to 32 bit signed int");
    }

    if (empty())
    {
        return tensor<_s32>(shape_);
    }

    std::vector<_s32> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[++i] = static_cast<_s32>(elem);
    }

    return tensor<_s32>(shape_, d);
}

template<class _Tp>
tensor<_u32> tensor<_Tp>::uint32_() const {
    if (!std::is_convertible_v<value_type, _u32>)
    {
        throw type_error("Type must be convertible to 32 bit unsigned int");
    }

    if (empty())
    {
        return tensor<_u32>(shape_);
    }

    std::vector<_u32> d(data_.size());

    for (index_type i = 0; i < data_.size(); ++i)
    {
        d[i] = static_cast<_u32>(data_[i]);
    }

    return tensor<_u32>(shape_, d);
}

template<class _Tp>
tensor<_f32> tensor<_Tp>::float32_() const {
    if (!std::is_convertible_v<value_type, _f32>)
    {
        throw type_error("Type must be convertible to 32 bit float");
    }

    if (empty())
    {
        return tensor<_f32>(shape_);
    }

    std::vector<_f32> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[++i] = static_cast<_f32>(elem);
    }

    return tensor<_f32>(shape_, d);
}

template<class _Tp>
tensor<_f64> tensor<_Tp>::double_() const {
    if (!std::is_convertible_v<value_type, _f64>)
    {
        throw type_error("Type must be convertible to 64 bit float");
    }

    if (empty())
    {
        return tensor<_f64>(shape_);
    }

    std::vector<_f64> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[++i] = static_cast<_f64>(elem);
    }

    return tensor<_f64>(shape_, d);
}

template<class _Tp>
tensor<uint64_t> tensor<_Tp>::unsigned_long_() const {
    if (!std::is_convertible_v<value_type, uint64_t>)
    {
        throw type_error("Type must be convertible to unsigned 64 bit int");
    }

    if (empty())
    {
        return tensor<uint64_t>(shape_);
    }

    std::vector<uint64_t> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[++i] = static_cast<uint64_t>(elem);
    }

    return tensor<uint64_t>(shape_, d);
}

template<class _Tp>
tensor<int64_t> tensor<_Tp>::long_() const {
    if (!std::is_convertible_v<value_type, int64_t>)
    {
        throw type_error("Type must be convertible to 64 bit signed int");
    }

    if (empty())
    {
        return tensor<int64_t>(shape_);
    }

    std::vector<int64_t> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[++i] = static_cast<int64_t>(elem);
    }

    return tensor<int64_t>(shape_, d);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::bool_() const {
    if (!std::is_convertible_v<value_type, bool>)
    {
        throw type_error("Type must be convertible to bool");
    }
    std::vector<bool> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[++i] = static_cast<bool>(elem);
    }

    return tensor<bool>(shape_, d);
}