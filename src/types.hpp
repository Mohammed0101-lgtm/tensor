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
tensor<int> tensor<_Tp>::int_() const {
    if (!std::is_convertible_v<value_type, int>)
    {
        throw type_error("Type must be convertible to 32 bit signed int");
    }

    if (empty())
    {
        return tensor<int>(shape_);
    }

    std::vector<int> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[i++] = int(elem);
    }

    return tensor<int>(shape_, d);
}

template<class _Tp>
tensor<unsigned int> tensor<_Tp>::unsigned_int_() const {
    if (!std::is_convertible_v<value_type, unsigned int>)
    {
        throw type_error("Type must be convertible to 32 bit unsigned int");
    }

    if (empty())
    {
        return tensor<unsigned int>(shape_);
    }

    std::vector<unsigned int> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[i++] = (unsigned int) (elem);
    }

    return tensor<unsigned int>(shape_, d);
}

template<class _Tp>
tensor<float> tensor<_Tp>::float_() const {
    if (!std::is_convertible_v<value_type, float>)
    {
        throw type_error("Type must be convertible to 32 bit float");
    }

    if (empty())
    {
        return tensor<float>(shape_);
    }

    std::vector<float> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[i++] = float(elem);
    }

    return tensor<float>(shape_, d);
}

template<class _Tp>
tensor<double> tensor<_Tp>::double_() const {
    if (!std::is_convertible_v<value_type, double>)
    {
        throw type_error("Type must be convertible to 64 bit float");
    }

    if (empty())
    {
        return tensor<double>(shape_);
    }

    std::vector<double> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[i++] = double(elem);
    }

    return tensor<double>(shape_, d);
}

template<class _Tp>
tensor<unsigned long> tensor<_Tp>::unsigned_long_() const {
    if (!std::is_convertible_v<value_type, unsigned long>)
    {
        throw type_error("Type must be convertible to unsigned 64 bit int");
    }

    if (empty())
    {
        return tensor<unsigned long>(shape_);
    }

    std::vector<unsigned long> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[i++] = (unsigned long) (elem);
    }

    return tensor<unsigned long>(shape_, d);
}

template<class _Tp>
inline tensor<short> tensor<_Tp>::short_() const {
    if (!std::is_convertible_v<value_type, short>)
    {
        throw type_error("Type must be convertible to short (aka 16 bit int)");
    }

    if (empty())
    {
        return tensor<short>(shape_);
    }

    std::vector<short> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[i++] = short(elem);
    }

    return tensor<short>(shape_, d);
}

template<class _Tp>
inline tensor<long long> tensor<_Tp>::long_long_() const {
    if (!std::is_convertible_v<value_type, long long>)
    {
        throw type_error("Type must be convertible to 64 bit int (aka long long)");
    }

    if (empty())
    {
        return tensor<long long>(shape_);
    }

    std::vector<long long> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[i++] = (long long) (elem);
    }

    return tensor<long long>(shape_, d);
}

template<class _Tp>
tensor<long> tensor<_Tp>::long_() const {
    if (!std::is_convertible_v<value_type, long>)
    {
        throw type_error("Type must be convertible to 64 bit signed int");
    }

    if (empty())
    {
        return tensor<long>(shape_);
    }

    std::vector<long> d(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
    {
        d[i++] = long(elem);
    }

    return tensor<long>(shape_, d);
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
        d[i++] = bool(elem);
    }

    return tensor<bool>(shape_, d);
}