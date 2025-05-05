#pragma once

#include "tensorbase.hpp"
#include "types.hpp"

template<class _Tp>
tensor<bool> tensor<_Tp>::not_equal(const tensor& other) const {
    if constexpr (!has_not_equal_operator_v<value_type>)
        throw operator_error("Value type must have an equal to operator");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    std::vector<bool> ret(data_.size());

    for (index_type i = 0, n = data_.size(); i < n; ++i)
        ret[i] = (data_[i] != other[i]);

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::not_equal(const value_type value) const {
    if constexpr (!has_equal_operator_v<value_type>)
        throw operator_error("Value type must have an equal to operator");

    std::vector<bool> ret(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
        ret[i++] = (elem != value);

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less(const tensor& other) const {
    if constexpr (!has_less_operator_v<value_type>)
        throw operator_error("Value type must have a less than operator");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    std::vector<bool> ret(data_.size());

    for (index_type i = 0, n = data_.size(); i < n; ++i)
        ret[i] = (data_[i] < other[i]);

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less(const value_type value) const {
    if constexpr (!has_less_operator_v<value_type>)
        throw operator_error("Value type must have a less  than operator");

    std::vector<bool> ret(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
        ret[i++] = (elem < value);

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater(const tensor& other) const {
    return other.less(*this);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater(const value_type value) const {
    if constexpr (!has_greater_operator_v<value_type>)
        throw operator_error("Value type must have a greater than operator");

    std::vector<bool> ret(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
        ret[i++] = (elem > value);

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const tensor& other) const {
    tensor<bool> ret = not_equal(other);
    ret.logical_not_();
    return ret;
}

template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const value_type value) const {
    return !(not_equal(value));
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const tensor& other) const {
    if constexpr (!has_less_equal_operator_v<value_type>)
        throw operator_error("Value type must have a less than or equal to operator");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    std::vector<bool> ret(data_.size());

    for (index_type i = 0; i < ret.size(); ++i)
        ret[i] = (data_[i] <= other[i]);

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const value_type value) const {
    if constexpr (!has_less_equal_operator_v<value_type>)
        throw operator_error("Value type must have a less than or equal to operator");
    std::vector<bool> ret(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
        ret[i++] = (elem <= value);

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const tensor& other) const {
    return other.less_equal(*this);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const value_type value) const {
    if constexpr (!has_greater_equal_operator_v<value_type>)
        throw operator_error("Value type must have a greater than or equal to operator");

    std::vector<bool> ret(data_.size());

    index_type i = 0;
    for (const auto& elem : data_)
        ret[i++] = (elem >= value);

    return tensor<bool>(shape(), ret);
}
