#pragma once

#include "tensorbase.hpp"
#include "types.hpp"


template<class _Tp>
tensor<bool> tensor<_Tp>::not_equal(const tensor& other) const {
    if (internal::types::using_neon())
    {
        return internal::neon::not_equal(*this, other);
    }

    if constexpr (!internal::types::has_not_equal_operator_v<value_type>)
    {
        throw error::operator_error("Value type must have an equal to operator");
    }

    if (!shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    std::vector<bool> ret(data_.size());

    for (index_type i = 0, n = data_.size(); i < n; ++i)
    {
        ret[i] = (data_[i] != other[i]);
    }

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::not_equal(const value_type value) const {
    if (internal::types::using_neon())
    {
        return internal::neon::not_equal(*this, value);
    }

    if constexpr (!internal::types::has_equal_operator_v<value_type>)
    {
        throw error::operator_error("Value type must have an equal to operator");
    }

    std::vector<bool> ret(data_.size());
    index_type        i = 0;

    for (const auto& elem : data_)
    {
        ret[i++] = (elem != value);
    }

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less(const tensor& other) const {
    if (internal::types::using_neon())
    {
        return internal::neon::less(*this, other);
    }

    if constexpr (!internal::types::has_less_operator_v<value_type>)
    {
        throw error::operator_error("Value type must have a less than operator");
    }

    if (!shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    std::vector<bool> ret(data_.size());

    for (index_type i = 0, n = data_.size(); i < n; ++i)
    {
        ret[i] = (data_[i] < other[i]);
    }

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less(const value_type value) const {
    if (internal::types::using_neon())
    {
        return internal::neon::less(*this, value);
    }

    if constexpr (!internal::types::has_less_operator_v<value_type>)
    {
        throw error::operator_error("Value type must have a less  than operator");
    }

    std::vector<bool> ret(data_.size());
    index_type        i = 0;

    for (const auto& elem : data_)
    {
        ret[i] = (elem < value);
        ++i;
    }

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater(const tensor& other) const {
    return other.less(*this);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater(const value_type value) const {
    if (internal::types::using_neon())
    {
        return internal::neon::greater(*this, value);
    }

    if constexpr (!internal::types::has_greater_operator_v<value_type>)
    {
        throw error::operator_error("Value type must have a greater than operator");
    }

    std::vector<bool> ret(data_.size());
    index_type        i = 0;

    for (const auto& elem : data_)
    {
        ret[i++] = (elem > value);
    }

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const tensor& other) const {
    if (internal::types::using_neon())
    {
        return internal::neon::equal(*this, other);
    }

    if constexpr (!internal::types::has_equal_operator_v<value_type>)
    {
        throw error::operator_error("Value type must have equal operator");
    }

    if (!shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors must have the same shape");
    }

    std::vector<bool> ret_data(data_.size());
    index_type        i = 0;

    for (auto& elem : data_)
    {
        ret_data[i] = (elem == other[i]);
        ++i;
    }

    return tensor<bool>(shape(), ret_data);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const value_type value) const {
    if constexpr (!internal::types::has_equal_operator_v<value_type>)
    {
        throw error::operator_error("Value type must have equal operator");
    }

    std::vector<bool> ret_data(data_.size());
    index_type        i = 0;

    for (auto& elem : data_)
    {
        ret_data[i++] = (elem == value);
    }

    return tensor<bool>(shape(), ret_data);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const tensor& other) const {
    if (internal::types::using_neon())
    {
        return internal::neon::less_equal(*this, other);
    }

    if constexpr (!internal::types::has_less_equal_operator_v<value_type>)
    {
        throw error::operator_error("Value type must have a less than or equal to operator");
    }

    if (!shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    std::vector<bool> ret(data_.size());

    for (index_type i = 0; i < ret.size(); ++i)
    {
        ret[i] = (data_[i] <= other[i]);
    }

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const value_type value) const {
    if (internal::types::using_neon())
    {
        return internal::neon::less_equal(value);
    }

    if constexpr (!internal::types::has_less_equal_operator_v<value_type>)
    {
        throw error::operator_error("Value type must have a less than or equal to operator");
    }

    std::vector<bool> ret(data_.size());
    index_type        i = 0;

    for (const auto& elem : data_)
    {
        ret[i++] = (elem <= value);
    }

    return tensor<bool>(shape(), ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const tensor& other) const {
    return other.less_equal(*this);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const value_type value) const {
    if (internal::types::using_neon())
    {
        return internal::neon::greater_equal(*this, value);
    }

    if constexpr (!internal::types::has_greater_equal_operator_v<value_type>)
    {
        throw error::operator_error("Value type must have a greater than or equal to operator");
    }

    std::vector<bool> ret(data_.size());
    index_type        i = 0;

    for (const auto& elem : data_)
    {
        ret[i++] = (elem >= value);
    }

    return tensor<bool>(shape(), ret);
}
