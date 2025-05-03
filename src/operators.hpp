#pragma once

#include "tensorbase.hpp"

template<class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::operator()(std::initializer_list<index_type> index_list) {
    return data_[compute_index(shape_type(index_list))];
}

template<class _Tp>
typename tensor<_Tp>::const_reference tensor<_Tp>::operator()(std::initializer_list<index_type> index_list) const {
    return data_[compute_index(shape_type(index_list))];
}

template<class _Tp>
bool tensor<_Tp>::operator!=(const tensor& other) const {
    return !(*this == other);
}

template<class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::operator[](const index_type idx) {
    if (idx >= data_.size() || idx < 0)
        throw index_error("Access index is out of range");

    return data_[idx];
}

template<class _Tp>
typename tensor<_Tp>::const_reference tensor<_Tp>::operator[](const index_type idx) const {
    if (idx >= data_.size() || idx < 0)
        throw index_error("Access index is out of range");

    return data_[idx];
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const tensor& other) const {
#if defined(__ARM_NEON)
    return neon_operator_plus(other);
#endif
    if constexpr (!has_plus_operator_v<value_type>)
        throw operator_error("Value type must have a plus operator");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    data_t d(data_.size());

    for (index_type i = 0; i < data_.size(); ++i)
        d[i] = data_[i] + other[i];

    return self(shape_, d);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const value_type value) const {
#if defined(__ARM_NEON)
    return neon_operator_plus(value);
#endif

    if constexpr (!has_plus_operator_v<value_type>)
        throw operator_error("Value type must have a plus operator");

    data_t d(data_.size());

    for (index_type i = 0; i < data_.size(); ++i)
        d[i] = data_[i] + value;

    return self(d, shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator*(const value_type value) const {
    if constexpr (!has_times_operator_v<value_type>)
        throw operator_error("Value type must have a times operator");

    data_t d(data_.size());

    for (index_type i = 0; i < data_.size(); ++i)
        d[i] = data_[i] + value;

    return self(shape_, d);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator*(const tensor& other) const {
    if constexpr (!has_times_operator_v<value_type>)
        throw operator_error("Value type must have a times operator");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    data_t d(data_.size());

    for (index_type i = 0; i < data_.size(); ++i)
        d[i] = data_[i] * other[i];

    return self(shape_, d);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator+=(const tensor& other) const {
#if defined(__ARM_NEON)
    return neon_operator_plus_eq(other);
#endif
    if constexpr (!has_plus_operator_v<value_type>)
        throw operator_error("Value type must have a plus equal to operator");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    index_type i = 0;
    for (auto& elem : data_)
        elem = elem + other[i++];

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator+=(const_reference value) const {
#if defined(__ARM_NEON)
    return neon_operator_plus(value);
#endif
    if constexpr (!has_plus_operator_v<value_type>)
        throw operator_error("Value type must have a plus operator");

    for (index_type i = 0; i < data_.size(); ++i)
        data_[i] = data_[i] + value;

    for (auto& elem : data_)
        elem = elem + value;

    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const tensor& other) const {
#if defined(__ARM_NEON)
    return neon_operator_minus(other);
#endif
    if constexpr (!has_minus_operator_v<value_type>)
        throw operator_error("Value type must have a minus operator");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    data_t d(data_.size());

    for (index_type i = 0; i < data_[i]; ++i)
        d[i] = data_[i] - other[i];

    return self(shape_, d);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const value_type value) const {
#if defined(__ARM_NEON)
    return neon_operator_minus(value);
#endif
    if constexpr (!has_minus_operator_v<value_type>)
        throw operator_error("Value type must have a minus operator");

    data_t d(data_.size());

    for (index_type i = 0; i < data_.size(); ++i)
        d[i] = data_[i] - value;

    return self(*this);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator-=(const tensor& other) const {
#if defined(__ARM_NEON)
    return neon_operator_minus_eq(other);
#endif
    if constexpr (!has_minus_operator_v<value_type>)
        throw operator_error("Value type must have a minus operator");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    index_type i = 0;
    for (auto& elem : data_)
        elem = elem - other[i++];

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator*=(const tensor& other) const {
    if constexpr (!has_times_operator_v<value_type>)
        throw operator_error("Value type must have a times operator");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    index_type i = 0;
    for (auto& elem : data_)
        elem = elem * other[i++];

    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator/(const_reference value) const {
    if constexpr (!has_divide_operator_v<value_type>)
        throw operator_error("Value type must have a divide operator");

    if (value == value_type(0))
        throw std::logic_error("Cannot divide by zero : undefined operation");

    data_t d(data_.size());

    index_type i = 0;
    for (auto& elem : data_)
        d[i++] = elem / value;

    return self(shape_, d);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator*=(const_reference value) const {
#if defined(__ARM_NEON)
    return neon_operator_minus_eq(value);
#endif
    if constexpr (!has_times_operator_v<value_type>)
        throw operator_error("Value type must have a times operator");

    for (auto& elem : data_)
        elem = elem * value;

    return *this;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::operator=(const tensor& other) const {
    shape_ = other.shape();
    data_  = other.storage();
    compute_strides();
    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator/=(const tensor& other) const {
    if constexpr (!has_divide_operator_v<value_type>)
        throw operator_error("Value type must have a divide operator");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    if (other.count_nonzero(0) != other.size(0))
        throw std::logic_error("Cannot divide by zero : undefined operation");

    index_type i = 0;
    for (auto& elem : data_)
        elem = elem / other[i++];

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator/=(const_reference value) const {
    if constexpr (!has_divide_operator_v<value_type>)
        throw operator_error("Value type must have a divide operator");

    if (value == value_type(0))
        throw std::invalid_argument("Cannot divide by zero : undefined operation");

    for (auto& elem : data_)
        elem = elem / value;

    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator/(const tensor& other) const {
    if constexpr (!has_divide_operator_v<value_type>)
        throw operator_error("Value type must have a divide operator");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    if (other.count_nonzero(0) != other.size(0))
        throw std::logic_error("Cannot divide by zero : undefined operation");

    data_t d(data_.size());

    for (index_type i = 0; i < data_.size(); ++i)
        d[i] = data_[i] / other[i];

    return self(shape_, d);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator-=(const_reference value) const {
#if defined(__ARM_NEON)
    return neon_operator_minus_eq(value);
#endif
    if constexpr (!has_minus_operator_v<value_type>)
        throw operator_error("Value type must have a minus operator");

    for (auto& elem : data_)
        elem = elem - value;

    return *this;
}

template<class _Tp>
bool tensor<_Tp>::operator==(const tensor& other) const {
    if (equal_shape(shape(), other.shape()) && strides_ == other.strides() && data_ == other.storage())
        return true;
    return false;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator=(tensor&& other) const noexcept {
    if (this != &other)
    {
        data_    = std::move(other.storage());
        shape_   = std::move(other.shape());
        strides_ = std::move(other.strides());
    }
    return *this;
}

template<class _Tp>
const tensor<bool>& tensor<_Tp>::operator!() const {
    return logical_not_();
}

template<class _Tp>
tensor<bool>& tensor<_Tp>::operator!() {
    return logical_not_();
}