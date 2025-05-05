#pragma once

#include "tensorbase.hpp"

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::bitwise_right_shift_(const int amount) {
    if constexpr (!std::is_integral_v<value_type>)
        throw type_error("Type must be integral");

    for (auto& elem : data_)
        elem >>= amount;

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_right_shift_(const int amount) const {
    if (!std::is_integral_v<value_type>)
        throw type_error("Type must be integral");

    for (auto& elem : data_)
        elem >>= amount;

    return *this;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::bitwise_left_shift_(const int amount) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Type must be integral");

    for (auto& elem : data_)
        elem <<= amount;

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_left_shift_(const int amount) const {
    if (!std::is_integral_v<value_type>)
        throw type_error("Type must be integral");

    for (auto& elem : data_)
        elem <<= amount;

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_or_(const value_type value) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise OR on non-integral values");

    for (auto& elem : data_)
        elem |= value;

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_or_(const value_type value) const {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise OR on non-integral values");

    for (auto& elem : data_)
        elem |= value;

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const value_type value) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise XOR on non-integral values");

    for (auto& elem : data_)
        elem ^= value;

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const value_type value) const {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise XOR on non-integral");

    for (auto& elem : data_)
        elem ^= value;

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_xor(const value_type value) const {
    self ret = clone();
    ret.bitwise_xor_(value);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_not() const {
    self ret = clone();
    ret.bitwise_not_();
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_not_() {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise NOT on non integral values");

    for (auto& elem : data_)
        elem = ~elem;

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_not_() const {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise NOT on non integral values");

    for (auto& elem : data_)
        elem = ~elem;

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_and_(const value_type value) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise AND on non-integral values");

    for (auto& elem : data_)
        elem &= value;

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_and_(const value_type value) const {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise AND on non-integral values");

    for (auto& elem : data_)
        elem &= value;

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_and(const value_type value) const {
    self ret = clone();
    ret.bitwise_and_(value);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_and(const tensor& other) const {
    self ret = clone();
    ret.bitwise_and_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_left_shift(const int amount) const {
    self ret = clone();
    ret.bitwise_left_shift_(amount);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_xor(const tensor& other) const {
    self ret = clone();
    ret.bitwise_xor_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_right_shift(const int amount) const {
    self ret = clone();
    ret.bitwise_right_shift_(amount);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_and_(const tensor& other) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise AND on non-integral values");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensor shapes must be equal");

    index_type i = 0;
    for (auto& elem : data_)
        elem &= other[i++];

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_and_(const tensor& other) const {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise AND on non-integral values");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    index_type i = 0;
    for (auto& elem : data_)
        elem &= other[i++];

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_or(const tensor& other) const {
    self ret = clone();
    ret.bitwise_or_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_or(const value_type value) const {
    self ret = clone();
    ret.bitwise_or_(value);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_or_(const tensor& other) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise OR on non-integral values");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    index_type i = 0;
    for (auto& elem : data_)
        elem |= other[i++];

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_or_(const tensor& other) const {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise OR on non-integral values");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    index_type i = 0;
    for (auto& elem : data_)
        elem |= other[i++];

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const tensor& other) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise XOR on non-integral values");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    index_type i = 0;
    for (auto& elem : data_)
        elem ^= other[i++];

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const tensor& other) const {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform a bitwise XOR on non-integral values");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    index_type i = 0;
    for (auto& elem : data_)
        elem ^= other[i++];

    return *this;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::fill_(const value_type value) {
    for (auto& elem : data_)
        elem = value;

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fill_(const value_type value) const {
    for (auto& elem : data_)
        elem = value;

    return *this;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::fill_(const tensor& other) {
    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    index_type i = 0;
    for (auto& elem : data_)
        elem = other[i++];

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fill_(const tensor& other) const {
    return fill_(other);
}