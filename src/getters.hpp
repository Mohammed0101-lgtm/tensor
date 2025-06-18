#pragma once

#include "tensorbase.hpp"

template<class _Tp>
inline typename tensor<_Tp>::data_t tensor<_Tp>::storage() const noexcept {
    return data_;
}

template<class _Tp>
inline typename tensor<_Tp>::data_t& tensor<_Tp>::storage_() const {
    return std::ref<data_t>(data_);
}

template<class _Tp>
inline shape::Shape tensor<_Tp>::shape() const noexcept {
    return shape_;
}

template<class _Tp>
inline tensor<_Tp>::Device tensor<_Tp>::device() const noexcept {
    return device_;
}

template<class _Tp>
inline std::size_t tensor<_Tp>::n_dims() const noexcept {
    return shape_.size();
}

template<class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::capacity() const noexcept {
    return data_.capacity();
}

template<class _Tp>
bool tensor<_Tp>::empty() const {
    return data_.empty();
}