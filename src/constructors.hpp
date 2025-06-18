#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>::tensor(const shape::Shape& sh, const_reference v, Device d) :
    shape_(sh),
    data_(sh.flatten_size(), v),
    device_(d) {
    shape_.compute_strides();
}

template<class _Tp>
tensor<_Tp>::tensor(const shape::Shape& sh, Device d) :
    shape_(sh),
    device_(d) {
    data_ = data_t(shape_.flatten_size());
    shape_.compute_strides();
}

template<class _Tp>
tensor<_Tp>::tensor(const shape::Shape& sh, const data_t& d, Device dev) :
    shape_(sh),
    device_(dev) {

    if (d.size() != static_cast<std::size_t>(shape_.flatten_size()))
    {
        throw std::invalid_argument("Initial data vector must match the tensor size : " + std::to_string(d.size())
                                    + " != " + std::to_string(shape_.flatten_size()));
    }

    data_ = d;
    shape_.compute_strides();
}

template<class _Tp>
tensor<_Tp>::tensor(const shape::Shape& sh, std::initializer_list<value_type> init_list, Device d) :
    shape_(sh),
    device_(d) {

    if (static_cast<index_type>(init_list.size()) != sh.flatten_size())
    {
        throw std::invalid_argument("Initializer list size must match the tensor size");
    }

    data_ = data_t(init_list);
    shape_.compute_strides();
}

template<class _Tp>
tensor<_Tp>::tensor(const shape::Shape& sh, const tensor& other) :
    data_(other.storage()),
    shape_(sh),
    device_(other.device()) {
    shape_.compute_strides();
}

template<class _Tp>
tensor<_Tp>::tensor(const tensor& t) :
    data_(t.storage()),
    shape_(t.shape()),
    device_(t.device()) {}

template<class _Tp>
tensor<_Tp>::tensor(tensor&& t) noexcept :
    data_(std::move(t.storage())),
    shape_(std::move(t.shape())),
    device_(std::move(t.device())) {}