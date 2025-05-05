#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>::tensor(const shape_type& shape_, const_reference v, Device d) :
    shape_(shape_),
    data_(computeSize(shape_), v),
    device_(d) {
    compute_strides();
}

template<class _Tp>
tensor<_Tp>::tensor(const shape_type& shape_, Device d) :
    shape_(shape_),
    device_(d) {
    index_type s = computeSize(shape_);
    data_        = data_t(s);
    compute_strides();
}

template<class _Tp>
tensor<_Tp>::tensor(const shape_type& shape_, const data_t& d, Device dev) :
    shape_(shape_),
    device_(dev) {
    index_type s = computeSize(shape_);
    
    if (d.size() != static_cast<std::size_t>(s))
        throw std::invalid_argument("Initial data vector must match the tensor size");

    data_ = d;
    compute_strides();
}

template<class _Tp>
tensor<_Tp>::tensor(const shape_type& shape_, std::initializer_list<value_type> init_list, Device d) :
    shape_(shape_),
    device_(d) {
    if (static_cast<index_type>(init_list.size()) != computeSize(shape_))
        throw std::invalid_argument("Initializer list size must match the tensor size");

    data_ = data_t(init_list);
    compute_strides();
}

template<class _Tp>
tensor<_Tp>::tensor(const shape_type& shape_, const tensor& other) :
    data_(other.storage()),
    shape_(shape_),
    device_(other.device()) {
    compute_strides();
}

template<class _Tp>
tensor<_Tp>::tensor(const tensor& t) :
    data_(t.storage()),
    shape_(t.shape()),
    strides_(t.strides()),
    device_(t.device()) {}

template<class _Tp>
tensor<_Tp>::tensor(tensor&& t) noexcept :
    data_(std::move(t.storage())),
    shape_(std::move(t.shape())),
    strides_(std::move(t.strides())),
    device_(std::move(t.device())) {}