#pragma once

#include "tensor.hpp"


template<class _Tp>
tensor<_Tp>::tensor(const shape::Shape& sh, const_reference v, Device d) :
    TensorBase<_Tp>(sh, v, d) {}

template<class _Tp>
tensor<_Tp>::tensor(const shape::Shape& sh, Device d) :
    TensorBase<_Tp>(sh, d) {}

template<class _Tp>
tensor<_Tp>::tensor(const shape::Shape& sh, const container_type& d, Device dev) :
    TensorBase<_Tp>(sh, d, dev) {}

template<class _Tp>
tensor<_Tp>::tensor(const shape::Shape& sh, std::initializer_list<value_type> init_list, Device d) :
    TensorBase<_Tp>(sh, init_list, d) {}

template<class _Tp>
tensor<_Tp>::tensor(const shape::Shape& sh, const tensor& other) :
    TensorBase<_Tp>(sh, other) {}

template<class _Tp>
tensor<_Tp>::tensor(const tensor& t) :
    TensorBase<_Tp>(t) {}

template<class _Tp>
tensor<_Tp>::tensor(tensor&& t) noexcept :
    TensorBase<_Tp>(std::move(t)) {}