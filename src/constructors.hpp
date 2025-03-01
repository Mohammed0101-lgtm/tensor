#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>::tensor(const shape_type& __sh, const_reference __v, Device __d)
    : __shape_(__sh), __data_(this->__computeSize(__sh), __v), __device_(__d) {
  this->__compute_strides();
}

template <class _Tp>
tensor<_Tp>::tensor(const shape_type& __sh, Device __d) : __shape_(__sh), __device_(__d) {
  index_type __s = this->__computeSize(__sh);
  this->__data_  = data_t(__s);
  this->__compute_strides();
}

template <class _Tp>
tensor<_Tp>::tensor(const shape_type& __sh, const data_t& __d, Device __dev)
    : __shape_(__sh), __device_(__dev) {
  index_type __s = this->__computeSize(__sh);
  assert(__d.size() == static_cast<size_t>(__s) &&
         "Initial data vector must match the tensor size");
  this->__data_ = __d;
  this->__compute_strides();
}

template <class _Tp>
tensor<_Tp>::tensor(const shape_type& __sh, std::initializer_list<value_type> init_list, Device __d)
    : __shape_(__sh), __device_(__d) {
  index_type __s = this->__computeSize(__sh);
  assert(init_list.size() == static_cast<size_t>(__s) &&
         "Initializer list size must match tensor size");
  this->__data_ = data_t(init_list);
  this->__compute_strides();
}

template <class _Tp>
tensor<_Tp>::tensor(const shape_type& __sh, const tensor& __other)
    : __data_(__other.storage()), __shape_(__sh), __device_(__other.device()) {
  this->__compute_strides();
}

template <class _Tp>
tensor<_Tp>::tensor(const tensor& __t)
    : __data_(__t.storage()),
      __shape_(__t.shape()),
      __strides_(__t.strides()),
      __device_(__t.device()) {}

template <class _Tp>
tensor<_Tp>::tensor(tensor&& __t) noexcept
    : __data_(std::move(__t.storage())),
      __shape_(std::move(__t.shape())),
      __strides_(std::move(__t.strides())),
      __device_(std::move(__t.device())) {}