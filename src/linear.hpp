#pragma once

#include "tensorbase.hpp"
#include "types.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::reshape(const shape_type __sh) const {
  data_t     __d = this->__data_;
  index_type __s = this->__computeSize(__sh);

  if (__s != this->__data_.size())
    throw __shape_error__(
        "input shape must have size of elements equal to the current number of elements in the "
        "tensor data");

  return __self(__d, __sh);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::absolute(const tensor& __tensor) const {
  index_type __s = __tensor.storage().size();
  data_t     __a;
  __a.reserve(__s);
  index_type __i = 0;
  for (; __i < __s; ++__i)
    __a.push_back(static_cast<value_type>(std::fabs(_f32(__tensor.storage()[__i]))));

  return __self(__a, __tensor.__shape_);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::pop_back() const {
  if (__equal_shape(this->__shape_, shape_type(1, this->__shape_[0])))
    throw __index_error__("push_back is only supported for one dimensional tensors");

  this->__data_.pop_back();
  --(this->__shape_[0]);
  this->__compute_strides();
  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::cat(const std::vector<tensor<_Tp>>& __others, index_type __dim) const {
  for (const tensor& __t : __others) {
    index_type __i = 0;
    for (; __i < this->__shape_.size(); ++__i)
      if (__i != __dim && this->__shape_[__i] != __t.__shape_[__i])
        throw __shape_error__(
            "Cannot concatenate tensors with different shapes along non-concatenation "
            "dimensions");
  }

  shape_type __ret_sh = this->__shape_;
  for (const tensor& __t : __others) __ret_sh[__dim] += __t.__shape_[__dim];

  data_t __c;
  __c.reserve(this->__data_.size());
  __c.insert(__c.end(), this->__data_.begin(), this->__data_.end());
  for (const tensor& __t : __others) __c.insert(__c.end(), __t.__data_.begin(), __t.__data_.end());

  return __self(__ret_sh, __c);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::log_softmax_(const index_type __dim) {
  assert(__dim < this->__shape_.size() && "Dimension out of range for log_softmax");
  tensor __max_values  = this->argmax_(__dim);
  tensor __shifted     = *this - __max_values.expand_as(this->__shape_, __dim);
  tensor __exp_values  = __shifted.exp();
  tensor __sum_exp     = __exp_values.sum(__dim);
  tensor __log_sum_exp = __sum_exp.log();
  *this                = __shifted - __log_sum_exp.expand_as(this->__shape_, __dim);
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log_softmax_(const index_type __dim) const {
  return this->log_softmax_(__dim);
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::resize_as_(const shape_type __sh) {
  // TODO: implement in place resize as here
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::resize_as_(const shape_type __sh) const {
  return this->resize_as_(__sh);
}
