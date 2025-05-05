#pragma once

#include "tensorbase.hpp"
#include "types.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::reshape(const shape_type shape_) const {
    data_t     d = data_;
    index_type s = computeSize(shape_);

    if (s != data_.size())
        throw shape_error("input shape must have size of elements equal to the current number of elements in the "
                          "tensor data");

    return self(shape_, d);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::absolute(const tensor& other) const {
    index_type s = other.storage().size();
    data_t     a(s);

    index_type i = 0;
    for (; i < s; ++i)
        a[i] = static_cast<value_type>(std::fabs(static_cast<_f32>(other[i])));

    return self(other.shape(), a);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::pop_back() const {
    if (equal_shape(shape_, shape_type(1, shape_[0])))
        throw index_error("push_back is only supported for one dimensional tensors");

    data_.pop_back();
    --(shape_[0]);
    compute_strides();
    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cat(const std::vector<tensor<_Tp>>& others, index_type dimension) const {
    for (const tensor& t : others)
    {
        index_type i = 0;
        for (; i < shape_.size(); ++i)
            if (i != dimension && shape_[i] != t.shape_[i])
                throw shape_error("Cannot concatenate tensors with different shapes along non-concatenation "
                                  "dimensions");
    }

    shape_type ret_sh = shape_;
    for (const tensor& t : others)
        ret_sh[dimension] += t.shape_[dimension];

    data_t c;
    c.reserve(data_.size());
    c.insert(c.end(), data_.begin(), data_.end());
    for (const tensor& t : others)
        c.insert(c.end(), t.data_.begin(), t.data_.end());

    return self(ret_sh, c);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::log_softmax_(const index_type dimension) {
    if (dimension < shape_.size())
        throw std::invalid_argument("Dimension out of range for log_softmax");
    tensor max_values  = argmax_(dimension);
    tensor shifted     = *this - max_values.expand_as(shape_, dimension);
    tensor exp_values  = shifted.exp();
    tensor sum_exp     = exp_values.sum(dimension);
    tensor log_sum_exp = sum_exp.log();
    *this              = shifted - log_sum_exp.expand_as(shape_, dimension);
    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log_softmax_(const index_type dimension) const {
    return log_softmax_(dimension);
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::resize_as_(const shape_type shape_) {
    // TODO: implement in place resize as here
    shape_ = shape_;
    compute_strides();
    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::resize_as_(const shape_type shape_) const {
    shape_ = shape_;
    compute_strides();
    return *this;
}
