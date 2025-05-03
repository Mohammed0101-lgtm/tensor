#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::unsqueeze(index_type dimension) const {
    if (dimension < 0 || dimension > static_cast<index_type>(shape_.size()))
    {
        throw index_error("Dimension out of range in unsqueeze");
    }

    shape_type s = shape_;
    s.insert(s.begin() + dimension, 1);

    tensor ret;
    ret.shape_ = s;
    ret.data_  = data_;

    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::unsqueeze_(index_type dimension) {
    if (dimension < 0 || dimension > static_cast<index_type>(shape_.size()))
    {
        throw index_error("Dimension out of range in unsqueeze");
    }

    shape_.insert(shape_.begin() + dimension, 1);

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::unsqueeze_(index_type dimension) const {
    if (dimension < 0 || dimension > static_cast<index_type>(shape_.size()))
    {
        throw index_error("Dimension out of range in unsqueeze");
    }

    shape_.insert(shape_.begin() + dimension, 1);

    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::squeeze(index_type dimension) const {
    self ret = clone();
    ret.squeeze_(dimension);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::squeeze_(index_type dimension) {
    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::squeeze_(index_type dimension) const {
    return squeeze_(dimension);
}
