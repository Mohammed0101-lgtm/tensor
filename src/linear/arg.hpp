#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argmax_(index_type dimension) const {
    if (dimension < 0 || dimension >= shape_.size())
    {
        throw error::index_error("Dimension out of range in argmax");
    }

    shape::Shape       ret_sh = shape_;
    ret_sh.value_.erase(ret_sh.value_.begin() + dimension);
    tensor<index_type> ret(ret_sh);
    ret.storage_().resize(ret_sh.flatten_size(), 0);

    index_type outer_size = 1;
    index_type inner_size = 1;
    index_type i          = 0;

    for (; i < dimension; ++i)
    {
        outer_size *= shape_[i];
    }

    for (i = dimension + 1; i < shape_.size(); ++i)
    {
        inner_size *= shape_[i];
    }

    for (i = 0; i < outer_size; ++i)
    {
        index_type j = 0;

        for (; j < inner_size; ++j)
        {
            index_type max_index = 0;
            value_type max_value = data_[i * shape_[dimension] * inner_size + j];
            index_type k         = 1;

            for (; k < shape_[dimension]; ++k)
            {
                value_type v = data_[(i * shape_[dimension] + k) * inner_size + j];

                if (v > max_value)
                {
                    max_value = v;
                    max_index = k;
                }
            }

            ret[i * inner_size + j] = max_index;
        }
    }

    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::argmax(index_type dimension) const {
    if (dimension < 0 || dimension >= shape_.size())
    {
        throw error::index_error("Dimension out of range in argmax");
    }

    tensor       ret;
    shape::Shape ret_sh = shape_;

    ret_sh.value_.erase(ret_sh.value_.begin() + dimension);
    ret.shape_ = ret_sh;
    ret.data_.resize(ret_sh.flatten_size(), value_type(0));

    index_type outer_size = 1;
    index_type inner_size = 1;
    index_type i          = 0;

    for (; i < dimension; ++i)
    {
        outer_size *= shape_[i];
    }

    for (i = dimension + 1; i < static_cast<index_type>(shape_.size()); ++i)
    {
        inner_size *= shape_[i];
    }

    for (i = 0; i < outer_size; ++i)
    {
        index_type j = 0;

        for (; j < inner_size; ++j)
        {
            value_type max_value = data_[i * shape_[dimension] * inner_size + j];
            index_type k         = 1;

            for (; k < shape_[dimension]; ++k)
            {
                value_type v = data_[(i * shape_[dimension] + k) * inner_size + j];

                if (v > max_value)
                {
                    max_value = v;
                }
            }

            ret.data_[i * inner_size + j] = max_value;
        }
    }

    return ret;
}

template<class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argsort(index_type d, bool ascending) const {
    index_type adjusted = (d < 0) ? d + data_.size() : d;

    if (adjusted != 0)
    {
        throw error::index_error("Invalid dimension for argsort: only 1D tensors are supported");
    }

    index_type size = static_cast<index_type>(data_.size());
    shape_type indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](index_type a, index_type b) { return ascending ? data_[a] < data_[b] : data_[a] > data_[b]; });

    return tensor<index_type>(indices);
}