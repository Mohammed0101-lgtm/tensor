#pragma once

#include "tensorbase.hpp"
/*
template<class _Tp>
tensor<_Tp>
tensor<_Tp>::slice(index_type dimension, std::optional<index_type> start, std::optional<index_type> end, int64_t step) const {
    if (empty())
    {
        return self();
    }

    if (dimension < 0 or dimension >= static_cast<index_type>(shape_.size()))
    {
        throw error::index_error("Invalid dimension provided");
    }

    if (step == 0)
    {
        throw std::invalid_argument("Step cannot be zero");
    }

    index_type s         = shape_[dimension];
    index_type start_val = start.value_or(step > 0 ? 0 : s - 1);
    index_type end_val   = end.value_or(step > 0 ? s : -1);

    if (start_val < 0)
    {
        start_val += s;
    }

    if (end_val < 0)
    {
        end_val += s;
    }

    start_val = std::clamp(start_val, index_type(0), s);
    end_val   = std::clamp(end_val, index_type(0), s);

    if ((step > 0 and start_val >= end_val) or (step < 0 and start_val <= end_val))
    {
        return self();
    }

    shape_type ret_sh     = shape_;
    index_type slice_size = 0;
    data_t     ret_data;

    for (index_type idx = start_val; (step > 0 ? idx < end_val : idx > end_val); idx += step)
    {
        ret_data.push_back(data_[idx]);
        slice_size++;
    }

    ret_sh[dimension] = slice_size;

    return self(ret_sh, ret_data);
}
*/