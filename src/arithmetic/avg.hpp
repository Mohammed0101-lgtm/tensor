#pragma once

#include "tensorbase.hpp"

template<class _Tp>
double tensor<_Tp>::mode(const index_type dimension) const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    if (dimension >= n_dims() || dimension < -1)
    {
        throw error::index_error("given dimension is out of range of the tensor dimensions");
    }

    index_type stride = (dimension == -1) ? 0 : strides_[dimension];
    index_type end    = (dimension == -1) ? data_.size() : strides_[dimension];

    if (data_.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::unordered_map<value_type, std::size_t> counts;

    for (index_type i = stride; i < end; ++i)
    {
        ++counts[data_[i]];
    }

    value_type  ret  = 0;
    std::size_t most = 0;

    for (const auto& pair : counts)
    {
        if (pair.second > most)
        {
            ret  = pair.first;
            most = pair.second;
        }
    }

    return static_cast<double>(ret);
}

template<class _Tp>
double tensor<_Tp>::median(const index_type dimension) const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    if (dimension >= n_dims() || dimension < -1)
    {
        throw error::index_error("given dimension is out of range of the tensor dimensions");
    }

    index_type stride = (dimension == -1) ? 0 : strides_[dimension];
    index_type end    = (dimension == -1) ? data_.size() : strides_[dimension];

    data_t d(data_.begin() + stride, data_.begin() + end);

    if (d.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());

    if (d.size() % 2 == 0)
    {
        std::nth_element(d.begin(), d.begin() + d.size() / 2 - 1, d.end());
        return (static_cast<double>(d[d.size() / 2]) + d[d.size() / 2 - 1]) / 2.0;
    }

    return d[d.size() / 2];
}

template<class _Tp>
double tensor<_Tp>::mean() const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    double m = 0.0;

    if (empty())
    {
        return m;
    }

    for (const auto& elem : data_)
    {
        m += elem;
    }

    return static_cast<double>(m) / static_cast<double>(data_.size());
}
