#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cross_product(const tensor& other) const {
    if (empty() || other.empty())
    {
        throw std::invalid_argument("Cannot cross product an empty vector");
    }

    if (shape() != shape_type{3} || other.shape() != shape_type{3})
    {
        throw std::logic_error("Cross product can only be performed on 3-element vectors");
    }

    tensor ret({3});

    // return neon_cross_product(other);
#if defined(CUDACC)
    pointer d_a;
    pointer d_b;
    pointer d_c;

    cudaMalloc(&d_a, 3 * sizeof(value_type));
    cudaMalloc(&d_b, 3 * sizeof(value_type));
    cudaMalloc(&d_c, 3 * sizeof(value_type));

    cudaMemcpy(d_a, data_.data(), 3 * sizeof(value_type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, other.storage().data(), 3 * sizeof(value_type), cudaMemcpyHostToDevice);

    dim3 block(1);
    dim3 grid(1);
    cross_product_kernel<<<grid, block>>>(d_a, d_b, d_c);

    cudaMemcpy(ret.storage().data(), d_c, 3 * sizeof(value_type), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

#else
    const_reference a1 = data_[0];
    const_reference a2 = data_[1];
    const_reference a3 = data_[2];

    const_reference b1 = other[0];
    const_reference b2 = other[1];
    const_reference b3 = other[2];

    ret[0] = a2 * b3 - a3 * b2;
    ret[1] = a3 * b1 - a1 * b3;
    ret[2] = a1 * b2 - a2 * b1;
#endif

    return ret;
}

#ifdef CUDACC
template<class _Tp>
global void cross_product_kernel(_Tp* a, _Tp* b, _Tp* c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}
#endif

template<class _Tp>
tensor<_Tp> tensor<_Tp>::dot(const tensor& other) const {
    if (empty() || other.empty())
    {
        throw std::invalid_argument("Cannot dot product an empty vector");
    }

    if (shape_.size() == 1 && other.shape().size() == 1)
    {
        if (shape_[0] != other.shape()[0])
        {
            throw shape_error("Vectors must have the same size for dot product");
        }

        const_pointer     this_data     = data_.data();
        data_t            other_storage = other.storage();
        const_pointer     other_data    = other_storage.data();
        const std::size_t size          = data_.size();
        value_type        ret           = std::inner_product(this_data, this_data + size, other_data, value_type(0));

        return self({1}, {ret});
    }

    if (shape_.size() == 2 && other.shape().size() == 2)
    {
        return matmul(other);
    }

    if (shape_.size() == 3 && other.shape().size() == 3)
    {
        return cross_product(other);
    }

    return self();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cumprod(index_type dimension) const {
    if (dimension == -1)
    {
        data_t flat = data_;
        data_t ret(flat.size());
        ret[0] = flat[0];

        index_type i = 1;

        for (; i < flat.size(); ++i)
        {
            ret[i] = ret[i - 1] * flat[i];
        }

        return self(ret, {flat.size()});
    }
    else
    {
        if (dimension < 0 || dimension >= static_cast<index_type>(shape_.size()))
        {
            throw index_error("Invalid dimension provided for cumprod.");
        }

        data_t ret(data_);
        // TODO : compute_outer_size() implementation
        index_type outer_size = compute_outer_size(dimension);
        index_type inner_size = shape_[dimension];
        index_type st         = strides_[dimension];

        index_type i = 0;

        for (; i < outer_size; ++i)
        {
            index_type base = i * st;
            ret[base]       = data_[base];
            index_type j    = 1;

            for (; j < inner_size; ++j)
            {
                index_type curr = base + j;
                ret[curr]       = ret[base + j - 1] * data_[curr];
            }
        }

        return self(ret, shape_);
    }
}
