#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::transpose() const {
    if (shape_.size() != 2)
        throw error::shape_error("Matrix transposition can only be done on 2D tensors");

    tensor           ret({shape_[1], shape_[0]});
    const index_type rows = shape_[0];
    const index_type cols = shape_[1];

#ifdef CUDACC
    if (is_cuda_tensor)
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
        transpose_kernel<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(data_.data()),
                                                thrust::raw_pointer_cast(ret.data_.data()), rows, cols);
        cudaDeviceSynchronize();
        return ret;
    }
#endif

    index_type i = 0;

    for (; i < rows; ++i)
    {
        index_type j = 0;

        for (; j < cols; ++j)
            ret.at({j, i}) = at({i, j});
    }

    return ret;
}

#ifdef CUDACC
template<class _Tp>
global void transpose_kernel(_Tp* input, _Tp* output, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols)
        output[j * rows + i] = input[i * cols + j];
}
#endif

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::transpose_() {
    if (shape_.size() != 2)
        throw error::shape_error("Transpose operation is only valid for 2D tensors");

    const index_type rows = shape_[0];
    const index_type cols = shape_[1];

    if (rows != cols)
        throw error::shape_error("In-place transpose is only supported for square tensors");

    for (index_type i = 0; i < rows; ++i)
        for (index_type j = i + 1; j < cols; ++j)
            std::swap(data_[i * cols + j], data_[j * cols + i]);

    return *this;
}