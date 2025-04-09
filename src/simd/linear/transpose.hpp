#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_transpose() const {
    if (!equal_shape(shape_, shape_type({shape_[0], shape_[1]})))
        throw shape_error("Matrix transposition can only be done on 2D tensors");

    tensor           ret({shape_[1], shape_[0]});
    const index_type rows = shape_[0];
    const index_type cols = shape_[1];

    if constexpr (std::is_same_v<_Tp, _f32>) {
        for (index_type i = 0; i < rows; i += _ARM64_REG_WIDTH) {
            for (index_type j = 0; j < cols; j += _ARM64_REG_WIDTH) {
                if (i + _ARM64_REG_WIDTH <= rows and j + _ARM64_REG_WIDTH <= cols) {
                    float32x4x4_t input;

                    for (index_type k = 0; k < _ARM64_REG_WIDTH; ++k) {
                        input.val[k] =
                            vld1q_f32(reinterpret_cast<const _f32*>(&data_[(i + k) * cols + j]));
                    }

                    float32x4x4_t output = vld4q_f32(reinterpret_cast<const _f32*>(&input));

                    for (index_type k = 0; k < _ARM64_REG_WIDTH; ++k) {
                        vst1q_f32(&ret.data_[(j + k) * rows + i], output.val[k]);
                    }
                } else {
                    for (index_type ii = i;
                         ii < std::min(static_cast<index_type>(i + _ARM64_REG_WIDTH), rows); ++ii) {
                        for (index_type jj = j;
                             jj < std::min(static_cast<index_type>(j + _ARM64_REG_WIDTH), cols);
                             ++jj) {
                            ret.at({jj, ii}) = at({ii, jj});
                        }
                    }
                }
            }
        }
    } else if constexpr (std::is_signed_v<value_type>) {
        for (index_type i = 0; i < rows; i += _ARM64_REG_WIDTH) {
            for (index_type j = 0; j < cols; j += _ARM64_REG_WIDTH) {
                if (i + _ARM64_REG_WIDTH <= rows and j + _ARM64_REG_WIDTH <= cols) {
                    int32x4x4_t input;

                    for (index_type k = 0; k < _ARM64_REG_WIDTH; ++k)
                        input.val[k] =
                            vld1q_s32(reinterpret_cast<const _s32*>(&data_[(i + k) * cols + j]));

                    int32x4x4_t output = vld4q_s32(reinterpret_cast<const _s32*>(&input));

                    for (index_type k = 0; k < _ARM64_REG_WIDTH; k++)
                        vst1q_s32(&ret.data_[(j + k) * rows + i], output.val[k]);
                } else {
                    for (index_type ii = i;
                         ii < std::min(static_cast<index_type>(i + _ARM64_REG_WIDTH), rows); ++ii) {
                        for (index_type jj = j;
                             jj < std::min(static_cast<index_type>(j + _ARM64_REG_WIDTH), cols);
                             ++jj) {
                            ret.at({jj, ii}) = at({ii, jj});
                        }
                    }
                }
            }
        }
    } else if constexpr (std::is_unsigned_v<value_type>) {
        for (index_type i = 0; i < rows; i += _ARM64_REG_WIDTH) {
            for (index_type j = 0; j < cols; j += _ARM64_REG_WIDTH) {
                if (i + _ARM64_REG_WIDTH <= rows and j + _ARM64_REG_WIDTH <= cols) {
                    uint32x4x4_t input;

                    for (index_type k = 0; k < _ARM64_REG_WIDTH; ++k)
                        input.val[k] =
                            vld1q_u32(reinterpret_cast<const _u32*>(&data_[(i + k) * cols + j]));

                    uint32x4x4_t output = vld4q_u32(reinterpret_cast<const _u32*>(&input));

                    for (index_type k = 0; k < _ARM64_REG_WIDTH; ++k)
                        vst1q_u32(&ret.data_[(j + k) * rows + i], output.val[k]);
                } else {
                    for (index_type ii = i;
                         ii < std::min(static_cast<index_type>(i + _ARM64_REG_WIDTH), rows); ++ii) {
                        for (index_type jj = j;
                             jj < std::min(static_cast<index_type>(j + _ARM64_REG_WIDTH), cols);
                             ++jj) {
                            ret.at({jj, ii}) = at({ii, jj});
                        }
                    }
                }
            }
        }
    } else {
        index_type i = 0;

        for (; i < rows; ++i) {
            index_type j = 0;
            for (; j < cols; ++j) {
                ret.at({j, i}) = at({i, j});
            }
        }
    }

    return ret;
}
