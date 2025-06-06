#pragma once

#include "tensorbase.hpp"

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::logical_not_() {
    bitwise_not_();
    bool_();
    return *this;
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_not() const {
    tensor<bool> ret = bool_();
    ret.logical_not_();
    return ret;
}
