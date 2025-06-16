#pragma once

#include "tensorbase.hpp"


template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::logical_not_() {
    if (empty())
    {
        return *this;
    }

    bitwise_not_();
    bool_();
    return *this;
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_not() const {
    if (empty())
    {
        return tensor<bool>({0});
    }

    tensor<bool> ret = bool_();
    ret.logical_not_();
    return ret;
}
