#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::relu() const {
    return clamp_min(value_type(0));
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::relu_() {
    return clamp_min_(value_type(0));
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::clipped_relu_(const value_type clip_limit) {
    if (internal::types::using_neon())
    {
        return internal::neon::clipped_relu_(clip_limit);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    clamp_(value_type(0), std::numeric_limits<value_type>::max());
    clamp_(std::numeric_limits<value_type>::lowest(), clip_limit);

    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::clipped_relu(const value_type clip_limit) const {
    self ret = clone();
    ret.clipped_relu_(clip_limit);
    return ret;
}