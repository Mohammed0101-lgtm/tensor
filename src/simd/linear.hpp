#pragma once

#include "tensorbase.hpp"
#include "types.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_absolute(const tensor& __tensor) const {
  index_type __s = __tensor.storage().size();
  data_t     __a;
  __a.reserve(__s);
  index_type __i = 0;

  // TODO : implement neon acceleration for absolute
#pragma omp parallel
  for (; __i < __s; ++__i)
    __a.push_back(static_cast<value_type>(std::fabs(_f32(__tensor.storage()[__i]))));

  return __self(__a, __tensor.__shape_);
}