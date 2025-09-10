#pragma once

#include "error.hpp"
#include <vector>


namespace shape {

struct Strides
{
  using index        = uint64_t;
  using strides_type = std::vector<index>;

  strides_type __value_;

  Strides() { __value_ = strides_type(); }

  Strides(const std::vector<index>& shape) {
    /*
         if (shape.empty())
         {
         throw error::shape_error("Shape must be initialized before computing strides");
         }
         */

    if (shape.empty())
    {
      std::cerr << "[Warning] Empty shape passed to compute strides!" << std::endl;
    }

    __value_   = std::vector<index>(shape.size(), 1);
    int stride = 1;

    for (int i = static_cast<int>(shape.size() - 1); i >= 0; i--)
    {
      __value_[i] = stride;
      stride *= shape[i];
    }
  }

  Strides(const Strides& other) noexcept :
      __value_(other.__value_) {}

  Strides& operator=(const Strides& other) noexcept {
    if (this != &other)
    {
      __value_ = other.__value_;
    }

    return *this;
  }

  bool operator==(const Strides& other) const { return this->__value_ == other.__value_; }

  index operator[](const index at) const { return __value_[at]; }

  index& operator[](const index at) { return __value_[at]; }

  strides_type get() const { return __value_; }

  index n_dims() const noexcept { return __value_.size(); }

  void compute_strides(const std::vector<index>& shape_) noexcept {
    if (shape_.empty())
    {
      __value_ = strides_type();
      return;
    }

    __value_ = strides_type(this->n_dims(), 1);
    int st = 1, i = static_cast<int>(this->n_dims() - 1);

    for (; i >= 0; i--)
    {
      __value_[i] = st;
      st *= shape_[i];
    }
  }
};

}  // namespace shape
