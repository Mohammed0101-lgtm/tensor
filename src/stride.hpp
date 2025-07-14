#pragma once

#include "error.hpp"
#include <vector>


namespace shape {

struct Strides
{
  using index        = uint64_t;
  using strides_type = std::vector<index>;

  strides_type value_;

  Strides() { value_ = strides_type(); }

  Strides(const std::vector<index>& shape) {
    if (shape.empty())
    {
      throw error::shape_error("Shape must be initialized before computing strides");
    }

    value_     = std::vector<index>(shape.size(), 1);
    int stride = 1;

    for (int i = static_cast<int>(shape.size() - 1); i >= 0; i--)
    {
      value_[i] = stride;
      stride *= shape[i];
    }
  }

  Strides(const Strides& other) noexcept :
      value_(other.value_) {}

  Strides& operator=(const Strides& other) noexcept {
    if (this != &other)
    {
      value_ = other.value_;
    }

    return *this;
  }

  bool operator==(const Strides& other) const { return this->value_ == other.value_; }

  index operator[](const index at) const { return value_[at]; }

  index& operator[](const index at) { return value_[at]; }

  strides_type get() const { return value_; }

  void compute_strides(const std::vector<index>& shape_) noexcept {
    if (shape_.empty())
    {
      value_ = strides_type();
      return;
    }

    value_ = strides_type(shape_.size(), 1);
    int st = 1, i = static_cast<int>(shape_.size() - 1);

    for (; i >= 0; i--)
    {
      value_[i] = st;
      st *= shape_[i];
    }
  }
};

}  // namespace shape