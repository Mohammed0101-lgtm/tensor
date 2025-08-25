#pragma once

#include "stride.hpp"
#include <vector>


namespace shape {

struct Shape
{
 public:
  using index      = uint64_t;
  using shape_type = std::vector<index>;

  shape_type __value_;
  Strides    __strides_;

  Shape() {
    __value_   = shape_type();
    __strides_ = Strides();
  }

  Shape(const shape_type sh) noexcept :
      __value_(sh),
      __strides_(sh) {}

  Shape(std::initializer_list<index> list) noexcept :
      __value_(list),
      __strides_(__value_) {}

  explicit Shape(const std::size_t size) noexcept :
      __value_(size) {}


  index size(const index dim) const {
    if (dim >= __value_.size())
    {
      throw error::shape_error("Dimension out of range");
    }

    if (dim == 0)
    {
      return compute_size();
    }

    return __value_[dim];
  }

  shape_type get() const { return __value_; }

  Strides strides() const { return __strides_; }

  index size() const { return __value_.size(); }

  index flatten_size() const { return compute_size(); }

  bool operator==(const Shape& other) const {
    return this->__value_ == other.__value_ && this->__strides_ == other.__strides_;
  }

  index operator[](const index at) const { return __value_[at]; }

  index& operator[](const index at) { return __value_[at]; }

  bool empty() const { return __value_.empty(); }

  bool equal(const Shape& other) const {
    std::size_t size_x = size();
    std::size_t size_y = other.size();

    if (size_x == size_y)
    {
      return __value_ == other.__value_;
    }

    if (size_x < size_y)
    {
      return other.equal(*this);
    }

    int diff = size_x - size_y;

    for (std::size_t i = 0; i < size_y; ++i)
    {
      if (__value_[i + diff] != other.__value_[i] && __value_[i + diff] != 1 && other.__value_[i] != 1)
      {
        return false;
      }
    }

    return true;
  }

  void compute_strides() noexcept { __strides_.compute_strides(__value_); }

  index compute_index(const shape_type& idx) const {
    if (idx.size() != __value_.size())
    {
      throw error::index_error("compute_index : input indices does not match the tensor shape");
    }

    index at = 0;
    index i  = 0;

    for (; i < __value_.size(); ++i)
    {
      at += idx[i] * __strides_[i];
    }

    return at;
  }


  inline std::size_t computeStride(std::size_t dimension, const shape::Shape& shape) const noexcept {
    std::size_t stride = 1;

    for (const auto& elem : shape.__value_)
    {
      stride *= elem;
    }

    return stride;
  }

  // implicit conversion to std::vector<uint64_t> needed

 private:
  int compute_size() const {
    int size = 1;

    for (const auto& dim : __value_)
    {
      size *= dim;
    }

    return size;
  }
};

}  // namespace tensor