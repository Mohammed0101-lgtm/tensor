#pragma once

#include "stride.hpp"
#include <vector>


namespace shape {

struct Shape
{
   public:
    using index      = uint64_t;
    using shape_type = std::vector<index>;

    shape_type value_;
    Strides    strides_;

    Shape() {
        value_   = shape_type();
        strides_ = Strides();
    }

    Shape(const shape_type sh) noexcept :
        value_(sh),
        strides_(sh) {}

    Shape(std::initializer_list<index> list) noexcept :
        value_(list),
        strides_(value_) {}

    explicit Shape(const std::size_t size) noexcept :
        value_(size) {}


    index size(const index dim) const {
        if (dim >= value_.size())
        {
            throw error::shape_error("Dimension out of range");
        }

        if (dim == 0)
        {
            return compute_size();
        }

        return value_[dim];
    }

    shape_type get() const { return value_; }

    Strides strides() const { return strides_; }

    index size() const { return value_.size(); }

    index flatten_size() const { return compute_size(); }

    bool operator==(const Shape& other) const {
        return this->value_ == other.value_ && this->strides_ == other.strides_;
    }

    index operator[](const index at) const { return value_[at]; }

    index& operator[](const index at) { return value_[at]; }

    bool empty() const { return value_.empty(); }

    bool equal(const Shape& other) const {
        std::size_t size_x = size();
        std::size_t size_y = other.size();

        if (size_x == size_y)
        {
            return value_ == other.value_;
        }

        if (size_x < size_y)
        {
            return other.equal(*this);
        }

        int diff = size_x - size_y;

        for (std::size_t i = 0; i < size_y; ++i)
        {
            if (value_[i + diff] != other.value_[i] && value_[i + diff] != 1 && other.value_[i] != 1)
            {
                return false;
            }
        }

        return true;
    }

    void compute_strides() noexcept { strides_.compute_strides(value_); }

    index compute_index(const shape_type& idx) const {
        if (idx.size() != value_.size())
        {
            throw error::index_error("compute_index : input indices does not match the tensor shape");
        }

        index at = 0;
        index i  = 0;

        for (; i < value_.size(); ++i)
        {
            at += idx[i] * strides_[i];
        }

        return at;
    }


    inline std::size_t computeStride(std::size_t dimension, const shape::Shape& shape) const noexcept {
        std::size_t stride = 1;

        for (const auto& elem : shape.value_)
        {
            stride *= elem;
        }

        return stride;
    }

    // implicit conversion to std::vector<uint64_t> needed

   private:
    int compute_size() const {
        int size = 1;

        for (const auto& dim : value_)
        {
            size *= dim;
        }

        return size;
    }
};

}  // namespace tensor