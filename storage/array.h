#include "allocator.h"
#include "macros.hpp"


namespace storage {

template<typename T, typename Alloc = storage::alloc::arena_allocator>
struct array
{
  using size_type      = std::size_t;
  using value_type     = T;
  using pointer        = value_type*;
  using const_pointer  = const pointer;
  using allocator_type = Alloc;

  size_type      __size_;
  pointer        __begin_;
  pointer        __end_;
  allocator_type __allocator_;

  TENSOR_LIBRARY_API array() :
      __size_(0),
      __begin_(nullptr),
      __end_(nullptr) {}

  TENSOR_LIBRARY_API array(const size_type _s, const value_type _v = static_cast<value_type>(0)) :
      __size_(_s) {
    __begin_ = reinterpret_cast<pointer>(__allocator_.allocate(_s));
    __end_   = __begin_ + _s;
    fill(_v);
  }

  TENSOR_LIBRARY_API array(const array& other) :
      __size_(other.__size_) {
    __begin_ = reinterpret_cast<pointer>(__allocator_.allocate(__size_));
    std::uninitialized_copy(other.__begin_, other.__end_, __begin_);
    __end_ = __begin_ + __size_;
  }

  TENSOR_LIBRARY_API array& operator=(const array& other) {
    if (this != &other)
    {
      if (this->__begin_)
      {
        this->__allocator_.deallocate(this->__begin_, this->__size_);
      }

      this->__size_  = other.__size_;
      this->__begin_ = reinterpret_cast<pointer>(this->__allocator_.allocate(this->__size_));
      std::uninitialized_copy(other.__begin_, other.__end_, this->__begin_);
      this->__end_ = this->__begin_ + this->__size_;
    }
    return *this;
  }

  TENSOR_LIBRARY_API array(array&& other) TENSOR_NOEXCEPT: __size_(other.__size_),
                                                           __begin_(other.__begin_),
                                                           __end_(other.__end_) {
    other.__begin_ = nullptr;
    other.__end_   = nullptr;
    other.__size_  = 0;
  }

  TENSOR_LIBRARY_API array& operator=(array&& other) TENSOR_NOEXCEPT {
    if (this != &other)
    {
      if (this->__begin_)
      {
        this->__allocator_.deallocate(this->__begin_, this->__size_);
      }

      this->__size_  = other.__size_;
      this->__begin_ = other.__begin_;
      this->__end_   = other.__end_;
      other.__begin_ = nullptr;
      other.__end_   = nullptr;
      other.__size_  = 0;
    }
    return *this;
  }

  TENSOR_LIBRARY_API ~array() {
    if constexpr (!std::is_trivially_destructible_v<T>)
    {
      for (pointer p = this->__begin_; p != this->__end_; ++p)
      {
        p->~T();
      }
    }
  }

  TENSOR_LIBRARY_API T&       operator[](size_type i) { return this->__begin_[i]; }
  TENSOR_LIBRARY_API const T& operator[](size_type i) const { return this->__begin_[i]; }

  TENSOR_LIBRARY_API T& at(size_type i) {
    if (i >= this->__size_)
    {
      throw std::out_of_range("index out of range");
    }

    return this->__begin_[i];
  }

  TENSOR_LIBRARY_API T& front() { return *this->__begin_; }
  TENSOR_LIBRARY_API T& back() { return *(this->__end_ - 1); }

  TENSOR_LIBRARY_API pointer begin() { return this->__begin_; }
  TENSOR_LIBRARY_API pointer end() { return this->__end_; }

  TENSOR_LIBRARY_API const_pointer begin() const { return this->__begin_; }
  TENSOR_LIBRARY_API const_pointer end() const { return this->__end_; }

  TENSOR_LIBRARY_API pointer data() const { return this->__begin_; }

  TENSOR_LIBRARY_API size_type size() const { return this->__size_; }
  TENSOR_LIBRARY_API bool      empty() const { return this->__size_ == 0; }

  TENSOR_LIBRARY_API void fill(const T& value) { std::fill(this->__begin_, this->__end_, value); }

  TENSOR_LIBRARY_API void clear() {
    if (this->__begin_)
    {
      this->__allocator_.deallocate(this->__begin_, this->__size_);
      this->__begin_ = this->__end_ = nullptr;
      this->__size_                 = 0;
    }
  }

  TENSOR_LIBRARY_API void resize(const size_type _s) {
    if (_s == this->__size_)
    {
      return;
    }

    pointer new_begin = reinterpret_cast<pointer>(this->__allocator_.allocate(_s));
    pointer new_end   = new_begin + _s;

    if constexpr (std::is_trivially_copyable_v<T>)
    {
      size_type copy_count = std::min(this->__size_, _s);
      std::memcpy(new_begin, this->__begin_, copy_count * sizeof(T));
    }
    else
    {
      size_type copy_count = std::min(this->__size_, _s);
      std::uninitialized_copy_n(this->__begin_, copy_count, new_begin);

      if (_s > this->__size_)
      {
        std::uninitialized_value_construct_n(new_begin + this->__size_, _s - this->__size_);
      }
    }

    if constexpr (!std::is_trivially_destructible_v<T>)
    {
      if (_s < this->__size_)
      {
        for (pointer p = this->__begin_ + _s; p != this->__end_; ++p)
        {
          p->~T();
        }
      }
    }

    if (this->__begin_)
    {
      this->__allocator_.deallocate(this->__begin_, this->__size_);
    }

    this->__begin_ = new_begin;
    this->__end_   = new_end;
    this->__size_  = _s;
  }

  TENSOR_LIBRARY_API bool operator==(const array& other) const {
    return this->__size_ == other.__size_ && std::equal(this->__begin_, this->__end_, other.__begin_);
  }

  TENSOR_LIBRARY_API bool operator!=(const array& other) const { return !(*this == other); }

  TENSOR_LIBRARY_API void swap(array& other) TENSOR_NOEXCEPT {
    std::swap(this->__size_, other.__size_);
    std::swap(this->__begin_, other.__begin_);
    std::swap(this->__end_, other.__end_);
  }

 private:
  // TODO: remove later after testing
  void print() const {
    for (size_type i = 0; i < this->__size_; ++i)
    {
      std::cout << this->__begin_[i] << ' ';
    }

    std::cout << '\n';
  }
};

}