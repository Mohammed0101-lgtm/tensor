#pragma once

#include "macros.hpp"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <memory>


namespace storage::alloc {

using byte          = uint8_t;
using pointer       = byte*;
using const_pointer = const pointer;

const std::size_t DEFAULT_BLOCK_SIZE = 1024 * 1024 * sizeof(byte);

struct Block
{
  using size_type = std::size_t;

 public:
  size_type __size_;
  size_type __remaining_;
  pointer   __begin_;
  Block*    __next_;

  Block() = delete;

  pointer end() const { return __begin_ + __size_; }
  pointer next_free() const { return __begin_ + (__size_ - __remaining_); }
  bool    filled() const { return __remaining_ == static_cast<size_type>(0); }
};

struct TENSOR_LIBRARY_API arena_allocator
{
 private:
  using size_type = std::size_t;

  mutable Block* __list_          = nullptr;
  mutable Block* __current_block_ = nullptr;
  mutable Block* __free_list_     = nullptr;
  size_type      __block_size_    = DEFAULT_BLOCK_SIZE;

 public:
  explicit arena_allocator(const size_type _s = DEFAULT_BLOCK_SIZE) :
      __block_size_(_s) {
    allocate_block();
    __current_block_ = __list_;
    __free_list_     = __list_;
  }

  ~arena_allocator() {
    free_all();
    free_freelist();
  }

  void* allocate(size_type _s, size_type _alignment = alignof(std::max_align_t)) {
    uintptr_t current = reinterpret_cast<uintptr_t>(__current_block_->next_free());
    uintptr_t aligned = (current + _alignment - 1) & ~(_alignment - 1);
    size_type padding = aligned - current;

    if (padding + _s > __current_block_->__remaining_)
    {
      allocate_block(std::max(__block_size_, _s + _alignment));
      return allocate(_s, _alignment);
    }

    __current_block_->__remaining_ -= padding + _s;
    void* ret = reinterpret_cast<void*>(aligned);
    return ret;
  }

  void reset() {
    Block* block = __list_;
    while (block)
    {
      block->__remaining_ = block->__size_;

      Block* next = block->__next_;

      block->__next_ = __free_list_;
      __free_list_   = block;
      block          = next;
    }

    __list_          = nullptr;
    __current_block_ = nullptr;

    if (__free_list_)
    {
      __list_          = __free_list_;
      __free_list_     = __free_list_->__next_;
      __list_->__next_ = nullptr;
      __current_block_ = __list_;
    }
    else
    {
      allocate_block();
    }
  }

  void deallocate(std::uintptr_t _begin, std::uintptr_t _size) {}

 private:
  Block* allocate_block(size_type _s = 0) const {
    size_type alloc_size = (_s ? _s : __block_size_) + sizeof(Block);
    pointer   raw        = static_cast<pointer>(std::malloc(alloc_size));
    assert(raw && "Memory Allocation (std::malloc) failed");

    Block* block = reinterpret_cast<Block*>(raw);

    block->__size_      = alloc_size - sizeof(Block);
    block->__begin_     = raw + sizeof(Block);
    block->__remaining_ = block->__size_;
    block->__next_      = nullptr;

    if (!__list_)
    {
      __list_ = block;
    }
    else
    {
      Block* last = __list_;

      while (last->__next_)
      {
        last = last->__next_;
      }

      last->__next_ = block;
    }

    return block;
  }

  void free_all() {
    Block* ptr = __list_;

    while (ptr)
    {
      Block* next = ptr->__next_;
      std::free(ptr);
      ptr = next;
    }

    __list_          = nullptr;
    __current_block_ = nullptr;
  }

  void free_freelist() {
    while (__free_list_)
    {
      Block* next = __free_list_->__next_;
      std::free(__free_list_);
      __free_list_ = next;
    }
  }

  arena_allocator(const arena_allocator&)            = delete;
  arena_allocator& operator=(const arena_allocator&) = delete;
};
};
