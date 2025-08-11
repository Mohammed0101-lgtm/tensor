#include "macros.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <unordered_map>
#include <vector>


template<typename _Tp>
struct arena_allocator
{
  using value_type = _Tp;
  using pointer    = value_type*;
  using size_type  = std::size_t;

  pointer                                __memory_;
  size_type                              __offset_;
  size_type                              __size_;
  size_type                              __max_bytes_ = std::numeric_limits<size_type>::digits10;
  std::unordered_map<pointer, size_type> __allocations_table_;

  arena_allocator(size_type _s) :
      __memory_(new value_type[_s]),
      __offset_(0),
      __size_(_s) {}

  arena_allocator(const arena_allocator&)            = delete;
  arena_allocator& operator=(const arena_allocator&) = delete;

  pointer allocate(size_type _bytes, size_type _alignment = alignof(std::max_align_t)) {
    size_type current = __memory_ + __offset_;
    size_type aligned = align_up(current, _alignment);
    size_type padding = aligned - reinterpret_cast<std::uintptr_t>(__memory_);

    if (padding + _bytes > __size_)
    {
      throw std::bad_alloc();
    }

    __offset_                     = padding + _bytes;
    __allocations_table_[pointer] = _bytes;

    return __memory_ + padding;
  }

  void      reset() { __offset_ = 0; }
  size_type used() const { return __offset_; }
  size_type capacity() const { return __size_; }

  ~arena_allocator() { delete[] __memory_; }
};  // arena_allocator

template<typename _Tp>
struct TENSOR_LIBRARY_API array
{
  using value_type      = _Tp;
  using pointer         = value_type*;
  using reference       = value_type&;
  using const_pointer   = const pointer;
  using const_reference = const reference;
  using size_type       = std::size_t;

  size_type __size_;
  _Tp*      __begin_;
  _Tp*      __end_;

  array() :
      __size_(0),
      __begin_(nullptr),
      __end_(nullptr) {}

  array(const std::size_t s) :
      __size_(s) {}
};  // array


class TENSOR_LIBRARY_API ArenaAllocator
{
 public:
  explicit ArenaAllocator(std::size_t _block_size = 1024 * 1024)  // default 1 MB
      :
      __block_size_(_block_size),
      __current_block_(nullptr),
      __freelist_(nullptr) {
    allocate_block();
  }

  ~ArenaAllocator() {
    free_all_blocks();
    free_freelist();
  }

  void* allocate(std::size_t _size, std::size_t _alignment = alignof(std::max_align_t)) {
    uintptr_t   current_ptr = reinterpret_cast<uintptr_t>(__current_block_->__ptr_);
    uintptr_t   aligned_ptr = (current_ptr + _alignment - 1) & ~(_alignment - 1);
    std::size_t padding     = aligned_ptr - current_ptr;

    if (padding + _size > __current_block_->__remaining_)
    {
      allocate_block(std::max(__block_size_, _size + _alignment));
      return allocate(_size, _alignment);
    }

    void* result = reinterpret_cast<void*>(aligned_ptr);
    __current_block_->__ptr_ += padding + _size;
    __current_block_->__remaining_ -= padding + _size;
    return result;
  }

  void reset() {
    Block* b = __blocks_;
    while (b)
    {
      b->__ptr_       = b->__start_;
      b->__remaining_ = b->__size_;
      Block* __next_  = b->__next_;
      b->__next_      = __freelist_;
      __freelist_     = b;
      b               = __next_;
    }

    __blocks_        = nullptr;
    __current_block_ = nullptr;

    if (__freelist_)
    {
      __blocks_          = __freelist_;
      __freelist_        = __freelist_->__next_;
      __blocks_->__next_ = nullptr;
      __current_block_   = __blocks_;
    }
    else
    {
      allocate_block();
    }
  }

 private:
  struct Block
  {
    std::size_t __size_;
    std::size_t __remaining_;
    uint8_t*    __start_;
    uint8_t*    __ptr_;
    Block*      __next_;
  };

  std::size_t __block_size_;
  Block*      __blocks_        = nullptr;
  Block*      __current_block_ = nullptr;
  Block*      __freelist_      = nullptr;

  void allocate_block(std::size_t __size_ = 0) {
    std::size_t alloc_size = (__size_ ? __size_ : __block_size_) + sizeof(Block);
    uint8_t*    raw        = static_cast<uint8_t*>(std::malloc(alloc_size));
    assert(raw != nullptr && "Out of memory!");

    Block* block        = reinterpret_cast<Block*>(raw);
    block->__size_      = alloc_size - sizeof(Block);
    block->__start_     = raw + sizeof(Block);
    block->__ptr_       = block->__start_;
    block->__remaining_ = block->__size_;
    block->__next_      = nullptr;

    if (__blocks_ == nullptr)
    {
      __blocks_ = block;
    }
    else
    {
      Block* last = __blocks_;

      while (last->__next_)
      {
        last = last->__next_;
      }

      last->__next_ = block;
    }

    __current_block_ = block;
  }

  void free_all_blocks() {
    Block* b = __blocks_;
    while (b)
    {
      Block* __next_ = b->__next_;
      std::free(b);
      b = __next_;
    }
    __blocks_        = nullptr;
    __current_block_ = nullptr;
  }

  void free_freelist() {
    while (__freelist_)
    {
      Block* __next_ = __freelist_->__next_;
      std::free(__freelist_);
      __freelist_ = __next_;
    }
  }

  ArenaAllocator(const ArenaAllocator&)            = delete;
  ArenaAllocator& operator=(const ArenaAllocator&) = delete;
};  // ArenaAllocator