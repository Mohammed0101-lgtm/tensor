#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

template<class _Tp>
class __container__
{
   private:
    using value_t         = _Tp;
    using size_type       = uint64_t;
    using container       = std::vector<value_t>;
    using reference       = value_t&;
    using const_reference = const value_t&;

   protected:
    std::vector<container> __data_;
    size_type              __size_ = 0;

    size_type max_vector_size() const noexcept { return std::vector<value_t>().max_size(); }

    std::tuple<int, size_t> __compute_index(const size_type __idx) const {
        assert(__idx <= this->__size_ && "Index out of range");
        int    __vec   = static_cast<int>(__idx / this->max_vector_size());
        size_t __index = static_cast<size_t>(__idx % this->max_vector_size());
        return std::make_tuple(__vec, __index);
    }

   public:
    __container__() = default;

    __container__(const size_type __s, value_t __v) {
        // TODO
    }

    __container__(const size_type __s) noexcept {
        assert(__s >= 0 && __s <= INTMAX_MAX && "Invalid size initializer");
        this->__size_ = __s;

        int    __vecs = static_cast<int>(__s / this->max_vector_size());
        size_t __rem  = static_cast<size_t>(__s % this->max_vector_size());

        this->__data_.resize(__vecs + (__rem > 0 ? 1 : 0));
        for (int i = 0; i < __vecs; ++i)
            this->__data_[i].resize(max_vector_size());

        if (__rem > 0)

            this->__data_[__vecs].resize(__rem);
    }

    reference operator[](const size_type __idx) {
        auto [__vec, __index] = this->__compute_index(__idx);
        return this->__data_[__vec][__index];
    }

    const_reference operator[](const size_type __idx) const {
        auto [__vec, __index] = this->__compute_index(__idx);
        return this->__data_[__vec][__index];
    }

    size_type size() const noexcept { return this->__size_; }

    void push_back(const_reference __value) {
        if (this->__size_ == this->max_vector_size() * this->__data_.size())
            this->__data_.emplace_back();

        auto [__vec, __index] = this->__compute_index(this->__size_);
        this->__data_[__vec].push_back(__value);
        this->__size_++;
    }

    void pop_back() {
        if (this->__size_ == 0)
            throw std::underflow_error("Container is empty.");

        auto [__vec, __index] = this->__compute_index(this->__size_ - 1);
        this->__data_[__vec].pop_back();
        this->__size_--;

        if (this->__data_.back().empty())
            this->__data_.pop_back();
    }
};