#pragma once

#include "tensorbase.hpp"


template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::reshape_as(const tensor& other) const {
    return reshape(other.shape());
}

template<class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::size(const index_type dimension) const {
    if (dimension < 0 || dimension > static_cast<index_type>(shape_.size()))
    {
        throw error::index_error("dimension input is out of range");
    }

    if (!dimension)
    {
        return data_.size();
    }

    return shape()[dimension - 1];
}

template<class _Tp>
inline typename tensor<_Tp>::reference tensor<_Tp>::at(shape::Shape idx) {
    if (idx.empty())
    {
        throw std::invalid_argument("Passing an empty vector as indices for a tensor");
    }

    index_type i = shape().compute_index(idx.get());

    if (i < 0 || i >= data_.size())
    {
        throw error::index_error("input indices are out of bounds");
    }

    return data_[i];
}

template<class _Tp>
inline typename tensor<_Tp>::const_reference tensor<_Tp>::at(const shape::Shape idx) const {
    if (idx.empty())
    {
        throw std::invalid_argument("Passing an empty vector as indices for a tensor");
    }

    index_type i = idx.compute_index(idx.get());

    if (i < 0 || i >= data_.size())
    {
        throw error::index_error("input indices are out of bounds");
    }

    return data_[i];
}

template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::count_nonzero(index_type dimension) const {
    if (internal::types::using_neon())
    {
        return internal::neon::count_nonzero(*this, dimension);
    }

    index_type c           = 0;
    index_type local_count = 0;
    index_type i           = 0;

    if (dimension == 0)
    {
        for (const auto& elem : data_)
        {
            if (elem)
            {
                ++local_count;
            }
        }

        c += local_count;
    }
    else
    {
        if (dimension < 0 || dimension >= static_cast<index_type>(shape_.size()))
        {
            throw error::index_error("Invalid dimension provided.");
        }

        throw std::runtime_error("Dimension-specific non-zero counting is not implemented yet.");
    }

    return c;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::push_back(value_type v) const {
    if (shape_.size() != 1)
    {
        throw std::range_error("push_back is only supported for one dimensional tensors");
    }

    data_.push_back(v);
    ++shape_[0];
    shape_.compute_strides();
    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::zeros(const shape::Shape& shape_) {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.zeros_(shape_);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::zeros_(shape::Shape sh) {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::zeros_(*this, sh);
    }

    if (!sh.empty())
    {
        shape_ = sh;
    }

    std::size_t s = shape_.flatten_size();
    data_.resize(s);
    shape_.compute_strides();

    for (auto& elem : data_)
    {
        elem = value_type(0.0);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::ones_(shape::Shape sh) {
    if (empty())
    {
        return self({0});
    }

    if (internal::types::using_neon())
    {
        return internal::neon::ones_(*this, sh);
    }

    if (sh.empty())
    {
        sh = shape();
    }
    else
    {
        shape_ = sh;
    }

    data_.resize(shape().flatten_size());
    shape_.compute_strides();

    for (auto& elem : data_)
    {
        elem = value_type(1.0);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::ones(const shape::Shape& shape_) {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.ones_(shape_);
    return ret;
}

template<class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::hash() const {
    index_type            hash_val = 0;
    std::hash<value_type> hasher;

    for (const auto& elem : data_)
    {
        hash_val ^= hasher(elem) + 0x9e3779b9 + (hash_val << 6) + (hash_val >> 2);
    }

    return hash_val;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::row(const index_type index) const {
    if (empty())
    {
        return self({0});
    }

    if (shape_.size() != 2)
    {
        throw error::shape_error("Cannot get a row from a non two-dimensional tensor");
    }

    if (index < 0 || index >= shape_[0])
    {
        throw error::index_error("Index is out of range");
    }

    data_t row_data;
    row_data.reserve(shape_[1]);
    const index_type offset = index * shape_[1];

    for (index_type j = 0; j < shape_[1]; ++j)
    {
        row_data.push_back(data_[offset + j]);
    }

    return tensor<_Tp>({shape_[1]}, row_data);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::col(const index_type index) const {
    if (empty())
    {
        return self({0});
    }

    if (shape_.size() != 2)
    {
        throw error::shape_error("Cannot get a column from a non two-dimensional tensor");
    }

    if (index < 0 || index >= shape_[1])
    {
        throw error::index_error("Index is out of range");
    }

    data_t col_data;
    col_data.reserve(shape_[0]);

    for (index_type i = 0; i < shape_[0]; ++i)
    {
        col_data.push_back(data_[shape_.compute_index({i, index})]);
    }

    return tensor<_Tp>({shape_[0]}, col_data);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::view(std::initializer_list<index_type> sh) {
    shape::Shape sh_(sh);
    index_type   s = sh_.flatten_size();

    if (s != data_.size())
    {
        throw std::invalid_argument("Total elements do not match for new shape");
    }

    shape_ = sh_;
    shape_.compute_strides();

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::randomize(const shape::Shape& shape_, bool bounded) {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.randomize_(shape_, bounded);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::get_minor(index_type a, index_type b) const {
    // not implemented yet
    return tensor();
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::randomize_(const shape::Shape& sh, bool bounded) {
    if (empty())
    {
        return self({0});
    }

    if (bounded && !std::is_floating_point_v<value_type>)
    {
        throw error::type_error("Cannot bound non floating point data type");
    }

    if (sh.empty() && sh.empty())
    {
        throw error::shape_error("randomize_ : Shape must be initialized");
    }

    if (sh.empty() && sh.equal(shape_))
    {
        shape_ = sh;
    }

    index_type s = shape_.flatten_size();
    data_.resize(s);
    shape_.compute_strides();
    std::random_device                   rd;
    std::mt19937                         gen(rd());
    std::uniform_real_distribution<_f32> unbounded_dist(1.0f, static_cast<_f32>(RAND_MAX));
    std::uniform_real_distribution<_f32> bounded_dist(0.0f, 1.0f);

    for (auto& elem : data_)
    {
        elem = value_type(bounded ? bounded_dist(gen) : unbounded_dist(gen));
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::clone() const {
    self ret(shape_, data_, device_);
    ret.is_cuda_tensor_ = is_cuda_tensor_;
    ret.compute_strides();
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::negative_() {
    if (empty())
    {
        return self({0});
    }

    for (auto& elem : data_)
    {
        elem = -elem;
    }

    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::negative() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.negative_();
    return ret;
}

void _permutations(std::vector<std::vector<int>>& res, std::vector<int>& arr, int idx) {
    if (idx == arr.size() - 1)
    {
        res.push_back(arr);
        return;
    }

    for (int i = idx; i < arr.size(); ++i)
    {
        std::swap(arr[idx], arr[i]);
        _permutations(res, arr, idx + 1);
        std::swap(arr[idx], arr[i]);
    }
}

void _nextPermutation(std::vector<int>& arr) {
    std::vector<std::vector<int>> ret;
    _permutations(ret, arr, 0);
    std::sort(ret.begin(), ret.end());

    for (int i = 0; i < ret.size(); ++i)
    {
        if (ret[i] == arr)
        {
            if (i < ret.size() - 1)
            {
                arr = ret[i + 1];
            }

            if (i == ret.size() - 1)
            {
                arr = ret[0];
            }

            break;
        }
    }
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::permute_(const index_type dimension) {
    if (empty())
    {
        return *this;
    }

    if (dimension < 0 || dimension > n_dims())
    {
        throw error::index_error("Dimension index is out of range");
    }

    if (dimension == 0)
    {
        _nextPermutation(data_);
        return *this;
    }

    index_type start = shape_.strides_[dimension - 1];
    index_type end   = shape_.strides_[dimension];

    data_t p(data_.begin() + start, data_.begin() + end);
    _nextPermutation(p);

    for (index_type i = start, pi = 0; i < end && pi < p.size(); ++i, ++pi)
    {
        data_[i] = p[pi];
    }

    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::permute(const index_type dimension) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.permute_(dimension);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::repeat_(const data_t& d, int dimension) {
    if (d.empty())
    {
        throw std::invalid_argument("Cannot repeat an empty data tensor.");
    }

    if (size(0) < d.size())
    {
        data_ = data_t(d.begin(), d.end());
    }

    index_type start      = 0;
    index_type end        = d.size();
    index_type total_size = size(0);

    if (total_size < d.size())
    {
        return *this;
    }

    unsigned int nbatches  = total_size / d.size();
    index_type   remainder = total_size % d.size();

    for (unsigned int i = 0; i < nbatches; ++i)
    {
        for (index_type j = start, k = 0; k < d.size(); ++j, ++k)
        {
            data_[j] = d[k];
        }

        start += d.size();
    }

    for (index_type j = start, k = 0; j < total_size && k < remainder; ++j, ++k)
    {
        data_[j] = d[k];
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::repeat(const data_t& d, int dimension) const {
    self ret = clone();
    ret.repeat_(d, dimension);
    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sort(index_type dimension, bool descending) const {
    if (empty())
    {
        return self({0});
    }

    if (dimension < 0 || dimension >= n_dims())
    {
        throw error::index_error("Dimension index is out of range");
    }

    if (dimension == 0)
    {
        std::sort(data_.begin(), data_.end(), descending ? std::greater<value_type>() : std::less<value_type>());
        return *this;
    }

    index_type start = shape_.strides_[dimension - 1];
    index_type end   = shape_.strides_[dimension];

    data_t p(data_.begin() + start, data_.begin() + end);
    std::sort(p.begin(), p.end(), descending ? std::greater<value_type>() : std::less<value_type>());

    for (index_type i = start, pi = 0; i < end && pi < p.size(); ++i, ++pi)
    {
        data_[i] = p[pi];
    }

    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fill(const value_type value) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.fill_(value);
    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fill(const tensor& other) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.fill_(other);
    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::resize_as(const shape::Shape shape_) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.resize_as_(shape_);
    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::all() const {
    bool       result = true;
    index_type i      = 0;

    for (; i < data_.size(); ++i)
    {
        if (data_[i] == static_cast<value_type>(0))
        {
            result = false;
            break;
        }
    }

    tensor ret;
    ret.data_ = {result ? static_cast<value_type>(1) : static_cast<value_type>(0)};

    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::any() const {
    bool result = false;

    for (index_type i = 0; i < data_.size(); ++i)
    {
        if (data_[i] != static_cast<value_type>(0))
        {
            result = true;
            break;
        }
    }

    tensor ret;
    ret.data_ = {result ? static_cast<value_type>(1) : static_cast<value_type>(0)};

    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::gcd(const tensor& other) const {
    if (empty())
    {
        return self({0});
    }

    if (!shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    tensor     ret = clone();
    index_type i   = 0;

    for (auto& elem : data_)
    {
        index_type gcd  = static_cast<index_type>(elem * other[i]);
        index_type _lcm = lcm(static_cast<index_type>(elem), static_cast<index_type>(other[i]));
        gcd /= _lcm;
        ret[i] = gcd;
        i++;
    }

    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::gcd(const value_type value) const {
    if (empty())
    {
        return self({0});
    }

    tensor     ret = clone();
    index_type i   = 0;

    for (; i < data_.size(); ++i)
    {
        index_type gcd  = static_cast<index_type>(data_[i] * value);
        index_type _lcm = lcm(static_cast<index_type>(data_[i]), static_cast<index_type>(value));
        gcd /= _lcm;
        ret[i] = gcd;
    }

    return ret;
}