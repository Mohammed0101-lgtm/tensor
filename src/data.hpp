#pragma once

#include "internal/simd/neon/data.hpp"
#include "math/lcm.hpp"
#include "tensor.hpp"


template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::reshape_as(const tensor& other) const {
  return reshape(other.shape());
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::push_back(value_type v) const {
  if (this->n_dims() != 1)
  {
    throw std::range_error("push_back is only supported for one dimensional tensors");
  }

  this->storage_().push_back(v);
  ++this->shape_()[0];
  this->shape_().compute_strides();
  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::zeros(const shape::Shape& shape_) {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.zeros_(shape_);
  return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::zeros_(shape::Shape sh) {
  if (this->empty())
  {
    return *this;
  }

  if (internal::types::using_neon())
  {
    return internal::simd::neon::zeros_(*this, sh);
  }

  if (!sh.empty())
  {
    this->shape_() = sh;
  }

  std::size_t s = this->shape().flatten_size();
  this->storage_().resize(s);
  this->shape_().compute_strides();

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = static_cast<value_type>(0.0);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::ones_(shape::Shape sh) {
  if (this->empty())
  {
    return *this;
  }

  if (internal::types::using_neon())
  {
    return internal::simd::neon::ones_(*this, sh);
  }

  if (sh.empty())
  {
    sh = this->shape();
  }
  else
  {
    this->shape_() = sh;
  }

  this->storage_().resize(this->shape().flatten_size());
  this->shape_().compute_strides();
  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = value_type(1.0);
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::ones(const shape::Shape& shape_) {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.ones_(shape_);
  return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::row(const index_type index) const {
  if (this->empty())
  {
    return self({0});
  }

  if (this->n_dims() != 2)
  {
    throw error::shape_error("Cannot get a row from a non two-dimensional tensor");
  }

  if (index < 0 || index >= this->shape()[0])
  {
    throw error::index_error("Index is out of range");
  }

  container_type row_data;
  row_data.reserve(this->shape()[1]);
  const index_type offset = index * this->shape()[1];

  for (index_type j = 0; j < this->shape()[1]; ++j)
  {
    row_data.push_back((*this)[offset + j]);
  }

  return tensor<_Tp>({this->shape()[1]}, row_data);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::col(const index_type index) const {
  if (this->empty())
  {
    return self({0});
  }

  if (this->n_dims() != 2)
  {
    throw error::shape_error("Cannot get a column from a non two-dimensional tensor");
  }

  if (index < 0 || index >= this->shape()[1])
  {
    throw error::index_error("Index is out of range");
  }

  container_type col_data;
  col_data.reserve(this->shape()[0]);

  for (index_type i = 0; i < this->shape()[0]; ++i)
  {
    col_data.push_back((*this)[this->shape().compute_index({i, index})]);
  }

  return tensor<_Tp>({this->shape()[0]}, col_data);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::view(std::initializer_list<index_type> sh) {
  shape::Shape sh_(sh);
  index_type   s = sh_.flatten_size();

  if (s != this->size(0))
  {
    throw std::invalid_argument("Total elements do not match for new shape");
  }

  this->shape_() = sh_;
  this->shape_().compute_strides();

  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::randomize(const shape::Shape& shape_, bool bounded) {
  if (this->empty())
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
  if (this->empty())
  {
    return *this;
  }

  if (bounded && !std::is_floating_point_v<value_type>)
  {
    throw error::type_error("Cannot bound non floating point data type");
  }

  if (sh.empty() && sh.empty())
  {
    throw error::shape_error("randomize_ : Shape must be initialized");
  }

  if (sh.empty() && sh.equal(this->shape()))
  {
    this->shape_() = sh;
  }

  index_type s = this->size(0);
  this->storage_().resize(s);
  this->shape().compute_strides();
  std::random_device                   rd;
  std::mt19937                         gen(rd());
  std::uniform_real_distribution<_f32> unbounded_dist(1.0f, static_cast<_f32>(RAND_MAX));
  std::uniform_real_distribution<_f32> bounded_dist(0.0f, 1.0f);
  container_type&                      a = this->storage_();

  for (auto& elem : a)
  {
    elem = value_type(bounded ? bounded_dist(gen) : unbounded_dist(gen));
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::clone() const {
  self ret(this->shape_(), this->storage_(), this->device());
  ret.compute_strides();
  return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::negative_() {
  if (this->empty())
  {
    return *this;
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = -elem;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::negative() const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.negative_();
  return ret;
}

inline void _permutations(std::vector<std::vector<int>>& res, std::vector<int>& arr, int idx) {
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

inline void _nextPermutation(std::vector<int>& arr) {
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
tensor<_Tp>& tensor<_Tp>::repeat_(const container_type& d, int dimension) {
  if (d.empty())
  {
    throw std::invalid_argument("Cannot repeat an empty data tensor.");
  }

  if (this->size(0) < d.size())
  {
    this->storage_() = container_type(d.begin(), d.end());
  }

  index_type start      = 0;
  index_type end        = d.size();
  index_type total_size = this->size(0);

  if (total_size < d.size())
  {
    return *this;
  }

  unsigned int    nbatches  = total_size / d.size();
  index_type      remainder = total_size % d.size();
  container_type& a         = this->storage_();

  for (unsigned int i = 0; i < nbatches; ++i)
  {
    for (index_type j = start, k = 0; k < d.size(); ++j, ++k)
    {
      a[j] = d[k];
    }

    start += d.size();
  }

  for (index_type j = start, k = 0; j < total_size && k < remainder; ++j, ++k)
  {
    a[j] = d[k];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fill(const value_type value) const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.fill_(value);
  return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fill(const tensor& other) const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.fill_(other);
  return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::resize_as(const shape::Shape shape_) const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.resize_as_(shape_);
  return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::all() const {
  bool            result = true;
  container_type& a      = this->storage_();
  index_type      i      = 0;

  for (; i < this->size(0); ++i)
  {
    if (a[i] == static_cast<value_type>(0))
    {
      result = false;
      break;
    }
  }

  tensor ret;
  ret.storage_() = {result ? static_cast<value_type>(1) : static_cast<value_type>(0)};

  return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::any() const {
  bool            result = false;
  container_type& a      = this->storage_();

  for (index_type i = 0; i < this->size(0); ++i)
  {
    if (a[i] != static_cast<value_type>(0))
    {
      result = true;
      break;
    }
  }

  tensor ret;
  ret.storage_() = {result ? static_cast<value_type>(1) : static_cast<value_type>(0)};

  return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::gcd(const tensor& other) const {
  if (this->empty())
  {
    return self({0});
  }

  if (!this->shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  tensor          ret = clone();
  container_type& a   = this->storage_();
  index_type      i   = 0;

  for (auto& elem : a)
  {
    index_type gcd  = static_cast<index_type>(elem * other[i]);
    index_type _lcm = __lcm(static_cast<index_type>(elem), static_cast<index_type>(other[i]));
    gcd /= _lcm;
    ret[i] = gcd;
    i++;
  }

  return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::gcd(const value_type value) const {
  if (this->empty())
  {
    return self({0});
  }

  tensor     ret = clone();
  index_type i   = 0;

  for (; i < this->size(0); ++i)
  {
    index_type gcd  = static_cast<index_type>((*this)[i] * value);
    index_type _lcm = lcm(static_cast<index_type>((*this)[i]), static_cast<index_type>(value));
    gcd /= _lcm;
    ret[i] = gcd;
  }

  return ret;
}

template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::count_nonzero(index_type dimension) const {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::count_nonzero(*this, dimension);
  }

  index_type c           = 0;
  index_type local_count = 0;
  index_type i           = 0;

  const container_type& a = this->storage_();

  if (dimension == 0)
  {
    for (const auto& elem : a)
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
    if (dimension < 0 || dimension >= static_cast<index_type>(this->n_dims()))
    {
      throw error::index_error("Invalid dimension provided.");
    }

    throw std::runtime_error("Dimension-specific non-zero counting is not implemented yet.");
  }

  return c;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::fill_(const value_type value) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::fill_(*this, value);
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = value;
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::fill_(const tensor& other) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::fill_(*this, other);
  }

  if (!this->shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  container_type& a = this->storage_();
  index_type      i = 0;

  for (auto& elem : a)
  {
    elem = other[i++];
  }

  return *this;
}