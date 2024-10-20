#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>
#include <random>
#include <cassert>

template<typename T>
class Tensor {
   private:
    std::vector<T>       data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;

   public:
    Tensor() = default;

    Tensor(const std::vector<int64_t> __sh) :
        shape_(__sh) {
        int64_t _size = 1;
        for (int64_t dim : __sh)
        {
            _size *= dim;
        }

        this->data_ = std::vector<T>(_size);
        compute_strides();
    }

    Tensor(const std::vector<T> __d, const std::vector<int64_t> __sh) :
        data_(__d),
        shape_(__sh) {
        compute_strides();
    }

    T& begin() const { return this->data_.begin(); }

    T& end() const { return this->data_.end(); }

    std::vector<T> storage() const { return this->data_; }

    std::vector<int64_t> shape() const { return this->shape_; }

    std::vector<int64_t> strides() const { return this->strides_; }

    // for modifiable direct access
    T& operator()(std::vector<int64_t> _idx) {
        if (_idx.empty())
        {
            throw std::invalid_argument("Passing an empty vector as indices for a tensor");
        }
        return this->data_[compute_index(_idx)];
    }

    // for read only direct access
    const T& operator()(const std::vector<int64_t> _idx) const {
        if (_idx.empty())
        {
            throw std::invalid_argument("Passing an empty vector as indices for a tensor");
        }
        return this->data_[compute_index(_idx)];
    }

    // same as above
    T& operator[](const size_t __in) {
        if (__in >= this->data_.size() || __in < 0)
        {
            throw std::out_of_range("Access index is out of range");
        }
        è return this->data_[__in];
    }

    const T& operator[](const size_t __in) const {
        if (__in >= this->data_.size() || __in < 0)
        {
            throw std::out_of_range("Access index is out of range");
        }
        return this->data_[__in];
    }

    Tensor<T> operator+(const Tensor<T>& _other) const {
        if (_other.shape() != this->shape_)
        {
            throw std::invalid_argument("Cannot add two tensors with different shapes");
        }

        std::vector<T> other_data = _other.storage();
        std::vector<T> new_data(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            new_data[i] = this->data_[i] + other_data[i];
        }
        return Tensor(new_data, this->shape_);
    }

    Tensor<T> operator-(const Tensor<T>& _other) const {
        if (_other.shape() != this->shape_)
        {
            throw std::invalid_argument("Cannot add two tensors with different shapes");
        }

        std::vector<T> other_data = _other.storage();
        std::vector<T> new_data(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            new_data[i] = this->data_[i] - other_data[i];
        }
        return Tensor(new_data, this->shape_);
    }

    Tensor<T> operator*(const Tensor<T>& _other) const {
        if (this->empty() || _other.empty())
        {
            throw std::invalid_argument("Cannot multiply empty tensors");
        }
        return this->matmul(_other);
    }

    Tensor<T> operator*(const T _scalar) const {
        if (!std::is_scalar<T>::value)
        {
            throw std::invalid_argument("Cannot multiply a tensor by a non scalar type");
        }

        if (this->empty())
        {
            throw std::invalid_argument("Cannot scale an empty tensor");
        }

        for (int i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] *= _scalar;
        }
        return Tensor(*this);
    }

    Tensor<T> operator-=(const Tensor<T>& _other) const { return Tensor(*this - _other); }

    Tensor<T> operator+=(const Tensor<T>& _other) const { return Tensor(*this + _other); }

    Tensor<T> operator+(const T _scalar) const {
        if (!std::is_scalar<T>::value)
        {
            throw std::invalid_argument("Cannot perform addition with a non scalar type");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] += _scalar;
        }
        return *this;
    }
    Tensor<T> operator-(const T _scalar) const {
        if (!std::is_scalar<T>::value)
        {
            throw std::invalid_argument("Cannot perform substraction with a non scalar type");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] -= _scalar;
        }
        return *this;
    }

    Tensor<T> operator*=(const Tensor<T>& _other) const { return Tensor(*this * _other); }

    Tensor<T> operator+=(const T _scalar) const { return Tensor(*this + _scalar); }
    Tensor<T> operator-=(const T _scalar) const { return Tensor(*this - _scalar); }
    Tensor<T> operator*=(const T _scalar) const { return Tensor(*this * _scalar); }

    bool operator==(const Tensor<T>& _other) const {
        if ((this->shape_ != _other.shape()) && (this->strides_ != _other.strides()))
        {
            return false;
        }
        return this->data_ == _other.storage();
    }

    bool operator!=(const Tensor<T>& other) const { return !(*this == other); }

    Tensor<T> matmul(const Tensor<T>& _other) const {
        if (this->shape_.size() != 2 || _other.shape().size() != 2)
        {
            throw std::invalid_argument("matmul is only supported for 2D tensors");
        }

        if (this->shape_[1] != other.shape()[0])
        {
            throw std::invalid_argument("Shape mismatch for matrix multiplication");
        }

        std::vector<int64_t> result_shape = {this->shape_[0], _other.shape()[1]};
        std::vector<T>       result_data(result_shape[0] * result_shape[1]);

        for (int64_t i = 0; i < result_shape[0]; ++i)
        {
            for (int64_t j = 0; j < result_shape[1]; ++j)
            {
                T sum = 0.0f;

                for (int64_t k = 0; k < this->shape_[1]; ++k)
                {
                    sum += (*this)({i, k}) * _other({k, j});
                }
                result_data[i * result_shape[1] + j] = sum;
            }
        }
        return Tensor(result_data, result_shape);
    }

    Tensor<T> relu() {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot relu non scalar type");
        }

        size_t         _size = this->data_.size();
        std::vector<T> new_data(_size);

        for (int i = 0; i < _size; i++)
        {
            new_data[i] = std::max(this->data_[i].data_, 0.0f);
        }
        return Tensor(new_data, this->shape_);
    }

    Tensor<T> transpose() const {
        if (this->shape_.size() != 2)
        {
            std::cerr << "Matrix transposition can only be done on 2D tensors" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        Tensor<T> transposed({this->shape_[1], this->shape_[0]});

        for (unsigned int i = 0; i < this->shape_[0]; i++)
        {
            for (unsigned int j = 0; j < this->shape_[1]; j++)
            {
                transposed({j, i}) = this->data_[this->compute_index({i, j})];
            }
        }
        return transposed;
    }

    Tensor<T> sum(const int64_t _axis) const {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot reduce tensor with non scalar type");
        }

        if (_axis < 0 || _axis >= this->shape_.size())
        {
            throw std::invalid_argument("Invalid axis for sum");
        }

        std::vector<int64_t> result_shape = this->shape_;
        result_shape[_axis]               = 1;

        int64_t result_size =
          std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<int64_t>());
        std::vector<T> result_data(result_size, T(0.0f));

        for (int64_t i = 0; i < this->data_.size(); ++i)
        {
            std::vector<int64_t> original_indices(this->shape_.size());
            int64_t              index = i;

            for (int64_t j = this->shape_.size() - 1; j >= 0; --j)
            {
                original_indices[j] = index % this->shape_[j];
                index /= this->shape_[j];
            }

            original_indices[_axis] = 0;

            int64_t result_index = 0;
            int64_t stride       = 1;
            for (int64_t j = this->shape_.size() - 1; j >= 0; --j)
            {
                result_index += original_indices[j] * stride;
                stride *= result_shape[j];
            }

            result_data[result_index] += this->data_[i];
        }

        return Tensor(result_data, result_shape);
    }

    Tensor<T> row(const int64_t _index) const {
        if (this->shape_.size() != 2)
        {
            throw std::runtime_error("Cannot get a row from a non two dimensional tensor");
        }

        if (this->shape_[0] <= _index || _index < 0)
        {
            throw std::invalid_argument("Index input is out of range");
        }

        std::vector<T> R(this->data_.begin() + (this->shape_[1] * _index),
                         this->data_.begin() + (this->shape_[1] * _index + this->shape_[1]));

        return Tensor(R, {this->shape_[1]});
    }

    Tensor<T> col(const int64_t _index) const {
        if (this->shape_.size() != 2)
        {
            throw std::runtime_error("Cannot get a column from a non two dimensional tensor");
        }

        if (this->shape_[1] <= _index || _index < 0)
        {
            throw std::invalid_argument("Index input out of range");
        }

        std::vector<T> C;

        for (int64_t i = 0; i < this->shape_[0]; i++)
        {
            C.push_back(this->data_[this->compute_index({i, _index})]);
        }

        return Tensor(C, {this->shape_[0]});
    }

    Tensor<T> slice(int64_t                _dim,
                    std::optional<int64_t> _start,
                    std::optional<int64_t> _end,
                    int64_t                _step) const {
        Tensor<T> ret;

        if (_dim < 0ULL || _dim >= static_cast<int64_t>(this->shape_.size()))
        {
            throw std::out_of_range("Dimension out of range.");
        }

        int64_t size      = this->shape_[_dim];
        int64_t start_idx = _start.value_or(0ULL);
        int64_t end_idx   = _end.value_or(size);

        if (start_idx < 0ULL)
        {
            start_idx += size;
        }
        if (end_idx < 0ULL)
        {
            end_idx += size;
        }

        start_idx = std::max(int64_t(0ULL), std::min(start_idx, size));
        end_idx   = std::max(int64_t(0ULL), std::min(end_idx, size));

        int64_t              slice_size  = (end_idx - start_idx + _step - 1) / _step;
        std::vector<int64_t> result_dims = this->shape_;

        result_dims[_dim] = slice_size;
        ret               = Tensor<T>(result_dims);

        for (int64_t i = start_idx, j = 0ULL; i < end_idx; i += _step, ++j)
        {
            ret({j}) = this->data_[compute_index({i})];
        }

        return ret;
    }

    Tensor<T> cat(const std::vector<Tensor<T>>& _others, int64_t _dim) const {
        for (const Tensor<T>& t : _others)
        {
            for (int64_t i = 0; i < this->shape_.size(); ++i)
            {
                if (i != _dim && this->shape_[i] != t.shape_[i])
                {
                    throw std::invalid_argument(
                      "Cannot concatenate tensors with different shapes along non-concatenation dimensions");
                }
            }
        }

        std::vector<int64_t> ret_sh = this->shape_;
        for (const Tensor<T>& t : _others)
        {
            ret_sh[_dim] += t.shape_[_dim];
        }

        std::vector<T> c_data;
        c_data.reserve(this->data_.size());
        c_data.insert(c_data.end(), this->data_.begin(), this->data_.end());

        for (const Tensor<T>& t : _others)
        {
            c_data.insert(c_data.end(), t.data_.begin(), t.data_.end());
        }

        Tensor<T> ret(c_data, ret_sh);
        return ret;
    }

    void view(std::initializer_list<int64_t> _new_sh) {
        int64_t new_size = 1;
        for (int64_t d : _new_sh)
        {
            new_size *= d;
        }

        if (new_size != this->data_.size())
        {
            throw std::invalid_argument("Total elements do not match for new shape");
        }

        this->shape_ = _new_sh;
        compute_strides();
    }

    void print() const {
        std::cout << "Shape: [";
        for (size_t i = 0; i < this->shape_.size(); ++i)
        {
            std::cout << this->shape_[i] << (i < this->shape_.size() - 1 ? ", " : "");
        }

        std::cout << "]\nData: ";

        for (const T& val : this->data_)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    bool empty() const {
        // just checks for the underlying storage for now
        return this->data_.empty();
    }

    size_t size(const int64_t _dim) const {
        if (_dim < 0 || _dim >= static_cast<int64_t>(this->shape_.size()))
        {
            throw std::invalid_argument("dimension input is out of range");
        }
        return this->shape_[_dim];
    }

    static Tensor<T> zeros(const std::vector<int64_t>& _sh) {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot zero non scalar tensor");
        }

        std::vector<T> d(std::accumulate(_sh.begin(), _sh.end(), 1, std::multiplies<int64_t>()),
                         T(0.0f));
        return Tensor(d, _sh);
    }

    static Tensor<T> ones(const std::vector<int64_t>& _sh) {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot one non scalar tensor");
        }

        std::vector<T> d(std::accumulate(_sh.begin(), _sh.end(), 1, std::multiplies<int64_t>()),
                         T(1));
        return Tensor(d, _sh);
    }

    static Tensor<T> randomize(const std::vector<int64_t>& _sh, bool _bounded = false) {
        assert(std::is_scalar<T>::value);

        std::srand(time(nullptr));
        int64_t _size = 1;
        for (int64_t dim : _sh)
        {
            _size *= dim;
        }

        std::vector<T> d(_size);
        for (int64_t i = 0; i < _size; i++)
        {
            d[i] = _bounded ? static_cast<float>(rand() % RAND_MAX) : static_cast<float>(rand());
        }
        return Tensor(d, _sh);
    }

    Tensor<int64_t> argmax_(int64_t _dim) const {
        if (_dim < 0 || _dim >= this->shape_.size())
        {
            throw std::out_of_range("Dimension out of range in argmax");
        }

        std::vector<int64_t> result_shape = this->shape_;
        result_shape.erase(result_shape.begin() + _dim);

        Tensor<int64_t> result;
        result.shape_ = result_shape;
        result.data_.resize(computeSize(result_shape), 0);

        int64_t outer_size = 1;
        int64_t inner_size = 1;

        for (int64_t i = 0; i < _dim; ++i)
        {
            outer_size *= this->shape_[i];
        }
        for (int64_t i = _dim + 1; i < this->shape_.size(); ++i)
        {
            inner_size *= this->shape_[i];
        }

        for (int64_t i = 0; i < outer_size; ++i)
        {
            for (int64_t j = 0; j < inner_size; ++j)
            {
                int64_t max_index = 0;
                T       max_value = this->data_[i * this->shape_[_dim] * inner_size + j];

                for (int64_t k = 1; k < this->shape_[_dim]; ++k)
                {
                    T value = this->data_[(i * this->shape_[_dim] + k) * inner_size + j];
                    if (value > max_value)
                    {
                        max_value = value;
                        max_index = k;
                    }
                }

                result.data_[i * inner_size + j] = max_index;
            }
        }

        return result;
    }

    Tensor<T> argmax(int64_t _dim) const {
        if (_dim < 0 || _dim >= this->shape_.size())
        {
            throw std::out_of_range("Dimension out of range in argmax");
        }

        std::vector<int64_t> result_shape = this->shape_;
        result_shape.erase(result_shape.begin() + _dim);

        Tensor<T> result;
        result.shape_ = result_shape;
        result.data_.resize(computeSize(result_shape), 0.0f);

        int64_t outer_size = 1;
        int64_t inner_size = 1;

        for (int64_t i = 0; i < _dim; ++i)
        {
            outer_size *= this->shape_[i];
        }
        for (int64_t i = _dim + 1; i < static_cast<int64_t>(this->shape_.size()); ++i)
        {
            inner_size *= this->shape_[i];
        }

        for (int64_t i = 0; i < outer_size; ++i)
        {
            for (int64_t j = 0; j < inner_size; ++j)
            {
                T max_value = this->data_[i * this->shape_[_dim] * inner_size + j];

                for (int64_t k = 1; k < this->shape_[_dim]; ++k)
                {
                    T value = this->data_[(i * this->shape_[_dim] + k) * inner_size + j];
                    if (value > max_value)
                    {
                        max_value = value;
                    }
                }

                result.data_[i * inner_size + j] = max_value;
            }
        }
        return result;
    }

    Tensor<T> unsqueeze(int64_t _dim) const {
        if (_dim < 0 || _dim > static_cast<int64_t>(this->shape_.size()))
        {
            throw std::out_of_range("Dimension out of range in unsqueeze");
        }

        std::vector<int64_t> new_shape = this->shape_;

        new_shape.insert(new_shape.begin() + _dim, 1);

        Tensor<T> result;
        result.shape_ = new_shape;
        result.data_  = this->data_;

        return result;
    }

   private:
    void compute_strides() {
        if (this->shape_.empty())
        {
            std::cerr << "Shape must be initialized before computing strides" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        this->strides_ = std::vector<int64_t>(this->shape_.size(), 1);
        int64_t stride = 1;

        for (int64_t i = this->shape_.size() - 1; i >= 0; i--)
        {
            this->strides_[i] = stride;
            stride *= this->shape_[i];
        }
    }

    size_t compute_index(const std::vector<int64_t>& _idx) const {
        if (_idx.size() != this->shape_.size())
        {
            throw std::out_of_range("input indices does not match the tensor shape_");
        }

        size_t index = 0;
        for (size_t i = 0; i < this->shape_.size(); i++)
        {
            index += _idx[i] * this->strides_[i];
        }
        return index;
    }

    int64_t computeSize(const std::vector<int64_t>& _dims) const {
        int64_t ret = 0ULL;

        for (const int64_t& d : _dims)
        {
            ret *= d;
        }
        return ret;
    }
};
