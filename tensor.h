#include <iostream>
#include <stdexcept>
#include <random>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <type_traits>


template<typename _Tp>
class tensor {
   private:
    std::vector<_Tp>     data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;

   public:
    tensor() = default;

    tensor(const std::vector<int64_t> __sh, const _Tp val) :
        shape_(__sh) {
        int64_t _size = 1LL;
        for (int64_t d : __sh)
        {
            _size *= d;
        }

        this->data_ = std::vector<_Tp>(_size, val);
        compute_strides();
    }

    tensor(const std::vector<int64_t> __sh) :
        shape_(__sh) {
        int64_t _size = 1;
        for (int64_t dim : __sh)
        {
            _size *= dim;
        }

        this->data_ = std::vector<_Tp>(_size);
        compute_strides();
    }

    tensor(const std::vector<_Tp> __d, const std::vector<int64_t> __sh) :
        data_(__d),
        shape_(__sh) {
        compute_strides();
    }

    _Tp& begin() const { return this->data_.begin(); }

    _Tp& end() const { return this->data_.end(); }

    std::vector<_Tp> storage() const { return this->data_; }

    std::vector<int64_t> shape() const { return this->shape_; }

    std::vector<int64_t> strides() const { return this->strides_; }

    size_t n_dims() const { return this->shape_.size(); }

    _Tp& at(std::vector<int64_t> _idx) {
        if (_idx.empty())
        {
            throw std::invalid_argument("Passing an empty vector as indices for a tensor");
        }
        return this->data_[compute_index(_idx)];
    }

    const _Tp& at(const std::vector<int64_t> _idx) const {
        if (_idx.empty())
        {
            throw std::invalid_argument("Passing an empty vector as indices for a tensor");
        }
        return this->data_[compute_index(_idx)];
    }

    _Tp& operator[](const size_t __in) {
        if (__in >= this->data_.size() || __in < 0)
        {
            throw std::out_of_range("Access index is out of range");
        }
        return this->data_[__in];
    }

    const _Tp& operator[](const size_t __in) const {
        if (__in >= this->data_.size() || __in < 0)
        {
            throw std::out_of_range("Access index is out of range");
        }
        return this->data_[__in];
    }

    tensor<_Tp> operator+(const tensor<_Tp>& _other) const {
        if (_other.shape() != this->shape_)
        {
            throw std::invalid_argument("Cannot add two tensors with different shapes");
        }

        std::vector<_Tp> other_data = _other.storage();
        std::vector<_Tp> new_data(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            new_data[i] = this->data_[i] + other_data[i];
        }
        return tensor<_Tp>(new_data, this->shape_);
    }

    tensor<_Tp> operator-(const tensor<_Tp>& _other) const {
        if (_other.shape() != this->shape_)
        {
            throw std::invalid_argument("Cannot add two tensors with different shapes");
        }

        std::vector<_Tp> other_data = _other.storage();
        std::vector<_Tp> new_data(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            new_data[i] = this->data_[i] - other_data[i];
        }
        return tensor<_Tp>(new_data, this->shape_);
    }

    tensor<_Tp> operator*(const tensor<_Tp>& _other) const {
        if (this->empty() || _other.empty())
        {
            throw std::invalid_argument("Cannot multiply empty tensors");
        }
        return this->matmul(_other);
    }

    tensor<_Tp> operator*(const _Tp _scalar) const {
        if (!std::is_scalar<_Tp>::value)
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
        return tensor<_Tp>(*this);
    }

    tensor<_Tp> operator-=(const tensor<_Tp>& _other) const { return tensor<_Tp>(*this - _other); }

    tensor<_Tp> operator+=(const tensor<_Tp>& _other) const { return tensor<_Tp>(*this + _other); }

    tensor<_Tp> operator+(const _Tp _scalar) const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::invalid_argument("Cannot perform addition with a non scalar type");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] += _scalar;
        }
        return tensor<_Tp>(*this);
    }

    tensor<_Tp> operator-(const _Tp _scalar) const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::invalid_argument("Cannot perform substraction with a non scalar type");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] -= _scalar;
        }
        return tensor<_Tp>(*this);
    }

    tensor<_Tp> operator*=(const tensor<_Tp>& _other) const { return tensor<_Tp>(*this * _other); }

    tensor<_Tp> operator+=(const _Tp& _scalar) const { return tensor<_Tp>(*this + _scalar); }

    tensor<_Tp> operator-=(const _Tp& _scalar) const { return tensor<_Tp>(*this - _scalar); }

    tensor<_Tp> operator*=(const _Tp& _scalar) const { return tensor<_Tp>(*this * _scalar); }

    bool operator==(const tensor<_Tp>& _other) const {
        if ((this->shape_ != _other.shape()) && (this->strides_ != _other.strides()))
        {
            return false;
        }
        return this->data_ == _other.storage();
    }

    bool operator!=(const tensor<_Tp>& other) const { return !(*this == other); }

    tensor<_Tp> matmul(const tensor<_Tp>& _other) const {
        if (this->shape_.size() != 2 || _other.shape().size() != 2)
        {
            throw std::invalid_argument("matmul is only supported for 2D tensors");
        }

        if (this->shape_[1] != other.shape()[0])
        {
            throw std::invalid_argument("Shape mismatch for matrix multiplication");
        }

        std::vector<int64_t> ret_shape = {this->shape_[0], _other.shape()[1]};
        std::vector<_Tp>     ret_data(ret_shape[0] * ret_shape[1]);

        for (int64_t i = 0; i < ret_shape[0]; ++i)
        {
            for (int64_t j = 0; j < ret_shape[1]; ++j)
            {
                _Tp sum = 0.0f;

                for (int64_t k = 0; k < this->shape_[1]; ++k)
                {
                    sum += this->at({i, k}) * _other.at({k, j});
                }
                ret_data[i * ret_shape[1] + j] = sum;
            }
        }
        return tensor<_Tp>(ret_data, ret_shape);
    }

    tensor<_Tp> cross_product(const tensor<_Tp>& _other) const {
        if (this->empty() || _other.empty())
        {
            throw std::invalid_argument("Cannot cross product an empty vector");
        }

        if (!std::is_arithmetic<_Tp>::value)
        {
            throw std::runtime_error("Cannot perform a cross product on non-scalar data types");
        }

        if (this->shape() != std::vector<int>{3} || _other.shape() != std::vector<int>{3})
        {
            throw std::invalid_argument("Cross product can only be performed on 3-element vectors");
        }

        tensor<_Tp> ret({3});

        const _Tp& a1 = this->storage()[0];
        const _Tp& a2 = this->storage()[1];
        const _Tp& a3 = this->storage()[2];

        const _Tp& b1 = _other.storage()[0];
        const _Tp& b2 = _other.storage()[1];
        const _Tp& b3 = _other.storage()[2];

        ret.storage()[0] = a2 * b3 - a3 * b2;
        ret.storage()[1] = a3 * b1 - a1 * b3;
        ret.storage()[2] = a1 * b2 - a2 * b1;

        return ret;
    }

    tensor<_Tp> absolute(const tensor<_Tp>& _tensor) const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::invalid_argument("Cannot call absolute on non scalar value");
        }

        std::vector<_Tp> a;
        for (const _Tp& v : _tensor.storage())
        {
            a.push_back(static_cast<_Tp>(std::fabs(float(v))));
        }
        return tensor<_Tp>(a, _tensor.shape_);
    }

    tensor<_Tp> dot(const tensor<_Tp>& _other) const {
        if (this->empty() || _other.empty())
        {
            throw std::invalid_argument("Cannot dot product an empty vector");
        }

        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot perform a dot product on non scalar data types");
        }

        if (this->shape().size() == 1 && _other.shape().size())
        {
            assert(this->shape()[0] == _other.shape()[0]);

            _Tp ret = 0;

            for (int64_t i = 0; i < this->data_.size(); i++)
            {
                ret_data += this->data_[i] * _other[i];
            }

            return tensor<_Tp>({ret}, {1});
        }

        if (this->shape().size() == 2 && _other.shape().size())
        {
            return this->matmul(_other);
        }

        if (this->shape().size() == 3 && _other.shape().size())
        {
            return this->cross_product(_other);
        }

        return tensor<_Tp>();
    }

    tensor<_Tp> relu() const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot relu non-scalar type");
        }

        size_t           _size = this->data_.size();
        std::vector<_Tp> new_data(_size);

        for (size_t i = 0; i < _size; i++)
        {
            new_data[i] = std::max(this->data_[i], _Tp(0));
        }
        return tensor<_Tp>(new_data, this->shape_);
    }

    void relu_() {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot relu non-scalar type");
        }

        size_t _size = this->data_.size();

        for (size_t i = 0; i < _size; i++)
        {
            this->data_[i] = std::max(this->data_[i], _Tp(0));
        }
    }

    tensor<_Tp> transpose() const {
        if (this->shape_.size() != 2)
        {
            std::cerr << "Matrix transposition can only be done on 2D tensors" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        tensor<_Tp> transposed({this->shape_[1], this->shape_[0]});

        for (int64_t i = 0; i < this->shape_[0]; i++)
        {
            for (int64_t j = 0; j < this->shape_[1]; j++)
            {
                transposed.at({j, i}) = this->at({i, j});
            }
        }
        return transposed;
    }

    tensor<int64_t> argsort(int64_t _dim = -1LL, bool _ascending = true) const {
        int64_t adjusted_dim = (_dim < 0LL) ? _dim + data_.size() : _dim;

        if (adjusted_dim < 0LL || adjusted_dim >= static_cast<int64_t>(data_.size()))
        {
            throw std::out_of_range("Invalid dimension for argsort");
        }

        std::vector<std::vector<int64_t>> indices = this->data_;

        for (size_t i = 0; i < data_.size(); i++)
        {
            std::vector<int64_t> idx(data_[i].size());
            std::iota(idx.begin(), idx.end(), 0);

            std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
                return _ascending ? data_[i][a] < data_[i][b] : data_[i][a] > data_[i][b];
            });

            indices[i] = idx;
        }
        return tensor<int64_t>(indices);
    }

    tensor<_Tp> bitwise_not() const {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise not on non integral or boolean value");
        }
        std::vector<_Tp> d;

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            d.push_back(~(this->data_[i]));
        }
        return tensor<_Tp>(d, this->shape_);
    }

    tensor<_Tp> bitwise_and(const _Tp value) const {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise AND on non-integral or non-boolean values");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = this->data_[i] & value;
        }
        return tensor<_Tp>(ret, this->shape_);
    }

    tensor<_Tp> bitwise_or(const _Tp value) const {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise OR on non-integral or non-boolean values");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = this->data_[i] | value;
        }
        return tensor<_Tp>(ret, this->shape_);
    }

    tensor<_Tp> bitwise_xor(const _Tp value) const {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise XOR on non-integral or non-boolean values");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = this->data_[i] ^ value;
        }
        return tensor<_Tp>(ret, this->shape_);
    }

    tensor<_Tp> bitwise_left_shift(const int _amount) const {
        if (!std::is_integral<_Tp>::value)
        {
            throw std::runtime_error("Cannot perform a bitwise left shift on non-integral values");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = this->data_[i] << _amount;
        }
        return tensor<_Tp>(ret, this->shape_);
    }

    tensor<_Tp> bitwise_right_shift(const int _amount) const {
        if (!std::is_integral<_Tp>::value)
        {
            throw std::runtime_error("Cannot perform a bitwise right shift on non-integral values");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = this->data_[i] >> _amount;
        }
        return tensor<_Tp>(ret, this->shape_);
    }

    void bitwise_left_shift_(const int _amount) {
        if (!std::is_integral<_Tp>::value)
        {
            throw std::runtime_error("Cannot perform a bitwise left shift on non-integral values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] <<= _amount;
        }
    }

    void bitwise_right_shift_(const int _amount) {
        if (!std::is_integral<_Tp>::value)
        {
            throw std::runtime_error("Cannot perform a bitwise right shift on non-integral values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] >>= amount;
        }
    }

    void bitwise_and_(const _Tp value) {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise AND on non-integral or non-boolean values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] &= value;
        }
    }

    void bitwise_or_(const _Tp value) {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise OR on non-integral or non-boolean values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] |= value;
        }
    }

    void bitwise_xor_(const _Tp value) {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise XOR on non-integral or non-boolean values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] ^= value;
        }
    }

    void bitwise_not_() {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise not on non integral or boolean value");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = ~this->data_[i];
        }
    }

    tensor<bool> logical_not() const {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot get the element wise not of non-integral and non-boolean value");
        }

        std::vector<bool> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = ~(this->data_[i]);
        }

        return tensor<bool>(ret, this->shape_);
    }

    tensor<bool> logical_or(const tensor<_Tp>& _other) const {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot get the element wise or of non-integral and non-boolean value");
        }

        assert(this->shape_ == _other.shape());

        std::vector<bool> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = (this->data_[i] || _other[i]);
        }

        return tensor<bool>(ret, this->shape_);
    }

    tensor<_Tp> logical_xor(const tensor<_Tp>& _other) const {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot get the element wise xor of non-integral and non-boolean value");
        }

        assert(this->shape_ == _other.shape());

        std::vector<bool> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = (this->data_[i] ^ _other[i]);
        }

        return tensor<bool>(ret, this->shape_);
    }

    tensor<_Tp> logical_and(const tensor<_Tp>& _other) const {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot get the element wise and of non-integral and non-boolean value");
        }

        assert(this->shape_ == _other.shape());

        std::vector<bool> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = (this->data_[i] && _other[i]);
        }

        return tensor<bool>(ret, this->shape_);
    }

    void logical_not_() const {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot get the element wise not of non-integral and non-boolean value");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = ~(this->data_[i]);
        }
    }

    void logical_or_(const tensor<_Tp>& _other) {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot get the element wise not of non-integral and non-boolean value");
        }

        assert(this->shape_ == _other.shape());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = (this->data_[i] || other[i]);
        }
    }

    void logical_xor_(const tensor<_Tp>& _other) {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot get the element wise xor of non-integral and non-boolean value");
        }

        assert(this->shape_ == _other.shape());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = (this->data_[i] ^ other[i]);
        }
    }

    void logical_and_(const tensor<_Tp>& _other) {
        if (!std::is_integral<_Tp>::value && !std::is_same<_Tp, bool>::value)
        {
            throw std::runtime_error(
              "Cannot get the element wise and of non-integral and non-boolean value");
        }

        assert(this->shape_ == _other.shape());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = (this->data_[i] && other[i]);
        }
    }

    tensor<_Tp> sum(const int64_t _axis) const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot reduce tensor with non scalar type");
        }

        if (_axis < 0LL || _axis >= static_cast<int64_t>(this->shape_.size()))
        {
            throw std::invalid_argument("Invalid axis for sum");
        }

        std::vector<int64_t> ret_shape = this->shape_;
        ret_shape[_axis]               = 1LL;

        int64_t ret_size =
          std::accumulate(ret_shape.begin(), ret_shape.end(), 1, std::multiplies<int64_t>());
        std::vector<_Tp> ret_data(ret_size, _Tp(0.0f));

        for (int64_t i = 0; i < static_cast<int64_t>(this->data_.size()); ++i)
        {
            std::vector<int64_t> original_indices(this->shape_.size());
            int64_t              index = i;

            for (int64_t j = static_cast<int64_t>(this->shape_.size()) - 1LL; j >= 0LL; j--)
            {
                original_indices[j] = index % this->shape_[j];
                index /= this->shape_[j];
            }

            original_indices[_axis] = 0LL;

            int64_t ret_index = 0LL;
            int64_t stride    = 1LL;
            for (int64_t j = static_cast<int64_t>(this->shape_.size()) - 1LL; j >= 0LL; j--)
            {
                ret_index += original_indices[j] * stride;
                stride *= ret_shape[j];
            }

            ret_data[ret_index] += this->data_[i];
        }

        return tensor<_Tp>(ret_data, ret_shape);
    }

    tensor<_Tp> row(const int64_t _index) const {
        if (this->shape_.size() != 2)
        {
            throw std::runtime_error("Cannot get a row from a non two dimensional tensor");
        }

        if (this->shape_[0] <= _index || _index < 0LL)
        {
            throw std::invalid_argument("Index input is out of range");
        }

        std::vector<_Tp> R(this->data_.begin() + (this->shape_[1] * _index),
                           this->data_.begin() + (this->shape_[1] * _index + this->shape_[1]));

        return tensor<_Tp>(R, {this->shape_[1]});
    }

    tensor<_Tp> col(const int64_t _index) const {
        if (this->shape_.size() != 2)
        {
            throw std::runtime_error("Cannot get a column from a non two dimensional tensor");
        }

        if (this->shape_[1] <= _index || _index < 0LL)
        {
            throw std::invalid_argument("Index input out of range");
        }

        std::vector<_Tp> C;

        for (int64_t i = 0LL; i < this->shape_[0]; i++)
        {
            C.push_back(this->data_[this->compute_index({i, _index})]);
        }

        return tensor(C, {this->shape_[0]});
    }

    tensor<_Tp> ceil() const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot get the ceiling of a non scalar value");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = std::ceil(this->data_[i]);
        }
        return tensor<_Tp>(ret, this->shape_);
    }

    tensor<_Tp> floor() const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot get the floor of a non scalar value");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = std::floor(this->data_[i]);
        }

        return tensor<_Tp>(ret, this->shape_);
    }

    void ceil_() {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot get the ceiling of a non scalar value");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = std::ceil(this->data_[i]);
        }
    }

    void ceil_() {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot get the floor of a non scalar value");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = std::floor(this->data_[i]);
        }
    }

    tensor<_Tp> clone() const {
        std::vector<_Tp>    new_data  = this->data_;
        std::vector<size_t> new_shape = this->shape_;

        return tensor<_Tp>(new_data, new_shape);
    }

    tensor<_Tp> clamp(const _Tp* _min_val = nullptr, const _Tp* _max_val = nullptr) const {
        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            _Tp value = this->data_[i];

            if (_min_val)
            {
                value = std::max(*_min_val, value);
            }

            if (_max_val)
            {
                value = std::min(*_max_val, value);
            }

            ret[i] = value;
        }

        return tensor<_Tp>(ret, this->shape_);
    }

    void clamp_(const _Tp* _min_val = nullptr, const _Tp* _max_val = nullptr) {
        for (size_t i = 0; i < this->data_.size(); i++)
        {
            if (_min_val)
            {
                this->data_[i] = std::max(*_min_val, this->data_[i]);
            }

            if (_max_val)
            {
                this->data_[i] = std::min(*_max_val, this->data_[i]);
            }
        }
    }

    tensor<_Tp> cos() const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot perform a cosine on non-scalar data type");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<_Tp>(std::cos(static_cast<double>(this->data_[i])));
        }

        return tensor<_Tp>(ret, this->shape_);
    }

    tensor<_Tp> cosh() const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot perform a cosh on non-scalar data type");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<_Tp>(std::cosh(static_cast<double>(this->data_[i])));
        }

        return tensor<_Tp>(ret, this->shape_);
    }

    tensor<_Tp> acosh() const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot perform a acosh on non-scalar data type");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<_Tp>(std::acosh(static_cast<double>(this->data_[i])));
        }

        return tensor<_Tp>(ret, this->shape_);
    }

    void acosh_() {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot perform a acosh on non-scalar data type");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<_Tp>(std::acosh(static_cast<double>(this->data_[i])));
        }
    }

    void cosh_() {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot perform a cosh on non-scalar data type");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<_Tp>(std::cosh(static_cast<double>(this->data_[i])));
        }
    }

    void cos_() {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot perform a cosine on non-scalar data type");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<_Tp>(std::cos(static_cast<double>(this->data_[i])));
        }
    }

    size_t count_nonzero(int64_t dim = -1LL) const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot compare a non scalar value to zero");
        }

        size_t count = 0;

        if (dim == -1)
        {
            for (const _Tp& elem : data_)
            {
                if (elem != 0)
                {
                    count++;
                }
            }
        }
        else
        {
            if (dim < 0 || dim >= static_cast<int64_t>(shape_.size()))
            {
                throw std::invalid_argument("Invalid dimension provided.");
            }

            throw std::runtime_error(
              "Dimension-specific non-zero counting is not implemented yet.");
        }

        return count;
    }

    tensor<_Tp> fmax(const tensor<_Tp>& _other) const {
        if (this->shape_ != _other.shape() || this->data_.size() != _other.size(0))
        {
            throw std::invalid_argument("Cannot compare two tensors of different shapes : fmax");
        }

        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot deduce the maximum of non scalar values");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<_Tp>(std::fmax(double(this->data_[i]), double(_other[i])));
        }

        return tensor<_Tp>(ret, this->shape_);
    }

    tensor<_Tp> fmod(const tensor<_Tp>& _other) const {
        if (this->shape_ != _other.shape() || this->data_.size() != _other.size(0))
        {
            throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");
        }

        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot divide non scalar values");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<_Tp>(std::fmod(double(this->data_[i]), double(_other[i])));
        }

        return tensor<_Tp>(ret, this->shape_);
    }

    void fmod_(const tensor<_Tp>& _other) {
        if (this->shape_ != _other.shape() || this->data_.size() != _other.size(0))
        {
            throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");
        }

        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot divide non scalar values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<_Tp>(std::fmax(double(this->data_[i]), double(_other[i])));
        }
    }

    tensor<_Tp> frac() const {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot get the fraction of a non-scalar type");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<_Tp>(this->__frac(this->data_[i]));
        }

        return tensor<_Tp>(ret, this->shape_);
    }

    void frac_() {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot get the fraction of a non-scalar type");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<_Tp>(this->__frac(this->data_[i]));
        }
    }

    int64_t lcm() const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Given template type must be an int");
        }

        int64_t ret = static_cast<int64_t>(this->data_[0]);
        for (size_t i = 1; i < this->data_.size(); i++)
        {
            ret = this->__lcm(static_cast<int64_t>(this->data_[i]), ret);
        }

        return ret;
    }

    tensor<_Tp> log() const {
        if (!std::is_integral<_Tp>::value)
        {
            throw std::runtime_error("Given data type must be an integral");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<_Tp>(std::log(double(this->data_[i])));
        }

        return tensor<_Tp>(ret, this->shape_);
    }

    void log_() {
        if (!std::is_integral<_Tp>::value)
        {
            throw std::runtime_error("Given data type must be an integral");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<_Tp>(std::log(double(this->data_[i])));
        }
    }

    tensor<_Tp> log10() const {
        if (!std::is_integral<_Tp>::value)
        {
            throw std::runtime_error("Given data type must be an integral");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<_Tp>(std::log10(double(this->data_[i])));
        }

        return tensor<_Tp>(ret, this->shape_);
    }

    void log10_() {
        if (!std::is_integral<_Tp>::value)
        {
            throw std::runtime_error("Given data type must be an integral");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<_Tp>(std::log10(double(this->data_[i])));
        }
    }

    tensor<_Tp> log2() const {
        if (!std::is_integral<_Tp>::value)
        {
            throw std::runtime_error("Given data type must be an integral");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<_Tp>(std::log2(double(this->data_[i])));
        }

        return tensor<_Tp>(ret, this->shape_);
    }

    void log2_() {
        if (!std::is_integral<_Tp>::value)
        {
            throw std::runtime_error("Given data type must be an integral");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<_Tp>(std::log2(double(this->data_[i])));
        }
    }

    tensor<_Tp> exp() const {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot get the exponential of non scalar values");
        }

        std::vector<_Tp> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<_Tp>(std::exp(double(this->data_[i]), double(_other[i])));
        }

        return tensor<_Tp>(ret, this->shape_);
    }

    void exp_() {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot get the exponential of non scalar values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<_Tp>(std::exp(double(this->data_[i]), double(_other[i])));
        }
    }

    tensor<_Tp> slice(int64_t                _dim,
                      std::optional<int64_t> _start,
                      std::optional<int64_t> _end,
                      int64_t                _step) const {
        tensor<_Tp> ret;

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

        int64_t              slice_size = (end_idx - start_idx + _step - 1) / _step;
        std::vector<int64_t> ret_dims   = this->shape_;

        ret_dims[_dim] = slice_size;
        ret            = tensor<_Tp>(ret_dims);

        for (int64_t i = start_idx, j = 0ULL; i < end_idx; i += _step, ++j)
        {
            ret({j}) = this->data_[compute_index({i})];
        }

        return ret;
    }

    tensor<_Tp> cumprod(int64_t _dim = -1) const {
        if (_dim == -1)
        {
            std::vector<_Tp> flattened_data = this->data_;
            std::vector<_Tp> ret(flattened_data.size());

            ret[0] = flattened_data[0];
            for (size_t i = 1; i < flattened_data.size(); ++i)
            {
                ret[i] = ret[i - 1] * flattened_data[i];
            }

            return tensor<_Tp>(ret, {flattened_data.size()});
        }
        else
        {
            if (_dim < 0 || _dim >= static_cast<int64_t>(this->shape_.size()))
            {
                throw std::invalid_argument("Invalid dimension provided.");
            }

            std::vector<_Tp> ret(this->data_);
            // TODO : compute_outer_size() implementation
            size_t outer_size = this->compute_outer_size(_dim);
            size_t inner_size = this->shape_[_dim];
            size_t stride     = this->strides_[_dim];

            for (size_t i = 0; i < outer_size; ++i)
            {
                size_t base_index = i * stride;
                ret[base_index]   = data_[base_index];

                for (size_t j = 1; j < inner_size; ++j)
                {
                    size_t current_index = base_index + j;
                    ret[current_index]   = ret[base_index + j - 1] * data_[current_index];
                }
            }

            return tensor<_Tp>(ret, this->shape_);
        }
    }

    tensor<_Tp> cat(const std::vector<tensor<_Tp>>& _others, int64_t _dim) const {
        for (const tensor<_Tp>& t : _others)
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
        for (const tensor<_Tp>& t : _others)
        {
            ret_sh[_dim] += t.shape_[_dim];
        }

        std::vector<_Tp> c_data;
        c_data.reserve(this->data_.size());
        c_data.insert(c_data.end(), this->data_.begin(), this->data_.end());

        for (const tensor<_Tp>& t : _others)
        {
            c_data.insert(c_data.end(), t.data_.begin(), t.data_.end());
        }

        tensor<_Tp> ret(c_data, ret_sh);
        return ret;
    }

    void view(std::initializer_list<int64_t> _new_sh) {
        int64_t new_size = 1LL;
        for (int64_t d : _new_sh)
        {
            new_size *= d;
        }

        if (new_size != this->data_.size())
        {
            throw std::invalid_argument("_Tpotal elements do not match for new shape");
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

        for (const _Tp& val : this->data_)
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

    static tensor<_Tp> zeros(const std::vector<int64_t>& _sh) {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot zero non scalar tensor");
        }

        std::vector<_Tp> d(std::accumulate(_sh.begin(), _sh.end(), 1, std::multiplies<int64_t>()),
                           _Tp(0));
        return tensor(d, _sh);
    }

    static tensor<_Tp> ones(const std::vector<int64_t>& _sh) {
        if (!std::is_scalar<_Tp>::value)
        {
            throw std::runtime_error("Cannot one non scalar tensor");
        }

        std::vector<_Tp> d(std::accumulate(_sh.begin(), _sh.end(), 1, std::multiplies<int64_t>()),
                           _Tp(1));
        return tensor(d, _sh);
    }

    static tensor<_Tp> randomize(const std::vector<int64_t>& _sh, bool _bounded = false) {
        assert(std::is_scalar<_Tp>::value);

        std::srand(time(nullptr));
        int64_t _size = 1;
        for (int64_t dim : _sh)
        {
            _size *= dim;
        }

        std::vector<_Tp> d(_size);
        for (int64_t i = 0; i < _size; i++)
        {
            d[i] =
              _bounded ? _Tp(static_cast<float>(rand() % RAND_MAX) : static_cast<float>(rand());)
        }
        return tensor(d, _sh);
    }

    tensor<int64_t> argmax_(int64_t _dim) const {
        if (_dim < 0 || _dim >= this->shape_.size())
        {
            throw std::out_of_range("Dimension out of range in argmax");
        }

        std::vector<int64_t> ret_shape = this->shape_;
        ret_shape.erase(ret_shape.begin() + _dim);

        tensor<int64_t> ret;
        ret.shape_ = ret_shape;
        ret.data_.resize(this->computeSize(ret_shape), 0);

        int64_t outer_size = 1LL;
        int64_t inner_size = 1LL;

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
                _Tp     max_value = this->data_[i * this->shape_[_dim] * inner_size + j];

                for (int64_t k = 1; k < this->shape_[_dim]; ++k)
                {
                    _Tp value = this->data_[(i * this->shape_[_dim] + k) * inner_size + j];
                    if (value > max_value)
                    {
                        max_value = value;
                        max_index = k;
                    }
                }

                ret.data_[i * inner_size + j] = max_index;
            }
        }

        return ret;
    }

    tensor<_Tp> argmax(int64_t _dim) const {
        if (_dim < 0 || _dim >= this->shape_.size())
        {
            throw std::out_of_range("Dimension out of range in argmax");
        }

        std::vector<int64_t> ret_shape = this->shape_;
        ret_shape.erase(ret_shape.begin() + _dim);

        tensor<_Tp> ret;
        ret.shape_ = ret_shape;
        ret.data_.resize(this->computeSize(ret_shape), _Tp(0));

        int64_t outer_size = 1;
        int64_t inner_size = 1;

        for (int64_t i = 0; i < _dim; ++i)
        {
            outer_size *= this->shape_[i];
        }
        for (int64_t i = _dim + 1; i < static_cast<int64_t>(this->shape_.size()); i++)
        {
            inner_size *= this->shape_[i];
        }

        for (int64_t i = 0; i < outer_size; i++)
        {
            for (int64_t j = 0; j < inner_size; j++)
            {
                _Tp max_value = this->data_[i * this->shape_[_dim] * inner_size + j];

                for (int64_t k = 1; k < this->shape_[_dim]; k++)
                {
                    _Tp value = this->data_[(i * this->shape_[_dim] + k) * inner_size + j];
                    if (value > max_value)
                    {
                        max_value = value;
                    }
                }

                ret.data_[i * inner_size + j] = max_value;
            }
        }
        return ret;
    }

    tensor<_Tp> unsqueeze(int64_t _dim) const {
        if (_dim < 0 || _dim > static_cast<int64_t>(this->shape_.size()))
        {
            throw std::out_of_range("Dimension out of range in unsqueeze");
        }

        std::vector<int64_t> new_shape = this->shape_;

        new_shape.insert(new_shape.begin() + _dim, 1);

        tensor<_Tp> ret;
        ret.shape_ = new_shape;
        ret.data_  = this->data_;

        return ret;
    }

   private:
    void compute_strides() {
        if (this->shape_.empty())
        {
            std::cerr << "Shape must be initialized before computing strides" << std::endl;
            std::exit(EXI_Tp_FAILURE);
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

    uint64_t computeSize(const std::vector<int64_t>& _dims) const {
        uint64_t ret = 0ULL;

        for (const int64_t& d : _dims)
        {
            ret *= d;
        }
        return ret;
    }

    float __frac(const _Tp& scalar) { return std::fmod(static_cast<float>(scalar), 1.0lf); }

    int64_t __lcm(const int64_t a, const int64_t b) { return (a * b) / std::gcd(a, b); }
};
