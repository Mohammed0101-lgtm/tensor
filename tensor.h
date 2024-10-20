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

    Tensor(const std::vector<int64_t> __sh, const T val) :
        shape_(__sh) {
        int64_t _size = 1LL;
        for (int64_t d : __sh)
        {
            _size *= d;
        }

        this->data_ = std::vector<T>(_size, val);
        compute_strides();
    }

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

    size_t n_dims() const { return this->shape_.size(); }

    T& operator()(std::vector<int64_t> _idx) {
        if (_idx.empty())
        {
            throw std::invalid_argument("Passing an empty vector as indices for a tensor");
        }
        return this->data_[compute_index(_idx)];
    }

    const T& operator()(const std::vector<int64_t> _idx) const {
        if (_idx.empty())
        {
            throw std::invalid_argument("Passing an empty vector as indices for a tensor");
        }
        return this->data_[compute_index(_idx)];
    }

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

        std::vector<int64_t> ret_shape = {this->shape_[0], _other.shape()[1]};
        std::vector<T>       ret_data(ret_shape[0] * ret_shape[1]);

        for (int64_t i = 0; i < ret_shape[0]; ++i)
        {
            for (int64_t j = 0; j < ret_shape[1]; ++j)
            {
                T sum = 0.0f;

                for (int64_t k = 0; k < this->shape_[1]; ++k)
                {
                    sum += (*this)({i, k}) * _other({k, j});
                }
                ret_data[i * ret_shape[1] + j] = sum;
            }
        }
        return Tensor(ret_data, ret_shape);
    }

    Tensor<T> cross_product(const Tensor<T>& _other) const {
        if (this->empty() || _other.empty())
        {
            throw std::invalid_argument("Cannot cross product an empty vector");
        }

        if (!std::is_arithmetic<T>::value)
        {
            throw std::runtime_error("Cannot perform a cross product on non-scalar data types");
        }

        if (this->shape() != std::vector<int>{3} || _other.shape() != std::vector<int>{3})
        {
            throw std::invalid_argument("Cross product can only be performed on 3-element vectors");
        }

        Tensor<T> ret({3});

        const T& a1 = this->data()[0];
        const T& a2 = this->data()[1];
        const T& a3 = this->data()[2];

        const T& b1 = _other.data()[0];
        const T& b2 = _other.data()[1];
        const T& b3 = _other.data()[2];

        ret.data()[0] = a2 * b3 - a3 * b2;
        ret.data()[1] = a3 * b1 - a1 * b3;
        ret.data()[2] = a1 * b2 - a2 * b1;

        return ret;
    }

    Tensor<T> absolute(const Tensor<T>& _tensor) const {
        if (!std::is_scalar<T>::value)
        {
            throw std::invalid_argument("Cannot call absolute on non scalar value");
        }

        std::vector<T> a;
        for (const T& v : _tensor.storage())
        {
            a.push_back(std::abs(v));
        }
        return Tensor<T>(a, _tensor.shape_);
    }

    Tensor<T> dot(const Tensor<T>& _other) const {
        if (this->empty() || _other.empty())
        {
            throw std::invalid_argument("Cannot dot product an empty vector");
        }

        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot perform a dot product on non scalar data types");
        }

        if (this->shape().size() == 1 && _other.shape().size())
        {
            assert(this->shape()[0] == _other.shape()[0]);

            T ret = 0;

            for (int64_t i = 0; i < this->data_.size(); i++)
            {
                ret_data += this->data_[i] * _other.storage()[i];
            }

            return Tensor(ret, {1});
        }

        if (this->shape().size() == 2 && _other.shape().size())
        {
            return this->matmul(_other);
        }

        if (this->shape().size() == 3 && _other.shape().size())
        {
            return this->cross_product(_other);
        }

        return Tensor();
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

    Tensor<int64_t> argsort(int64_t _dim = -1LL, bool _ascending = true) const {
        int64_t adjusted_dim = (_dim < 0LL) ? _dim + data_.size() : _dim;

        if (adjusted_dim < 0LL || adjusted_dim >= static_cast<int64_t>(data_.size()))
        {
            throw std::out_of_range("Invalid dimension for argsort");
        }

        std::vector<std::vector<int64_t>> indices = data_;

        for (size_t i = 0UL; i < data_.size(); i++)
        {
            std::vector<int64_t> idx(data_[i].size());
            std::iota(idx.begin(), idx.end(), 0);

            std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
                return ascending ? data_[i][a] < data_[i][b] : data_[i][a] > data_[i][b];
            });

            indices[i] = idx;
        }
        return Tensor<int64_t>(indices);
    }

    Tensor<T> bitwise_not() const {
        if (!std::is_integral<T>::value && !std::is_same<T, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise not on non integral or boolean value");
        }
        std::vector<T> d;

        for (size_t i = 0UL; i < this->data_.size(); i++)
        {
            d.push_back(~(this->data_[i]));
        }

        return Tensor<T>(d, this->shape_);
    }

    Tensor<T> bitwise_and(T value) const {
        if (!std::is_integral<T>::value && !std::is_same<T, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise AND on non-integral or non-boolean values");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = this->data_[i] & value;
        }

        return Tensor<T>(ret, this->shape_);
    }

    Tensor<T> bitwise_or(T value) const {
        if (!std::is_integral<T>::value && !std::is_same<T, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise OR on non-integral or non-boolean values");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = this->data_[i] | value;
        }

        return Tensor<T>(ret, this->shape_);
    }

    Tensor<T> bitwise_xor(T value) const {
        if (!std::is_integral<T>::value && !std::is_same<T, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise XOR on non-integral or non-boolean values");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = this->data_[i] ^ value;
        }

        return Tensor<T>(ret, this->shape_);
    }

    Tensor<T> bitwise_left_shift(int _amount) const {
        if (!std::is_integral<T>::value)
        {
            throw std::runtime_error("Cannot perform a bitwise left shift on non-integral values");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = this->data_[i] << _amount;
        }

        return Tensor<T>(ret, this->shape_);
    }

    Tensor<T> bitwise_right_shift(int _amount) const {
        if (!std::is_integral<T>::value)
        {
            throw std::runtime_error("Cannot perform a bitwise right shift on non-integral values");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = this->data_[i] >> _amount;
        }

        return Tensor<T>(ret, this->shape_);
    }

    void bitwise_left_shift_(int _amount) {
        if (!std::is_integral<T>::value)
        {
            throw std::runtime_error("Cannot perform a bitwise left shift on non-integral values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] <<= _amount;
        }
    }

    void bitwise_right_shift_(int _amount) {
        if (!std::is_integral<T>::value)
        {
            throw std::runtime_error("Cannot perform a bitwise right shift on non-integral values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] >>= amount;
        }
    }

    void bitwise_and_(T value) {
        if (!std::is_integral<T>::value && !std::is_same<T, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise AND on non-integral or non-boolean values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] &= value;
        }
    }

    void bitwise_or_(T value) {
        if (!std::is_integral<T>::value && !std::is_same<T, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise OR on non-integral or non-boolean values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] |= value;
        }
    }

    void bitwise_xor_(T value) {
        if (!std::is_integral<T>::value && !std::is_same<T, bool>::value)
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
        if (!std::is_integral<T>::value && !std::is_same<T, bool>::value)
        {
            throw std::runtime_error(
              "Cannot perform a bitwise not on non integral or boolean value");
        }

        for (size_t i = 0UL; i < this->data_.size(); i++)
        {
            this->data_[i] = ~this->data_[i];
        }
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

        std::vector<int64_t> ret_shape = this->shape_;
        ret_shape[_axis]               = 1;

        int64_t ret_size =
          std::accumulate(ret_shape.begin(), ret_shape.end(), 1, std::multiplies<int64_t>());
        std::vector<T> ret_data(ret_size, T(0.0f));

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

            int64_t ret_index = 0;
            int64_t stride    = 1;
            for (int64_t j = this->shape_.size() - 1; j >= 0; --j)
            {
                ret_index += original_indices[j] * stride;
                stride *= ret_shape[j];
            }

            ret_data[ret_index] += this->data_[i];
        }

        return Tensor<T>(ret_data, ret_shape);
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

    Tensor<T> ceil() const {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot get the ceiling of a non scalar value");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = std::ceil(this->data_[i]);
        }

        return Tensor<T>(ret, this->shape_);
    }

    Tensor<T> floor() const {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot get the floor of a non scalar value");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = std::floor(this->data_[i]);
        }

        return Tensor<T>(ret, this->shape_);
    }

    void ceil_() {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot get the ceiling of a non scalar value");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = std::ceil(this->data_[i]);
        }
    }

    void ceil_() {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot get the floor of a non scalar value");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = std::floor(this->data_[i]);
        }
    }

    Tensor<T> clone() const {
        std::vector<T>      new_data  = this->data_;
        std::vector<size_t> new_shape = this->shape_;

        return Tensor<T>(new_data, new_shape);
    }

    Tensor<T> clamp(const T* _min_val = nullptr, const T* _max_val = nullptr) const {
        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            T value = this->data_[i];

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

        return Tensor<T>(ret, this->shape_);
    }

    void clamp_(const T* _min_val = nullptr, const T* _max_val = nullptr) {
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

    Tensor<T> cos() const {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot perform a cosine on non-scalar data type");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<T>(std::cos(static_cast<double>(this->data_[i])));
        }

        return Tensor<T>(ret, this->shape_);
    }

    Tensor<T> cosh() const {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot perform a cosh on non-scalar data type");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<T>(std::cosh(static_cast<double>(this->data_[i])));
        }

        return Tensor<T>(ret, this->shape_);
    }

    Tensor<T> acosh() const {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot perform a acosh on non-scalar data type");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<T>(std::acosh(static_cast<double>(this->data_[i])));
        }

        return Tensor<T>(ret, this->shape_);
    }

    void acosh_() {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot perform a acosh on non-scalar data type");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<T>(std::acosh(static_cast<double>(this->data_[i])));
        }
    }

    void cosh_() {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot perform a cosh on non-scalar data type");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<T>(std::cosh(static_cast<double>(this->data_[i])));
        }
    }

    void cos_() {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot perform a cosine on non-scalar data type");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<T>(std::cos(static_cast<double>(this->data_[i])));
        }
    }

    size_t count_nonzero(int64_t dim = -1LL) const {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot compare a non scalar value to zero");
        }

        size_t count = 0;

        if (dim == -1)
        {
            for (const T& elem : data_)
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

    Tensor<T> fmax(const Tensor<T>& _other) const {
        if (this->shape_ != _other.shape() || this->data_.size() != _other.size(0))
        {
            throw std::invalid_argument("Cannot compare two tensors of different shapes : fmax");
        }

        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot deduce the maximum of non scalar values");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<T>(std::fmax(double(this->data_[i]), double(_other[i])));
        }

        return Tensor<T>(ret, this->shape_);
    }

    Tensor<T> fmod(const Tensor<T>& _other) const {
        if (this->shape_ != _other.shape() || this->data_.size() != _other.size(0))
        {
            throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");
        }

        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot divide non scalar values");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<T>(std::fmod(double(this->data_[i]), double(_other[i])));
        }

        return Tensor<T>(ret, this->shape_);
    }

    void fmod_(const Tensor<T>& _other) {
        if (this->shape_ != _other.shape() || this->data_.size() != _other.size(0))
        {
            throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");
        }

        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot divide non scalar values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<T>(std::fmax(double(this->data_[i]), double(_other[i])));
        }
    }

    Tensor<T> exp() const {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot get the exponential of non scalar values");
        }

        std::vector<T> ret(this->data_.size());

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            ret[i] = static_cast<T>(std::exp(double(this->data_[i]), double(_other[i])));
        }

        return Tensor<T>(ret, this->shape_);
    }

    void exp_() {
        if (!std::is_scalar<T>::value)
        {
            throw std::runtime_error("Cannot get the exponential of non scalar values");
        }

        for (size_t i = 0; i < this->data_.size(); i++)
        {
            this->data_[i] = static_cast<T>(std::exp(double(this->data_[i]), double(_other[i])));
        }
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

        int64_t              slice_size = (end_idx - start_idx + _step - 1) / _step;
        std::vector<int64_t> ret_dims   = this->shape_;

        ret_dims[_dim] = slice_size;
        ret            = Tensor<T>(ret_dims);

        for (int64_t i = start_idx, j = 0ULL; i < end_idx; i += _step, ++j)
        {
            ret({j}) = this->data_[compute_index({i})];
        }

        return ret;
    }

    Tensor<T> cumprod(int64_t _dim = -1) const {
        if (_dim == -1)
        {
            std::vector<T> flattened_data = this->data_;
            std::vector<T> ret(flattened_data.size());

            ret[0] = flattened_data[0];
            for (size_t i = 1; i < flattened_data.size(); ++i)
            {
                ret[i] = ret[i - 1] * flattened_data[i];
            }

            return Tensor<T>(ret, {flattened_data.size()});
        }
        else
        {
            if (_dim < 0 || _dim >= static_cast<int64_t>(this->shape_.size()))
            {
                throw std::invalid_argument("Invalid dimension provided.");
            }

            std::vector<T> ret(this->data_);
            size_t         outer_size = compute_outer_size(_dim);
            size_t         inner_size = this->shape_[_dim];
            size_t         stride     = this->strides_[_dim];

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

            return Tensor<T>(ret, this->shape_);
        }
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

        std::vector<int64_t> ret_shape = this->shape_;
        ret_shape.erase(ret_shape.begin() + _dim);

        Tensor<int64_t> ret;
        ret.shape_ = ret_shape;
        ret.data_.resize(computeSize(ret_shape), 0);

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

                ret.data_[i * inner_size + j] = max_index;
            }
        }

        return ret;
    }

    Tensor<T> argmax(int64_t _dim) const {
        if (_dim < 0 || _dim >= this->shape_.size())
        {
            throw std::out_of_range("Dimension out of range in argmax");
        }

        std::vector<int64_t> ret_shape = this->shape_;
        ret_shape.erase(ret_shape.begin() + _dim);

        Tensor<T> ret;
        ret.shape_ = ret_shape;
        ret.data_.resize(computeSize(ret_shape), 0.0f);

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

                ret.data_[i * inner_size + j] = max_value;
            }
        }
        return ret;
    }

    Tensor<T> unsqueeze(int64_t _dim) const {
        if (_dim < 0 || _dim > static_cast<int64_t>(this->shape_.size()))
        {
            throw std::out_of_range("Dimension out of range in unsqueeze");
        }

        std::vector<int64_t> new_shape = this->shape_;

        new_shape.insert(new_shape.begin() + _dim, 1);

        Tensor<T> ret;
        ret.shape_ = new_shape;
        ret.data_  = this->data_;

        return ret;
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
