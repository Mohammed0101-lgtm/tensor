#include "tensor.hpp"
#include <gtest/gtest.h>


template<typename T>
class TensorLogicTest: public ::testing::Test
{
 protected:
  using tensor_t = arch::tensor<T>;

  tensor_t make_tensor(std::vector<T> data, std::vector<uint64_t> shape = {})
  {
    if (shape.empty())
    {
      shape = {data.size()};
    }

    return tensor_t(shape::Shape(shape), data);
  }
};


using MyTypes = ::testing::Types<bool, int, float>;
TYPED_TEST_SUITE(TensorLogicTest, MyTypes);


TYPED_TEST(TensorLogicTest, LogicalNotBasic)
{
  auto t      = this->make_tensor({1, 0, 1, 0});
  auto result = t.logical_not();
  EXPECT_EQ(result[0], false);
  EXPECT_EQ(result[1], true);
  EXPECT_EQ(result[2], false);
  EXPECT_EQ(result[3], true);
}

TYPED_TEST(TensorLogicTest, LogicalNotAllZeros)
{
  auto t      = this->make_tensor({0, 0, 0});
  auto result = t.logical_not();
  auto data   = result.storage_();
  for (auto v : data)
  {
    EXPECT_TRUE(v);
  }
}

TYPED_TEST(TensorLogicTest, LogicalNotAllOnes)
{
  auto t      = this->make_tensor({1, 1, 1});
  auto result = t.logical_not();
  auto data   = result.storage_();
  for (auto v : data)
  {
    EXPECT_FALSE(v);
  }
}

TYPED_TEST(TensorLogicTest, LogicalNotInplace)
{
  auto t = this->make_tensor({1, 0});
  t.logical_not_();

  EXPECT_EQ(t[0], false);
  EXPECT_EQ(t[1], true);
}

TYPED_TEST(TensorLogicTest, LogicalOrScalar)
{
  auto t      = this->make_tensor({0, 1, 0});
  auto result = t.logical_or(1);
  auto data   = result.storage_();

  for (auto v : data)
  {
    EXPECT_TRUE(v);
  }
}

TYPED_TEST(TensorLogicTest, LogicalOrScalarFalse)
{
  auto t      = this->make_tensor({0, 0, 0});
  auto result = t.logical_or(0);
  auto data   = result.storage_();

  for (auto v : data)
  {
    EXPECT_FALSE(v);
  }
}

TYPED_TEST(TensorLogicTest, LogicalOrScalarTrue)
{
  auto t      = this->make_tensor({0, 1});
  auto result = t.logical_or(1);

  EXPECT_EQ(result[0], true);
  EXPECT_EQ(result[1], true);
}

TYPED_TEST(TensorLogicTest, LogicalOrTensorSameShape)
{
  auto a      = this->make_tensor({1, 0, 0});
  auto b      = this->make_tensor({0, 0, 1});
  auto result = a.logical_or(b);

  EXPECT_EQ(result[0], true);
  EXPECT_EQ(result[1], false);
  EXPECT_EQ(result[2], true);
}

TYPED_TEST(TensorLogicTest, LogicalOrInplaceScalar)
{
  auto t = this->make_tensor({0, 1});
  t.logical_or_(0);
  EXPECT_EQ(t[0], false);
  EXPECT_EQ(t[1], true);
}

TYPED_TEST(TensorLogicTest, LogicalOrInplaceTensor)
{
  auto a = this->make_tensor({0, 1});
  auto b = this->make_tensor({1, 0});
  a.logical_or_(b);

  EXPECT_EQ(a[0], true);
  EXPECT_EQ(a[1], true);
}

TYPED_TEST(TensorLogicTest, LogicalXorScalar)
{
  auto t      = this->make_tensor({1, 0, 1});
  auto result = t.logical_xor(1);
  EXPECT_EQ(result[0], false);
  EXPECT_EQ(result[1], true);
  EXPECT_EQ(result[2], false);
}

TYPED_TEST(TensorLogicTest, LogicalXorTensor)
{
  auto a      = this->make_tensor({1, 0, 1});
  auto b      = this->make_tensor({1, 1, 0});
  auto result = a.logical_xor(b);
  EXPECT_EQ(result[0], false);
  EXPECT_EQ(result[1], true);
  EXPECT_EQ(result[2], true);
}

TYPED_TEST(TensorLogicTest, LogicalXorInplaceScalar)
{
  auto t = this->make_tensor({1, 1});
  t.logical_xor_(1);
  auto data = t.storage_();
  for (auto v : data)
  {
    EXPECT_FALSE(v);
  }
}

TYPED_TEST(TensorLogicTest, LogicalXorInplaceTensor)
{
  auto a = this->make_tensor({1, 0});
  auto b = this->make_tensor({0, 0});
  a.logical_xor_(b);
  EXPECT_EQ(a[0], true);
  EXPECT_EQ(a[1], false);
}

TYPED_TEST(TensorLogicTest, LogicalAndScalarTrue)
{
  auto t      = this->make_tensor({1, 0, 1});
  auto result = t.logical_and(1);
  EXPECT_EQ(result[0], true);
  EXPECT_EQ(result[1], false);
  EXPECT_EQ(result[2], true);
}

TYPED_TEST(TensorLogicTest, LogicalAndScalarFalse)
{
  auto t      = this->make_tensor({1, 1, 1});
  auto result = t.logical_and(0);
  auto data   = result.storage_();
  for (auto v : data)
  {
    EXPECT_FALSE(v);
  }
}

TYPED_TEST(TensorLogicTest, LogicalAndTensor)
{
  auto a      = this->make_tensor({1, 0, 1});
  auto b      = this->make_tensor({1, 1, 0});
  auto result = a.logical_and(b);
  EXPECT_EQ(result[0], true);
  EXPECT_EQ(result[1], false);
  EXPECT_EQ(result[2], false);
}

TYPED_TEST(TensorLogicTest, LogicalAndInplaceScalar)
{
  auto t = this->make_tensor({1, 0});
  t.logical_and_(1);
  EXPECT_EQ(t[0], true);
  EXPECT_EQ(t[1], false);
}

TYPED_TEST(TensorLogicTest, LogicalAndInplaceTensor)
{
  auto a = this->make_tensor({1, 0});
  auto b = this->make_tensor({1, 1});
  a.logical_and_(b);
  EXPECT_EQ(a[0], true);
  EXPECT_EQ(a[1], false);
}