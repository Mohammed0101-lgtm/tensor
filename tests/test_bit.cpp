#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(BitTest, BitwiseNot)
{
  arch::tensor<int> t({3}, {0b0000, 0b1010, 0b1111});
  arch::tensor<int> expected({3}, {~0b0000, ~0b1010, ~0b1111});
  arch::tensor<int> result = t.bitwise_not();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseAndScalar)
{
  arch::tensor<int> t({3}, {0b1010, 0b1100, 0b1111});
  int               value = 0b0101;
  arch::tensor<int> expected({3}, {0b1010 & value, 0b1100 & value, 0b1111 & value});
  arch::tensor<int> result = t.bitwise_and(value);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseAndTensor)
{
  arch::tensor<int> t1({3}, {0b1010, 0b1100, 0b1111});
  arch::tensor<int> t2({3}, {0b0110, 0b1010, 0b0001});
  arch::tensor<int> expected({3}, {0b1010 & 0b0110, 0b1100 & 0b1010, 0b1111 & 0b0001});
  arch::tensor<int> result = t1.bitwise_and(t2);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseOrScalar)
{
  arch::tensor<int> t({3}, {0b0001, 0b1000, 0b1111});
  int               value = 0b0101;
  arch::tensor<int> expected({3}, {0b0001 | value, 0b1000 | value, 0b1111 | value});
  arch::tensor<int> result = t.bitwise_or(value);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseOrTensor)
{
  arch::tensor<int> t1({3}, {0b0001, 0b1000, 0b1111});
  arch::tensor<int> t2({3}, {0b0110, 0b0011, 0b0000});
  arch::tensor<int> expected({3}, {0b0001 | 0b0110, 0b1000 | 0b0011, 0b1111 | 0b0000});
  arch::tensor<int> result = t1.bitwise_or(t2);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseXorScalar)
{
  arch::tensor<int> t({3}, {0b1100, 0b1010, 0b1111});
  int               value = 0b0101;
  arch::tensor<int> expected({3}, {0b1100 ^ value, 0b1010 ^ value, 0b1111 ^ value});
  arch::tensor<int> result = t.bitwise_xor(value);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseXorTensor)
{
  arch::tensor<int> t1({3}, {0b1100, 0b1010, 0b1111});
  arch::tensor<int> t2({3}, {0b0011, 0b0101, 0b0000});
  arch::tensor<int> expected({3}, {0b1100 ^ 0b0011, 0b1010 ^ 0b0101, 0b1111 ^ 0b0000});
  arch::tensor<int> result = t1.bitwise_xor(t2);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseLeftShift)
{
  arch::tensor<int> t({3}, {1, 2, 4});
  arch::tensor<int> expected({3}, {1 << 1, 2 << 1, 4 << 1});
  arch::tensor<int> result = t.bitwise_left_shift(1);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseRightShift)
{
  arch::tensor<int> t({3}, {4, 8, 16});
  arch::tensor<int> expected({3}, {4 >> 1, 8 >> 1, 16 >> 1});
  arch::tensor<int> result = t.bitwise_right_shift(1);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseNotEmptyTensor)
{
  arch::tensor<int> t({0});
  arch::tensor<int> result = t.bitwise_not();

  EXPECT_EQ(result.shape(), t.shape());
  EXPECT_EQ(result.size(0), 0);
}

TEST(BitTest, BitwiseNotNegativeValues)
{
  arch::tensor<int> t({3}, {-1, -2, -3});
  arch::tensor<int> expected({3}, {~(-1), ~(-2), ~(-3)});
  arch::tensor<int> result = t.bitwise_not();

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseAndShapeMismatchThrows)
{
  arch::tensor<int> t1({3}, {1, 2, 3});
  arch::tensor<int> t2({4}, {1, 2, 3, 4});

  EXPECT_THROW(t1.bitwise_and(t2), error::shape_error);
}

TEST(BitTest, BitwiseOrWithMaxInt)
{
  arch::tensor<int> t({3}, {1, 2, 3});
  int               max_int = std::numeric_limits<int>::max();
  arch::tensor<int> expected({3}, {1 | max_int, 2 | max_int, 3 | max_int});
  arch::tensor<int> result = t.bitwise_or(max_int);

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseXorWithZeros)
{
  arch::tensor<int> t({3}, {0, 0, 0});
  int               value = 0b101010;
  arch::tensor<int> expected({3}, {value, value, value});
  arch::tensor<int> result = t.bitwise_xor(value);

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseLeftShiftByZero)
{
  arch::tensor<int> t({3}, {1, 2, 4});
  arch::tensor<int> expected = t;
  arch::tensor<int> result   = t.bitwise_left_shift(0);

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseRightShiftLarge)
{
  arch::tensor<int> t({3}, {1024, 2048, 4096});
  arch::tensor<int> expected({3}, {1024 >> 10, 2048 >> 10, 4096 >> 10});
  arch::tensor<int> result = t.bitwise_right_shift(10);

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseAnd2DTensor)
{
  arch::tensor<int> t1({2, 2}, {0b1010, 0b1100, 0b1111, 0b1001});
  arch::tensor<int> t2({2, 2}, {0b0101, 0b0011, 0b1111, 0b0000});
  arch::tensor<int> expected({2, 2}, {0b1010 & 0b0101, 0b1100 & 0b0011, 0b1111 & 0b1111, 0b1001 & 0b0000});
  arch::tensor<int> result = t1.bitwise_and(t2);

  for (std::size_t i = 0; i < t1.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseLeftShiftNegativeValues)
{
  arch::tensor<int> t({3}, {-1, -2, -4});
  arch::tensor<int> expected({3}, {-1 << 2, -2 << 2, -4 << 2});
  arch::tensor<int> result = t.bitwise_left_shift(2);

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseRightShiftNegativeValues)
{
  arch::tensor<int> t({3}, {-1, -2, -4});
  arch::tensor<int> expected({3}, {-1 >> 1, -2 >> 1, -4 >> 1});
  arch::tensor<int> result = t.bitwise_right_shift(1);

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}