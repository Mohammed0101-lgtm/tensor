#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(BitTest, BitwiseNot) {
  tensor<int> t({3}, {0b0000, 0b1010, 0b1111});
  tensor<int> expected({3}, {~0b0000, ~0b1010, ~0b1111});
  tensor<int> result = t.bitwise_not();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseAndScalar) {
  tensor<int> t({3}, {0b1010, 0b1100, 0b1111});
  int         value = 0b0101;
  tensor<int> expected({3}, {0b1010 & value, 0b1100 & value, 0b1111 & value});
  tensor<int> result = t.bitwise_and(value);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseAndTensor) {
  tensor<int> t1({3}, {0b1010, 0b1100, 0b1111});
  tensor<int> t2({3}, {0b0110, 0b1010, 0b0001});
  tensor<int> expected({3}, {0b1010 & 0b0110, 0b1100 & 0b1010, 0b1111 & 0b0001});
  tensor<int> result = t1.bitwise_and(t2);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseOrScalar) {
  tensor<int> t({3}, {0b0001, 0b1000, 0b1111});
  int         value = 0b0101;
  tensor<int> expected({3}, {0b0001 | value, 0b1000 | value, 0b1111 | value});
  tensor<int> result = t.bitwise_or(value);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseOrTensor) {
  tensor<int> t1({3}, {0b0001, 0b1000, 0b1111});
  tensor<int> t2({3}, {0b0110, 0b0011, 0b0000});
  tensor<int> expected({3}, {0b0001 | 0b0110, 0b1000 | 0b0011, 0b1111 | 0b0000});
  tensor<int> result = t1.bitwise_or(t2);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseXorScalar) {
  tensor<int> t({3}, {0b1100, 0b1010, 0b1111});
  int         value = 0b0101;
  tensor<int> expected({3}, {0b1100 ^ value, 0b1010 ^ value, 0b1111 ^ value});
  tensor<int> result = t.bitwise_xor(value);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseXorTensor) {
  tensor<int> t1({3}, {0b1100, 0b1010, 0b1111});
  tensor<int> t2({3}, {0b0011, 0b0101, 0b0000});
  tensor<int> expected({3}, {0b1100 ^ 0b0011, 0b1010 ^ 0b0101, 0b1111 ^ 0b0000});
  tensor<int> result = t1.bitwise_xor(t2);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseLeftShift) {
  tensor<int> t({3}, {1, 2, 4});
  tensor<int> expected({3}, {1 << 1, 2 << 1, 4 << 1});
  tensor<int> result = t.bitwise_left_shift(1);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseRightShift) {
  tensor<int> t({3}, {4, 8, 16});
  tensor<int> expected({3}, {4 >> 1, 8 >> 1, 16 >> 1});
  tensor<int> result = t.bitwise_right_shift(1);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseNotEmptyTensor) {
  tensor<int> t({0});
  tensor<int> result = t.bitwise_not();

  EXPECT_EQ(result.shape(), t.shape());
  EXPECT_EQ(result.size(0), 0);
}

TEST(BitTest, BitwiseNotNegativeValues) {
  tensor<int> t({3}, {-1, -2, -3});
  tensor<int> expected({3}, {~(-1), ~(-2), ~(-3)});
  tensor<int> result = t.bitwise_not();

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseAndShapeMismatchThrows) {
  tensor<int> t1({3}, {1, 2, 3});
  tensor<int> t2({4}, {1, 2, 3, 4});

  EXPECT_THROW(t1.bitwise_and(t2), error::shape_error);
}

TEST(BitTest, BitwiseOrWithMaxInt) {
  tensor<int> t({3}, {1, 2, 3});
  int         max_int = std::numeric_limits<int>::max();
  tensor<int> expected({3}, {1 | max_int, 2 | max_int, 3 | max_int});
  tensor<int> result = t.bitwise_or(max_int);

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseXorWithZeros) {
  tensor<int> t({3}, {0, 0, 0});
  int         value = 0b101010;
  tensor<int> expected({3}, {value, value, value});
  tensor<int> result = t.bitwise_xor(value);

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseLeftShiftByZero) {
  tensor<int> t({3}, {1, 2, 4});
  tensor<int> expected = t;
  tensor<int> result   = t.bitwise_left_shift(0);

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseRightShiftLarge) {
  tensor<int> t({3}, {1024, 2048, 4096});
  tensor<int> expected({3}, {1024 >> 10, 2048 >> 10, 4096 >> 10});
  tensor<int> result = t.bitwise_right_shift(10);

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseAnd2DTensor) {
  tensor<int> t1({2, 2}, {0b1010, 0b1100, 0b1111, 0b1001});
  tensor<int> t2({2, 2}, {0b0101, 0b0011, 0b1111, 0b0000});
  tensor<int> expected({2, 2}, {0b1010 & 0b0101, 0b1100 & 0b0011, 0b1111 & 0b1111, 0b1001 & 0b0000});
  tensor<int> result = t1.bitwise_and(t2);

  for (std::size_t i = 0; i < t1.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseLeftShiftNegativeValues) {
  tensor<int> t({3}, {-1, -2, -4});
  tensor<int> expected({3}, {-1 << 2, -2 << 2, -4 << 2});
  tensor<int> result = t.bitwise_left_shift(2);

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(BitTest, BitwiseRightShiftNegativeValues) {
  tensor<int> t({3}, {-1, -2, -4});
  tensor<int> expected({3}, {-1 >> 1, -2 >> 1, -4 >> 1});
  tensor<int> result = t.bitwise_right_shift(1);

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}