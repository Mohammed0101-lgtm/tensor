#include "tensor.hpp"
#include <gtest/gtest.h>
#include <iostream>


TEST(TensorTest, BitwiseNot) {
  tensor<int> t({3}, {0b0000, 0b1010, 0b1111});
  tensor<int> expected({3}, {~0b0000, ~0b1010, ~0b1111});
  tensor<int> result = t.bitwise_not();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseAndScalar) {
  tensor<int> t({3}, {0b1010, 0b1100, 0b1111});
  int         value = 0b0101;
  tensor<int> expected({3}, {0b1010 & value, 0b1100 & value, 0b1111 & value});
  tensor<int> result = t.bitwise_and(value);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseAndTensor) {
  tensor<int> t1({3}, {0b1010, 0b1100, 0b1111});
  tensor<int> t2({3}, {0b0110, 0b1010, 0b0001});
  tensor<int> expected({3}, {0b1010 & 0b0110, 0b1100 & 0b1010, 0b1111 & 0b0001});
  tensor<int> result = t1.bitwise_and(t2);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseOrScalar) {
  tensor<int> t({3}, {0b0001, 0b1000, 0b1111});
  int         value = 0b0101;
  tensor<int> expected({3}, {0b0001 | value, 0b1000 | value, 0b1111 | value});
  tensor<int> result = t.bitwise_or(value);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseOrTensor) {
  tensor<int> t1({3}, {0b0001, 0b1000, 0b1111});
  tensor<int> t2({3}, {0b0110, 0b0011, 0b0000});
  tensor<int> expected({3}, {0b0001 | 0b0110, 0b1000 | 0b0011, 0b1111 | 0b0000});
  tensor<int> result = t1.bitwise_or(t2);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseXorScalar) {
  tensor<int> t({3}, {0b1100, 0b1010, 0b1111});
  int         value = 0b0101;
  tensor<int> expected({3}, {0b1100 ^ value, 0b1010 ^ value, 0b1111 ^ value});
  tensor<int> result = t.bitwise_xor(value);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseXorTensor) {
  tensor<int> t1({3}, {0b1100, 0b1010, 0b1111});
  tensor<int> t2({3}, {0b0011, 0b0101, 0b0000});
  tensor<int> expected({3}, {0b1100 ^ 0b0011, 0b1010 ^ 0b0101, 0b1111 ^ 0b0000});
  tensor<int> result = t1.bitwise_xor(t2);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseLeftShift) {
  tensor<int> t({3}, {1, 2, 4});
  tensor<int> expected({3}, {1 << 1, 2 << 1, 4 << 1});
  tensor<int> result = t.bitwise_left_shift(1);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseRightShift) {
  tensor<int> t({3}, {4, 8, 16});
  tensor<int> expected({3}, {4 >> 1, 8 >> 1, 16 >> 1});
  tensor<int> result = t.bitwise_right_shift(1);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}
