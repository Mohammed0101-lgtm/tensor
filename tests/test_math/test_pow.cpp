#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorPowTest, POWTest) {
  tensor<int> t({3}, {2, 3, 4});
  int         exponent = 2;
  auto        result   = t.pow(exponent);

  tensor<int> expected({3}, {4, 9, 16});

  EXPECT_EQ(result.shape(), expected.shape());
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorPowTest, POWTensorTest) {
  tensor<int> t1({3}, {2, 3, 4});
  tensor<int> t2({3}, {2, 3, 4});
  auto        result = t1.pow(t2);

  tensor<int> expected({3}, {4, 27, 256});

  EXPECT_EQ(result.shape(), expected.shape());
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}