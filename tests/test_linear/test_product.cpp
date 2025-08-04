#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(LinearTest, DotProduct) {
  tensor<int> A({3}, {1, 3, -5});
  tensor<int> B({3}, {4, -2, -1});
  int         dot_result = 1 * 4 + 3 * (-2) + (-5) * (-1);
  tensor<int> result     = A.dot(B);

  EXPECT_EQ(result.shape(), std::vector<unsigned long long>{1});
  EXPECT_EQ(result[0], dot_result);
}

TEST(LinearTest, CrossProduct3D) {
  tensor<int> A({3}, {1, 2, 3});
  tensor<int> B({3}, {4, 5, 6});
  tensor<int> expected({3}, {2 * 6 - 3 * 5, 3 * 4 - 1 * 6, 1 * 5 - 2 * 4});
  tensor<int> result = A.cross_product(B);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}