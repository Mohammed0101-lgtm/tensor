#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(LinearTest, MatmulBasic)
{
  arch::tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> B({3, 2}, {7, 8, 9, 10, 11, 12});

  arch::tensor<int> expected(
    {2, 2}, {1 * 7 + 2 * 9 + 3 * 11, 1 * 8 + 2 * 10 + 3 * 12, 4 * 7 + 5 * 9 + 6 * 11, 4 * 8 + 5 * 10 + 6 * 12});

  arch::tensor<int> result = A.matmul(B);
  EXPECT_EQ(result.shape(), expected.shape());
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}