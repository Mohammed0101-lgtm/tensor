#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorAbsTest, AbsBasic)
{
  arch::tensor<double> t({6}, {-2.5, -1.0, 0.0, 0.5, 1.0, 2.5});
  arch::tensor<double> expected({6}, {2.5, 1.0, 0.0, 0.5, 1.0, 2.5});
  arch::tensor<double> result = t.abs();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}

TEST(TensorAbsTest, AbsWithInts)
{
  arch::tensor<int> t({4}, {-3, -1, 0, 5});
  arch::tensor<int> expected({4}, {3, 1, 0, 5});
  arch::tensor<int> result = t.abs();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}