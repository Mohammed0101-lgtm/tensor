#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorAsinhTest, AsinhBasic) {
  tensor<double> t({4}, {0.0, 0.5, 1.0, 2.0});
  tensor<double> expected({4}, {std::asinh(0.0), std::asinh(0.5), std::asinh(1.0), std::asinh(2.0)});
  tensor<double> result = t.asinh();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}

TEST(TensorAsinhTest, AsinhNegative) {
  tensor<double> t({3}, {-0.5, -1.0, -2.0});
  tensor<double> expected({3}, {std::asinh(-0.5), std::asinh(-1.0), std::asinh(-2.0)});
  tensor<double> result = t.asinh();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}