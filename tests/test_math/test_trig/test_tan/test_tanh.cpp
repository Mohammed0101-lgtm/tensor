#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorTanhTest, TanhBasic) {
  tensor<double>      t({3}, {-2.0, 0.0, 2.0});
  tensor<double>      result   = t.tanh();
  std::vector<double> expected = {std::tanh(-2.0), std::tanh(0.0), std::tanh(2.0)};

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-7);
  }
}