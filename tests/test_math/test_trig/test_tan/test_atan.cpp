#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorAtanTest, AtanBasic)
{
  arch::tensor<double> t({3}, {-1.0, 0.0, 1.0});
  arch::tensor<double> result   = t.atan();
  std::vector<double>  expected = {std::atan(-1.0), std::atan(0.0), std::atan(1.0)};

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-7);
  }
}