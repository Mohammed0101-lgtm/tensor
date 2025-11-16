#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorSinhTest, SinhBasic)
{
  arch::tensor<double> t({3}, {-1.0, 0.0, 1.0});
  arch::tensor<double> result   = t.sinh();
  std::vector<double>  expected = {std::sinh(-1.0), std::sinh(0.0), std::sinh(1.0)};

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-7);
  }
}