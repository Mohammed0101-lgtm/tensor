#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorTanTest, TanValues)
{
  arch::tensor<float>  t({2}, {0.0, M_PI / 4});
  arch::tensor<float>  result = t.tan();
  arch::tensor<double> expected({2}, {std::tan(0.0), std::tan(M_PI / 4)});

  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}