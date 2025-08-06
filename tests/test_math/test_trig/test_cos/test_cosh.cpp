#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorCoshTest, CoshValues) {
  tensor<float>  t({2}, {0.0, 1.0});
  tensor<float>  result = t.cosh();
  tensor<double> expected({2}, {std::cosh(0.0), std::cosh(1.0)});

  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}