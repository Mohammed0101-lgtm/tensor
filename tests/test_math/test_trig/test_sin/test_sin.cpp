#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorsinTest, SinBasic) {
  tensor<double>      t({3}, {0.0, M_PI / 2, M_PI});
  tensor<double>      result   = t.sin();
  std::vector<double> expected = {0.0, 1.0, 0.0};

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}