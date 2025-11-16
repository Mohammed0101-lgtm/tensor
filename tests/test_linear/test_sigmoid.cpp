#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(LinearTest, Sigmoid)
{
  arch::tensor<float> t({2}, {0.0f, 2.0f});
  auto                result = t.sigmoid();

  EXPECT_NEAR(result[0], 0.5f, 1e-5);
  EXPECT_NEAR(result[1], 1.0f / (1.0f + std::exp(-2.0f)), 1e-5);
}
