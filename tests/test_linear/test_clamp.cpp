#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(LinearTest, ClampWithinRange)
{
  arch::tensor<float> t({3}, {-2.0, 0.5, 3.0});
  arch::tensor<float> result = t.clamp(0.0, 2.0);
  arch::tensor<float> expected({3}, {0.0, 0.5, 2.0});
  EXPECT_EQ(result, expected);
}

TEST(LinearTest, ClampWithDefaultMinMax)
{
  arch::tensor<float> t({3}, {-100.0, 0.0, 100.0});
  arch::tensor<float> result = t.clamp();
  EXPECT_EQ(result, t);
}

TEST(LinearTest, ClampNegativeOnly)
{
  arch::tensor<float> t({3}, {-5.0, -1.0, 0.0});
  arch::tensor<float> result = t.clamp(-2.0, 0.0);
  arch::tensor<float> expected({3}, {-2.0, -1.0, 0.0});
  EXPECT_EQ(result, expected);
}