#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(LinearTest, Relu)
{
  arch::tensor<float> t({4}, {-1.0, 0.0, 2.5, -3.3});
  arch::tensor<float> expected({4}, {0.0, 0.0, 2.5, 0.0});
  arch::tensor<float> result = t.relu();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_FLOAT_EQ(result[i], expected[i]);
  }
}

TEST(LinearTest, ClippedReLU)
{
  arch::tensor<float> t({6}, {-3.0f, -1.0f, 0.0f, 2.5f, 5.0f, 10.0f});
  float               clip_limit = 4.0f;
  std::vector<float>  expected   = {0.0f, 0.0f, 0.0f, 2.5f, 4.0f, 4.0f};
  arch::tensor<float> result     = t.clipped_relu(clip_limit);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_FLOAT_EQ(result[i], expected[i]);
  }
}
