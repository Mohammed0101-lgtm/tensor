#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorCosTest, CosValues) {
  tensor<float> t({3}, {0.0, M_PI / 2, M_PI});
  tensor<float> result = t.cos();
  tensor<float> expected({3}, {1.0, 0.0, -1.0});

  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}