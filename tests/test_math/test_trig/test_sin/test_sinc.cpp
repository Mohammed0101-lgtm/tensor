#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorSincTest, InplaceSincBasic) {
  tensor<double> t({3}, {0.0, 0.5, 1.0});
  t.sinc_();
  std::vector<double> expected = {1.0, std::sin(M_PI * 0.5) / (M_PI * 0.5), std::sin(M_PI * 1.0) / (M_PI * 1.0)};

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_NEAR(t[i], expected[i], 1e-7);
  }
}