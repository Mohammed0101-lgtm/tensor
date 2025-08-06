#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorAcosTest, AcosValues) {
  tensor<float>  t({3}, {-1.0, 0.0, 1.0});
  tensor<float>  result = t.acos();
  tensor<double> expected({3}, {M_PI, M_PI / 2, 0.0});

  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(TensorAcosTest, AcosOutOfDomain) {
  tensor<float> t({2}, {-2.0, 2.0});

  EXPECT_THROW(t.acos(), std::domain_error);
}