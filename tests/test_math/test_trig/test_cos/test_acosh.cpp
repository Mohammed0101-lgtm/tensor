#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorAcoshTest, AcoshValidRange) {
  tensor<float>  t({3}, {1.0, 2.0, 10.0});
  tensor<float>  result = t.acosh();
  tensor<double> expected({3}, {std::acosh(1.0), std::acosh(2.0), std::acosh(10.0)});

  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(TensorAcoshTest, AcoshOutOfDomain) {
  tensor<float> t({1}, {0.5});
  EXPECT_THROW(t.acosh(), std::domain_error);
}