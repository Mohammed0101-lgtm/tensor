#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorAtanhTest, AtanhInDomain)
{
  arch::tensor<double> t({3}, {-0.5, 0.0, 0.5});
  arch::tensor<double> result   = t.atanh();
  std::vector<double>  expected = {std::atanh(-0.5), std::atanh(0.0), std::atanh(0.5)};

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-7);
  }
}

TEST(TensorAtanhTest, AtanhOutOfDomain)
{
  arch::tensor<double> t({2}, {-1.5, 2.0});
  EXPECT_THROW(t.atanh(), std::domain_error);
}