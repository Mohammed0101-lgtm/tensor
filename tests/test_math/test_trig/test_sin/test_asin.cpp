#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorAsinTest, AsinBasic)
{
  arch::tensor<double> t({3}, {0.0, 0.5, 1.0});
  arch::tensor<double> expected({3}, {std::asin(0.0), std::asin(0.5), std::asin(1.0)});
  arch::tensor<double> result = t.asin();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}

TEST(TensorAsinTest, AsinNegativeInputs)
{
  arch::tensor<double> t({3}, {-0.5, -1.0, -0.25});
  arch::tensor<double> expected({3}, {std::asin(-0.5), std::asin(-1.0), std::asin(-0.25)});
  arch::tensor<double> result = t.asin();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}

TEST(TensorAsinTest, AsinOutOfDomain)
{
  arch::tensor<double> t({2}, {1.5, -1.5});
  EXPECT_THROW(t.asin(), std::domain_error);
}