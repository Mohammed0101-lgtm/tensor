#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorTest, EmptyTensorOps)
{
  arch::tensor<double> t;
  EXPECT_NO_THROW(t.sin());
  EXPECT_NO_THROW(t.tanh());
  EXPECT_EQ(t.sin().size(0), 0);
}