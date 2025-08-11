#include "tensor.hpp"
#include <gtest/gtest.h>






TEST(TensorTest, EmptyTensorOps) {
  tensor<double> t;
  EXPECT_NO_THROW(t.sin());
  EXPECT_NO_THROW(t.tanh());
  EXPECT_EQ(t.sin().size(0), 0);
}