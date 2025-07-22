#include "tensor.hpp"
#include <gtest/gtest.h>
#include <iostream>

TEST(TensorTest, LogicalTest) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  tensor<bool> bool_       = t.bool_();
  tensor<bool> logical_not = bool_.logical_not();
  tensor<bool> logical_or  = bool_.logical_or(true);
  tensor<bool> logical_and = bool_.logical_and(false);
  tensor<bool> logical_xor = bool_.logical_xor(false);

  EXPECT_EQ(logical_not, tensor<bool>({2, 2}, {false, false, false, false}));
  EXPECT_EQ(logical_or, tensor<bool>({2, 2}, {true, true, true, true}));
  EXPECT_EQ(logical_xor, tensor<bool>({2, 2}, {false, false, false, false}));
  EXPECT_EQ(logical_and, tensor<bool>({2, 2}, {false, false, false, false}));
}