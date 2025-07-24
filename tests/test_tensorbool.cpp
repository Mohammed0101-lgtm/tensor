#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(BoolTest, NotBoolTest) {
  tensor<bool> t({2, 3}, {true, true, true, true, true, true});
  tensor<bool> expected({2, 3}, {false, false, false, false, false, false});

  tensor<bool> op = !t;

  EXPECT_EQ(op, expected);
}

TEST(BoolTest, BoolTest) {
  tensor<int>  vals({2, 2}, {1, 2, 0, 0});
  tensor<bool> bools = vals.bool_();
  tensor<bool> expected({2, 2}, {true, true, false, false});

  EXPECT_EQ(bools, expected);
}

TEST(BoolTest, BoolRowTest) {
  tensor<bool> t({2, 3}, {true, false, true, false, false, true});
  tensor<bool> expected_row({3}, {true, false, true});
  tensor<bool> expected_col({2}, {false, false});

  EXPECT_EQ(t.row(0), expected_row);
  EXPECT_EQ(t.col(1), expected_col);
}