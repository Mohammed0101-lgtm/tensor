#include "stride.hpp"
#include "tensor.hpp"

#include <gtest/gtest.h>
#include <vector>


using strides = shape::Strides;
using vec     = std::vector<uint64_t>;


vec univ_shape = {2, 2};


TEST(StridesTest, ConstructorsTest)
{
  strides default_constructor;
  strides shape_constructor(univ_shape);
  strides copy_constructor(shape_constructor);

  EXPECT_NO_THROW();
}

TEST(StridesTest, CopyOperatorTest)
{
  strides other(univ_shape);
  strides copy = other;

  EXPECT_EQ(other, copy);  // this tests the equal operator automatically for free
}

TEST(StridesTest, GetTest)
{
  struct
  {
    vec shape;
    vec expected_strides;
  } test_cases[] = {
    {{}, {}},
    {{1}, {1}},
    {{2}, {1}},
    {{3}, {1}},
    {{1, 1}, {1, 1}},
    {{2, 3}, {3, 1}},
    {{3, 2}, {2, 1}},
    {{3, 4, 5}, {20, 5, 1}},
    {{5, 4, 3}, {12, 3, 1}},
    {{1, 3, 2}, {6, 2, 1}},
    {{2, 1, 4}, {4, 4, 1}},
    {{4, 1, 1}, {1, 1, 1}},
    {{10, 10}, {10, 1}},
    {{3, 3, 3}, {9, 3, 1}},
    {{5, 1, 2}, {2, 2, 1}},
    {{1, 1, 1, 1}, {1, 1, 1, 1}},
    {{2, 3, 4, 5}, {60, 20, 5, 1}},
    {{6, 1, 1, 1}, {1, 1, 1, 1}},
    {{4, 2, 2}, {4, 2, 1}},
    {{7, 1, 2}, {2, 2, 1}},
  };

  for (const auto& test : test_cases)
  {
    strides s(test.shape);

    EXPECT_EQ(s.get(), test.expected_strides) << "Failed on shape: " << ::testing::PrintToString(test.shape);
  }
}

TEST(StridesTest, NdimsTest)
{
  strides one({1});
  strides two({1, 2});
  strides three({1, 2, 3});
  strides four({1, 2, 3, 4});
  strides ten({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  // and so on

  EXPECT_EQ(one.n_dims(), 1);
  EXPECT_EQ(two.n_dims(), 2);
  EXPECT_EQ(three.n_dims(), 3);
  EXPECT_EQ(four.n_dims(), 4);
  EXPECT_EQ(ten.n_dims(), 10);
}
