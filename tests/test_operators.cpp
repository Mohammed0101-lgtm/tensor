#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(OperatorsTest, LinearAccessOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});

  int expected = 4;

  EXPECT_EQ(t[3], expected);
}

TEST(OperatorsTest, MultiDimensionalAccessOperator)
{
  arch::tensor<int> ten({2, 3}, {1, 2, 3, 4, 5, 6});

  int expected  = 6;
  int expected1 = 2;

  EXPECT_EQ(ten({0, 1}), expected1);
  EXPECT_EQ(ten({1, 2}), expected);
}

TEST(OperatorsTest, EmptyTest)
{
  arch::tensor<int> t;
  arch::tensor<int> q({1, 2}, {1, 2});

  EXPECT_TRUE(t.empty());
  EXPECT_FALSE(q.empty());
}

TEST(OperatorsTest, EqualOperatorTest)
{
  arch::tensor<int> t;
  arch::tensor<int> q;

  arch::tensor<int> a({1, 2}, {1, 2});
  arch::tensor<int> b({2, 1}, {1, 2});
  arch::tensor<int> c({1, 2}, {1, 2});

  EXPECT_TRUE(t == q);
  EXPECT_TRUE(a == c);
  EXPECT_FALSE(t == a);
  EXPECT_FALSE(a == b);
}

TEST(OperatorsTest, NotEqualOperatorTest)
{
  arch::tensor<int> t;
  arch::tensor<int> q;

  arch::tensor<int> a({1, 2}, {1, 2});
  arch::tensor<int> b({2, 1}, {1, 2});
  arch::tensor<int> c({1, 2}, {1, 2});

  EXPECT_TRUE(a != b);
  EXPECT_TRUE(t != a);
  EXPECT_FALSE(t != q);
  EXPECT_FALSE(a != c);
}

TEST(OperatorsTest, PlusOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  arch::tensor<int> expected({2, 3}, {2, 4, 6, 8, 10, 12});

  EXPECT_EQ(t + q, expected);
}

TEST(OperatorsTest, PlusValueOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> expected({2, 3}, {2, 3, 4, 5, 6, 7});

  int q = 1;

  EXPECT_EQ(t + q, expected);
}

TEST(OperatorsTest, PlusEqualOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  arch::tensor<int> expected({2, 3}, {2, 4, 6, 8, 10, 12});

  t += q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, PlusEqualValueOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> expected({2, 3}, {2, 3, 4, 5, 6, 7});

  int q = 1;
  t += q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, MinusOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  arch::tensor<int> expected({2, 3}, {0, 0, 0, 0, 0, 0});

  EXPECT_EQ(t - q, expected);
}

TEST(OperatorsTest, MinusValueOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> expected({2, 3}, {0, 1, 2, 3, 4, 5});
  int               q = 1;

  EXPECT_EQ(t - q, expected);
}

TEST(OperatorsTest, MinusEqualOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  arch::tensor<int> expected({2, 3}, {0, 0, 0, 0, 0, 0});

  t -= q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, MinusEqualValueOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> expected({2, 3}, {0, 1, 2, 3, 4, 5});

  int q = 1;
  t -= q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, TimesOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  arch::tensor<int> expected({2, 3}, {1, 4, 9, 16, 25, 36});

  EXPECT_EQ(t * q, expected);
}

TEST(OperatorsTest, TimesValueOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  int               q = 2;

  arch::tensor<int> expected({2, 3}, {2, 4, 6, 8, 10, 12});

  EXPECT_EQ(t * q, expected);
}

TEST(OperatorsTest, TimesEqualOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  arch::tensor<int> expected({2, 3}, {1, 4, 9, 16, 25, 36});

  t *= q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, TimesEqualValueOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> expected({2, 3}, {2, 4, 6, 8, 10, 12});

  int q = 2;
  t *= q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, DivideOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  arch::tensor<int> expected({2, 3}, {1, 1, 1, 1, 1, 1});

  EXPECT_EQ(t / q, expected);
}

TEST(OperatorsTest, DivideValueOperatorTest)
{
  arch::tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<float> expected({2, 3}, {0.5, 1, 1.5, 2, 2.5, 3});

  float q = 2;

  EXPECT_EQ(t / q, expected);
}

TEST(OperatorsTest, DivideValueOperatorExceptionTest)
{
  arch::tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
  float               q = 0;

  EXPECT_THROW(t / q, std::logic_error);
}

TEST(OperatorsTest, DivideEqualOperatorTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  arch::tensor<int> expected({2, 3}, {1, 1, 1, 1, 1, 1});

  t /= q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, DivideEqualValueOperatorTest)
{
  arch::tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
  arch::tensor<float> expected({2, 3}, {0.5, 1, 1.5, 2, 2.5, 3});

  float q = 2;
  t /= q;

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_NEAR(t[i], expected[i], 1e-5);
  }
}

TEST(OperatorsTest, DivideEqualValueOperatorExceptionTest)
{
  arch::tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
  float               q = 0;

  EXPECT_THROW(t /= q, std::logic_error);
}