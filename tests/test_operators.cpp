#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(OperatorsTest, LinearAccessOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});

  int expected = 4;

  EXPECT_EQ(t[3], expected);
}

TEST(OperatorsTest, MultiDimensionalAccessOperator) {
  tensor<int> ten({2, 3}, {1, 2, 3, 4, 5, 6});

  int expected  = 6;
  int expected1 = 2;

  EXPECT_EQ(ten({0, 1}), expected1);
  EXPECT_EQ(ten({1, 2}), expected);
}

TEST(OperatorsTest, EmptyTest) {
  tensor<int> t;
  tensor<int> q({1, 2}, {1, 2});

  EXPECT_TRUE(t.empty());
  EXPECT_FALSE(q.empty());
}

TEST(OperatorsTest, EqualOperatorTest) {
  tensor<int> t;
  tensor<int> q;

  tensor<int> a({1, 2}, {1, 2});
  tensor<int> b({2, 1}, {1, 2});
  tensor<int> c({1, 2}, {1, 2});

  EXPECT_TRUE(t == q);
  EXPECT_TRUE(a == c);
  EXPECT_FALSE(t == a);
  EXPECT_FALSE(a == b);
}

TEST(OperatorsTest, NotEqualOperatorTest) {
  tensor<int> t;
  tensor<int> q;

  tensor<int> a({1, 2}, {1, 2});
  tensor<int> b({2, 1}, {1, 2});
  tensor<int> c({1, 2}, {1, 2});

  EXPECT_TRUE(a != b);
  EXPECT_TRUE(t != a);
  EXPECT_FALSE(t != q);
  EXPECT_FALSE(a != c);
}

TEST(OperatorsTest, PlusOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  tensor<int> expected({2, 3}, {2, 4, 6, 8, 10, 12});

  EXPECT_EQ(t + q, expected);
}

TEST(OperatorsTest, PlusValueOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> expected({2, 3}, {2, 3, 4, 5, 6, 7});

  int q = 1;

  EXPECT_EQ(t + q, expected);
}

TEST(OperatorsTest, PlusEqualOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  tensor<int> expected({2, 3}, {2, 4, 6, 8, 10, 12});

  t += q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, PlusEqualValueOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> expected({2, 3}, {2, 3, 4, 5, 6, 7});

  int q = 1;
  t += q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, MinusOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  tensor<int> expected({2, 3}, {0, 0, 0, 0, 0, 0});

  EXPECT_EQ(t - q, expected);
}

TEST(OperatorsTest, MinusValueOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> expected({2, 3}, {0, 1, 2, 3, 4, 5});
  int         q = 1;

  EXPECT_EQ(t - q, expected);
}

TEST(OperatorsTest, MinusEqualOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  tensor<int> expected({2, 3}, {0, 0, 0, 0, 0, 0});

  t -= q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, MinusEqualValueOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> expected({2, 3}, {0, 1, 2, 3, 4, 5});

  int q = 1;
  t -= q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, TimesOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  tensor<int> expected({2, 3}, {1, 4, 9, 16, 25, 36});

  EXPECT_EQ(t * q, expected);
}

TEST(OperatorsTest, TimesValueOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  int         q = 2;

  tensor<int> expected({2, 3}, {2, 4, 6, 8, 10, 12});

  EXPECT_EQ(t * q, expected);
}

TEST(OperatorsTest, TimesEqualOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  tensor<int> expected({2, 3}, {1, 4, 9, 16, 25, 36});

  t *= q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, TimesEqualValueOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> expected({2, 3}, {2, 4, 6, 8, 10, 12});

  int q = 2;
  t *= q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, DivideOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  tensor<int> expected({2, 3}, {1, 1, 1, 1, 1, 1});

  EXPECT_EQ(t / q, expected);
}

TEST(OperatorsTest, DivideValueOperatorTest) {
  tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<float> expected({2, 3}, {0.5, 1, 1.5, 2, 2.5, 3});

  float q = 2;

  EXPECT_EQ(t / q, expected);
}

TEST(OperatorsTest, DivideValueOperatorExceptionTest) {
  tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
  float         q = 0;

  EXPECT_THROW(t / q, std::logic_error);
}

TEST(OperatorsTest, DivideEqualOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  tensor<int> expected({2, 3}, {1, 1, 1, 1, 1, 1});

  t /= q;

  EXPECT_EQ(t, expected);
}

TEST(OperatorsTest, DivideEqualValueOperatorTest) {
  tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<float> expected({2, 3}, {0.5, 1, 1.5, 2, 2.5, 3});

  float q = 2;
  t /= q;

  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_NEAR(t[i], expected[i], 1e-5);
  }
}

TEST(OperatorsTest, DivideEqualValueOperatorExceptionTest) {
  tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
  float         q = 0;

  EXPECT_THROW(t /= q, std::logic_error);
}