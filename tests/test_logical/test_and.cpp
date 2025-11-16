#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TestLogical, TensorTestAndFalse)
{
  arch::tensor<int> t({3, 2}, {0, 0, 0, 0, 0, 0});
  arch::tensor<int> other({3, 2}, {2, 3, 4, 5, 6, 7});
  arch::tensor<int> exp({3, 2}, {0, 0, 0, 0, 0, 0});

  EXPECT_EQ(t.logical_and(other), exp);
  EXPECT_EQ(t.logical_and_(other), exp);
}

TEST(TestLogical, TensorTestAndTrue)
{
  arch::tensor<int> t({3, 2}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> other({3, 2}, {1, 2, 3, 4, 5, 6});
  arch::tensor<int> exp({3, 2}, {1, 1, 1, 1, 1, 1});

  EXPECT_EQ(t.logical_and(other), exp);
  EXPECT_EQ(t.logical_and_(other), exp);
}

TEST(TestLogical, TensorTestScalarAndTrueAndFalse)
{
  arch::tensor<int> t({3, 2}, {0, 2, 3, 0, 5, 6});
  int               other = 1;
  arch::tensor<int> exp({3, 2}, {0, 1, 1, 0, 1, 1});

  EXPECT_EQ(t.logical_and(other), exp);
  EXPECT_EQ(t.logical_and_(other), exp);
}
