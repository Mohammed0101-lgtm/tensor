#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(CompareTest, LessTest)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  arch::tensor<int> other({2, 2}, {1, 2, 3, 4});
  arch::tensor<int> other1({2, 2}, {2, 3, 4, 5});
  arch::tensor<int> other2({2, 2}, {0, 3, 2, 6});

  arch::tensor<bool> expected({2, 2}, {false, false, false, false});
  arch::tensor<bool> expected1({2, 2}, {true, true, true, true});
  arch::tensor<bool> expected2({2, 2}, {false, true, false, true});

  EXPECT_EQ(t.less(other), expected);
  EXPECT_EQ(t.less(other1), expected1);
  EXPECT_EQ(t.less(other2), expected2);
}

TEST(CompareTest, LessTest1)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  int other  = 1;
  int other1 = 2;
  int other2 = 3;

  arch::tensor<bool> expected({2, 2}, {false, false, false, false});
  arch::tensor<bool> expected1({2, 2}, {true, false, false, false});
  arch::tensor<bool> expected2({2, 2}, {true, true, false, false});

  EXPECT_EQ(t.less(other), expected);
  EXPECT_EQ(t.less(other1), expected1);
  EXPECT_EQ(t.less(other2), expected2);
}

TEST(CompareTest, EqualTest)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  arch::tensor<int> other({2, 2}, {1, 2, 3, 4});
  arch::tensor<int> other1({2, 2}, {2, 3, 4, 5});
  arch::tensor<int> other2({2, 2}, {0, 3, 2, 6});

  arch::tensor<bool> expected({2, 2}, {true, true, true, true});
  arch::tensor<bool> expected1({2, 2}, {false, false, false, false});
  arch::tensor<bool> expected2({2, 2}, {false, false, false, false});

  EXPECT_EQ(t.equal(other), expected);
  EXPECT_EQ(t.equal(other1), expected1);
  EXPECT_EQ(t.equal(other2), expected2);
}

TEST(CompareTest, EqualTest1)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  int other  = 1;
  int other1 = 2;
  int other2 = 3;

  arch::tensor<bool> expected({2, 2}, {true, false, false, false});
  arch::tensor<bool> expected1({2, 2}, {false, true, false, false});
  arch::tensor<bool> expected2({2, 2}, {false, false, true, false});

  EXPECT_EQ(t.equal(other), expected);
  EXPECT_EQ(t.equal(other1), expected1);
  EXPECT_EQ(t.equal(other2), expected2);
}

TEST(CompareTest, NotEqualTest)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  arch::tensor<int> other({2, 2}, {1, 2, 3, 4});
  arch::tensor<int> other1({2, 2}, {2, 3, 4, 5});
  arch::tensor<int> other2({2, 2}, {0, 3, 2, 6});

  arch::tensor<bool> expected({2, 2}, {false, false, false, false});
  arch::tensor<bool> expected1({2, 2}, {true, true, true, true});
  arch::tensor<bool> expected2({2, 2}, {true, true, true, true});

  EXPECT_EQ(t.not_equal(other), expected);
  EXPECT_EQ(t.not_equal(other1), expected1);
  EXPECT_EQ(t.not_equal(other2), expected2);
}

TEST(CompareTest, NotEqualTest1)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  int other  = 1;
  int other1 = 2;
  int other2 = 3;

  arch::tensor<bool> expected({2, 2}, {false, true, true, true});
  arch::tensor<bool> expected1({2, 2}, {true, false, true, true});
  arch::tensor<bool> expected2({2, 2}, {true, true, false, true});

  EXPECT_EQ(t.not_equal(other), expected);
  EXPECT_EQ(t.not_equal(other1), expected1);
  EXPECT_EQ(t.not_equal(other2), expected2);
}

TEST(CompareTest, GreaterTest)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  arch::tensor<int> other({2, 2}, {1, 2, 3, 4});
  arch::tensor<int> other1({2, 2}, {2, 3, 4, 5});
  arch::tensor<int> other2({2, 2}, {0, 3, 2, 6});

  arch::tensor<bool> expected({2, 2}, {false, false, false, false});
  arch::tensor<bool> expected1({2, 2}, {false, false, false, false});
  arch::tensor<bool> expected2({2, 2}, {true, false, true, false});

  EXPECT_EQ(t.greater(other), expected);
  EXPECT_EQ(t.greater(other1), expected1);
  EXPECT_EQ(t.greater(other2), expected2);
}

TEST(CompareTest, GreaterTest1)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  int other  = 1;
  int other1 = 2;
  int other2 = 3;

  arch::tensor<bool> expected({2, 2}, {false, true, true, true});
  arch::tensor<bool> expected1({2, 2}, {false, false, true, true});
  arch::tensor<bool> expected2({2, 2}, {false, false, false, true});

  EXPECT_EQ(t.greater(other), expected);
  EXPECT_EQ(t.greater(other1), expected1);
  EXPECT_EQ(t.greater(other2), expected2);
}

TEST(CompareTest, LessEqualTest)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  arch::tensor<int> other({2, 2}, {1, 2, 3, 4});
  arch::tensor<int> other1({2, 2}, {2, 3, 4, 5});
  arch::tensor<int> other2({2, 2}, {0, 3, 2, 6});

  arch::tensor<bool> expected({2, 2}, {true, true, true, true});
  arch::tensor<bool> expected1({2, 2}, {true, true, true, true});
  arch::tensor<bool> expected2({2, 2}, {false, true, false, true});

  EXPECT_EQ(t.less_equal(other), expected);
  EXPECT_EQ(t.less_equal(other1), expected1);
  EXPECT_EQ(t.less_equal(other2), expected2);
}

TEST(CompareTest, GreaterEqualTest)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  arch::tensor<int> other({2, 2}, {1, 2, 3, 4});
  arch::tensor<int> other1({2, 2}, {2, 3, 4, 5});
  arch::tensor<int> other2({2, 2}, {0, 3, 2, 6});

  arch::tensor<bool> expected({2, 2}, {true, true, true, true});
  arch::tensor<bool> expected1({2, 2}, {false, false, false, false});
  arch::tensor<bool> expected2({2, 2}, {true, false, true, false});

  EXPECT_EQ(t.greater_equal(other), expected);
  EXPECT_EQ(t.greater_equal(other1), expected1);
  EXPECT_EQ(t.greater_equal(other2), expected2);
}