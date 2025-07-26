#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(DataTest, CountNonZeros) {
  tensor<int> t({10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
  tensor<int> q;
  tensor<int> a = t.zeros(t.shape());

  int expected  = 9;
  int expected1 = 0;
  int expected2 = 0;

  EXPECT_EQ(t.count_nonzero(0), expected);
  EXPECT_EQ(a.count_nonzero(0), expected2);
  EXPECT_EQ(q.count_nonzero(0), expected1);
}

TEST(DataTest, ZerosTest) {
  tensor<int> t;

  t.zeros_({10});

  EXPECT_EQ(t.count_nonzero(0), 0);
}

TEST(DataTest, RowTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> expected_row({3}, {1, 2, 3});
  tensor<int> expected_col({2}, {2, 5});

  EXPECT_EQ(t.row(0), expected_row);
  EXPECT_EQ(t.col(1), expected_col);
}

TEST(tensorTest, CloneTest) {
  tensor<float> t({2, 3}, {1.5, 2.7, 3.4, 1.1, 2.0, 0.9});
  tensor<float> q({1, 5}, {1.7, 9.7, 0.4, 4.1, 2.7});

  tensor<float> clone1 = t.clone();
  tensor<float> clone2 = q.clone();

  EXPECT_EQ(t.clone(), clone1);
  EXPECT_EQ(q.clone(), clone2);
}

TEST(DataTest, ReshapeBasic) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> expected({3, 2}, {1, 2, 3, 4, 5, 6});
  tensor<int> reshaped = t.reshape({3, 2});

  EXPECT_EQ(reshaped.shape(), expected.shape());
  for (std::size_t i = 0; i < reshaped.size(0); ++i)
  {
    EXPECT_EQ(reshaped[i], expected[i]);
  }
}

TEST(DataTest, ReshapeAs) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> ref({3, 2});
  tensor<int> expected({3, 2}, {1, 2, 3, 4, 5, 6});

  tensor<int> reshaped = t.reshape_as(ref);

  EXPECT_EQ(reshaped.shape(), expected.shape());
  for (std::size_t i = 0; i < reshaped.size(0); ++i)
  {
    EXPECT_EQ(reshaped[i], expected[i]);
  }
}

TEST(DataTest, FillWithScalar) {
  tensor<float> t({2, 3});
  auto          filled = t.fill(5.0f);

  for (std::size_t i = 0; i < filled.size(0); ++i)
  {
    EXPECT_EQ(filled[i], 5.0f);
  }
}

TEST(DataTest, FillWithTensor) {
  tensor<float> t1({2, 2}, {1, 2, 3, 4});
  tensor<float> t2({2, 2});

  auto filled = t2.fill(t1);
  EXPECT_EQ(filled.shape(), t1.shape());

  for (std::size_t i = 0; i < filled.size(0); ++i)
  {
    EXPECT_EQ(filled[i], t1[i]);
  }
}

TEST(DataTest, ResizeAs) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  auto        resized = t.resize_as({3, 2});

  EXPECT_EQ(resized.shape(), std::vector<unsigned long long>({3, 2}));
  for (std::size_t i = 0; i < resized.size(0); ++i)
  {
    EXPECT_EQ(resized[i], t[i]);
  }
}

TEST(DataTest, AllTrue) {
  tensor<int> t({3}, {1, 2, 3});
  auto        result = t.all();

  EXPECT_TRUE(result.shape().equal(std::vector<unsigned long long>{1}));
  EXPECT_TRUE(result[0]);
}

TEST(DataTest, AllFalse) {
  tensor<int> t({3}, {1, 0, 3});
  auto        result = t.all();

  EXPECT_FALSE(result[0]);
}

TEST(DataTest, AnyTrue) {
  tensor<int> t({4}, {0, 0, 1, 0});
  auto        result = t.any();

  EXPECT_TRUE(result[0]);
}

TEST(DataTest, AnyFalse) {
  tensor<int> t({3}, {0, 0, 0});
  auto        result = t.any();

  EXPECT_FALSE(result[0]);
}