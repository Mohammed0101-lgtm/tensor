#include "src/tensor.hpp"
#include <gtest/gtest.h>


TEST(TensorTest, StorageTest) {
  tensor<int> t({5}, {1, 2, 3, 4, 5});

  std::vector<int> expected = {1, 2, 3, 4, 5};

  EXPECT_EQ(t.storage(), expected);
}

TEST(TensorTest, IteratorsTest) {
  tensor<int> t({5}, {1, 2, 3, 4, 5});
  auto        it = t.begin();
  ++it;

  EXPECT_EQ(*t.begin(), 1);
  EXPECT_EQ(*(t.end() - 1), 5);

  EXPECT_EQ(*it, 2);
}

TEST(TensorTest, ConstIteratorsTest) {
  const tensor<int> t({3}, {10, 20, 30});

  EXPECT_EQ(*t.begin(), 10);
  EXPECT_EQ(*(t.end() - 1), 30);
}

TEST(TensorTest, ReverseIteratorsTest) {
  tensor<int> t({5}, {1, 2, 3, 4, 5});

  EXPECT_EQ(*t.rbegin(), 5);
  EXPECT_EQ(*(t.rend() - 1), 1);
}

TEST(TensorTest, ConstReverseIteratorsTest) {
  const tensor<int> t({3}, {10, 20, 30});

  EXPECT_EQ(*t.rbegin(), 30);
  EXPECT_EQ(*(t.rend() - 1), 10);
}

TEST(TensorTest, DataTypeConversionTest) {
  tensor<int> t({3}, {1, 2, 3});

  tensor<long long>          t_long    = t.long_();
  tensor<int32_t>            t_int32   = t.int32_();
  tensor<uint32_t>           t_uint32  = t.uint32_();
  tensor<unsigned long long> t_ulong   = t.unsigned_long_();
  tensor<float32_t>          t_float32 = t.float32_();
  tensor<float64_t>          t_double  = t.double_();

  EXPECT_EQ(t_long.storage(), (std::vector<long long>{1, 2, 3}));
  EXPECT_EQ(t_int32.storage(), (std::vector<int32_t>{1, 2, 3}));
  EXPECT_EQ(t_uint32.storage(), (std::vector<uint32_t>{1, 2, 3}));
  EXPECT_EQ(t_ulong.storage(), (std::vector<unsigned long long>{1, 2, 3}));
  EXPECT_EQ(t_float32.storage(), (std::vector<float32_t>{1.0, 2.0, 3.0}));
  EXPECT_EQ(t_double.storage(), (std::vector<float64_t>{1.0, 2.0, 3.0}));
}

TEST(TensorTest, ShapeTest) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  EXPECT_EQ(t.shape(), (std::vector<long long>{2, 2}));
}

TEST(TensorTest, StridesTest) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  EXPECT_EQ(t.strides(), (std::vector<long long>{2, 1}));
}

TEST(TensorTest, DeviceTest) {
  tensor<int> t({4}, {1, 2, 3, 4});
  tensor<int> q({4}, {1, 2, 3, 4}, Device::CUDA);

  EXPECT_EQ(t.device(), Device::CPU);
  EXPECT_EQ(q.device(), Device::CUDA);
}

TEST(TensorTest, NdimsTest) {
  tensor<int> t({4}, {1, 2, 3, 4});
  tensor<int> q({1, 3}, {1, 2, 3});

  int expected_1 = 1;
  int expected_2 = 2;

  EXPECT_EQ(t.n_dims(), expected_1);
  EXPECT_EQ(q.n_dims(), expected_2);
}

TEST(TensorTest, SizeTest) {
  tensor<int> t({4}, {1, 2, 3, 4});
  tensor<int> q({3, 3}, {1, 1, 8, 1, 3, 4, 5, 6, 6});

  int expected_total1 = 4;
  int expected_total2 = 9;

  EXPECT_EQ(t.size(1), expected_total1);
  EXPECT_EQ(q.size(0), expected_total2);
}

TEST(TensorTest, CapacityTest) {
  tensor<int> a({2, 5});
  tensor<int> b({3, 3});
  tensor<int> c({3, 6, 4, 5});
  tensor<int> d({9, 10});
  tensor<int> e({3, 4, 2});

  int expected1 = 10;
  int expected2 = 9;
  int expected3 = 360;
  int expected4 = 90;
  int expected5 = 24;

  EXPECT_EQ(a.capacity(), expected1);
  EXPECT_EQ(b.capacity(), expected2);
  EXPECT_EQ(c.capacity(), expected3);
  EXPECT_EQ(d.capacity(), expected4);
  EXPECT_EQ(e.capacity(), expected5);
}

TEST(TensorTest, CountNonZeros) {
  tensor<int> t({10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
  tensor<int> q;
  tensor<int> a = t.zeros(t.shape());

  int expected  = 9;
  int expected1 = 0;
  int expected2 = 0;

  EXPECT_EQ(t.count_nonzero(), expected);
  EXPECT_EQ(a.count_nonzero(), expected2);
  EXPECT_EQ(q.count_nonzero(), expected1);
}

TEST(TensorTest, ZerosTest) {
  tensor<int> t;

  t.zeros_({10});

  EXPECT_EQ(t.count_nonzero(), 0);
}

TEST(TensorTest, LcmTest) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  int expected = 12;

  EXPECT_EQ(t.lcm(), expected);
}

TEST(TensorTest, AtTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});

  int expected  = 2;
  int expected1 = 6;

  EXPECT_EQ(t.at({0, 1}), expected);
  EXPECT_EQ(t.at({1, 2}), expected1);
}

TEST(TensorTest, LinearAccessOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});

  int expected = 4;

  EXPECT_EQ(t[3], expected);
}

TEST(TensorTest, MultiDimensionalAccessOperator) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});

  int expected  = 6;
  int expected1 = 2;

  EXPECT_EQ(t({0, 1}), expected1);
  EXPECT_EQ(t({1, 2}), expected);
}

TEST(TensorTest, EmptyTest) {
  tensor<int> t;
  tensor<int> q({1, 2}, {1, 2});

  EXPECT_TRUE(t.empty());
  EXPECT_FALSE(q.empty());
}

TEST(TensorTest, EqualOperatorTest) {
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

TEST(TensorTest, NotEqualOperatorTest) {
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}