#include "../src/tensor.hpp"
#include <gtest/gtest.h>


/*
  TODO :
  this file should be broken down into smaller test files for each tensor method
*/

TEST(TensorTest, StorageTest)
{
  arch::tensor<int> t({5}, {1, 2, 3, 4, 5});

  std::vector<int> expected = {1, 2, 3, 4, 5};

  EXPECT_EQ(t.storage(), expected);
}

TEST(TensorTest, ShapeTest)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  EXPECT_EQ(t.shape(), (std::vector<unsigned long long>{2, 2}));
}

TEST(TensorTest, StridesTest)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  EXPECT_EQ(t.shape().strides().get(), (std::vector<uint64_t>{2, 1}));
}

TEST(TensorTest, DeviceTest)
{
  arch::tensor<int> t({4}, {1, 2, 3, 4});
  arch::tensor<int> q({4}, {1, 2, 3, 4}, Device::CUDA);

  EXPECT_EQ(t.device(), Device::CPU);
  EXPECT_EQ(q.device(), Device::CUDA);
}

TEST(TensorTest, NdimsTest)
{
  arch::tensor<int> t({4}, {1, 2, 3, 4});
  arch::tensor<int> q({1, 3}, {1, 2, 3});

  int expected_1 = 1;
  int expected_2 = 2;

  EXPECT_EQ(t.n_dims(), expected_1);
  EXPECT_EQ(q.n_dims(), expected_2);
}

TEST(TensorTest, SizeTest)
{
  arch::tensor<int> t({4}, {1, 2, 3, 4});
  arch::tensor<int> q({3, 3}, {1, 1, 8, 1, 3, 4, 5, 6, 6});

  int expected_total1 = 4;
  int expected_total2 = 9;

  EXPECT_EQ(t.size(1), expected_total1);
  EXPECT_EQ(q.size(0), expected_total2);
}

TEST(TensorTest, CapacityTest)
{
  arch::tensor<int> a({2, 5});
  arch::tensor<int> b({3, 3});
  arch::tensor<int> c({3, 6, 4, 5});
  arch::tensor<int> d({9, 10});
  arch::tensor<int> e({3, 4, 2});

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

TEST(TensorTest, LcmTest)
{
  arch::tensor<int> t({2, 2}, {1, 2, 3, 4});

  int expected = 12;

  EXPECT_EQ(t.lcm(), expected);
}

TEST(TensorTest, AtTest)
{
  arch::tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});

  int expected  = 2;
  int expected1 = 6;

  EXPECT_EQ(t.at({0, 1}), expected);
  EXPECT_EQ(t.at({1, 2}), expected1);
}

/*
TEST(TensorTest, SliceTest) {
    arch::tensor<int> t1({2, 2}, {1, 2, 3, 4});

    arch::tensor<int> expected1({1, 2}, {1, 2});
    EXPECT_EQ(t1.slice(0, 0, 1, 1), expected1);

    arch::tensor<int> expected2({1, 2}, {3, 4});
    EXPECT_EQ(t1.slice(0, 1, 2, 1), expected2);

    arch::tensor<int> expected3({2, 1}, {1, 3});
    EXPECT_EQ(t1.slice(1, 0, 1, 1), expected3);

    arch::tensor<int> expected4({2, 1}, {2, 4});
    EXPECT_EQ(t1.slice(1, 1, 2, 1), expected4);

    arch::tensor<int> expected5({1}, {3});
    EXPECT_EQ(t1.slice(0, 1, 2, 1).slice(1, 0, 1, 1), expected5);

    arch::tensor<int> t2({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    arch::tensor<int> expected6({2, 3}, {1, 2, 3, 7, 8, 9});
    EXPECT_EQ(t2.slice(0, 0, 3, 2), expected6);

    arch::tensor<int> expected7({1, 3}, {4, 5, 6});
    EXPECT_EQ(t2.slice(0, 1, 2, 1), expected7);

    arch::tensor<int> expected8({3, 1}, {2, 5, 8});
    EXPECT_EQ(t2.slice(1, 1, 2, 1), expected8);

    arch::tensor<int> expected9({1, 3}, {7, 8, 9});
    EXPECT_EQ(t2.slice(0, 2, 3, 1), expected9);
}
*/

TEST(TensorTest, CeilTest)
{
  arch::tensor<float> t({2, 3}, {1.5, 2.7, 3.4, 1.1, 2.0, 0.9});
  arch::tensor<float> q({1, 5}, {1.7, 9.7, 0.4, 4.1, 2.7});

  arch::tensor<float> expected1({2, 3}, {2.0, 3.0, 4.0, 2.0, 2.0, 1.0});
  arch::tensor<float> expected2({1, 5}, {2.0, 10.0, 1.0, 5.0, 3.0});

  EXPECT_EQ(t.ceil(), expected1);
  EXPECT_EQ(q.ceil(), expected2);
}

TEST(TensorTest, FloorTest)
{
  arch::tensor<float> t({2, 3}, {1.5, 2.7, 3.4, 1.1, 2.0, 0.9});
  arch::tensor<float> q({1, 5}, {1.7, 9.7, 0.4, 4.1, 2.7});

  arch::tensor<float> expected1({2, 3}, {1.0, 2.0, 3.0, 1.0, 2.0, 0.0});
  arch::tensor<float> expected2({1, 5}, {1.0, 9.0, 0.0, 4.0, 2.0});

  EXPECT_EQ(t.floor(), expected1);
  EXPECT_EQ(q.floor(), expected2);
}

TEST(TensorTest, AbsoluteValues)
{
  arch::tensor<int> t({4}, {-1, 0, 5, -7});
  arch::tensor<int> expected({4}, {1, 0, 5, 7});
  arch::tensor<int> result = t.absolute(t);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

/*
TEST(TensorTest, Determinant3x3) {
    arch::tensor<float> t({3, 3}, {6, 1, 1, 4, -2, 5, 2, 8, 7});
    auto          result = t.det();
    EXPECT_NEAR(result[0], 6 * (-2 * 7 - 5 * 8) - 1 * (4 * 7 - 5 * 2) + 1 * (4 * 8 + 2 * 2), 1e-5);
}
*/

TEST(TensorTest, Square)
{
  arch::tensor<int> t({3}, {2, -3, 4});
  auto              result = t.square();
  EXPECT_EQ(result.shape(), std::vector<unsigned long long>({3}));
  EXPECT_EQ(result[0], 4);
  EXPECT_EQ(result[1], 9);
  EXPECT_EQ(result[2], 16);
}

TEST(TensorTest, RemainderTest)
{
  arch::tensor<int> t({5}, {10, 20, 30, 40, 50});
  int               divisor = 7;
  auto              result  = t.remainder(divisor);

  arch::tensor<int> expected({5}, {3, 6, 2, 5, 1});

  EXPECT_EQ(result.shape(), expected.shape());
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, MaximumTest)
{
  arch::tensor<int> t({5}, {1, 3, 2, 5, 4});
  arch::tensor<int> other({5}, {2, 3, 1, 4, 5});
  auto              result = t.maximum(other);

  arch::tensor<int> expected({5}, {2, 3, 2, 5, 5});

  EXPECT_EQ(result.shape(), expected.shape());
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, DistTest)
{
  arch::tensor<float> t1({3}, {1.0f, 2.0f, 3.0f});
  arch::tensor<float> t2({3}, {4.0f, 5.0f, 6.0f});
  arch::tensor<float> expected_distance({3}, {3.0f, 3.0f, 3.0f});  // Euclidean distance
  auto                result = t1.dist(t2);

  EXPECT_EQ(result, expected_distance);
}

TEST(TensorTest, NegativeTest)
{
  arch::tensor<int> t({3}, {1, -2, 3});
  auto              negated = t.negative();

  arch::tensor<int> expected({3}, {-1, 2, -3});

  EXPECT_EQ(negated.shape(), expected.shape());
  for (std::size_t i = 0; i < negated.size(0); ++i)
  {
    EXPECT_EQ(negated[i], expected[i]);
  }
}

TEST(TensorTest, GCDTest)
{
  arch::tensor<int> t1({3}, {48, 64, 80});
  arch::tensor<int> t2({3}, {18, 24, 30});
  auto              result = t1.gcd(t2);

  arch::tensor<int> expected({3}, {6, 8, 10});

  EXPECT_EQ(result.shape(), expected.shape());
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}