#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(LinearTest, ClampWithinRange) {
  tensor<float> t({3}, {-2.0, 0.5, 3.0});
  tensor<float> result = t.clamp(0.0, 2.0);
  tensor<float> expected({3}, {0.0, 0.5, 2.0});
  EXPECT_EQ(result, expected);
}

TEST(LinearTest, ClampWithDefaultMinMax) {
  tensor<float> t({3}, {-100.0, 0.0, 100.0});
  tensor<float> result = t.clamp();
  EXPECT_EQ(result, t);
}

TEST(LinearTest, ClampNegativeOnly) {
  tensor<float> t({3}, {-5.0, -1.0, 0.0});
  tensor<float> result = t.clamp(-2.0, 0.0);
  tensor<float> expected({3}, {-2.0, -1.0, 0.0});
  EXPECT_EQ(result, expected);
}

TEST(LinearTest, MatmulBasic) {
  tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> B({3, 2}, {7, 8, 9, 10, 11, 12});

  tensor<int> expected(
    {2, 2}, {1 * 7 + 2 * 9 + 3 * 11, 1 * 8 + 2 * 10 + 3 * 12, 4 * 7 + 5 * 9 + 6 * 11, 4 * 8 + 5 * 10 + 6 * 12});

  tensor<int> result = A.matmul(B);
  EXPECT_EQ(result.shape(), expected.shape());
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(LinearTest, CrossProduct3D) {
  tensor<int> A({3}, {1, 2, 3});
  tensor<int> B({3}, {4, 5, 6});
  tensor<int> expected({3}, {2 * 6 - 3 * 5, 3 * 4 - 1 * 6, 1 * 5 - 2 * 4});
  tensor<int> result = A.cross_product(B);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(LinearTest, DotProduct) {
  tensor<int> A({3}, {1, 3, -5});
  tensor<int> B({3}, {4, -2, -1});
  int         dot_result = 1 * 4 + 3 * (-2) + (-5) * (-1);
  tensor<int> result     = A.dot(B);

  EXPECT_EQ(result.shape(), std::vector<unsigned long long>{1});
  EXPECT_EQ(result[0], dot_result);
}

TEST(LinearTest, Relu) {
  tensor<float> t({4}, {-1.0, 0.0, 2.5, -3.3});
  tensor<float> expected({4}, {0.0, 0.0, 2.5, 0.0});
  tensor<float> result = t.relu();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_FLOAT_EQ(result[i], expected[i]);
  }
}

TEST(LinearTest, Determinant2x2) {
  tensor<float> t({2, 2}, {4, 6, 3, 8});
  auto          result = t.det();

  EXPECT_NEAR(result[0], 4 * 8 - 6 * 3, 1e-5);
}

TEST(LinearTest, Sigmoid) {
  tensor<float> t({2}, {0.0f, 2.0f});
  auto          result = t.sigmoid();

  EXPECT_NEAR(result[0], 0.5f, 1e-5);
  EXPECT_NEAR(result[1], 1.0f / (1.0f + std::exp(-2.0f)), 1e-5);
}

TEST(LinearTest, ClippedReLU) {
  tensor<float>      t({6}, {-3.0f, -1.0f, 0.0f, 2.5f, 5.0f, 10.0f});
  float              clip_limit = 4.0f;
  std::vector<float> expected   = {0.0f, 0.0f, 0.0f, 2.5f, 4.0f, 4.0f};
  tensor<float>      result     = t.clipped_relu(clip_limit);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_FLOAT_EQ(result[i], expected[i]);
  }
}
