#include "tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>

using ten = arch::tensor<double>;


TEST(TensorLog10Test, ScalarValue)
{
  ten  t({1}, {1000.0f});
  auto result = t.log10();

  EXPECT_FLOAT_EQ(result[0], 3.0f);
}

TEST(TensorLog10Test, ZeroValue)
{
  ten  t({1}, {0.0f});
  auto result = t.log10();

  EXPECT_TRUE(std::isinf(result[0]) && result[0] < 0);
}

TEST(TensorLog10Test, NegativeValue)
{
  ten  t({1}, {-10.0f});
  auto result = t.log10();

  EXPECT_TRUE(std::isnan(result[0]));
}

TEST(TensorLog10Test, OneValue)
{
  ten  t({1}, {1.0f});
  auto result = t.log10();

  EXPECT_FLOAT_EQ(result[0], 0.0f);
}

TEST(TensorLog10Test, PowersOfTen)
{
  ten  t({5}, {1, 10, 100, 1000, 10000});
  auto result = t.log10();

  for (size_t i = 0; i < 5; ++i)
  {
    EXPECT_FLOAT_EQ(result[i], static_cast<float>(i));
  }
}

TEST(TensorLog10Test, FloatPrecision)
{
  ten  t({3}, {1e-1f, 1e-2f, 1e-3f});
  auto result = t.log10();

  EXPECT_NEAR(result[0], -1.0f, 1e-6);
  EXPECT_NEAR(result[1], -2.0f, 1e-6);
  EXPECT_NEAR(result[2], -3.0f, 1e-6);
}

TEST(TensorLog10Test, DoublePrecision)
{
  ten  t({3}, {1e-5, 1e-6, 1e-7});
  auto result = t.log10();

  EXPECT_NEAR(result[0], -5.0, 1e-10);
  EXPECT_NEAR(result[1], -6.0, 1e-10);
  EXPECT_NEAR(result[2], -7.0, 1e-10);
}

TEST(TensorLog10Test, SmallValues)
{
  ten  t({3}, {0.1f, 0.01f, 0.001f});
  auto result = t.log10();

  EXPECT_NEAR(result[0], -1.0f, 1e-6);
  EXPECT_NEAR(result[1], -2.0f, 1e-6);
  EXPECT_NEAR(result[2], -3.0f, 1e-6);
}

TEST(TensorLog10Test, HighPrecisionRandom)
{
  ten  t({3}, {3.16227766, 31.6227766, 316.227766});
  auto result = t.log10();

  EXPECT_NEAR(result[0], 0.5, 1e-6);
  EXPECT_NEAR(result[1], 1.5, 1e-6);
  EXPECT_NEAR(result[2], 2.5, 1e-6);
}

TEST(TensorLog10Test, NegativeMixedValues)
{
  ten  t({5}, {-1.0f, 0.0f, 1.0f, 10.0f, -100.0f});
  auto result = t.log10();

  EXPECT_TRUE(std::isnan(result[0]));
  EXPECT_TRUE(std::isinf(result[1]));
  EXPECT_FLOAT_EQ(result[2], 0.0f);
  EXPECT_FLOAT_EQ(result[3], 1.0f);
  EXPECT_TRUE(std::isnan(result[4]));
}

TEST(TensorLog10Test, LargeMatrix)
{
  ten                t({2, 3}, {1.0f, 10.0f, 100.0f, 1000.0f, 0.1f, 0.01f});
  auto               result   = t.log10();
  std::vector<float> expected = {0, 1, 2, 3, -1, -2};

  for (int i = 0; i < 6; ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-5);
  }
}

TEST(TensorLog10Test, EmptyTensor)
{
  ten  t({0});
  auto result = t.log10();

  EXPECT_EQ(result.size(0), 0);
}

TEST(TensorLog10Test, MultiDim3D)
{
  ten                t({2, 2, 2}, {1, 10, 100, 1000, 0.1, 0.01, 0.001, 0.0001});
  auto               result   = t.log10();
  std::vector<float> expected = {0, 1, 2, 3, -1, -2, -3, -4};

  for (int i = 0; i < 8; ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-5);
  }
}

TEST(TensorLog10Test, InPlaceLog10IfSupported)
{
  ten t({3}, {10.0f, 100.0f, 1000.0f});
  t.log10_();  // in-place overwrite

  EXPECT_FLOAT_EQ(t[0], 1.0f);
  EXPECT_FLOAT_EQ(t[1], 2.0f);
  EXPECT_FLOAT_EQ(t[2], 3.0f);
}

TEST(TensorLog10Test, LargeValues)
{
  ten  t({2}, {1e10f, 1e20f});
  auto result = t.log10();

  EXPECT_NEAR(result[0], 10.0f, 1e-4);
  EXPECT_NEAR(result[1], 20.0f, 1e-4);
}

TEST(TensorLog10Test, WithNaNs)
{
  ten  t({3}, {std::nanf(""), 1.0f, 10.0f});
  auto result = t.log10();

  EXPECT_TRUE(std::isnan(result[0]));
  EXPECT_FLOAT_EQ(result[1], 0.0f);
  EXPECT_FLOAT_EQ(result[2], 1.0f);
}

TEST(TensorLog10Test, IntCasting)
{
  arch::tensor<int> t({3}, {1, 10, 100});
  auto              result = t.log10();

  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[1], 1.0f);
  EXPECT_FLOAT_EQ(result[2], 2.0f);
}

TEST(TensorLog10Test, VeryTinyValues)
{
  ten  t({2}, {1e-100, 1e-200});
  auto result = t.log10();

  EXPECT_NEAR(result[0], -100.0, 1e-8);
  EXPECT_NEAR(result[1], -200.0, 1e-8);
}

TEST(TensorLog10Test, BroadcastingSimulation)
{
  ten  t({2, 1}, {10.0f, 100.0f});
  auto result = t.log10();

  EXPECT_FLOAT_EQ(result[0], 1.0f);
  EXPECT_FLOAT_EQ(result[1], 2.0f);
}

TEST(TensorLog10Test, IdentityProperty)
{
  ten  t({5}, {std::pow(10.0f, 1), std::pow(10.0f, 2), std::pow(10.0f, 3), std::pow(10.0f, -1), std::pow(10.0f, -2)});
  auto result = t.log10();

  EXPECT_NEAR(result[0], 1.0f, 1e-6);
  EXPECT_NEAR(result[1], 2.0f, 1e-6);
  EXPECT_NEAR(result[2], 3.0f, 1e-6);
  EXPECT_NEAR(result[3], -1.0f, 1e-6);
  EXPECT_NEAR(result[4], -2.0f, 1e-6);
}