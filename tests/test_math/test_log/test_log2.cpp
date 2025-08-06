#include "tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>


using ten = tensor<double>;


TEST(TensorLog2Test, ScalarValue) {
  ten  t({1}, {8.0f});
  auto result = t.log2();

  EXPECT_FLOAT_EQ(result[0], 3.0f);
}

TEST(TensorLog2Test, ZeroValue) {
  ten  t({1}, {0.0f});
  auto result = t.log2();

  EXPECT_TRUE(std::isinf(result[0]) && result[0] < 0);
}

TEST(TensorLog2Test, NegativeValue) {
  ten  t({1}, {-2.0f});
  auto result = t.log2();

  EXPECT_TRUE(std::isnan(result[0]));
}

TEST(TensorLog2Test, OneValue) {
  ten  t({1}, {1.0f});
  auto result = t.log2();

  EXPECT_FLOAT_EQ(result[0], 0.0f);
}

TEST(TensorLog2Test, PowersOfTwo) {
  ten  t({5}, {1, 2, 4, 8, 16});
  auto result = t.log2();

  for (size_t i = 0; i < 5; ++i)
  {
    EXPECT_FLOAT_EQ(result[i], static_cast<float>(i));
  }
}

TEST(TensorLog2Test, FloatPrecision) {
  ten  t({3}, {0.5f, 0.25f, 0.125f});
  auto result = t.log2();
  
  EXPECT_NEAR(result[0], -1.0f, 1e-6);
  EXPECT_NEAR(result[1], -2.0f, 1e-6);
  EXPECT_NEAR(result[2], -3.0f, 1e-6);
}

TEST(TensorLog2Test, DoublePrecision) {
  ten  t({3}, {1e-5, 1e-6, 1e-7});
  auto result = t.log2();
  
  EXPECT_NEAR(result[0], std::log2(1e-5), 1e-10);
  EXPECT_NEAR(result[1], std::log2(1e-6), 1e-10);
  EXPECT_NEAR(result[2], std::log2(1e-7), 1e-10);
}

TEST(TensorLog2Test, SmallValues) {
  ten  t({3}, {0.25f, 0.125f, 0.0625f});
  auto result = t.log2();
  
  EXPECT_NEAR(result[0], -2.0f, 1e-6);
  EXPECT_NEAR(result[1], -3.0f, 1e-6);
  EXPECT_NEAR(result[2], -4.0f, 1e-6);
}

TEST(TensorLog2Test, HighPrecisionRandom) {
  ten  t({3}, {sqrt(2), 8.0, 16.0});
  auto result = t.log2();
  
  EXPECT_NEAR(result[0], std::log2(sqrt(2)), 1e-10);
  EXPECT_NEAR(result[1], 3.0, 1e-10);
  EXPECT_NEAR(result[2], 4.0, 1e-10);
}

TEST(TensorLog2Test, NegativeMixedValues) {
  ten  t({5}, {-1.0f, 0.0f, 1.0f, 2.0f, -4.0f});
  auto result = t.log2();
  
  EXPECT_TRUE(std::isnan(result[0]));
  EXPECT_TRUE(std::isinf(result[1]));
  EXPECT_FLOAT_EQ(result[2], 0.0f);
  EXPECT_FLOAT_EQ(result[3], 1.0f);
  EXPECT_TRUE(std::isnan(result[4]));
}

TEST(TensorLog2Test, LargeMatrix) {
  ten                t({2, 3}, {1.0f, 2.0f, 4.0f, 8.0f, 0.5f, 0.25f});
  auto               result   = t.log2();
  std::vector<float> expected = {0, 1, 2, 3, -1, -2};

  for (int i = 0; i < 6; ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-5);
  }
}

TEST(TensorLog2Test, EmptyTensor) {
  ten  t({0});
  auto result = t.log2();
  
  EXPECT_EQ(result.size(0), 0);
}

TEST(TensorLog2Test, MultiDim3D) {
  ten                t({2, 2, 2}, {1, 2, 4, 8, 0.5, 0.25, 0.125, 0.0625});
  auto               result   = t.log2();
  std::vector<float> expected = {0, 1, 2, 3, -1, -2, -3, -4};
  
  for (int i = 0; i < 8; ++i)
    {
        EXPECT_NEAR(result[i], expected[i], 1e-5);
        }
}

TEST(TensorLog2Test, InPlaceLog2IfSupported) {
  ten t({3}, {2.0f, 4.0f, 8.0f});
  t.log2_(); 

  EXPECT_FLOAT_EQ(t[0], 1.0f);
  EXPECT_FLOAT_EQ(t[1], 2.0f);
  EXPECT_FLOAT_EQ(t[2], 3.0f);
}

TEST(TensorLog2Test, LargeValues) {
  ten  t({2}, {1e10f, 1e20f});
  auto result = t.log2();
  
  EXPECT_NEAR(result[0], std::log2(1e10f), 1e-4);
  EXPECT_NEAR(result[1], std::log2(1e20f), 1e-4);
}

TEST(TensorLog2Test, WithNaNs) {
  ten  t({3}, {std::nanf(""), 1.0f, 2.0f});
  auto result = t.log2();
  
  EXPECT_TRUE(std::isnan(result[0]));
  EXPECT_FLOAT_EQ(result[1], 0.0f);
  EXPECT_FLOAT_EQ(result[2], 1.0f);
}

TEST(TensorLog2Test, IntCasting) {
  tensor<int> t({3}, {1, 2, 4});
  auto        result = t.log2();
  
  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[1], 1.0f);
  EXPECT_FLOAT_EQ(result[2], 2.0f);
}

TEST(TensorLog2Test, VeryTinyValues) {
  ten  t({2}, {1e-100, 1e-200});
  auto result = t.log2();
  
  EXPECT_NEAR(result[0], std::log2(1e-100), 1e-8);
  EXPECT_NEAR(result[1], std::log2(1e-200), 1e-8);
}

TEST(TensorLog2Test, BroadcastingSimulation) {
  ten  t({2, 1}, {2.0f, 4.0f});
  auto result = t.log2();
  
  EXPECT_FLOAT_EQ(result[0], 1.0f);
  EXPECT_FLOAT_EQ(result[1], 2.0f);
}

TEST(TensorLog2Test, IdentityProperty) {
  ten  t({5}, {std::pow(2.0f, 1), std::pow(2.0f, 2), std::pow(2.0f, 3), std::pow(2.0f, -1), std::pow(2.0f, -2)});
  auto result = t.log2();
  
  EXPECT_NEAR(result[0], 1.0f, 1e-6);
  EXPECT_NEAR(result[1], 2.0f, 1e-6);
  EXPECT_NEAR(result[2], 3.0f, 1e-6);
  EXPECT_NEAR(result[3], -1.0f, 1e-6);
  EXPECT_NEAR(result[4], -2.0f, 1e-6);
}