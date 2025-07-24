#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(TypesTest, IntToInt32Conversion) {
  tensor<int>          t({3}, {1, 2, 3});
  tensor<int32_t>      result   = t.int32_();
  std::vector<int32_t> expected = {1, 2, 3};

  EXPECT_EQ(result.storage(), expected);
}

TEST(TypesTest, IntToInt16Conversion) {
  tensor<int>          t({3}, {1, 2, 3});
  tensor<int16_t>      result   = t.int16_();
  std::vector<int16_t> expected = {1, 2, 3};

  EXPECT_EQ(result.storage(), expected);
}

TEST(TypesTest, IntToUInt32Conversion) {
  tensor<int>           t({3}, {1, 2, 3});
  tensor<uint32_t>      result   = t.uint32_();
  std::vector<uint32_t> expected = {1, 2, 3};

  EXPECT_EQ(result.storage(), expected);
}

TEST(TypesTest, IntToUInt64Conversion) {
  tensor<int>           t({3}, {1, 2, 3});
  tensor<uint64_t>      result   = t.uint64_();
  std::vector<uint64_t> expected = {1, 2, 3};

  EXPECT_EQ(result.storage(), expected);
}

TEST(TypesTest, IntToFloat32Conversion) {
  tensor<int>            t({3}, {1, 2, 3});
  tensor<float32_t>      result   = t.float32_();
  std::vector<float32_t> expected = {1.0f, 2.0f, 3.0f};

  EXPECT_EQ(result.storage(), expected);
}

TEST(TypesTest, IntToFloat64Conversion) {
  tensor<int>            t({3}, {1, 2, 3});
  tensor<float64_t>      result   = t.float64_();
  std::vector<float64_t> expected = {1.0, 2.0, 3.0};

  EXPECT_EQ(result.storage(), expected);
}

TEST(TypesTest, IntToInt64Conversion) {
  tensor<int>          t({3}, {1, 2, 3});
  tensor<int64_t>      result   = t.int64_();
  std::vector<int64_t> expected = {1, 2, 3};

  EXPECT_EQ(result.storage(), expected);
}