#include "tensor.hpp"
#include <gtest/gtest.h>
#include <iostream>


TEST(TensorTest, DataTypeConversionTest) {
  tensor<int>       t({3}, {1, 2, 3});
  tensor<int32_t>   t_int       = t.int32_();
  tensor<int16_t>   t_short     = t.int16_();
  tensor<uint32_t>  t_uint      = t.uint32_();
  tensor<uint64_t>  t_ulong     = t.uint64_();
  tensor<float32_t> t_float     = t.float32_();
  tensor<float64_t> t_double    = t.float64_();
  tensor<int64_t>   t_long_long = t.int64_();

  EXPECT_EQ(t_int.storage(), (std::vector<int32_t>{1, 2, 3}));
  EXPECT_EQ(t_uint.storage(), (std::vector<uint32_t>{1, 2, 3}));
  EXPECT_EQ(t_ulong.storage(), (std::vector<uint64_t>{1, 2, 3}));
  EXPECT_EQ(t_float.storage(), (std::vector<float32_t>{1.0, 2.0, 3.0}));
  EXPECT_EQ(t_double.storage(), (std::vector<float64_t>{1.0, 2.0, 3.0}));
  EXPECT_EQ(t_short.storage(), std::vector<int16_t>({1, 2, 3}));
  EXPECT_EQ(t_long_long.storage(), std::vector<int64_t>({1, 2, 3}));
}
