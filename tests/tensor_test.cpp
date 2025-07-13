#include "../src/tensor.hpp"
#include <gtest/gtest.h>
#include <iostream>

/*
  TODO :
  this file should be broken down into smaller test files for each tensor method
*/

TEST(TensorTest, StorageTest) {
  tensor<int> t({5}, {1, 2, 3, 4, 5});

  std::vector<int> expected = {1, 2, 3, 4, 5};

  EXPECT_EQ(t.storage(), expected);
}

TEST(TensorTest, DataTypeConversionTest) {
  tensor<int>       t({3}, {1, 2, 3});
  tensor<int32_t>   t_int       = t.int32_();
  tensor<uint32_t>  t_uint      = t.uint32_();
  tensor<uint64_t>  t_ulong     = t.uint64_();
  tensor<float32_t> t_float     = t.float32_();
  tensor<float64_t> t_double    = t.float64_();
  tensor<int16_t>   t_short     = t.int16_();
  tensor<int64_t>   t_long_long = t.int64_();

  EXPECT_EQ(t_int.storage(), (std::vector<int32_t>{1, 2, 3}));
  EXPECT_EQ(t_uint.storage(), (std::vector<uint32_t>{1, 2, 3}));
  EXPECT_EQ(t_ulong.storage(), (std::vector<uint64_t>{1, 2, 3}));
  EXPECT_EQ(t_float.storage(), (std::vector<float32_t>{1.0, 2.0, 3.0}));
  EXPECT_EQ(t_double.storage(), (std::vector<float64_t>{1.0, 2.0, 3.0}));
  EXPECT_EQ(t_short.storage(), std::vector<int16_t>({1, 2, 3}));
  EXPECT_EQ(t_long_long.storage(), std::vector<int64_t>({1, 2, 3}));
}

TEST(TensorTest, ShapeTest) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  EXPECT_EQ(t.shape(), (std::vector<unsigned long long>{2, 2}));
}

TEST(TensorTest, StridesTest) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  EXPECT_EQ(t.shape().strides().get(), (std::vector<uint64_t>{2, 1}));
}

TEST(TensorTest, DeviceTest) {
  tensor<int> t({4}, {1, 2, 3, 4});
  tensor<int> q({4}, {1, 2, 3, 4}, tensor<int>::Device::CUDA);

  EXPECT_EQ(t.device(), tensor<int>::Device::CPU);
  EXPECT_EQ(q.device(), tensor<int>::Device::CUDA);
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

  EXPECT_EQ(t.count_nonzero(0), expected);
  EXPECT_EQ(a.count_nonzero(0), expected2);
  EXPECT_EQ(q.count_nonzero(0), expected1);
}

TEST(TensorTest, ZerosTest) {
  tensor<int> t;

  t.zeros_({10});

  EXPECT_EQ(t.count_nonzero(0), 0);
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
  tensor<int> ten({2, 3}, {1, 2, 3, 4, 5, 6});

  int expected  = 6;
  int expected1 = 2;

  EXPECT_EQ(ten({0, 1}), expected1);
  EXPECT_EQ(ten({1, 2}), expected);
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

TEST(TensorTest, PlusOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  tensor<int> expected({2, 3}, {2, 4, 6, 8, 10, 12});

  EXPECT_EQ(t + q, expected);
}

TEST(TensorTest, MinusOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  tensor<int> expected({2, 3}, {0, 0, 0, 0, 0, 0});

  EXPECT_EQ(t - q, expected);
}

TEST(TensorTest, TimesOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  tensor<int> expected({2, 3}, {1, 4, 9, 16, 25, 36});

  EXPECT_EQ(t * q, expected);
}

TEST(TensorTest, DivideOperatorTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> q({2, 3}, {1, 2, 3, 4, 5, 6});

  tensor<int> expected({2, 3}, {1, 1, 1, 1, 1, 1});

  EXPECT_EQ(t / q, expected);
}

TEST(TensorTest, BoolTest) {
  tensor<int>  vals({2, 2}, {1, 2, 0, 0});
  tensor<bool> bools = vals.bool_();
  tensor<bool> expected({2, 2}, {true, true, false, false});

  EXPECT_EQ(bools, expected);
}

TEST(TensorTest, LogicalTest) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  tensor<bool> bool_       = t.bool_();
  tensor<bool> logical_not = bool_.logical_not();
  tensor<bool> logical_or  = bool_.logical_or(true);
  tensor<bool> logical_and = bool_.logical_and(false);
  tensor<bool> logical_xor = bool_.logical_xor(false);

  EXPECT_EQ(logical_not, tensor<bool>({2, 2}, {false, false, false, false}));
  EXPECT_EQ(logical_or, tensor<bool>({2, 2}, {true, true, true, true}));
  EXPECT_EQ(logical_xor, tensor<bool>({2, 2}, {false, false, false, false}));
  EXPECT_EQ(logical_and, tensor<bool>({2, 2}, {false, false, false, false}));
}

TEST(TensorTest, LessEqualTest) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  tensor<int> other({2, 2}, {1, 2, 3, 4});
  tensor<int> other1({2, 2}, {2, 3, 4, 5});
  tensor<int> other2({2, 2}, {0, 3, 2, 6});

  tensor<bool> expected({2, 2}, {true, true, true, true});
  tensor<bool> expected1({2, 2}, {true, true, true, true});
  tensor<bool> expected2({2, 2}, {false, true, false, true});

  EXPECT_EQ(t.less_equal(other), expected);
  EXPECT_EQ(t.less_equal(other1), expected1);
  EXPECT_EQ(t.less_equal(other2), expected2);
}

TEST(TensorTest, GreaterEqualTest) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  tensor<int> other({2, 2}, {1, 2, 3, 4});
  tensor<int> other1({2, 2}, {2, 3, 4, 5});
  tensor<int> other2({2, 2}, {0, 3, 2, 6});

  tensor<bool> expected({2, 2}, {true, true, true, true});
  tensor<bool> expected1({2, 2}, {false, false, false, false});
  tensor<bool> expected2({2, 2}, {true, false, true, false});

  EXPECT_EQ(t.greater_equal(other), expected);
  EXPECT_EQ(t.greater_equal(other1), expected1);
  EXPECT_EQ(t.greater_equal(other2), expected2);
}

TEST(TensorTest, EqualTest) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  tensor<int> other({2, 2}, {1, 2, 3, 4});
  tensor<int> other1({2, 2}, {2, 3, 4, 5});
  tensor<int> other2({2, 2}, {0, 3, 2, 6});

  tensor<bool> expected({2, 2}, {true, true, true, true});
  tensor<bool> expected1({2, 2}, {false, false, false, false});
  tensor<bool> expected2({2, 2}, {false, false, false, false});

  EXPECT_EQ(t.equal(other), expected);
  EXPECT_EQ(t.equal(other1), expected1);
  EXPECT_EQ(t.equal(other2), expected2);
}

TEST(TensorTest, EqualTest1) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  int other  = 1;
  int other1 = 2;
  int other2 = 3;

  tensor<bool> expected({2, 2}, {true, false, false, false});
  tensor<bool> expected1({2, 2}, {false, true, false, false});
  tensor<bool> expected2({2, 2}, {false, false, true, false});

  EXPECT_EQ(t.equal(other), expected);
  EXPECT_EQ(t.equal(other1), expected1);
  EXPECT_EQ(t.equal(other2), expected2);
}

TEST(TensorTest, NotEqualTest) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  tensor<int> other({2, 2}, {1, 2, 3, 4});
  tensor<int> other1({2, 2}, {2, 3, 4, 5});
  tensor<int> other2({2, 2}, {0, 3, 2, 6});

  tensor<bool> expected({2, 2}, {false, false, false, false});
  tensor<bool> expected1({2, 2}, {true, true, true, true});
  tensor<bool> expected2({2, 2}, {true, true, true, true});

  EXPECT_EQ(t.not_equal(other), expected);
  EXPECT_EQ(t.not_equal(other1), expected1);
  EXPECT_EQ(t.not_equal(other2), expected2);
}

TEST(TensorTest, NotEqualTest1) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  int other  = 1;
  int other1 = 2;
  int other2 = 3;

  tensor<bool> expected({2, 2}, {false, true, true, true});
  tensor<bool> expected1({2, 2}, {true, false, true, true});
  tensor<bool> expected2({2, 2}, {true, true, false, true});

  EXPECT_EQ(t.not_equal(other), expected);
  EXPECT_EQ(t.not_equal(other1), expected1);
  EXPECT_EQ(t.not_equal(other2), expected2);
}

TEST(TensorTest, LessTest) {
  tensor<int>  t({2, 2}, {1, 2, 3, 4});
  tensor<int>  other({2, 2}, {1, 2, 3, 4});
  tensor<int>  other1({2, 2}, {2, 3, 4, 5});
  tensor<int>  other2({2, 2}, {0, 3, 2, 6});
  tensor<bool> expected({2, 2}, {false, false, false, false});
  tensor<bool> expected1({2, 2}, {true, true, true, true});
  tensor<bool> expected2({2, 2}, {false, true, false, true});

  EXPECT_EQ(t.less(other), expected);
  EXPECT_EQ(t.less(other1), expected1);
  EXPECT_EQ(t.less(other2), expected2);
}

TEST(TensorTest, LessTest1) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  int other  = 1;
  int other1 = 2;
  int other2 = 3;

  tensor<bool> expected({2, 2}, {false, false, false, false});
  tensor<bool> expected1({2, 2}, {true, false, false, false});
  tensor<bool> expected2({2, 2}, {true, true, false, false});

  EXPECT_EQ(t.less(other), expected);
  EXPECT_EQ(t.less(other1), expected1);
  EXPECT_EQ(t.less(other2), expected2);
}

TEST(TensorTest, GreaterTest) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  tensor<int> other({2, 2}, {1, 2, 3, 4});
  tensor<int> other1({2, 2}, {2, 3, 4, 5});
  tensor<int> other2({2, 2}, {0, 3, 2, 6});

  tensor<bool> expected({2, 2}, {false, false, false, false});
  tensor<bool> expected1({2, 2}, {false, false, false, false});
  tensor<bool> expected2({2, 2}, {true, false, true, false});

  EXPECT_EQ(t.greater(other), expected);
  EXPECT_EQ(t.greater(other1), expected1);
  EXPECT_EQ(t.greater(other2), expected2);
}

TEST(TensorTest, GreaterTest1) {
  tensor<int> t({2, 2}, {1, 2, 3, 4});

  int other  = 1;
  int other1 = 2;
  int other2 = 3;

  tensor<bool> expected({2, 2}, {false, true, true, true});
  tensor<bool> expected1({2, 2}, {false, false, true, true});
  tensor<bool> expected2({2, 2}, {false, false, false, true});

  EXPECT_EQ(t.greater(other), expected);
  EXPECT_EQ(t.greater(other1), expected1);
  EXPECT_EQ(t.greater(other2), expected2);
}
/*
TEST(TensorTest, SliceTest) {
    tensor<int> t1({2, 2}, {1, 2, 3, 4});

    tensor<int> expected1({1, 2}, {1, 2});
    EXPECT_EQ(t1.slice(0, 0, 1, 1), expected1);

    tensor<int> expected2({1, 2}, {3, 4});
    EXPECT_EQ(t1.slice(0, 1, 2, 1), expected2);

    tensor<int> expected3({2, 1}, {1, 3});
    EXPECT_EQ(t1.slice(1, 0, 1, 1), expected3);

    tensor<int> expected4({2, 1}, {2, 4});
    EXPECT_EQ(t1.slice(1, 1, 2, 1), expected4);

    tensor<int> expected5({1}, {3});
    EXPECT_EQ(t1.slice(0, 1, 2, 1).slice(1, 0, 1, 1), expected5);

    tensor<int> t2({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    tensor<int> expected6({2, 3}, {1, 2, 3, 7, 8, 9});
    EXPECT_EQ(t2.slice(0, 0, 3, 2), expected6);

    tensor<int> expected7({1, 3}, {4, 5, 6});
    EXPECT_EQ(t2.slice(0, 1, 2, 1), expected7);

    tensor<int> expected8({3, 1}, {2, 5, 8});
    EXPECT_EQ(t2.slice(1, 1, 2, 1), expected8);

    tensor<int> expected9({1, 3}, {7, 8, 9});
    EXPECT_EQ(t2.slice(0, 2, 3, 1), expected9);
}
*/

TEST(TensorTest, RowTest) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> expected_row({3}, {1, 2, 3});
  tensor<int> expected_col({2}, {2, 5});

  EXPECT_EQ(t.row(0), expected_row);
  EXPECT_EQ(t.col(1), expected_col);
}

TEST(TensorTest, BoolRowTest) {
  tensor<int> t({2, 3}, {true, false, true, false, false, true});
  tensor<int> expected_row({3}, {true, false, true});
  tensor<int> expected_col({2}, {false, false});

  EXPECT_EQ(t.row(0), expected_row);
  EXPECT_EQ(t.col(1), expected_col);
}

TEST(TensorTest, CeilTest) {
  tensor<float> t({2, 3}, {1.5, 2.7, 3.4, 1.1, 2.0, 0.9});
  tensor<float> q({1, 5}, {1.7, 9.7, 0.4, 4.1, 2.7});

  tensor<float> expected1({2, 3}, {2.0, 3.0, 4.0, 2.0, 2.0, 1.0});
  tensor<float> expected2({1, 5}, {2.0, 10.0, 1.0, 5.0, 3.0});

  EXPECT_EQ(t.ceil(), expected1);
  EXPECT_EQ(q.ceil(), expected2);
}

TEST(TensorTest, FloorTest) {
  tensor<float> t({2, 3}, {1.5, 2.7, 3.4, 1.1, 2.0, 0.9});
  tensor<float> q({1, 5}, {1.7, 9.7, 0.4, 4.1, 2.7});

  tensor<float> expected1({2, 3}, {1.0, 2.0, 3.0, 1.0, 2.0, 0.0});
  tensor<float> expected2({1, 5}, {1.0, 9.0, 0.0, 4.0, 2.0});

  EXPECT_EQ(t.floor(), expected1);
  EXPECT_EQ(q.floor(), expected2);
}

TEST(tensorTest, CloneTest) {
  tensor<float> t({2, 3}, {1.5, 2.7, 3.4, 1.1, 2.0, 0.9});
  tensor<float> q({1, 5}, {1.7, 9.7, 0.4, 4.1, 2.7});

  tensor<float> clone1 = t.clone();
  tensor<float> clone2 = q.clone();

  EXPECT_EQ(t.clone(), clone1);
  EXPECT_EQ(q.clone(), clone2);
}

TEST(TensorTest, ClampWithinRange) {
  tensor<float> t({3}, {-2.0, 0.5, 3.0});
  tensor<float> result = t.clamp(0.0, 2.0);
  tensor<float> expected({3}, {0.0, 0.5, 2.0});
  EXPECT_EQ(result, expected);
}

TEST(TensorTest, ClampWithDefaultMinMax) {
  tensor t({3}, {-100.0, 0.0, 100.0});
  tensor result = t.clamp();
  EXPECT_EQ(result, t);
}

TEST(TensorTest, ClampNegativeOnly) {
  tensor<float> t({3}, {-5.0, -1.0, 0.0});
  tensor<float> result = t.clamp(-2.0, 0.0);
  tensor<float> expected({3}, {-2.0, -1.0, 0.0});
  EXPECT_EQ(result, expected);
}

TEST(TensorTest, CosValues) {
  tensor<float> t({3}, {0.0, M_PI / 2, M_PI});
  tensor<float> result = t.cos();
  tensor<float> expected({3}, {1.0, 0.0, -1.0});
  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(TensorTest, CoshValues) {
  tensor<float>  t({2}, {0.0, 1.0});
  tensor<float>  result = t.cosh();
  tensor<double> expected({2}, {std::cosh(0.0), std::cosh(1.0)});
  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(TensorTest, AcosValues) {
  tensor<float>  t({3}, {-1.0, 0.0, 1.0});
  tensor<float>  result = t.acos();
  tensor<double> expected({3}, {M_PI, M_PI / 2, 0.0});
  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(TensorTest, AcosOutOfDomain) {
  tensor<float> t({2}, {-2.0, 2.0});

  EXPECT_THROW(t.acos(), std::domain_error);
}

TEST(TensorTest, AcoshValidRange) {
  tensor<float>  t({3}, {1.0, 2.0, 10.0});
  tensor<float>  result = t.acosh();
  tensor<double> expected({3}, {std::acosh(1.0), std::acosh(2.0), std::acosh(10.0)});
  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(TensorTest, AcoshOutOfDomain) {
  tensor<float> t({1}, {0.5});
  EXPECT_THROW(t.acosh(), std::domain_error);
}

TEST(TensorTest, TanValues) {
  tensor<float>  t({2}, {0.0, M_PI / 4});
  tensor<float>  result = t.tan();
  tensor<double> expected({2}, {std::tan(0.0), std::tan(M_PI / 4)});
  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(TensorTest, TanhBasic) {
  tensor<double>      t({3}, {-2.0, 0.0, 2.0});
  tensor<double>      result   = t.tanh();
  std::vector<double> expected = {std::tanh(-2.0), std::tanh(0.0), std::tanh(2.0)};
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-7);
  }
}

TEST(TensorTest, AtanBasic) {
  tensor<double>      t({3}, {-1.0, 0.0, 1.0});
  tensor<double>      result   = t.atan();
  std::vector<double> expected = {std::atan(-1.0), std::atan(0.0), std::atan(1.0)};
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-7);
  }
}

TEST(TensorTest, AtanhInDomain) {
  tensor<double>      t({3}, {-0.5, 0.0, 0.5});
  tensor<double>      result   = t.atanh();
  std::vector<double> expected = {std::atanh(-0.5), std::atanh(0.0), std::atanh(0.5)};

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-7);
  }
}

TEST(TensorTest, AtanhOutOfDomain) {
  tensor<double> t({2}, {-1.5, 2.0});
  EXPECT_THROW(t.atanh(), std::domain_error);
}

TEST(TensorTest, SinBasic) {
  tensor<double>      t({3}, {0.0, M_PI / 2, M_PI});
  tensor<double>      result   = t.sin();
  std::vector<double> expected = {0.0, 1.0, 0.0};
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(TensorTest, InplaceSincBasic) {
  tensor<double> t({3}, {0.0, 0.5, 1.0});
  t.sinc_();
  std::vector<double> expected = {1.0, std::sin(M_PI * 0.5) / (M_PI * 0.5), std::sin(M_PI * 1.0) / (M_PI * 1.0)};
  for (std::size_t i = 0; i < t.size(0); ++i)
  {
    EXPECT_NEAR(t[i], expected[i], 1e-7);
  }
}

TEST(TensorTest, SinhBasic) {
  tensor<double>      t({3}, {-1.0, 0.0, 1.0});
  tensor<double>      result   = t.sinh();
  std::vector<double> expected = {std::sinh(-1.0), std::sinh(0.0), std::sinh(1.0)};
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-7);
  }
}

TEST(TensorTest, EmptyTensorOps) {
  tensor<double> t;
  EXPECT_NO_THROW(t.sin());
  EXPECT_NO_THROW(t.tanh());
  EXPECT_EQ(t.sin().size(0), 0);
}

TEST(TensorTest, AsinBasic) {
  tensor<double> t({3}, {0.0, 0.5, 1.0});
  tensor<double> expected({3}, {std::asin(0.0), std::asin(0.5), std::asin(1.0)});
  tensor<double> result = t.asin();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}

TEST(TensorTest, AsinNegativeInputs) {
  tensor<double> t({3}, {-0.5, -1.0, -0.25});
  tensor<double> expected({3}, {std::asin(-0.5), std::asin(-1.0), std::asin(-0.25)});
  tensor<double> result = t.asin();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}

TEST(TensorTest, AsinOutOfDomain) {
  tensor<double> t({2}, {1.5, -1.5});
  EXPECT_THROW(t.asin(), std::domain_error);
}

TEST(TensorTest, AsinhBasic) {
  tensor<double> t({4}, {0.0, 0.5, 1.0, 2.0});
  tensor<double> expected({4}, {std::asinh(0.0), std::asinh(0.5), std::asinh(1.0), std::asinh(2.0)});
  tensor<double> result = t.asinh();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}

TEST(TensorTest, AsinhNegative) {
  tensor<double> t({3}, {-0.5, -1.0, -2.0});
  tensor<double> expected({3}, {std::asinh(-0.5), std::asinh(-1.0), std::asinh(-2.0)});
  tensor<double> result = t.asinh();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}

TEST(TensorTest, AbsBasic) {
  tensor<double> t({6}, {-2.5, -1.0, 0.0, 0.5, 1.0, 2.5});
  tensor<double> expected({6}, {2.5, 1.0, 0.0, 0.5, 1.0, 2.5});
  tensor<double> result = t.abs();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, AbsWithInts) {
  tensor<int> t({4}, {-3, -1, 0, 5});
  tensor<int> expected({4}, {3, 1, 0, 5});
  tensor<int> result = t.abs();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseNot) {
  tensor<int> t({3}, {0b0000, 0b1010, 0b1111});
  tensor<int> expected({3}, {~0b0000, ~0b1010, ~0b1111});
  tensor<int> result = t.bitwise_not();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseAndScalar) {
  tensor<int> t({3}, {0b1010, 0b1100, 0b1111});
  int         value = 0b0101;
  tensor<int> expected({3}, {0b1010 & value, 0b1100 & value, 0b1111 & value});
  tensor<int> result = t.bitwise_and(value);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseAndTensor) {
  tensor<int> t1({3}, {0b1010, 0b1100, 0b1111});
  tensor<int> t2({3}, {0b0110, 0b1010, 0b0001});
  tensor<int> expected({3}, {0b1010 & 0b0110, 0b1100 & 0b1010, 0b1111 & 0b0001});
  tensor<int> result = t1.bitwise_and(t2);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseOrScalar) {
  tensor<int> t({3}, {0b0001, 0b1000, 0b1111});
  int         value = 0b0101;
  tensor<int> expected({3}, {0b0001 | value, 0b1000 | value, 0b1111 | value});
  tensor<int> result = t.bitwise_or(value);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseOrTensor) {
  tensor<int> t1({3}, {0b0001, 0b1000, 0b1111});
  tensor<int> t2({3}, {0b0110, 0b0011, 0b0000});
  tensor<int> expected({3}, {0b0001 | 0b0110, 0b1000 | 0b0011, 0b1111 | 0b0000});
  tensor<int> result = t1.bitwise_or(t2);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseXorScalar) {
  tensor<int> t({3}, {0b1100, 0b1010, 0b1111});
  int         value = 0b0101;
  tensor<int> expected({3}, {0b1100 ^ value, 0b1010 ^ value, 0b1111 ^ value});
  tensor<int> result = t.bitwise_xor(value);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseXorTensor) {
  tensor<int> t1({3}, {0b1100, 0b1010, 0b1111});
  tensor<int> t2({3}, {0b0011, 0b0101, 0b0000});
  tensor<int> expected({3}, {0b1100 ^ 0b0011, 0b1010 ^ 0b0101, 0b1111 ^ 0b0000});
  tensor<int> result = t1.bitwise_xor(t2);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseLeftShift) {
  tensor<int> t({3}, {1, 2, 4});
  tensor<int> expected({3}, {1 << 1, 2 << 1, 4 << 1});
  tensor<int> result = t.bitwise_left_shift(1);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, BitwiseRightShift) {
  tensor<int> t({3}, {4, 8, 16});
  tensor<int> expected({3}, {4 >> 1, 8 >> 1, 16 >> 1});
  tensor<int> result = t.bitwise_right_shift(1);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, MatmulBasic) {
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

TEST(TensorTest, ReshapeBasic) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  tensor<int> expected({3, 2}, {1, 2, 3, 4, 5, 6});
  tensor<int> reshaped = t.reshape({3, 2});

  EXPECT_EQ(reshaped.shape(), expected.shape());
  for (std::size_t i = 0; i < reshaped.size(0); ++i)
  {
    EXPECT_EQ(reshaped[i], expected[i]);
  }
}

TEST(TensorTest, ReshapeAs) {
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

TEST(TensorTest, CrossProduct3D) {
  tensor<int> A({3}, {1, 2, 3});
  tensor<int> B({3}, {4, 5, 6});
  tensor<int> expected({3}, {2 * 6 - 3 * 5, 3 * 4 - 1 * 6, 1 * 5 - 2 * 4});
  tensor<int> result = A.cross_product(B);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, AbsoluteValues) {
  tensor<int> t({4}, {-1, 0, 5, -7});
  tensor<int> expected({4}, {1, 0, 5, 7});
  tensor<int> result = t.absolute(t);

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, DotProduct) {
  tensor<int> A({3}, {1, 3, -5});
  tensor<int> B({3}, {4, -2, -1});
  int         dot_result = 1 * 4 + 3 * (-2) + (-5) * (-1);
  tensor<int> result     = A.dot(B);

  EXPECT_EQ(result.shape(), std::vector<unsigned long long>{1});
  EXPECT_EQ(result[0], dot_result);
}

TEST(TensorTest, Relu) {
  tensor<float> t({4}, {-1.0, 0.0, 2.5, -3.3});
  tensor<float> expected({4}, {0.0, 0.0, 2.5, 0.0});
  tensor<float> result = t.relu();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_FLOAT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, FillWithScalar) {
  tensor<float> t({2, 3});
  auto          filled = t.fill(5.0f);

  for (std::size_t i = 0; i < filled.size(0); ++i)
  {
    EXPECT_EQ(filled[i], 5.0f);
  }
}

TEST(TensorTest, FillWithTensor) {
  tensor<float> t1({2, 2}, {1, 2, 3, 4});
  tensor<float> t2({2, 2});

  auto filled = t2.fill(t1);
  EXPECT_EQ(filled.shape(), t1.shape());

  for (std::size_t i = 0; i < filled.size(0); ++i)
  {
    EXPECT_EQ(filled[i], t1[i]);
  }
}

TEST(TensorTest, ResizeAs) {
  tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  auto        resized = t.resize_as({3, 2});

  EXPECT_EQ(resized.shape(), std::vector<unsigned long long>({3, 2}));
  for (std::size_t i = 0; i < resized.size(0); ++i)
  {
    EXPECT_EQ(resized[i], t[i]);
  }
}

TEST(TensorTest, AllTrue) {
  tensor<int> t({3}, {1, 2, 3});
  auto        result = t.all();
  EXPECT_TRUE(result.shape().equal(std::vector<unsigned long long>{1}));
  EXPECT_TRUE(result[0]);
}

TEST(TensorTest, AllFalse) {
  tensor<int> t({3}, {1, 0, 3});
  auto        result = t.all();
  EXPECT_FALSE(result[0]);
}

TEST(TensorTest, AnyTrue) {
  tensor<int> t({4}, {0, 0, 1, 0});
  auto        result = t.any();
  EXPECT_TRUE(result[0]);
}

TEST(TensorTest, AnyFalse) {
  tensor<int> t({3}, {0, 0, 0});
  auto        result = t.any();
  EXPECT_FALSE(result[0]);
}

TEST(TensorTest, Determinant2x2) {
  tensor<float> t({2, 2}, {4, 6, 3, 8});
  auto          result = t.det();
  EXPECT_NEAR(result[0], 4 * 8 - 6 * 3, 1e-5);
}

/*
TEST(TensorTest, Determinant3x3) {
    tensor<float> t({3, 3}, {6, 1, 1, 4, -2, 5, 2, 8, 7});
    auto          result = t.det();
    EXPECT_NEAR(result[0], 6 * (-2 * 7 - 5 * 8) - 1 * (4 * 7 - 5 * 2) + 1 * (4 * 8 + 2 * 2), 1e-5);
}
*/

TEST(TensorTest, Square) {
  tensor<int> t({3}, {2, -3, 4});
  auto        result = t.square();
  EXPECT_EQ(result.shape(), std::vector<unsigned long long>({3}));
  EXPECT_EQ(result[0], 4);
  EXPECT_EQ(result[1], 9);
  EXPECT_EQ(result[2], 16);
}

TEST(TensorTest, Sigmoid) {
  tensor<float> t({2}, {0.0f, 2.0f});
  auto          result = t.sigmoid();
  EXPECT_NEAR(result[0], 0.5f, 1e-5);
  EXPECT_NEAR(result[1], 1.0f / (1.0f + std::exp(-2.0f)), 1e-5);
}

TEST(TensorTest, ClippedReLU) {
  tensor<float> t({6}, {-3.0f, -1.0f, 0.0f, 2.5f, 5.0f, 10.0f});
  float         clip_limit = 4.0f;
  tensor<float> result     = t.clipped_relu(clip_limit);

  std::vector<float> expected = {0.0f, 0.0f, 0.0f, 2.5f, 4.0f, 4.0f};
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_FLOAT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, RemainderTest) {
  tensor<int> t({5}, {10, 20, 30, 40, 50});
  int         divisor = 7;
  auto        result  = t.remainder(divisor);

  tensor<int> expected({5}, {3, 6, 2, 5, 1});
  EXPECT_EQ(result.shape(), expected.shape());
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, MaximumTest) {
  tensor<int> t({5}, {1, 3, 2, 5, 4});
  tensor<int> other({5}, {2, 3, 1, 4, 5});
  auto        result = t.maximum(other);

  tensor<int> expected({5}, {2, 3, 2, 5, 5});
  EXPECT_EQ(result.shape(), expected.shape());
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, DistTest) {
  tensor<float> t1({3}, {1.0f, 2.0f, 3.0f});
  tensor<float> t2({3}, {4.0f, 5.0f, 6.0f});
  tensor<float> expected_distance({3}, {3.0f, 3.0f, 3.0f});  // Euclidean distance
  auto          result = t1.dist(t2);

  EXPECT_EQ(result, expected_distance);
}

TEST(TensorTest, NegativeTest) {
  tensor<int> t({3}, {1, -2, 3});
  auto        negated = t.negative();

  tensor<int> expected({3}, {-1, 2, -3});
  EXPECT_EQ(negated.shape(), expected.shape());
  for (std::size_t i = 0; i < negated.size(0); ++i)
  {
    EXPECT_EQ(negated[i], expected[i]);
  }
}

TEST(TensorTest, GCDTest) {
  tensor<int> t1({3}, {48, 64, 80});
  tensor<int> t2({3}, {18, 24, 30});
  auto        result = t1.gcd(t2);

  tensor<int> expected({3}, {6, 8, 10});
  EXPECT_EQ(result.shape(), expected.shape());
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, POWTest) {
  tensor<int> t({3}, {2, 3, 4});
  int         exponent = 2;
  auto        result   = t.pow(exponent);

  tensor<int> expected({3}, {4, 9, 16});
  EXPECT_EQ(result.shape(), expected.shape());
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(TensorTest, POWTensorTest) {
  tensor<int> t1({3}, {2, 3, 4});
  tensor<int> t2({3}, {2, 3, 4});
  auto        result = t1.pow(t2);

  tensor<int> expected({3}, {4, 27, 256});
  EXPECT_EQ(result.shape(), expected.shape());
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}