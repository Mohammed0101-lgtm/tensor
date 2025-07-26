#include "tensor.hpp"
#include <gtest/gtest.h>


TEST(ArithmeticTest, CosValues) {
  tensor<float> t({3}, {0.0, M_PI / 2, M_PI});
  tensor<float> result = t.cos();
  tensor<float> expected({3}, {1.0, 0.0, -1.0});
  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(ArithmeticTest, CoshValues) {
  tensor<float>  t({2}, {0.0, 1.0});
  tensor<float>  result = t.cosh();
  tensor<double> expected({2}, {std::cosh(0.0), std::cosh(1.0)});
 
  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(ArithmeticTest, AcosValues) {
  tensor<float>  t({3}, {-1.0, 0.0, 1.0});
  tensor<float>  result = t.acos();
  tensor<double> expected({3}, {M_PI, M_PI / 2, 0.0});
  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(ArithmeticTest, AcosOutOfDomain) {
  tensor<float> t({2}, {-2.0, 2.0});

  EXPECT_THROW(t.acos(), std::domain_error);
}

TEST(ArithmeticTest, AcoshValidRange) {
  tensor<float>  t({3}, {1.0, 2.0, 10.0});
  tensor<float>  result = t.acosh();
  tensor<double> expected({3}, {std::acosh(1.0), std::acosh(2.0), std::acosh(10.0)});
  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(ArithmeticTest, AcoshOutOfDomain) {
  tensor<float> t({1}, {0.5});
  EXPECT_THROW(t.acosh(), std::domain_error);
}

TEST(ArithmeticTest, TanValues) {
  tensor<float>  t({2}, {0.0, M_PI / 4});
  tensor<float>  result = t.tan();
  tensor<double> expected({2}, {std::tan(0.0), std::tan(M_PI / 4)});
  for (std::size_t i = 0; i < expected.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(ArithmeticTest, TanhBasic) {
  tensor<double>      t({3}, {-2.0, 0.0, 2.0});
  tensor<double>      result   = t.tanh();
  std::vector<double> expected = {std::tanh(-2.0), std::tanh(0.0), std::tanh(2.0)};
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-7);
  }
}

TEST(ArithmeticTest, AtanBasic) {
  tensor<double>      t({3}, {-1.0, 0.0, 1.0});
  tensor<double>      result   = t.atan();
  std::vector<double> expected = {std::atan(-1.0), std::atan(0.0), std::atan(1.0)};
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-7);
  }
}

TEST(ArithmeticTest, AtanhInDomain) {
  tensor<double>      t({3}, {-0.5, 0.0, 0.5});
  tensor<double>      result   = t.atanh();
  std::vector<double> expected = {std::atanh(-0.5), std::atanh(0.0), std::atanh(0.5)};

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-7);
  }
}

TEST(ArithmeticTest, AtanhOutOfDomain) {
  tensor<double> t({2}, {-1.5, 2.0});
  EXPECT_THROW(t.atanh(), std::domain_error);
}

TEST(ArithmeticTest, SinBasic) {
  tensor<double>      t({3}, {0.0, M_PI / 2, M_PI});
  tensor<double>      result   = t.sin();
  std::vector<double> expected = {0.0, 1.0, 0.0};
  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-6);
  }
}

TEST(ArithmeticTest, InplaceSincBasic) {
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

TEST(ArithmeticTest, AsinNegativeInputs) {
  tensor<double> t({3}, {-0.5, -1.0, -0.25});
  tensor<double> expected({3}, {std::asin(-0.5), std::asin(-1.0), std::asin(-0.25)});
  tensor<double> result = t.asin();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}

TEST(ArithmeticTest, AsinOutOfDomain) {
  tensor<double> t({2}, {1.5, -1.5});
  EXPECT_THROW(t.asin(), std::domain_error);
}

TEST(ArithmeticTest, AsinhBasic) {
  tensor<double> t({4}, {0.0, 0.5, 1.0, 2.0});
  tensor<double> expected({4}, {std::asinh(0.0), std::asinh(0.5), std::asinh(1.0), std::asinh(2.0)});
  tensor<double> result = t.asinh();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}

TEST(ArithmeticTest, AsinhNegative) {
  tensor<double> t({3}, {-0.5, -1.0, -2.0});
  tensor<double> expected({3}, {std::asinh(-0.5), std::asinh(-1.0), std::asinh(-2.0)});
  tensor<double> result = t.asinh();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1e-9);
  }
}

TEST(ArithmeticTest, AbsBasic) {
  tensor<double> t({6}, {-2.5, -1.0, 0.0, 0.5, 1.0, 2.5});
  tensor<double> expected({6}, {2.5, 1.0, 0.0, 0.5, 1.0, 2.5});
  tensor<double> result = t.abs();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_DOUBLE_EQ(result[i], expected[i]);
  }
}

TEST(ArithmeticTest, AbsWithInts) {
  tensor<int> t({4}, {-3, -1, 0, 5});
  tensor<int> expected({4}, {3, 1, 0, 5});
  tensor<int> result = t.abs();

  for (std::size_t i = 0; i < result.size(0); ++i)
  {
    EXPECT_EQ(result[i], expected[i]);
  }
}

TEST(ArithmeticTest, POWTest) {
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

TEST(ArithmeticTest, POWTensorTest) {
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