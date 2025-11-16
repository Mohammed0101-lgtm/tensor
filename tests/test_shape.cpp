#include "tensor.hpp"
#include <gtest/gtest.h>


using shape::Shape;
using vec = std::vector<uint64_t>;


TEST(ShapeTest, DefaultConstructor)
{
  Shape s;

  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.size(), 0);
}

TEST(ShapeTest, InitializerListConstructor)
{
  Shape s{2, 3, 4};

  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s[0], 2);
  EXPECT_EQ(s[1], 3);
  EXPECT_EQ(s[2], 4);
}

TEST(ShapeTest, ShapeTypeConstructor)
{
  vec   vec = {5, 6};
  Shape s(vec);

  EXPECT_EQ(s.size(), 2);
  EXPECT_EQ(s[0], 5);
  EXPECT_EQ(s[1], 6);
}

TEST(ShapeTest, SizeOnlyConstructor)
{
  Shape s(3);

  EXPECT_EQ(s.size(), 3);
  for (int i = 0; i < 3; ++i)
  {
    EXPECT_EQ(s[i], 0);  // default initialized
  }
}

TEST(ShapeTest, IndexOperatorConst)
{
  const Shape s{7, 8, 9};

  EXPECT_EQ(s[0], 7);
  EXPECT_EQ(s[1], 8);
  EXPECT_EQ(s[2], 9);
}

TEST(ShapeTest, IndexOperatorMutable)
{
  Shape s{1, 2, 3};

  s[0] = 10;
  s[2] = 20;

  EXPECT_EQ(s[0], 10);
  EXPECT_EQ(s[2], 20);
}

TEST(ShapeTest, SizeWithDimZeroReturnsFlattenSize)
{
  Shape s{2, 3, 4};

  EXPECT_EQ(s.size(0), 2 * 3 * 4);
}

TEST(ShapeTest, SizeWithOutOfBoundsDimThrows)
{
  Shape s{2, 3};

  EXPECT_THROW(s.size(3), error::shape_error);
}

TEST(ShapeTest, FlattenSizeWorks)
{
  Shape s{2, 3, 4};

  EXPECT_EQ(s.flatten_size(), 24);
}

TEST(ShapeTest, EmptyCheck)
{
  Shape s1{};
  Shape s2{1, 2};

  EXPECT_TRUE(s1.empty());
  EXPECT_FALSE(s2.empty());
}

TEST(ShapeTest, GetReturnsShapeVector)
{
  Shape s{3, 4};
  auto  vec = s.get();

  EXPECT_EQ(vec.size(), 2);
  EXPECT_EQ(vec[0], 3);
  EXPECT_EQ(vec[1], 4);
}

TEST(ShapeTest, StridesComputedCorrectly)
{
  Shape s{2, 3, 4};
  s.compute_strides();
  auto strides = s.strides();

  EXPECT_EQ(strides[0], 12);  // 3*4
  EXPECT_EQ(strides[1], 4);   // 4
  EXPECT_EQ(strides[2], 1);
}

TEST(ShapeTest, ComputeIndexWorks)
{
  Shape s{2, 3, 4};
  auto  idx = vec{1, 2, 3};

  EXPECT_EQ(s.compute_index(idx), 1 * 12 + 2 * 4 + 3 * 1);
}

TEST(ShapeTest, ComputeIndexInvalidInputThrows)
{
  Shape s{2, 3};
  vec   bad_idx = {1, 2, 3};

  EXPECT_THROW(s.compute_index(bad_idx), error::index_error);
}

TEST(ShapeTest, EqualityOperatorTrue)
{
  Shape s1{2, 3};
  Shape s2{2, 3};

  EXPECT_TRUE(s1 == s2);
}

TEST(ShapeTest, EqualityOperatorFalse)
{
  Shape s1{2, 3};
  Shape s2{3, 2};

  EXPECT_FALSE(s1 == s2);
}

TEST(ShapeTest, EqualMethodTrueExact)
{
  Shape s1{2, 3};
  Shape s2{2, 3};

  EXPECT_TRUE(s1.equal(s2));
}

TEST(ShapeTest, EqualMethodBroadcastable)
{
  Shape s1{1, 3};
  Shape s2{2, 3};
  Shape s3{1, 3};
  Shape s4{5, 1, 3};

  EXPECT_FALSE(s1.equal(s2));
  EXPECT_TRUE(s3.equal(s4));
}

TEST(ShapeTest, EqualMethodIncompatible)
{
  Shape s1{2, 3};
  Shape s2{3, 2};

  EXPECT_FALSE(s1.equal(s2));
}

TEST(ShapeTest, EqualMethodDifferentRanksCompatible)
{
  Shape s1{1, 2, 3};
  Shape s2{2, 3};

  EXPECT_TRUE(s1.equal(s2));
}

TEST(ShapeTest, EqualMethodDifferentRanksIncompatible)
{
  Shape s1{2, 2, 3};
  Shape s2{2, 4};

  EXPECT_FALSE(s1.equal(s2));
}

TEST(ShapeTest, ComputeStrideWorksCorrectly)
{
  Shape       s{2, 3, 4};
  std::size_t stride = s.computeStride(1, s);

  EXPECT_EQ(stride, 24);  // 2*3*4
}