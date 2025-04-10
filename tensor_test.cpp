#include "src/tensor.hpp"

#include <gtest/gtest.h>

#include <iostream>

TEST(TensorTest, StorageTest) {
    tensor<int> t({5}, {1, 2, 3, 4, 5});

    std::vector<int> expected = {1, 2, 3, 4, 5};

    EXPECT_EQ(t.storage(), expected);
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

    EXPECT_EQ(t.shape(), (std::vector<unsigned long long>{2, 2}));
}

TEST(TensorTest, StridesTest) {
    tensor<int> t({2, 2}, {1, 2, 3, 4});

    EXPECT_EQ(t.strides(), (std::vector<unsigned long long>{2, 1}));
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

TEST(TensorTest, RepeatTest) {
    tensor<int>      t({20});
    std::vector<int> v = {1, 2};
    tensor<int>      expected({20}, {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2});

    tensor<int>      t1({20});
    std::vector<int> v1 = {1, 2, 3};
    tensor<int>      expected1({20}, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2});

    EXPECT_EQ(t.repeat_(v, 0), expected);
    EXPECT_EQ(t1.repeat_(v1, 0), expected1);
    EXPECT_EQ(t1.repeat(v1), expected1);
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
    tensor<int> t({2, 2}, {1, 2, 3, 4});

    tensor<int> other({2, 2}, {1, 2, 3, 4});
    tensor<int> other1({2, 2}, {2, 3, 4, 5});
    tensor<int> other2({2, 2}, {0, 3, 2, 6});

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

TEST(TensorTest, RowTest) {
    tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
    tensor<int> expected_row({3}, {1, 2, 3});
    tensor<int> expected_col({2}, {3, 6});

    EXPECT_EQ(t.row(0), expected_row);
    EXPECT_EQ(t.col(1), expected_col);
}

TEST(TensorTest, BoolRowTest) {
    tensor<int> t({2, 3}, {true, false, true, false, false, true});
    tensor<int> expected_row({3}, {true, false, true});
    tensor<int> expected_col({2}, {true, true});

    EXPECT_EQ(t.row(0), expected_row);
    EXPECT_EQ(t.col(1), expected_col);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}