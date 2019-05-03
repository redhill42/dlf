#include <cmath>
#include <complex>
#include "tensor.h"
#include "gtest/gtest.h"

using kneron::model::Tensor;

class TensorTest : public ::testing::Test {
protected:
    std::vector<int32_t> data1 {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,

        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23
    };

    std::vector<int32_t> data2 {
         2,  3,  5,  7,
        11, 13, 17, 19,
        23, 29, 31, 37,

        41, 43, 47, 53,
        57, 59, 61, 67,
        71, 73, 79, 83
    };

    Tensor<int32_t> t1, t2;

    TensorTest() : t1({2,3,4}, data1), t2({2,3,4}, data2) {}

    template<typename F>
    void testBinaryOp(const Tensor<int32_t> &t, const F &f);

    template<typename F>
    void testScalarOp(const Tensor<int32_t> &t, int32_t v, const F &f);
};

TEST_F(TensorTest, Init) {
    EXPECT_EQ(t1.dims(), std::vector<size_t>({2,3,4}));
    EXPECT_EQ(t1.size(), 2*3*4);

    int32_t next = 0;
    for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 3; j++)
    for (size_t k = 0; k < 4; k++) {
        EXPECT_EQ(next, (t1[{i,j,k}]));
        next++;
    }
}

TEST_F(TensorTest, InitializedToZero) {
    Tensor<int32_t> t({2,2,2});
    ASSERT_EQ(t.size(), 8);
    for (int i = 0; i < 8; i++)
        EXPECT_EQ(t.data()[i], 0);
}

TEST_F(TensorTest, Wrap) {
    int32_t data[2*3*4] = {0};
    auto t = Tensor<int32_t>::wrap({2,3,4}, data);
    t += 5;

    EXPECT_EQ(data, t.data());
    EXPECT_EQ((t[{1,1,1}]), 5);
    EXPECT_EQ(data[12], 5);
}

TEST_F(TensorTest, ElementAccess) {
    Tensor<int32_t> t({2, 3, 4});
    int32_t next = 0;

    for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 3; j++)
    for (size_t k = 0; k < 4; k++)
        t[{i, j, k}] = next++;

    next = 0;
    for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 3; j++)
    for (size_t k = 0; k < 4; k++) {
        ASSERT_EQ(next, (t[{i,j,k}]));
        next++;
    }

    ASSERT_EQ(t, t1);
    ASSERT_NE(t, t2);
}

template <typename F>
void TensorTest::testBinaryOp(const Tensor<int32_t>& t, const F& f) {
    size_t next = 0;
    for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 3; j++)
    for (size_t k = 0; k < 4; k++) {
        ASSERT_EQ(f(data1[next], data2[next]), (t[{i,j,k}]));
        next++;
    }
}

template <typename F>
void TensorTest::testScalarOp(const Tensor<int32_t>& t, int32_t v, const F& f) {
    size_t next = 0;
    for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 3; j++)
    for (size_t k = 0; k < 4; k++) {
        ASSERT_EQ(f(data1[next], v), (t[{i,j,k}]));
        next++;
    }
}

TEST_F(TensorTest, BinaryOp) {
    { SCOPED_TRACE("+"); testBinaryOp(t1 + t2, std::plus<>()); }
    { SCOPED_TRACE("-"); testBinaryOp(t1 - t2, std::minus<>()); }
    { SCOPED_TRACE("*"); testBinaryOp(t1 * t2, std::multiplies<>()); }
    { SCOPED_TRACE("/"); testBinaryOp(t1 / t2, std::divides<>()); }
}

TEST_F(TensorTest, BinaryAssignOp) {
    Tensor<int32_t> t;

    { SCOPED_TRACE("+="); t = t1; t += t2; testBinaryOp(t, std::plus<>()); }
    { SCOPED_TRACE("-="); t = t1; t -= t2; testBinaryOp(t, std::minus<>()); }
    { SCOPED_TRACE("*="); t = t1; t *= t2; testBinaryOp(t, std::multiplies<>()); }
    { SCOPED_TRACE("/="); t = t1; t /= t2; testBinaryOp(t, std::divides<>()); }
}

TEST_F(TensorTest, ScalarOp) {
    { SCOPED_TRACE("+5"); testScalarOp(t1+5, 5, std::plus<>()); }
    { SCOPED_TRACE("-5"); testScalarOp(t1-5, 5, std::minus<>()); }
    { SCOPED_TRACE("*5"); testScalarOp(t1*5, 5, std::multiplies<>()); }
    { SCOPED_TRACE("/5"); testScalarOp(t1/5, 5, std::divides<>()); }
}

TEST_F(TensorTest, ScalarAssignOp) {
    Tensor<int32_t> t;

    { SCOPED_TRACE("+=5"); t = t1; t += 5; testScalarOp(t, 5, std::plus<>()); }
    { SCOPED_TRACE("-=5"); t = t1; t -= 5; testScalarOp(t, 5, std::minus<>()); }
    { SCOPED_TRACE("*=5"); t = t1; t *= 5; testScalarOp(t, 5, std::multiplies<>()); }
    { SCOPED_TRACE("/=5"); t = t1; t /= 5; testScalarOp(t, 5, std::divides<>()); }
}

TEST_F(TensorTest, DotProduct) {
    Tensor<int32_t> a({3, 6}, {
        5, 7, 6, 10, 6, 2,
        9, 6, 6, 1, 6, 10,
        10, 1, 9, 3, 1, 3
    });

    Tensor<int32_t> b({6, 4}, {
        7, 1, 8, 7,
        9, 5, 2, 6,
        7, 8, 5, 7,
        6, 9, 1, 1,
        4, 10, 1, 10,
        3, 8, 8, 5
    });

    Tensor<int32_t> c({3, 4}, {
        230, 254, 116, 199,
        219, 236, 201, 252,
        173, 148, 155, 167
    });

    EXPECT_EQ(a.dot(b), c);
}

TEST_F(TensorTest, Transpose) {
    Tensor<int32_t> a({3, 4}, {
        230, 254, 116, 199,
        219, 236, 201, 252,
        173, 148, 155, 167
    });

    Tensor<int32_t> b({4, 3}, {
        230, 219, 173,
        254, 236, 148,
        116, 201, 155,
        199, 252, 167
    });

    EXPECT_EQ(a.transpose(), b);
}

TEST_F(TensorTest, Apply) {
    auto t = t1.cast<double>().apply(sin);
    for (int i = 0; i < t.size(); i++) {
        EXPECT_EQ(t.data()[i], sin(data1[i]));
    }
}

TEST_F(TensorTest, Complex) {
    using namespace std::complex_literals;
    Tensor<std::complex<double>> t({2,2}, {
        1.+2i, 3.+4i, -1.+1i, 2.-5i
    });
    t += 1.+1i;

    EXPECT_EQ((t[{0,0}]), 2.+3i);
    EXPECT_EQ((t[{0,1}]), 4.+5i);
    EXPECT_EQ((t[{1,0}]), 0.+2i);
    EXPECT_EQ((t[{1,1}]), 3.-4i);
}

TEST_F(TensorTest, Format) {
    std::stringstream out;
    out << t2;
    EXPECT_EQ(out.str(), "[[[2, 3, 5, 7],\n  [11, 13, 17, 19],\n  [23, 29, 31, 37]],\n [[41, 43, 47, 53],\n  [57, 59, 61, 67],\n  [71, 73, 79, 83]]]");
}
