#include <cmath>
#include <complex>
#include <random>
#include "tensor.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

using kneron::model::Shape;
using kneron::model::Tensor;

namespace T = ::testing;

class TensorTest : public ::testing::Test {
protected:
    std::vector<int32_t> data1 {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,

        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
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

    TensorTest()
        : t1({2,3,4}, data1.begin(), data1.end()),
          t2({2,3,4}, data2.begin(), data2.end()) {}

    template<typename F>
    void testBinaryOp(const Tensor<int32_t> &t, const F &f);

    template<typename F>
    void testScalarOp(const Tensor<int32_t> &t, int32_t v, const F &f);

    template<typename F>
    void testScalarOp(int32_t v, const Tensor<int32_t> &t, const F &f);

    template <typename T, typename F>
    void checkDataTransform(const Tensor<T>& t, F f);
};

TEST_F(TensorTest, Init) {
    EXPECT_EQ(t1.shape(), Shape({2,3,4}));
    EXPECT_EQ(t1.size(), 2*3*4);
    EXPECT_THAT(t1, T::ElementsAreArray(data1));
}

TEST_F(TensorTest, InitializedToZero) {
    Tensor<int32_t> t({2,2,2});
    ASSERT_EQ(t.size(), 8);
    EXPECT_THAT(t, T::Each(0));
}

TEST_F(TensorTest, InitializerList) {
    Tensor<int32_t> tv = {{1, 2, 3, 4, 5}};
    EXPECT_EQ(tv.shape(), Shape({5}));
    EXPECT_THAT(tv, T::ElementsAre(1, 2, 3, 4, 5));

    Tensor<int32_t> tm = {{{1, 2, 3, 4}, {5, 6, 7, 8}}};
    EXPECT_EQ(tm.shape(), Shape({2, 4}));
    EXPECT_THAT(tm, T::ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));

    Tensor<int32_t> tt = {{
        {
            { 1,  2,  3,  4},
            { 5,  6,  7,  8},
            { 9, 10, 11, 12}
        },
        {
            {13, 14, 15, 16},
            {17, 18, 19, 20},
            {21, 22, 23, 24}
        }
    }};
    EXPECT_EQ(tt.shape(), Shape({2,3,4}));
    EXPECT_EQ(tt, t1);
}

TEST_F(TensorTest, Wrap) {
    int32_t data[2*3*4] = {0};
    auto t = Tensor<int32_t>::wrap({2,3,4}, data);
    t += 5;

    EXPECT_EQ(data, t.data());
    EXPECT_THAT(t, T::Each(5));
    EXPECT_THAT(data, T::Each(5));
}

TEST_F(TensorTest, ElementAccess) {
    Tensor<int32_t> t({2, 3, 4});
    int32_t next = 1;

    for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 3; j++)
    for (size_t k = 0; k < 4; k++)
        t[{i, j, k}] = next++;

    next = 1;
    for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 3; j++)
    for (size_t k = 0; k < 4; k++) {
        ASSERT_EQ(next, (t[{i,j,k}]));
        next++;
    }

    ASSERT_EQ(t, t1);
    ASSERT_NE(t, t2);
}

TEST_F(TensorTest, Slice) {
    Tensor<int32_t> s1({3,4}, {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12
    });
    Tensor<int32_t> s2({3,4}, {
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    });

    EXPECT_EQ(t1[0], s1);
    EXPECT_EQ(t1[1], s2);

    auto ts1 = t1[1][1];
    auto ts2 = ts1; // make copy

    EXPECT_THAT(ts1, T::ElementsAre(17, 18, 19, 20));
    EXPECT_NE(ts1.data(), ts2.data());

    ts1[{2}] = 100; // shallow change original tensor
    EXPECT_EQ((t1[{1,1,2}]), 100);

    ts2[{3}] = 200; // should not change original tensor
    EXPECT_EQ((t1[{1,1,3}]), 20);

    // change on original tensor should reflect on sliced tensor
    t1[{1,1,1}] = 50;
    EXPECT_EQ(ts1[{1}], 50);
    EXPECT_EQ(ts2[{1}], 18); // no change on copy
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

template <typename F>
void TensorTest::testScalarOp(int32_t v, const Tensor<int32_t>& t, const F& f) {
    size_t next = 0;
    for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 3; j++)
    for (size_t k = 0; k < 4; k++) {
        ASSERT_EQ(f(v, data1[next]), (t[{i,j,k}]));
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
    { SCOPED_TRACE("+="); auto t = t1; t += t2; testBinaryOp(t, std::plus<>()); }
    { SCOPED_TRACE("-="); auto t = t1; t -= t2; testBinaryOp(t, std::minus<>()); }
    { SCOPED_TRACE("*="); auto t = t1; t *= t2; testBinaryOp(t, std::multiplies<>()); }
    { SCOPED_TRACE("/="); auto t = t1; t /= t2; testBinaryOp(t, std::divides<>()); }
}

TEST_F(TensorTest, ScalarOp) {
    { SCOPED_TRACE("+5"); testScalarOp(t1+5, 5, std::plus<>()); }
    { SCOPED_TRACE("-5"); testScalarOp(t1-5, 5, std::minus<>()); }
    { SCOPED_TRACE("*5"); testScalarOp(t1*5, 5, std::multiplies<>()); }
    { SCOPED_TRACE("/5"); testScalarOp(t1/5, 5, std::divides<>()); }

    { SCOPED_TRACE("100+"); testScalarOp(100, 100+t1, std::plus<>()); }
    { SCOPED_TRACE("100-"); testScalarOp(100, 100-t1, std::minus<>()); }
    { SCOPED_TRACE("100*"); testScalarOp(100, 100*t1, std::multiplies<>()); }
    { SCOPED_TRACE("100/"); testScalarOp(100, 100/t1, std::divides<>()); }
}

TEST_F(TensorTest, ScalarAssignOp) {
    { SCOPED_TRACE("+=5"); auto t = t1; t += 5; testScalarOp(t, 5, std::plus<>()); }
    { SCOPED_TRACE("-=5"); auto t = t1; t -= 5; testScalarOp(t, 5, std::minus<>()); }
    { SCOPED_TRACE("*=5"); auto t = t1; t *= 5; testScalarOp(t, 5, std::multiplies<>()); }
    { SCOPED_TRACE("/=5"); auto t = t1; t /= 5; testScalarOp(t, 5, std::divides<>()); }
}

TEST_F(TensorTest, Expression) {
    SCOPED_TRACE("5+x*y+x");
    testBinaryOp(5+t1*t2+t1, [](auto x, auto y) { return 5+x*y+x; });
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

template <typename T, typename F>
void TensorTest::checkDataTransform(const Tensor<T>& t, F f) {
    for (int i = 0; i < t.size(); i++) {
        EXPECT_EQ(t.data()[i], f(data1[i]));
    }
}

TEST_F(TensorTest, Apply) {
    SCOPED_TRACE("");
    auto f = [](auto x) { return x*x/2; };
    t1.apply(f);
    checkDataTransform(t1, f);
}

TEST_F(TensorTest, Transform) {
    {
        SCOPED_TRACE("");
        auto f = static_cast<double(*)(double)>(sin);
        auto t = t1.transform(f);
        static_assert(std::is_same_v<decltype(t), Tensor<double>>);
        checkDataTransform(t, f);
    }

    {
        SCOPED_TRACE("");
        auto f = [](auto x) { return std::complex<int>(x, x * 2); };
        auto t = t1.transform(f);
        static_assert(std::is_same_v<decltype(t), Tensor<std::complex<int>>>);
        checkDataTransform(t, f);
    }
}

TEST_F(TensorTest, TransformTo) {
    auto f = static_cast<double(*)(double)>(sin);
    Tensor<double> t({2,3,4});
    t1.transformTo(t, f);
    checkDataTransform(t, f);
}

TEST_F(TensorTest, Cast) {
    SCOPED_TRACE("");
    auto t = t1.cast<double>();
    static_assert(std::is_same_v<decltype(t), Tensor<double>>);
    checkDataTransform(t, [](auto x){ return static_cast<double>(x); });
}

TEST_F(TensorTest, Transform2) {
    auto f = [](auto x, auto y) { return (x+y)/2; };

    auto a = transform(t1, t2, f);

    Tensor<int32_t> b({2,3,4});
    transformTo(b, t1, t2, f);

    for (int i = 0; i < a.size(); i++) {
        auto v = f(data1[i], data2[i]);
        EXPECT_EQ(a.data()[i], v);
        EXPECT_EQ(b.data()[i], v);
    }
}

TEST_F(TensorTest, Complex) {
    using namespace std::complex_literals;
    Tensor<std::complex<double>> t({2,2}, {1.+2i, 3.+4i, -1.+1i, 2.-5i});
    t += 1.+1i;
    EXPECT_THAT(t, T::ElementsAre(2.+3i, 4.+5i, 0.+2i, 3.-4i));
}

/**
 * The Relu functor that compute Rectified Linear Unit.
 *
 * @tparam T the value type.
 */
template <typename T>
struct Relu {
    T max_v;
    static constexpr T zero_v = T();
    explicit constexpr Relu(T max_v = std::numeric_limits<T>::max()) : max_v(max_v) {}

    constexpr T operator()(T value) const noexcept {
        if (value <= zero_v)
            return zero_v;
        if (value > max_v)
            return max_v;
        return value;
    }
};

TEST_F(TensorTest, Relu) {
    // initialize random number generator
    std::random_device rdev;
    std::default_random_engine reng(rdev());
    std::uniform_int_distribution<int32_t> rand(-500, 500);

    // fill a tensor with random numbers in [-500,500]
    auto in = Tensor<int32_t>::build({100, 100}, std::bind(rand, reng));

    // relu with 100 as max value
    auto out = in.transform(Relu(100.0));

    EXPECT_THAT(in, T::Contains(T::Lt(0)));
    EXPECT_THAT(in, T::Contains(T::Gt(100)));
    EXPECT_THAT(out, T::Each(T::AllOf(T::Ge(0.0), T::Le(100.0))));
}

TEST_F(TensorTest, Format) {
    std::stringstream out;
    out << t2;
    EXPECT_EQ(out.str(), "[[[2, 3, 5, 7],\n  [11, 13, 17, 19],\n  [23, 29, 31, 37]],\n [[41, 43, 47, 53],\n  [57, 59, 61, 67],\n  [71, 73, 79, 83]]]");
}
