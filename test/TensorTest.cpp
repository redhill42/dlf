#include <cmath>
#include <complex>
#include <variant>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test_utility.h"

#ifndef NDEBUG
#define GRAINSIZE 1 // enforce parallel algorithm
#endif

#include "tensor.h"

#if HAS_GMP
#include <gmpxx.h>
#endif

using namespace tensor;
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

    template <typename T>
    std::string format(Tensor<T> t) {
        std::stringstream out;
        out << t;
        return out.str();
    }
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

TEST_F(TensorTest, Wrap) {
    int32_t data[2*3*4] = {0};
    auto t = Tensor<int32_t>::wrap({2,3,4}, data);
    t += 5;

    EXPECT_EQ(data, t.data());
    EXPECT_THAT(t, T::Each(5));
    EXPECT_THAT(data, T::Each(5));
}

TEST_F(TensorTest, Reshape) {
    // reordered dims
    {
        Tensor<int> A({2, 3, 4});
        EXPECT_TRUE(A.reshape({4, 2, 3}));
        EXPECT_EQ(A.shape(), Shape({4, 2, 3}));
    }
    // reduced dims
    {
        Tensor<int> A({2, 3, 4});
        EXPECT_TRUE(A.reshape({3, 8}));
        EXPECT_EQ(A.shape(), Shape({3, 8}));
    }
    // extended dims
    {
        Tensor<int> A({2, 3, 4});
        EXPECT_TRUE(A.reshape({3, 2, 2, 2}));
        EXPECT_EQ(A.shape(), Shape({3, 2, 2, 2}));
    }
    // one dim
    {
        Tensor<int> A({2, 3, 4});
        EXPECT_TRUE(A.reshape({24}));
        EXPECT_EQ(A.shape(), Shape({24}));
    }
    // negative dim
    {
        Tensor<int> A({2, 3, 4});
        EXPECT_TRUE(A.reshape({6, size_t(-1), 2}));
        EXPECT_EQ(A.shape(), Shape({6, 2, 2}));
    }
    // multiple negative dim
    {
        Tensor<int> A({2, 3, 4});
        EXPECT_FALSE(A.reshape({2, size_t(-1), size_t(-1)}));
        EXPECT_EQ(A.shape(), Shape({2, 3, 4}));
    }
    // zero dim
    {
        Tensor<int> A;
        EXPECT_TRUE(A.reshape({}));
        EXPECT_EQ(A.shape(), Shape({}));
    }
    // incompatible shape
    {
        Tensor<int> A({3, 7});
        EXPECT_FALSE(A.reshape({2, size_t(-1)}));
        EXPECT_EQ(A.shape(), Shape({3, 7}));
    }
}

TEST_F(TensorTest, ElementAccess) {
    Tensor<int32_t> t({2, 3, 4});
    int next = 1;

    for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++)
    for (int k = 0; k < 4; k++)
        t(i, j, k) = next++;

    next = 1;
    for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++)
    for (int k = 0; k < 4; k++) {
        ASSERT_EQ(next, t(i,j,k));
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

    ts1(2) = 100; // shallow change original tensor
    EXPECT_EQ(t1(1,1,2), 100);

    ts2(3) = 200; // should not change original tensor
    EXPECT_EQ(t1(1,1,3), 20);

    // change on original tensor should reflect on sliced tensor
    t1(1,1,1) = 50;
    EXPECT_EQ(ts1(1), 50);
    EXPECT_EQ(ts2(1), 18); // no change on copy
}

template <typename F>
void TensorTest::testBinaryOp(const Tensor<int32_t>& t, const F& f) {
    int next = 0;
    for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++)
    for (int k = 0; k < 4; k++) {
        ASSERT_EQ(f(data1[next], data2[next]), t(i,j,k));
        next++;
    }
}

template <typename F>
void TensorTest::testScalarOp(const Tensor<int32_t>& t, int32_t v, const F& f) {
    int next = 0;
    for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++)
    for (int k = 0; k < 4; k++) {
        ASSERT_EQ(f(data1[next], v), t(i,j,k));
        next++;
    }
}

template <typename F>
void TensorTest::testScalarOp(int32_t v, const Tensor<int32_t>& t, const F& f) {
    int next = 0;
    for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++)
    for (int k = 0; k < 4; k++) {
        ASSERT_EQ(f(v, data1[next]), t(i,j,k));
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

TEST_F(TensorTest, BinaryOpWithDifferentElementType) {
    Tensor<int> x({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<double> y({2, 3}, {3.1, 3.2, 3.3, 5.5, 5.6, 5.7});

    auto a = x; a += y;
    EXPECT_THAT(a, T::ElementsAre(4, 5, 6, 9, 10, 11));

    auto b = y; b += x;
    EXPECT_THAT(b, T::ElementsAre(3.1+1, 3.2+2, 3.3+3, 5.5+4, 5.6+5, 5.7+6));

    static_assert(std::is_same_v<decltype(x+y), Tensor<double>>);
    EXPECT_THAT(x+y, T::ElementsAre(1+3.1, 2+3.2, 3+3.3, 4+5.5, 5+5.6, 6+5.7));

    static_assert(std::is_same_v<decltype(y-x), Tensor<double>>);
    EXPECT_THAT(y-x, T::ElementsAre(3.1-1, 3.2-2, 3.3-3, 5.5-4, 5.6-5, 5.7-6));

    static_assert(std::is_same_v<decltype(x+5.5), Tensor<double>>);
    EXPECT_THAT(x+5.5, T::ElementsAre(1+5.5, 2+5.5, 3+5.5, 4+5.5, 5+5.5, 6+5.5));

    static_assert(std::is_same_v<decltype(5.5-x), Tensor<double>>);
    EXPECT_THAT(5.5-x, T::ElementsAre(5.5-1, 5.5-2, 5.5-3, 5.5-4, 5.5-5, 5.5-6));

    EXPECT_THAT(-x, T::ElementsAre(-1, -2, -3, -4, -5, -6));
    EXPECT_THAT(-y, T::ElementsAre(-3.1, -3.2, -3.3, -5.5, -5.6, -5.7));

    Tensor<std::string> greetings({4}, {"Hello", "Bonjour", "Ciao", "Aloha"});
    greetings += " world";
    EXPECT_THAT(greetings, T::ElementsAre("Hello world", "Bonjour world", "Ciao world", "Aloha world"));
}

TEST_F(TensorTest, FloatingOp) {
    Tensor<float> a({2, 2}, {1.f, 7.f, 5.f, 8.f});
    Tensor<float> b({2, 2}, {2.f, 9.f, 4.f, 3.f});
    EXPECT_EQ(a + b, Tensor<float>({2, 2}, {3.f, 16.f, 9.f, 11.f}));
    EXPECT_EQ(a - b, Tensor<float>({2, 2}, {-1.f, -2.f, 1.f, 5.f}));
}

TEST_F(TensorTest, FloatingAssignOp) {
    {
        Tensor<float> a({2, 2}, {1.f, 7.f, 5.f, 8.f});
        Tensor<float> b({2, 2}, {2.f, 9.f, 4.f, 3.f});
        a += b;
        EXPECT_EQ(a, Tensor<float>({2, 2}, {3.f, 16.f, 9.f, 11.f}));
    }
    {
        Tensor<float> a({2, 2}, {1.f, 7.f, 5.f, 8.f});
        Tensor<float> b({2, 2}, {2.f, 9.f, 4.f, 3.f});
        a -= b;
        EXPECT_EQ(a, Tensor<float>({2, 2}, {-1.f, -2.f, 1.f, 5.f}));
    }
}

TEST_F(TensorTest, FloatingOpRValueOptimization) {
    {
        Tensor<float> a({2, 2}, {1.f, 7.f, 5.f, 8.f});
        Tensor<float> b({2, 2}, {2.f, 9.f, 4.f, 3.f});
        EXPECT_EQ(std::move(a)+b, Tensor<float>({2, 2}, {3.f, 16.f, 9.f, 11.f}));
    }
    {
        Tensor<float> a({2, 2}, {1.f, 7.f, 5.f, 8.f});
        Tensor<float> b({2, 2}, {2.f, 9.f, 4.f, 3.f});
        EXPECT_EQ(std::move(a)-b, Tensor<float>({2, 2}, {-1.f, -2.f, 1.f, 5.f}));
    }
    {
        Tensor<float> a({2, 2}, {1.f, 7.f, 5.f, 8.f});
        Tensor<float> b({2, 2}, {2.f, 9.f, 4.f, 3.f});
        EXPECT_EQ(a+std::move(b), Tensor<float>({2, 2}, {3.f, 16.f, 9.f, 11.f}));
    }
    {
        Tensor<float> a({2, 2}, {1.f, 7.f, 5.f, 8.f});
        Tensor<float> b({2, 2}, {2.f, 9.f, 4.f, 3.f});
        EXPECT_EQ(a-std::move(b), Tensor<float>({2, 2}, {-1.f, -2.f, 1.f, 5.f}));
    }
    {
        Tensor<float> a({2, 2}, {1.f, 7.f, 5.f, 8.f});
        Tensor<float> b({2, 2}, {2.f, 9.f, 4.f, 3.f});
        EXPECT_EQ(std::move(a)+std::move(b), Tensor<float>({2, 2}, {3.f, 16.f, 9.f, 11.f}));
    }

    {
        Tensor<float> a({2, 2}, {1.f, 7.f, 5.f, 8.f});
        Tensor<float> b({2, 2}, {2.f, 9.f, 4.f, 3.f});
        EXPECT_EQ(std::move(a)-std::move(b), Tensor<float>({2, 2}, {-1.f, -2.f, 1.f, 5.f}));
    }
}

TEST_F(TensorTest, ComplexOp) {
    using namespace std::literals::complex_literals;
    Tensor<std::complex<double>> a({2, 2}, {1.0+2i, 2.0+3i, 3.0-4i, 4.0-5i});
    Tensor<std::complex<double>> b({2, 2}, {-3.0-4i, 5.0-2i, 4.0+1i, -3i});
    EXPECT_EQ(a + b, Tensor<std::complex<double>>({2, 2}, {-2.0-2i, 7.0+1i, 7.0-3i, 4.0-8i}));
}

TEST_F(TensorTest, Expression) {
    SCOPED_TRACE("(5+x)*y+x*3");
    testBinaryOp((5+t1)*t2+t1*3, [](auto x, auto y) { return (5+x)*y+x*3; });
}

template <typename T>
static void inner_test() {
    // Vector . Vector
    {
        Tensor<T> a({3}, {1, 2, 3});
        Tensor<T> b({3}, {4, 5, 6});
        Tensor<T> c({1}, {42});
        EXPECT_EQ(inner(a, b), Tensor<T>({1}, {32})); // computed by WolframAlpha
        inner(a, b, &c);
        EXPECT_EQ(c, Tensor<T>({1}, {32}));
    }

    // Vector . Matrix
    {
        Tensor<T> a({3}, {1, 2, 3});
        Tensor<T> b({{3, 2}, {4, 5, 6, 7, 8, 9}});
        Tensor<T> c({2}, {17, 53});
        EXPECT_EQ(inner(a, b), Tensor<T>({2}, {40, 46})); // computed by WolframAlpha
        inner(a, b, &c);
        EXPECT_EQ(c, Tensor<T>({2}, {40, 46}));
    }

    // Matrix . Vector
    {
        Tensor<T> a({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor<T> b({3}, {7, 8, 9});
        Tensor<T> c({2}, {17, 53});
        EXPECT_EQ(inner(a, b), Tensor<T>({2}, {50, 122})); // computed by WolframAlpha
        inner(a, b, &c);
        EXPECT_EQ(c, Tensor<T>({2}, {50, 122}));
    }

    // Matrix . Matrix
    {
        Tensor<T> a({3, 6}, {
            5, 7, 6, 10, 6, 2,
            9, 6, 6, 1, 6, 10,
            10, 1, 9, 3, 1, 3
        });

        Tensor<T> b({6, 4}, {
            7, 1, 8, 7,
            9, 5, 2, 6,
            7, 8, 5, 7,
            6, 9, 1, 1,
            4, 10, 1, 10,
            3, 8, 8, 5
        });

        Tensor<T> c({3, 4}, {
            1,  2,  3,  4,
            5,  6,  7,  8,
            9, 10, 11, 12
        });

        // computed by WolframAlpha
        Tensor<T> d({3, 4}, {
            230, 254, 116, 199,
            219, 236, 201, 252,
            173, 148, 155, 167
        });

        EXPECT_EQ(inner(a, b), d);
        inner(a, b, &c);
        EXPECT_EQ(c, d);
    }
}

TEST_F(TensorTest, Inner) {
    inner_test<int>();
    inner_test<float>();
    inner_test<double>();
    inner_test<std::complex<float>>();
    inner_test<std::complex<double>>();
}

template <typename T = int>
T quick_fibonacci(int n) {
    Tensor<T> A({2, 2}, {1, 1, 1, 0});
    return pow(A, n-1)(0, 0);
}

TEST(Tensor, Fibonacci) {
    EXPECT_EQ(quick_fibonacci(12), 144);

#if HAS_GMP
    auto z = quick_fibonacci<mpz_class>(12);
    EXPECT_EQ(z, mpz_class(144));
#endif
}

template <typename T>
static void gemm_test() {
    Tensor<T> a({3, 6}, {
        5, 10, 9, 1, 10, 3,
        7,  6, 6, 6,  1, 1,
        6,  2, 6, 10, 9, 3
    });

    Tensor<T> b({6, 4}, {
        7,  1, 8,  7,
        9,  5, 2,  6,
        7,  8, 5,  7,
        6,  9, 1,  1,
        4, 10, 1, 10,
        3,  8, 8,  5
    });

    Tensor<T> c({3, 4}, {
        230, 254, 116, 199,
        219, 236, 201, 252,
        173, 148, 155, 167
    });

    Tensor<T> r({3, 4}, {
        1176, 1282, 628, 1145,
        1033, 1022, 829, 1052,
         933,  980, 715,  923
    });

    EXPECT_THAT(gemm(a, b, c, T(2), T(3), false, false), r);
    EXPECT_THAT(gemm(transpose(a), b, c, T(2), T(3), true, false), r);
    EXPECT_THAT(gemm(a, transpose(b), c, T(2), T(3), false, true), r);
    EXPECT_THAT(gemm(transpose(a), transpose(b), c, T(2), T(3), true, true), r);
}

TEST_F(TensorTest, Gemm) {
    gemm_test<int>();
    gemm_test<float>();
    gemm_test<double>();
    gemm_test<std::complex<float>>();
    gemm_test<std::complex<double>>();
}

template <typename T>
static void transpose_test() {
    Tensor<T> a({3, 4}, {
        230, 254, 116, 199,
        219, 236, 201, 252,
        173, 148, 155, 167
    });

    Tensor<T> b({4, 3}, {
        230, 219, 173,
        254, 236, 148,
        116, 201, 155,
        199, 252, 167
    });

    EXPECT_EQ(transpose(a), b);
    transpose(a, &a);
    EXPECT_EQ(a, b);
}

TEST_F(TensorTest, Transpose) {
    transpose_test<int>();
    transpose_test<float>();
    transpose_test<double>();
    transpose_test<std::complex<float>>();
    transpose_test<std::complex<double>>();
}

template <typename T>
static void test_transpose_in_place() {
    for (size_t i = 0; i <= 5; i++) {
        for (size_t j = 0; j <= 5; j++) {
            auto a = Tensor<T>::range({i, j}, 1);
            auto b = transpose(a); // assume out-of-place transposition is correct
            transpose(a, &a); // in-place transpose
            EXPECT_EQ(a, b);
        }
    }

}
TEST_F(TensorTest, TransposeInPlace) {
    test_transpose_in_place<int>();
    test_transpose_in_place<float>();
    test_transpose_in_place<double>();
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

template <typename T>
static void ChainedApplyTest() {
    constexpr T PI  = 3.14159;
    constexpr T E   = 2.71828;
    constexpr T PHI = 1.61803;
    constexpr T SR2 = 1.41421;

    Tensor<T> t({2,2}, {PI, E, PHI, SR2});
    t.apply(sin).apply(sqrt);

    static_assert(std::is_same_v<decltype(sin(PI)), T>);
    auto f = [](auto x){ return sqrt(sin(x)); };
    EXPECT_THAT(t, testing::ElementsAre(f(PI), f(E), f(PHI), f(SR2)));
}

TEST_F(TensorTest, ChainedApply) {
    ChainedApplyTest<double>();
    ChainedApplyTest<float>();
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

    auto a = t1.transform(t2, f);

    Tensor<int32_t> b({2,3,4});
    t1.transformTo(b, t2, f);

    for (int i = 0; i < a.size(); i++) {
        auto v = f(data1[i], data2[i]);
        EXPECT_EQ(a.data()[i], v);
        EXPECT_EQ(b.data()[i], v);
    }
}

TEST_F(TensorTest, TransformRValueOptimization) {
    {
        Tensor<int> a({2, 2}, {1, 2, 3, 4});
        Tensor<int> b = std::move(a).transform([](auto x) { return x*2; });
        EXPECT_TRUE(a.empty());
        EXPECT_THAT(b, T::ElementsAre(2, 4, 6, 8));
    }

    {
        Tensor<int> a({2, 2}, {1, 2, 3, 4});
        Tensor<int> b({2, 2}, {5, 6, 7, 8});
        Tensor<int> c = std::move(a).transform(b, std::plus());
        EXPECT_TRUE(a.empty());
        EXPECT_THAT(b, T::ElementsAre(5, 6, 7, 8));
        EXPECT_THAT(c, T::ElementsAre(6, 8, 10, 12));
    }

    {
        Tensor<int> a({2, 2}, {1, 2, 3, 4});
        Tensor<int> b({2, 2}, {5, 6, 7, 8});
        Tensor<int> c = a.transform(std::move(b), std::plus());
        EXPECT_THAT(a, T::ElementsAre(1, 2, 3, 4));
        EXPECT_TRUE(b.empty());
        EXPECT_THAT(c, T::ElementsAre(6, 8, 10, 12));
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
    // fill a tensor with random numbers in [-500,500]
    auto in = Tensor<int32_t>::random({100, 100}, -500, 500);

    // relu with 100 as max value
    auto out = in.transform(Relu(100.0));
    static_assert(std::is_same_v<decltype(out), Tensor<double>>);

    EXPECT_THAT(in, T::Contains(T::Lt(0)));
    EXPECT_THAT(in, T::Contains(T::Gt(100)));
    EXPECT_THAT(out, T::Each(T::AllOf(T::Ge(0.0), T::Le(100.0))));
}

TEST_F(TensorTest, Format) {
    EXPECT_EQ(format(t2),
        "["
        "[[2,3,5,7],[11,13,17,19],[23,29,31,37]],"
        "[[41,43,47,53],[57,59,61,67],[71,73,79,83]]"
        "]");

    EXPECT_EQ(format(Tensor<int>()), "");
}

TEST(Tensor, MatrixMultiplicationPerformance) {
    {
        auto A = Tensor<double>::random({100, 100}, -100, 100);
        auto B = Tensor<double>::random({100, 100}, -100, 100);
        timing("Small matrix multiplication", 10000, [&]() {
            inner(A, B);
        });
    }
    {
        auto A = Tensor<double>::random({1024, 1024}, -100, 100);
        auto B = Tensor<double>::random({1024, 1024}, -100, 100);
        timing("Big matrix multiplication", 100, [&]() {
            inner(A, B);
        });
    }
}
