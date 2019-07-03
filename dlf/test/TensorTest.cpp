#include <cmath>
#include <complex>
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

using namespace dlf;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Each;

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
    EXPECT_THAT(t1, ElementsAreArray(data1));
}

TEST_F(TensorTest, InitializedToZero) {
    Tensor<int32_t> t({2,2,2});
    ASSERT_EQ(t.size(), 8);
    EXPECT_THAT(t, Each(0));
}

TEST_F(TensorTest, Wrap) {
    int32_t data[2*3*4] = {0};
    auto t = Tensor<int32_t>::wrap({2,3,4}, data);
    t += scalar(5);

    EXPECT_EQ(data, t.data());
    EXPECT_THAT(t, Each(5));
    EXPECT_THAT(data, Each(5));
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
    { SCOPED_TRACE("+5"); testScalarOp(t1+scalar(5), 5, std::plus<>()); }
    { SCOPED_TRACE("-5"); testScalarOp(t1-scalar(5), 5, std::minus<>()); }
    { SCOPED_TRACE("*5"); testScalarOp(t1*scalar(5), 5, std::multiplies<>()); }
    { SCOPED_TRACE("/5"); testScalarOp(t1/scalar(5), 5, std::divides<>()); }

    { SCOPED_TRACE("100+"); testScalarOp(100, scalar(100)+t1, std::plus<>()); }
    { SCOPED_TRACE("100-"); testScalarOp(100, scalar(100)-t1, std::minus<>()); }
    { SCOPED_TRACE("100*"); testScalarOp(100, scalar(100)*t1, std::multiplies<>()); }
    { SCOPED_TRACE("100/"); testScalarOp(100, scalar(100)/t1, std::divides<>()); }
}

TEST_F(TensorTest, ScalarAssignOp) {
    { SCOPED_TRACE("+=5"); auto t = t1; t += scalar(5); testScalarOp(t, 5, std::plus<>()); }
    { SCOPED_TRACE("-=5"); auto t = t1; t -= scalar(5); testScalarOp(t, 5, std::minus<>()); }
    { SCOPED_TRACE("*=5"); auto t = t1; t *= scalar(5); testScalarOp(t, 5, std::multiplies<>()); }
    { SCOPED_TRACE("/=5"); auto t = t1; t /= scalar(5); testScalarOp(t, 5, std::divides<>()); }
}

TEST_F(TensorTest, BinaryOpWithDifferentElementType) {
    Tensor<int> x({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<double> y({2, 3}, {3.1, 3.2, 3.3, 5.5, 5.6, 5.7});

    auto a = x; a += y;
    EXPECT_THAT(a, ElementsAre(4, 5, 6, 9, 10, 11));

    auto b = y; b += x;
    EXPECT_THAT(b, ElementsAre(3.1+1, 3.2+2, 3.3+3, 5.5+4, 5.6+5, 5.7+6));

    static_assert(std::is_same<decltype(x+y), Tensor<double>>::value, "");
    EXPECT_THAT(x+y, ElementsAre(1+3.1, 2+3.2, 3+3.3, 4+5.5, 5+5.6, 6+5.7));

    static_assert(std::is_same<decltype(y-x), Tensor<double>>::value, "");
    EXPECT_THAT(y-x, ElementsAre(3.1-1, 3.2-2, 3.3-3, 5.5-4, 5.6-5, 5.7-6));

    static_assert(std::is_same<decltype(x+scalar(5.5)), Tensor<double>>::value, "");
    EXPECT_THAT(x+scalar(5.5), ElementsAre(1+5.5, 2+5.5, 3+5.5, 4+5.5, 5+5.5, 6+5.5));

    static_assert(std::is_same<decltype(scalar(5.5)-x), Tensor<double>>::value, "");
    EXPECT_THAT(scalar(5.5)-x, ElementsAre(5.5-1, 5.5-2, 5.5-3, 5.5-4, 5.5-5, 5.5-6));

    EXPECT_THAT(-x, ElementsAre(-1, -2, -3, -4, -5, -6));
    EXPECT_THAT(-y, ElementsAre(-3.1, -3.2, -3.3, -5.5, -5.6, -5.7));

    Tensor<std::string> greetings({4}, {"Hello", "Bonjour", "Ciao", "Aloha"});
    greetings += scalar(" world");
    EXPECT_THAT(greetings, ElementsAre("Hello world", "Bonjour world", "Ciao world", "Aloha world"));
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
    Tensor<std::complex<double>> b({2, 2}, {-3.0-4i, 5.0-2i, 4.0+1i, 0.0-3i});
    EXPECT_EQ(a + b, Tensor<std::complex<double>>({2, 2}, {-2.0-2i, 7.0+1i, 7.0-3i, 4.0-8i}));
}

TEST_F(TensorTest, Expression) {
    SCOPED_TRACE("(5+x)*y+x*3");
    testBinaryOp((scalar(5)+t1)*t2+t1*scalar(3), [](auto x, auto y) { return (5+x)*y+x*3; });
}

TEST_F(TensorTest, ShapeBroadcastArthimetic) {
    {
        auto A = Tensor<int>({2, 3}, {1, 2, 3, 4, 5, 6});
        auto B = Tensor<int>({3}, {5, 8, 4});
        auto C = Tensor<int>({2, 3}, {6, 10, 7, 9, 13, 10});
        EXPECT_EQ(A + B, C);
    }
    {
        auto A = Tensor<int>({3}, {5, 8, 4});
        auto B = Tensor<int>({2, 3}, {1, 2, 3, 4, 5, 6});
        auto C = Tensor<int>({2, 3}, {4, 6, 1, 1, 3, -2});
        EXPECT_EQ(A - B, C);
    }
    {
        auto A = Tensor<int>({4, 1}, {3, 7, 5, 2});
        auto B = Tensor<int>({3}, {2, 6, 5});
        auto C = Tensor<int>({4, 3}, {5, 9, 8, 9, 13, 12, 7, 11, 10, 4, 8, 7});
        EXPECT_EQ(A + B, C);
    }
    {
        auto A = Tensor<int>({4});
        auto B = Tensor<int>({3});
        EXPECT_ANY_THROW(A + B);
    }

    {
        auto A = Tensor<int>::range({3, 1, 2, 1}, 1);
        auto B = Tensor<int>::range(   {4, 1, 5}, 1);

        // Computed by numpy
        auto C = Tensor<int>({3, 4, 2, 5}, {
              1,   2,   3,   4,   5,
              2,   4,   6,   8,  10,
              6,   7,   8,   9,  10,
             12,  14,  16,  18,  20,
             11,  12,  13,  14,  15,
             22,  24,  26,  28,  30,
             16,  17,  18,  19,  20,
             32,  34,  36,  38,  40,
              3,   6,   9,  12,  15,
              4,   8,  12,  16,  20,
             18,  21,  24,  27,  30,
             24,  28,  32,  36,  40,
             33,  36,  39,  42,  45,
             44,  48,  52,  56,  60,
             48,  51,  54,  57,  60,
             64,  68,  72,  76,  80,
              5,  10,  15,  20,  25,
              6,  12,  18,  24,  30,
             30,  35,  40,  45,  50,
             36,  42,  48,  54,  60,
             55,  60,  65,  70,  75,
             66,  72,  78,  84,  90,
             80,  85,  90,  95, 100,
             96, 102, 108, 114, 120
        });

        EXPECT_EQ(A * B, C);
    }
}

TEST_F(TensorTest, ShapeBroadcastCopy) {
    auto A = Tensor<int>({3, 1}, {1, 2, 3});
    auto B = Tensor<int>({2, 3, 4}, {
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
    });
    EXPECT_EQ(A.broadcast({2, 3, 4}), B);
}

template <typename T>
static void inner_test() {
    // Vector . Vector
    {
        Tensor<T> a({3}, {1, 2, 3});
        Tensor<T> b({3}, {4, 5, 6});
        Tensor<T> c({1}, {42});
        EXPECT_EQ(dot(a, b), Tensor<T>({1}, {32})); // computed by WolframAlpha
        dot(a, b, &c);
        EXPECT_EQ(c, Tensor<T>({1}, {32}));
    }

    // Vector . Matrix
    {
        Tensor<T> a({3}, {1, 2, 3});
        Tensor<T> b({{3, 2}, {4, 5, 6, 7, 8, 9}});
        Tensor<T> c({2}, {17, 53});
        EXPECT_EQ(dot(a, b), Tensor<T>({2}, {40, 46})); // computed by WolframAlpha
        dot(a, b, &c);
        EXPECT_EQ(c, Tensor<T>({2}, {40, 46}));
    }

    // Matrix . Vector
    {
        Tensor<T> a({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor<T> b({3}, {7, 8, 9});
        Tensor<T> c({2}, {17, 53});
        EXPECT_EQ(dot(a, b), Tensor<T>({2}, {50, 122})); // computed by WolframAlpha
        dot(a, b, &c);
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

        EXPECT_EQ(dot(a, b), d);
        dot(a, b, &c);
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

TEST_F(TensorTest, VectorOuter) {
    auto A = Tensor<int>({3}, {1, 2, 3});
    auto B = Tensor<int>({4}, {1, 2, 3, 4});
    auto C = Tensor<int>({3, 4}, {
        1, 2, 3, 4,
        2, 4, 6, 8,
        3, 6, 9, 12
    });
    EXPECT_EQ(outer(A, B), C);
}

TEST_F(TensorTest, MatrixOuter) {
    auto A = Tensor<int>({2, 2}, {1, 2, 3, 4});
    auto B = Tensor<int>({2, 3}, {5, 6, 7, 8, 9, 10});
    auto C = Tensor<int>({2, 2, 2, 3}, {
         5,  6,  7,  8,  9, 10,
        10, 12, 14, 16, 18, 20,
        15, 18, 21, 24, 27, 30,
        20, 24, 28, 32, 36, 40
    });
    EXPECT_EQ(outer(A, B), C);
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

    Tensor<T> t_a({6, 3}, {
         5,  7,  6,
        10,  6,  2,
         9,  6,  6,
         1,  6, 10,
        10,  1,  9,
         3,  1,  3
    });

    Tensor<T> b({6, 4}, {
        7,  1, 8,  7,
        9,  5, 2,  6,
        7,  8, 5,  7,
        6,  9, 1,  1,
        4, 10, 1, 10,
        3,  8, 8,  5
    });

    Tensor<T> t_b({4, 6}, {
        7, 9, 7, 6,  4, 3,
        1, 5, 8, 9, 10, 8,
        8, 2, 5, 1,  1, 8,
        7, 6, 7, 1, 10, 5
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

    EXPECT_EQ(gemm(T(2), a, b, T(3), c, false, false), r);
    EXPECT_EQ(gemm(T(2), t_a, b, T(3), c, true, false), r);
    EXPECT_EQ(gemm(T(2), a, t_b, T(3), c, false, true), r);
    EXPECT_EQ(gemm(T(2), t_a, t_b, T(3), c, true, true), r);

    gemm(T(2), a, b, T(3), &c);
    EXPECT_EQ(c, r);
}

TEST_F(TensorTest, Gemm) {
    gemm_test<int>();
    gemm_test<float>();
    gemm_test<double>();
    gemm_test<std::complex<float>>();
    gemm_test<std::complex<double>>();
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

TEST_F(TensorTest, ChainedApply) {
    constexpr double PI  = 3.14159;
    constexpr double E   = 2.71828;
    constexpr double PHI = 1.61803;
    constexpr double SR2 = 1.41421;

    auto sin_f  = [](double x) { return sin(x); };
    auto sqrt_f = [](double x) { return sqrt(x); };

    Tensor<double> t({2,2}, {PI, E, PHI, SR2});
    t.apply(sin_f).apply(sqrt_f);

    auto f = [](double x) { return sqrt(sin(x)); };
    EXPECT_THAT(t, testing::ElementsAre(f(PI), f(E), f(PHI), f(SR2)));
}

TEST_F(TensorTest, Transform) {
    {
        SCOPED_TRACE("");
        auto f = static_cast<double(*)(double)>(sin);
        auto t = transform(t1, f);
        static_assert(std::is_same<decltype(t), Tensor<double>>::value, "");
        checkDataTransform(t, f);
    }

    {
        SCOPED_TRACE("");
        auto f = [](auto x) { return std::complex<int>(x, x * 2); };
        auto t = transform(t1, f);
        static_assert(std::is_same<decltype(t), Tensor<std::complex<int>>>::value, "");
        checkDataTransform(t, f);
    }
}

TEST_F(TensorTest, TransformTo) {
    auto f = static_cast<double(*)(double)>(sin);
    Tensor<double> t({2,3,4});
    transformTo(t1, t, f);
    checkDataTransform(t, f);
}

TEST_F(TensorTest, Cast) {
    SCOPED_TRACE("");
    auto t = t1.cast<double>();
    static_assert(std::is_same<decltype(t), Tensor<double>>::value, "");
    checkDataTransform(t, [](auto x){ return static_cast<double>(x); });
}

TEST_F(TensorTest, Transform2) {
    auto f = [](auto x, auto y) { return (x+y)/2; };

    auto a = transform(t1, t2, f);

    Tensor<int32_t> b({2,3,4});
    transformTo(t1, t2, b, f);

    for (int i = 0; i < a.size(); i++) {
        auto v = f(data1[i], data2[i]);
        EXPECT_EQ(a.data()[i], v);
        EXPECT_EQ(b.data()[i], v);
    }
}

TEST_F(TensorTest, TransformRValueOptimization) {
    {
        Tensor<int> a({2, 2}, {1, 2, 3, 4});
        Tensor<int> b = transform(std::move(a), [](auto x) { return x*2; });
        EXPECT_TRUE(a.empty());
        EXPECT_THAT(b, ElementsAre(2, 4, 6, 8));
    }

    {
        Tensor<int> a({2, 2}, {1, 2, 3, 4});
        Tensor<int> b({2, 2}, {5, 6, 7, 8});
        Tensor<int> c = transform(std::move(a), b, std::plus<>());
        EXPECT_TRUE(a.empty());
        EXPECT_THAT(b, ElementsAre(5, 6, 7, 8));
        EXPECT_THAT(c, ElementsAre(6, 8, 10, 12));
    }

    {
        Tensor<int> a({2, 2}, {1, 2, 3, 4});
        Tensor<int> b({2, 2}, {5, 6, 7, 8});
        Tensor<int> c = transform(a, std::move(b), std::plus<>());
        EXPECT_THAT(a, ElementsAre(1, 2, 3, 4));
        EXPECT_TRUE(b.empty());
        EXPECT_THAT(c, ElementsAre(6, 8, 10, 12));
    }
}

TEST_F(TensorTest, Complex) {
    using namespace std::complex_literals;
    Tensor<std::complex<double>> t({2,2}, {1.+2i, 3.+4i, -1.+1i, 2.-5i});
    t += scalar<std::complex<double>>({1, 1});
    EXPECT_THAT(t, ElementsAre(2.+3i, 4.+5i, 0.+2i, 3.-4i));
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
    using testing::Contains;
    using testing::AllOf;
    using testing::Lt;
    using testing::Gt;
    using testing::Le;
    using testing::Ge;

    // fill a tensor with random numbers in [-500,500]
    auto in = Tensor<int32_t>::random({100, 100}, -500, 500);

    // relu with 100 as max value
    auto out = transform(in, Relu<double>(100.0));
    static_assert(std::is_same<decltype(out), Tensor<double>>::value, "");

    EXPECT_THAT(in, Contains(Lt(0)));
    EXPECT_THAT(in, Contains(Gt(100)));
    EXPECT_THAT(out, Each(AllOf(Ge(0.0), Le(100.0))));
}

TEST_F(TensorTest, Format) {
    EXPECT_EQ(format(t2),
        "["
        "[[2,3,5,7],[11,13,17,19],[23,29,31,37]],"
        "[[41,43,47,53],[57,59,61,67],[71,73,79,83]]"
        "]");

    EXPECT_EQ(format(Tensor<int>()), "");
}
