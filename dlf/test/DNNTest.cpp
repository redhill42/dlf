#include <cmath>
#include "tensor.h"
#include "gdnn.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace dlf;

template <typename T>
void ExpectEQ(T a, T b) {
    EXPECT_EQ(a, b);
}

void ExpectEQ(float a, float b) {
    if (std::isnan(a))
        EXPECT_TRUE(std::isnan(b));
    else if (std::isinf(a))
        EXPECT_TRUE(std::isinf(b));
    else
        EXPECT_FLOAT_EQ(a, b);
}

void ExpectEQ(double a, double b) {
    if (std::isnan(a))
        EXPECT_TRUE(std::isnan(b));
    else if (std::isinf(a))
        EXPECT_TRUE(std::isinf(b));
    else
        EXPECT_DOUBLE_EQ(a, b);
}

template <typename T>
void ExpectEQ(std::complex<T> a, std::complex<T> b) {
    ExpectEQ(a.real(), b.real());
    ExpectEQ(a.imag(), b.imag());
}

template <typename T, size_t N, typename Unary>
static void transform_test(const std::string& name, Unary op) {
    auto A = Tensor<T>::range({N}, N/2);
    auto B = Tensor<T>({N});

    auto dev_A = DevTensor<T>(A);
    auto dev_B = DevTensor<T>(B);

    gpgpu::dnn::transform(name, dev_A.size(), dev_A.data(), dev_B.data());
    dev_B.read(B);

    SCOPED_TRACE(name);
    for (size_t i = 0; i < N; i++) {
        T a = op(A(i)), b = B(i);
        ExpectEQ(a, b);
    }
}

template <typename T, size_t N>
static void transform_test_all() {
    transform_test<T,N>("reciprocal", [](T x){ return 1/x;});
    transform_test<T,N>("floor", [](T x){ return floor(x); });
    transform_test<T,N>("ceil", [](T x){ return ceil(x); });
    transform_test<T,N>("round", [](T x){ return round(x); });
    transform_test<T,N>("sqrt", [](T x){ return sqrt(x); });
    transform_test<T,N>("exp", [](T x){ return exp(x); });
    transform_test<T,N>("log", [](T x){ return log(x); });
    transform_test<T,N>("sin", [](T x){ return sin(x); });
    transform_test<T,N>("cos", [](T x){ return cos(x); });
    transform_test<T,N>("tan", [](T x){ return tan(x); });
    transform_test<T,N>("asin", [](T x){ return asin(x); });
    transform_test<T,N>("acos", [](T x){ return acos(x); });
    transform_test<T,N>("atan", [](T x){ return atan(x); });
    transform_test<T,N>("sinh", [](T x){ return sinh(x); });
    transform_test<T,N>("cosh", [](T x){ return cosh(x); });
    transform_test<T,N>("tanh", [](T x){ return tanh(x); });
    transform_test<T,N>("asinh", [](T x){ return asinh(x); });
    transform_test<T,N>("acosh", [](T x){ return acosh(x); });
    transform_test<T,N>("atanh", [](T x){ return atanh(x); });
    transform_test<T,N>("erf", [](T x){ return erf(x); });
}

TEST(DNNTest, Transform) {
    transform_test_all<float,200>();
    if (gpgpu::current::context().device().supportsFP64()) {
        transform_test_all<double,200>();
    }
}

template <typename T> struct TransformTest : public testing::Test {};
using TestTypes = testing::Types<int16_t, int32_t, int64_t, float>;
TYPED_TEST_CASE(TransformTest, TestTypes);

TYPED_TEST(TransformTest, Abs) {
    using T = TypeParam;
    constexpr size_t N = 200;

    auto A = Tensor<T>::range({N}, N/2);
    auto B = Tensor<T>({N});

    auto dev_A = DevTensor<T>(A);
    auto dev_B = transform(dev_A, xfn::abs<>());

    dev_B.read(B);
    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(B(i), std::abs(A(i)));
    }

    auto C = abs(dev_A).read();
    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(B(i), std::abs(A(i)));
    }
}

TYPED_TEST(TransformTest, Neg) {
    using T = TypeParam;
    constexpr size_t N = 200;

    auto A = Tensor<T>::range({N}, N/2);
    auto B = Tensor<T>({N});

    auto dev_A = DevTensor<T>(A);
    auto dev_B = transform(dev_A, xfn::negate<>());

    dev_B.read(B);
    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(B(i), -A(i));
    }
}

TYPED_TEST(TransformTest, Sign) {
    using T = TypeParam;
    constexpr size_t N = 200;

    auto A = Tensor<T>::range({N}, N/2);
    auto B = Tensor<T>({N});

    auto dev_A = DevTensor<T>(A);
    auto dev_B = transform(dev_A, xfn::sign<>());

    dev_B.read(B);
    for (size_t i = 0; i < N; i++) {
        EXPECT_EQ(B(i), A(i)<0 ? -1 : A(i)>0 ? 1 : 0);
    }
}

template <typename T> struct BinaryTest : public testing::Test {};
using BinaryTestTypes = testing::Types<int16_t, int32_t, int64_t, float, std::complex<float>>;
TYPED_TEST_CASE(BinaryTest, BinaryTestTypes);

TYPED_TEST(BinaryTest, Add) {
    using T = TypeParam;
    constexpr size_t N = 20;

    auto A = Tensor<int>::random({N}, -int(N), int(N)).template cast<T>();
    auto B = Tensor<int>::random({N}, -int(N), int(N)).template cast<T>();
    auto C = Tensor<T>({N});

    auto dev_A = DevTensor<T>(A);
    auto dev_B = DevTensor<T>(B);
    auto dev_C = transform(dev_A, dev_B, xfn::plus<>());

    dev_C.read(C);
    for (size_t i = 0; i < N; i++) {
        ExpectEQ(T(A(i) + B(i)), C(i));
    }
}

TYPED_TEST(BinaryTest, Sub) {
    using T = TypeParam;
    constexpr size_t N = 20;

    auto A = Tensor<int>::random({N}, -int(N), int(N)).template cast<T>();
    auto B = Tensor<int>::random({N}, -int(N), int(N)).template cast<T>();
    auto C = Tensor<T>({N});

    auto dev_A = DevTensor<T>(A);
    auto dev_B = DevTensor<T>(B);
    auto dev_C = transform(dev_A, dev_B, xfn::minus<>());

    dev_C.read(C);
    for (size_t i = 0; i < N; i++) {
        ExpectEQ(T(A(i) - B(i)), C(i));
    }
}

TYPED_TEST(BinaryTest, Mul) {
    using T = TypeParam;
    constexpr size_t N = 20;

    auto A = Tensor<int>::random({N}, -int(N), int(N)).template cast<T>();
    auto B = Tensor<int>::random({N}, -int(N), int(N)).template cast<T>();
    auto C = Tensor<T>({N});

    auto dev_A = DevTensor<T>(A);
    auto dev_B = DevTensor<T>(B);
    auto dev_C = transform(dev_A, dev_B, xfn::multiplies<>());

    dev_C.read(C);
    for (size_t i = 0; i < N; i++) {
        ExpectEQ(T(A(i) * B(i)), C(i));
    }
}

TYPED_TEST(BinaryTest, Div) {
    using T = TypeParam;
    constexpr size_t N = 20;

    auto A = Tensor<int>::random({N}, -int(N), int(N)).template cast<T>();
    auto B = Tensor<int>::random({N}, -int(N), int(N)).template cast<T>();
    auto C = Tensor<T>({N});

    auto dev_A = DevTensor<T>(A);
    auto dev_B = DevTensor<T>(B);
    auto dev_C = transform(dev_A, dev_B, xfn::divides<>());

    dev_C.read(C);
    for (size_t i = 0; i < N; i++) {
        if (B(i) == T(0))
            continue;
        ExpectEQ(T(A(i) / B(i)), C(i));
    }
}

TEST(BinaryTest, Pow) {
    auto A = Tensor<float>({7}, {-3, -2, -1, 0, 1, 2, 3});
    auto B = Tensor<float>({7}, {9, 4, 1, 0, 1, 4, 9});
    EXPECT_EQ(transform(A, scalar<float>(2), xfn::power<>()), B);
    EXPECT_EQ(transform(dev(A), dev<float>(2), xfn::power<>()).read(), B);
}

TEST(BinaryTest, ShapeBroadcastArthimetic) {
    {
        auto A = Tensor<int>({2, 3}, {1, 2, 3, 4, 5, 6});
        auto B = Tensor<int>({3}, {5, 8, 4});
        auto C = Tensor<int>({2, 3}, {6, 10, 7, 9, 13, 10});
        EXPECT_EQ((dev(A) + dev(B)).read(), C);
    }
    {
        auto A = Tensor<int>({3}, {5, 8, 4});
        auto B = Tensor<int>({2, 3}, {1, 2, 3, 4, 5, 6});
        auto C = Tensor<int>({2, 3}, {4, 6, 1, 1, 3, -2});
        EXPECT_EQ((dev(A) - dev(B)).read(), C);
    }
    {
        auto A = Tensor<int>({4, 1}, {3, 7, 5, 2});
        auto B = Tensor<int>({3}, {2, 6, 5});
        auto C = Tensor<int>({4, 3}, {5, 9, 8, 9, 13, 12, 7, 11, 10, 4, 8, 7});
        EXPECT_EQ((dev(A) + dev(B)).read(), C);
    }
    {
        auto A = Tensor<int>({4});
        auto B = Tensor<int>({3});
        EXPECT_ANY_THROW(dev(A) + dev(B));
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

        EXPECT_EQ((dev(A) * dev(B)).read(), C);
    }
}

TEST(DNNTest, ShapeBroadcastCopy) {
    auto A = Tensor<int>({3, 1}, {1, 2, 3});
    auto B = Tensor<int>({2, 3, 4}, {
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
    });

    auto dev_A = dev(A).broadcast({2, 3, 4});
    EXPECT_EQ(dev_A.read(), B);
}

TEST(ActivationTest, Relu) {
    constexpr size_t N = 20;

    auto A = Tensor<int>::random({N}, -int(N), int(N)).template cast<float>();
    auto dev_A = DevTensor<float>(A);
    auto B = transform(A, xfn::relu<float>());
    auto dev_B = transform(dev_A, xfn::relu<float>());
    ExpectEQ(B, dev_B.read());
}

TEST(ActivationTest, PRelu) {
    constexpr size_t N = 20;

    auto A = Tensor<int>::random({N}, -int(N), int(N)).template cast<float>();
    auto dev_A = DevTensor<float>(A);
    auto B = transform(A, scalar(0.01f), xfn::prelu<float>());
    auto dev_B = transform(dev_A, dev(0.01f), xfn::prelu<float>());
    ExpectEQ(B, dev_B.read());
}
