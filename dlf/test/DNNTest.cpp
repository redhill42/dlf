#include <cmath>
#include "tensor.h"
#include "gdnn.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test_utility.h"

using namespace dlf;

template <typename T>
void ExpectEQ(T a, T b) {
    EXPECT_EQ(a, b);
}

template <typename T>
bool isFloatEQ(T a, T b, T eps = T{1e-5}) {
    if (std::isnan(a))
        return std::isnan(b);
    if (std::isinf(a))
        return std::isinf(b);
    if (std::signbit(a) != std::signbit(b))
        return std::abs(a - b) <= eps;
    if (std::signbit(a)) {
        a = -a; b = -b;
    }

    int e = static_cast<int>(std::log10(a));
    if (e != static_cast<int>(std::log10(b)))
        return false;
    T scale = static_cast<T>(std::pow(10, std::abs(e)));
    return e >= 0
        ? std::abs(a/scale - b/scale) <= eps
        : std::abs(a*scale - b*scale) <= eps;
}

void ExpectEQ(float a, float b) {
    if (!isFloatEQ(a, b))
        FAIL() << a << " and " << b << " are not equal";
}

void ExpectEQ(double a, double b) {
    if (!isFloatEQ(a, b))
        FAIL() << a << " and " << b << " are not equal";
}

template <typename T>
void ExpectEQ(std::complex<T> a, std::complex<T> b) {
    ExpectEQ(a.real(), b.real());
    ExpectEQ(a.imag(), b.imag());
}

template <typename T>
void ExpectElementsEQ(const Tensor<T>& a, const Tensor<T>& b) {
    for (size_t i = 0; i < a.size(); i++) {
        ExpectEQ(a.data()[i], b.data()[i]);
    }
}

template <typename T, size_t N, typename Unary>
static void transform_test(const std::string& name, Unary op) {
    auto A = Tensor<T>::range({N}, N/2);
    auto B = Tensor<T>({N});

    auto dev_A = DevTensor<T>(A);
    auto dev_B = DevTensor<T>(B);

    gpgpu::dnn::transform(name, dev_A.size(), dev_A.data(), dev_B.data());
    dev_B.readTo(B);

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

    dev_B.readTo(B);
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

    dev_B.readTo(B);
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

    dev_B.readTo(B);
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

    dev_C.readTo(C);
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

    dev_C.readTo(C);
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

    dev_C.readTo(C);
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

    dev_C.readTo(C);
    for (size_t i = 0; i < N; i++) {
        if (B(i) == T(0))
            continue;
        ExpectEQ(T(A(i) / B(i)), C(i));
    }
}

TEST(BinaryTest, Pow) {
    auto A = Tensor<float>({7}, {-3, -2, -1, 0, 1, 2, 3});
    auto B = Tensor<float>({7}, {9, 4, 1, 0, 1, 4, 9});
    EXPECT_EQ(transform(A, Tensor<float>::scalar(2), xfn::power<>()), B);
    EXPECT_EQ(transform(dev(A), DevTensor<float>::scalar(2), xfn::power<>()).read(), B);
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

    auto dev_A = broadcast(dev(A), {2, 3, 4});
    EXPECT_EQ(dev_A.read(), B);
}

TEST(ActivationTest, Clip) {
    auto A = Tensor<float>({10}, {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5});
    auto B = transform(A, xfn::clip<float>(-2, 2));
    auto C = Tensor<float>({10}, {-2, -2, -2, -1, 0, 1, 2, 2, 2, 2});
    EXPECT_EQ(B, C);

    auto D = transform(dev(A), xfn::clip<float>(-2, 2)).read();
    EXPECT_EQ(D, C);
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
    auto B = transform(A, Tensor<float>::scalar(0.01f), xfn::prelu<float>());
    auto dev_B = transform(dev_A, DevTensor<float>::scalar(0.01f), xfn::prelu<float>());
    ExpectEQ(B, dev_B.read());
}

template <typename T>
Tensor<T> batch_norm_test(const Tensor<T>& x, Tensor<T> s, Tensor<T> b,
                           Tensor<T> m, Tensor<T> v, T epsilon = T(1e-5))
{
    int c = x.extent(1);
    s.reshape({1, c, 1, 1});
    b.reshape({1, c, 1, 1});
    m.reshape({1, c, 1, 1});
    v.reshape({1, c, 1, 1});
    return s * (x - m) / sqrt(v + epsilon) + b;
}

TEST(DNNTest, BatchNormalizationCPU) {
    {
        auto x = Tensor<float>({1, 2, 1, 3}, {-1, 0, 1, 2, 3, 4});
        auto s = Tensor<float>({2}, {1.0, 1.5});
        auto b = Tensor<float>({2}, {0, 1});
        auto m = Tensor<float>({2}, {0, 3});
        auto v = Tensor<float>({2}, {1, 1.5});
        auto t = batch_norm_test(x, s, b, m, v);

        auto y = Tensor<float>({1, 2, 1, 3});
        batch_norm(x, y, s, b, m, v);
        ExpectElementsEQ(t, y);
    }

    {
        auto x = Tensor<float>::random({2, 3, 4, 5}, -10, 10);
        auto s = Tensor<float>::random({3}, 0.5, 1.5);
        auto b = Tensor<float>::random({3}, 0, 1);
        auto m = Tensor<float>::random({3}, 0, 3);
        auto v = Tensor<float>::random({3}, 1, 1.5);
        auto t = batch_norm_test(x, s, b, m, v);

        auto y = Tensor<float>({2, 3, 4, 5});
        batch_norm(x, y, s, b, m, v);
        ExpectElementsEQ(t, y);
    }
}

TEST(DNNTest, BatchNormalizationGPU) {
    {
        auto x = Tensor<float>({1, 2, 1, 3}, {-1, 0, 1, 2, 3, 4});
        auto s = Tensor<float>({2}, {1.0, 1.5});
        auto b = Tensor<float>({2}, {0, 1});
        auto m = Tensor<float>({2}, {0, 3});
        auto v = Tensor<float>({2}, {1, 1.5});
        auto t = batch_norm_test(x, s, b, m, v);

        auto y = DevTensor<float>({1, 2, 1, 3});
        batch_norm(dev(x), y, dev(s), dev(b), dev(m), dev(v));
        ExpectElementsEQ(t, y.read());
    }

    {
        auto x = Tensor<float>::random({2, 3, 4, 5}, -10, 10);
        auto s = Tensor<float>::random({3}, 0.5, 1.5);
        auto b = Tensor<float>::random({3}, 0, 1);
        auto m = Tensor<float>::random({3}, 0, 3);
        auto v = Tensor<float>::random({3}, 1, 1.5);
        auto t = batch_norm_test(x, s, b, m, v);

        auto y = DevTensor<float>({2, 3, 4, 5});
        batch_norm(dev(x), y, dev(s), dev(b), dev(m), dev(v));
        ExpectElementsEQ(t, y.read());
    }
}

TEST(DNNTest, BatchNormalizationPerformanceCPU) {
    auto x = Tensor<float>::random({2, 3, 1024, 1024}, -10, 10);
    auto s = Tensor<float>::random({3}, 0.5, 1.5);
    auto b = Tensor<float>::random({3}, 0, 1);
    auto m = Tensor<float>::random({3}, 0, 3);
    auto v = Tensor<float>::random({3}, 1, 1.5);
    auto y = Tensor<float>({2, 3, 1024, 1024});

    for (int i = 0; i < 3; i++) {
        timing("Batch normalization CPU", 1, [&]() {
            for (int j = 0; j < 100; j++)
                batch_norm(x, y, s, b, m, v);
        });
    }
    std::cout << std::endl;
}

TEST(DNNTest, BatchNormalizationPerformanceGPU) {
    auto x = dev(Tensor<float>::random({2, 3, 1024, 1024}, -10, 10));
    auto s = dev(Tensor<float>::random({3}, 0.5, 1.5));
    auto b = dev(Tensor<float>::random({3}, 0, 1));
    auto m = dev(Tensor<float>::random({3}, 0, 3));
    auto v = dev(Tensor<float>::random({3}, 1, 1.5));
    auto y = dev(Tensor<float>({2, 3, 1024, 1024}));

    for (int i = 0; i < 3; i++)
        timing("Batch normalization GPU", 1, [&]() {
            for (int j = 0; j < 100; j++)
                batch_norm(x, y, s, b, m, v);
            y.read();
        });
    std::cout << std::endl;
}

TEST(Conv2D, basic_conv_with_padding) {
    auto X = Tensor<float>::range({1, 1, 5, 5}, 0);
    auto W = Tensor<float>({1, 1, 3, 3});
    std::fill(W.begin(), W.end(), 1);
    auto Y = Tensor<float>({1, 1, 5, 5});
    auto filter = FilterShape2D(X.shape(), W.shape()).pads(1, 1);
    auto R = Tensor<float>({1, 1, 5, 5}, {
        12, 21, 27, 33, 24,
        33, 54, 63, 72, 51,
        63, 99, 108, 117, 81,
        93, 144, 153, 162, 111,
        72, 111, 117, 123, 84
    });

    conv2d(X, W, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_Y = DevTensor<float>({1, 1, 5, 5});
    conv2d(dev(X), dev(W), dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(Conv2D, basic_conv_without_padding) {
    auto X = Tensor<float>::range({1, 1, 5, 5}, 0);
    auto W = Tensor<float>({1, 1, 3, 3});
    std::fill(W.begin(), W.end(), 1);
    auto Y = Tensor<float>({1, 1, 3, 3});
    auto filter = FilterShape2D(X.shape(), W.shape());
    auto R = Tensor<float>({1, 1, 3, 3}, {
        54, 63, 72,
        99, 108, 117,
        144, 153, 162
    });

    conv2d(X, W, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_Y = DevTensor<float>({1, 1, 3, 3});
    conv2d(dev(X), dev(W), dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(Conv2D, conv_with_strides_padding) {
    auto X = Tensor<float>::range({1, 1, 7, 5}, 0);
    auto W = Tensor<float>({1, 1, 3, 3});
    std::fill(W.begin(), W.end(), 1);
    auto Y = Tensor<float>({1, 1, 4, 3});
    auto filter = FilterShape2D(X.shape(), W.shape()).pads(1, 1).strides(2, 2);
    auto R = Tensor<float>({1, 1, 4, 3}, {
        12, 27, 24,
        63, 108, 81,
        123, 198, 141,
        112, 177, 124
    });

    conv2d(X, W, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_Y = DevTensor<float>({1, 1, 4, 3});
    conv2d(dev(X), dev(W), dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(Conv2D, conv_with_strides_no_padding) {
    auto X = Tensor<float>::range({1, 1, 7, 5}, 0);
    auto W = Tensor<float>({1, 1, 3, 3});
    std::fill(W.begin(), W.end(), 1);
    auto Y = Tensor<float>({1, 1, 3, 2});
    auto filter = FilterShape2D(X.shape(), W.shape()).strides(2, 2);
    auto R = Tensor<float>({1, 1, 3, 2}, {
        54, 72,
        144, 162,
        234, 252
    });

    conv2d(X, W, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_Y = DevTensor<float>({1, 1, 3, 2});
    conv2d(dev(X), dev(W), dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(Conv2D, conv_with_strides_and_asymmetric_padding) {
    auto X = Tensor<float>::range({1, 1, 7, 5}, 0);
    auto W = Tensor<float>({1, 1, 3, 3});
    std::fill(W.begin(), W.end(), 1);
    auto Y = Tensor<float>({1, 1, 4, 2});
    auto filter = FilterShape2D(X.shape(), W.shape()).pads(1, 0).strides(2, 2);
    auto R = Tensor<float>({1, 1, 4, 2}, {
        21, 33,
        99, 117,
        189, 207,
        171, 183
    });

    conv2d(X, W, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_Y = DevTensor<float>({1, 1, 4, 2});
    conv2d(dev(X), dev(W), dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(Conv2D, conv_with_multiple_channels) {
    auto X = Tensor<float>::range({2, 3, 5, 5}, 0);
    auto W = Tensor<float>({8, 3, 3, 3});
    std::fill(W.begin(), W.end(), 1);
    auto Y = Tensor<float>({2, 8, 5, 5});
    auto filter = FilterShape2D(X.shape(), W.shape()).pads(1, 1);
    auto dev_Y = DevTensor<float>({2, 8, 5, 5});

    conv2d(X, W, Y, filter);
    conv2d(dev(X), dev(W), dev_Y, filter);
    EXPECT_EQ(Y, dev_Y.read());
}

TEST(Conv2D, conv_with_strange_padding) {
    auto X = Tensor<float>::range({2, 3, 10, 10}, 0);
    auto W = Tensor<float>({8, 3, 3, 3});
    std::fill(W.begin(), W.end(), 1);
    auto filter = FilterShape2D(X.shape(), W.shape()).pads(1, 2, 2, 1);
    auto Y = Tensor<float>(filter.output_shape());
    auto dev_Y = DevTensor<float>(Y.shape());

    conv2d(X, W, Y, filter);
    conv2d(dev(X), dev(W), dev_Y, filter);
    EXPECT_EQ(Y, dev_Y.read());
}

TEST(Conv2D, performance_test) {
    auto X = Tensor<float>::range({1, 3, 1000, 1000}, 0);
    auto W = Tensor<float>::range({8, 3, 3, 3}, 0);
    auto Y = Tensor<float>({1, 8, 1000, 1000});
    auto filter = FilterShape2D(X.shape(), W.shape()).pads(1, 1);

    for (int i = 0; i < 3; i++) {
        timing("Conv2D CPU", 1, [&]() {
            conv2d(X, W, Y, filter);
        });
    }

    for (int i = 0; i < 3; i++) {
        auto dev_X = dev(X), dev_W = dev(W);
        auto dev_Y = DevTensor<float>({1, 8, 1000, 1000});
        timing("Conv2D GPU", 1, [&]() {
            conv2d(dev(X), dev(W), dev_Y, filter);
            gpgpu::current::queue().finish();
        });
    }
}

TEST(MaxPool, basic_2d_with_padding) {
    auto X = Tensor<float>::range({1, 1, 5, 5}, 1);
    auto Y = Tensor<float>({1, 1, 5, 5});
    auto filter = FilterShape2D(X.shape(), 3, 3).pads(1, 1);

    maxpool(X, Y, filter);
    EXPECT_EQ(Y, Tensor<float>({1, 1, 5, 5}, {
         7,  8,  9, 10, 10,
        12, 13, 14, 15, 15,
        17, 18, 19, 20, 20,
        22, 23, 24, 25, 25,
        22, 23, 24, 25, 25,
    }));

    auto dev_Y = DevTensor<float>({1, 1, 5, 5});
    maxpool(dev(X), dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(MaxPool, basic_2d_without_padding) {
    auto X = Tensor<float>::range({1, 1, 5, 5}, 1);
    auto Y = Tensor<float>({1, 1, 3, 3});
    auto filter = FilterShape2D(X.shape(), 3, 3);

    maxpool(X, Y, filter);
    EXPECT_EQ(Y, Tensor<float>({1, 1, 3, 3}, {
        13, 14, 15,
        18, 19, 20,
        23, 24, 25
    }));

    auto dev_Y = DevTensor<float>({1, 1, 3, 3});
    maxpool(dev(X), dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(MaxPool, basic_2d_with_multiple_channels) {
    auto X = Tensor<float>::range({2, 3, 100, 100}, 0);
    auto Y = Tensor<float>({2, 3, 100, 100});
    auto dev_Y = DevTensor<float>({2, 3, 100, 100});
    auto filter = FilterShape2D(X.shape(), 3, 3).pads(1, 1);

    maxpool(X, Y, filter);
    maxpool(dev(X), dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(AveragePool, basic_2d_with_multiple_channels) {
    auto X = Tensor<float>::range({2, 3, 100, 100}, 0);
    auto Y = Tensor<float>({2, 3, 100, 100});
    auto dev_Y = DevTensor<float>({2, 3, 100, 100});
    auto filter = FilterShape2D(X.shape(), 3, 3).pads(1, 1);

    avgpool(X, Y, filter, false);
    avgpool(dev(X), dev_Y, filter, false);
    ExpectElementsEQ(dev_Y.read(), Y);
}

TEST(DNNTest, GlobalPooling) {
    auto X = Tensor<float>::range({2, 3, 2, 2}, 1);
    auto Y = Tensor<float>({2, 3, 1, 1});

    auto max_R = Tensor<float>({2, 3, 1, 1}, {4, 8, 12, 16, 20, 24});
    auto avg_R = Tensor<float>({2, 3, 1, 1}, {2.5, 6.5, 10.5, 14.5, 18.5, 22.5});

    global_maxpool(X, Y);
    EXPECT_EQ(Y, max_R);

    global_avgpool(X, Y);
    ExpectElementsEQ(Y, avg_R);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 3, 1, 1});

    global_maxpool(dev_X, dev_Y);
    EXPECT_EQ(dev_Y.read(), max_R);

    global_avgpool(dev_X, dev_Y);
    ExpectElementsEQ(dev_Y.read(), avg_R);
}

TEST(DNNTest, Softmax) {
    auto X = Tensor<float>({2, 4}, {0, 1, 2, 3, 10000, 10001, 10002, 10003});
    auto R = Tensor<float>({2, 4}, {
        0.0320586, 0.08714432, 0.23688284, 0.64391428,
        0.0320586, 0.08714432, 0.23688284, 0.64391428
    });

    auto Y = softmax(X);
    EXPECT_EQ(Y.shape(), X.shape());
    ExpectElementsEQ(Y, R);
}
