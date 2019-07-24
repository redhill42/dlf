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
bool isFloatEQ(T a, T b, T eps = T{1e-4}) {
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
    ASSERT_EQ(a.shape(), b.shape());
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


TEST(DNNTest, TransformChannel_CPU) {
    auto A = Tensor<int>::range({2, 64, 32, 32}, 1);
    auto B = Tensor<int>::range({64, 1, 1}, 1);
    auto C = Tensor<int>(A.shape());
    auto D = Tensor<int>(A.shape());

    transformChannel(A, B, C, 1, xfn::plus<>());
    transformTo(A, broadcast(B, A.shape()), D, xfn::plus<>());
    EXPECT_EQ(C, D);
}

TEST(DNNTest, TransformChannel_GPU) {
    auto A = dev(Tensor<int>::range({2, 64, 32, 32}, 1));
    auto B = dev(Tensor<int>::range({64, 1, 1}, 1));
    auto C = DevTensor<int>(A.shape());
    auto D = DevTensor<int>(A.shape());

    transformChannel(A, B, C, 1, xfn::plus<>());

    B = broadcast(B, A.shape());
    transformTo(A, broadcast(B, A.shape()), D, xfn::plus<>());
    EXPECT_EQ(C.read(), D.read());
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
        dnn::batch_norm(x, y, s, b, m, v);
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
        dnn::batch_norm(x, y, s, b, m, v);
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
        dnn::batch_norm(dev(x), y, dev(s), dev(b), dev(m), dev(v));
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
        dnn::batch_norm(dev(x), y, dev(s), dev(b), dev(m), dev(v));
        ExpectElementsEQ(t, y.read());
    }
}

PERFORMANCE_TEST(DNNTest, BatchNormalizationPerformanceCPU) {
    auto x = Tensor<float>::random({2, 3, 1024, 1024}, -10, 10);
    auto s = Tensor<float>::random({3}, 0.5, 1.5);
    auto b = Tensor<float>::random({3}, 0, 1);
    auto m = Tensor<float>::random({3}, 0, 3);
    auto v = Tensor<float>::random({3}, 1, 1.5);
    auto y = Tensor<float>({2, 3, 1024, 1024});

    for (int i = 0; i < 3; i++) {
        timing("Batch normalization CPU", 1, [&]() {
            for (int j = 0; j < 100; j++)
                dnn::batch_norm(x, y, s, b, m, v);
        });
    }
    std::cout << std::endl;
}

PERFORMANCE_TEST(DNNTest, BatchNormalizationPerformanceGPU) {
    auto x = dev(Tensor<float>::random({2, 3, 1024, 1024}, -10, 10));
    auto s = dev(Tensor<float>::random({3}, 0.5, 1.5));
    auto b = dev(Tensor<float>::random({3}, 0, 1));
    auto m = dev(Tensor<float>::random({3}, 0, 3));
    auto v = dev(Tensor<float>::random({3}, 1, 1.5));
    auto y = dev(Tensor<float>({2, 3, 1024, 1024}));

    for (int i = 0; i < 3; i++)
        timing("Batch normalization GPU", 1, [&]() {
            for (int j = 0; j < 100; j++)
                dnn::batch_norm(x, y, s, b, m, v);
            y.read();
        });
    std::cout << std::endl;
}

TEST(DNNTest, LRN) {
    auto X = Tensor<float>::range({1, 5, 5, 5}, 1);
    auto Y = Tensor<float>({1, 5, 5, 5});
    auto R = Tensor<float>({1, 5, 5, 5}, {
         0.98340248,  1.96411627,  2.94186795,  3.91638736,  4.88740840,
         5.85467050,  6.81791520,  7.77689134,  8.73135138,  9.68105386,
        10.62576165, 11.56524689, 12.49927980, 13.42764720, 14.35013222,
        15.26652989, 16.17664199, 17.08027125, 17.97723423, 18.86735265,
        19.75044852, 20.62635898, 21.49492133, 22.35598418, 23.20940828,

        24.05450229, 24.89053469, 25.71728714, 26.53455021, 27.34213193,
        28.13984933, 28.92753747, 29.70503724, 30.47220112, 31.22889971,
        31.97500868, 32.71041834, 33.43503373, 34.14876410, 34.85154010,
        35.54329247, 36.22396578, 36.89351850, 37.55191914, 38.19914208,
        38.83517102, 39.46000255, 40.07364216, 40.67610388, 41.26740590,

        41.84757829, 42.41665458, 42.97468377, 43.52171359, 44.05780671,
        44.58301529, 45.09742275, 45.60110139, 46.09412451, 46.57659192,
        47.04858249, 47.51019673, 47.96153075, 48.40269764, 48.83379675,
        49.25493512, 49.66623608, 50.06780832, 50.45977634, 50.84225825,
        51.21538314, 51.57926004, 51.93403494, 52.27982898, 52.61676462,

        52.94497871, 53.26460905, 53.57578106, 53.87862083, 54.17327248,
        54.45985857, 54.73851936, 55.00939076, 55.27259994, 55.52828258,
        55.77656108, 56.01757908, 56.25146241, 56.47834493, 56.69834696,
        56.91160519, 57.11824264, 57.31838573, 57.51215985, 57.69968940,
        57.88109765, 58.05650673, 58.22602930, 58.38978926, 58.54790911,

        73.32580628, 73.62465776, 73.91614796, 74.20037664, 74.47743753,
        74.74743718, 75.01048222, 75.26666051, 75.51607875, 75.75883723,
        75.99503615, 76.22478164, 76.44815487, 76.66527377, 76.87623116,
        77.08111344, 77.28003711, 77.47309367, 77.66036203, 77.84196295,
        78.01798019, 78.18850301, 78.35362602, 78.51344901, 78.66805911
    });

    dnn::lrn(X, Y, 3, 0.0001f, 0.75f, 1.0f);
    ExpectElementsEQ(Y, R);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({1, 5, 5, 5});
    dnn::lrn(dev_X, dev_Y, 3, 0.0001f, 0.75f, 1.0f);
    ExpectElementsEQ(dev_Y.read(), R);
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

    dnn::conv2d(X, W, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_Y = DevTensor<float>({1, 1, 5, 5});
    dnn::conv2d(dev(X), dev(W), dev_Y, filter);
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

    dnn::conv2d(X, W, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_Y = DevTensor<float>({1, 1, 3, 3});
    dnn::conv2d(dev(X), dev(W), dev_Y, filter);
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

    dnn::conv2d(X, W, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_Y = DevTensor<float>({1, 1, 4, 3});
    dnn::conv2d(dev(X), dev(W), dev_Y, filter);
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

    dnn::conv2d(X, W, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_Y = DevTensor<float>({1, 1, 3, 2});
    dnn::conv2d(dev(X), dev(W), dev_Y, filter);
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

    dnn::conv2d(X, W, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_Y = DevTensor<float>({1, 1, 4, 2});
    dnn::conv2d(dev(X), dev(W), dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(Conv2D, conv_with_multiple_channels) {
    auto X = Tensor<float>::range({2, 3, 5, 5}, 0);
    auto W = Tensor<float>({8, 3, 3, 3});
    std::fill(W.begin(), W.end(), 1);
    auto Y = Tensor<float>({2, 8, 5, 5});
    auto filter = FilterShape2D(X.shape(), W.shape()).pads(1, 1);
    auto dev_Y = DevTensor<float>({2, 8, 5, 5});

    dnn::conv2d(X, W, Y, filter);
    dnn::conv2d(dev(X), dev(W), dev_Y, filter);
    EXPECT_EQ(Y, dev_Y.read());
}

TEST(Conv2D, conv_with_strange_padding) {
    auto X = Tensor<float>::range({2, 3, 10, 10}, 0);
    auto W = Tensor<float>({8, 3, 3, 3});
    std::fill(W.begin(), W.end(), 1);
    auto filter = FilterShape2D(X.shape(), W.shape()).pads(1, 2, 2, 1);
    auto Y = Tensor<float>(filter.output_shape());
    auto dev_Y = DevTensor<float>(Y.shape());

    dnn::conv2d(X, W, Y, filter);
    dnn::conv2d(dev(X), dev(W), dev_Y, filter);
    EXPECT_EQ(Y, dev_Y.read());
}

PERFORMANCE_TEST(Conv2D, performance_test) {
    auto X = Tensor<float>::range({1, 3, 1000, 1000}, 0);
    auto W = Tensor<float>::range({8, 3, 3, 3}, 0);
    auto Y = Tensor<float>({1, 8, 1000, 1000});
    auto filter = FilterShape2D(X.shape(), W.shape()).pads(1, 1);

    for (int i = 0; i < 3; i++) {
        timing("Conv2D CPU", 1, [&]() {
            dnn::conv2d(X, W, Y, filter);
        });
    }

    for (int i = 0; i < 3; i++) {
        auto dev_X = dev(X), dev_W = dev(W);
        auto dev_Y = DevTensor<float>({1, 8, 1000, 1000});
        timing("Conv2D GPU", 1, [&]() {
            dnn::conv2d(dev(X), dev(W), dev_Y, filter);
            gpgpu::current::queue().finish();
        });
    }
}

TEST(MaxPool, basic_2d_with_padding) {
    auto X = Tensor<float>::range({1, 1, 5, 5}, 1);
    auto Y = Tensor<float>({1, 1, 5, 5});
    auto R = Tensor<float>({1, 1, 5, 5}, {
         7,  8,  9, 10, 10,
        12, 13, 14, 15, 15,
        17, 18, 19, 20, 20,
        22, 23, 24, 25, 25,
        22, 23, 24, 25, 25,
    });
    auto filter = FilterShape2D(X.shape(), 3, 3).pads(1, 1);

    dnn::maxpool(X, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({1, 1, 5, 5});
    dnn::maxpool(dev_X, dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(MaxPool, basic_2d_without_padding) {
    auto X = Tensor<float>::range({1, 1, 5, 5}, 1);
    auto Y = Tensor<float>({1, 1, 3, 3});
    auto R = Tensor<float>({1, 1, 3, 3}, {
        13, 14, 15,
        18, 19, 20,
        23, 24, 25
    });
    auto filter = FilterShape2D(X.shape(), 3, 3);

    dnn::maxpool(X, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({1, 1, 3, 3});
    dnn::maxpool(dev_X, dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(MaxPool, basic_2d_with_dilations) {
    auto X = Tensor<float>::range({1, 1, 4, 4}, 1);
    auto Y = Tensor<float>({1, 1, 2, 2});
    auto R = Tensor<float>({1, 1, 2, 2}, {11, 12, 15, 16});
    auto filter = FilterShape2D(X.shape(), 2, 2).dilations(2, 2);

    dnn::maxpool(X, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({1, 1, 2, 2});
    dnn::maxpool(dev_X, dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(MaxPool, basic_2d_precomputed_pads) {
    auto X = Tensor<float>::range({1, 1, 5, 5}, 1);
    auto Y = Tensor<float>({1, 1, 5, 5});
    auto R = Tensor<float>({1, 1, 5, 5}, {
        13, 14, 15, 15, 15,
        18, 19, 20, 20, 20,
        23, 24, 25, 25, 25,
        23, 24, 25, 25, 25,
        23, 24, 25, 25, 25
    });
    auto filter = FilterShape2D(X.shape(), 5, 5).pads(2, 2);

    dnn::maxpool(X, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({1, 1, 5, 5});
    dnn::maxpool(dev_X, dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(MaxPool, basic_2d_precomputed_same_upper) {
    auto X = Tensor<float>::range({1, 1, 5, 5}, 1);
    auto Y = Tensor<float>({1, 1, 3, 3});
    auto R = Tensor<float>({1, 1, 3, 3}, {
        7, 9, 10, 17, 19, 20, 22, 24, 25
    });
    auto filter = FilterShape2D(X.shape(), 3, 3).strides(2, 2).auto_pad("SAME_UPPER");

    dnn::maxpool(X, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({1, 1, 3, 3});
    dnn::maxpool(dev_X, dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(MaxPool, basic_2d_precomputed_strides) {
    auto X = Tensor<float>::range({1, 1, 5, 5}, 1);
    auto Y = Tensor<float>({1, 1, 2, 2});
    auto R = Tensor<float>({1, 1, 2, 2}, {7, 9, 17, 19});
    auto filter = FilterShape2D(X.shape(), 2, 2).strides(2, 2);

    dnn::maxpool(X, Y, filter);
    EXPECT_EQ(Y, R);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({1, 1, 2, 2});
    dnn::maxpool(dev_X, dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(MaxPool, basic_2d_with_multiple_channels) {
    auto X = Tensor<float>::range({2, 3, 100, 100}, 0);
    auto Y = Tensor<float>({2, 3, 100, 100});
    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 3, 100, 100});
    auto filter = FilterShape2D(X.shape(), 3, 3).pads(1, 1);

    dnn::maxpool(X, Y, filter);
    dnn::maxpool(dev_X, dev_Y, filter);
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(AveragePool, basic_2d_with_multiple_channels) {
    auto X = Tensor<float>::range({2, 3, 100, 100}, 0);
    auto Y = Tensor<float>({2, 3, 100, 100});
    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 3, 100, 100});
    auto filter = FilterShape2D(X.shape(), 3, 3).pads(1, 1);

    dnn::avgpool(X, Y, filter, false);
    dnn::avgpool(dev_X, dev_Y, filter, false);
    ExpectElementsEQ(dev_Y.read(), Y);
}

TEST(LpPool, basic_2d_with_multiple_channels) {
    auto X = Tensor<float>::range({2, 3, 100, 100}, 0);
    auto Y = Tensor<float>({2, 3, 100, 100});
    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 3, 100, 100});
    auto filter = FilterShape2D(X.shape(), 3, 3).pads(1, 1);

    dnn::lppool(X, Y, filter, 2);
    dnn::lppool(dev_X, dev_Y, filter, 2);
    ExpectElementsEQ(dev_Y.read(), Y);
}

TEST(DNNTest, GlobalPooling) {
    auto X = Tensor<float>::range({2, 3, 2, 2}, 1);
    auto Y = Tensor<float>({2, 3, 1, 1});

    auto max_R = Tensor<float>({2, 3, 1, 1}, {4, 8, 12, 16, 20, 24});
    auto avg_R = Tensor<float>({2, 3, 1, 1}, {2.5, 6.5, 10.5, 14.5, 18.5, 22.5});

    dnn::global_maxpool(X, Y);
    EXPECT_EQ(Y, max_R);

    dnn::global_avgpool(X, Y);
    ExpectElementsEQ(Y, avg_R);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 3, 1, 1});

    dnn::global_maxpool(dev_X, dev_Y);
    EXPECT_EQ(dev_Y.read(), max_R);

    dnn::global_avgpool(dev_X, dev_Y);
    ExpectElementsEQ(dev_Y.read(), avg_R);

    dnn::global_lppool(X, Y, 2);
    dnn::global_lppool(dev_X, dev_Y, 2);
    ExpectElementsEQ(dev_Y.read(), Y);
}

TEST(DNNTest, Softmax) {
    auto X = Tensor<float>({2, 4}, {0, 1, 2, 3, 10000, 10001, 10002, 10003});
    auto R = Tensor<float>({2, 4}, {
        0.0320586, 0.08714432, 0.23688284, 0.64391428,
        0.0320586, 0.08714432, 0.23688284, 0.64391428
    });

    auto Y = dnn::softmax(X);
    EXPECT_EQ(Y.shape(), X.shape());
    ExpectElementsEQ(Y, R);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 4});
    dnn::softmax(dev_X, dev_Y);
    ExpectElementsEQ(dev_Y.read(), R);
}

TEST(DNNTest, LogSoftmax) {
    auto X = Tensor<float>({2, 4}, {0, 1, 2, 3, 10000, 10001, 10002, 10003});
    auto R = Tensor<float>({2, 4}, {
        -3.4401896, -2.4401896, -1.44018972, -0.44018969,
        -3.4401896, -2.4401896, -1.44018972, -0.44018969
    });

    auto Y = dnn::logsoftmax(X);
    EXPECT_EQ(Y.shape(), X.shape());
    ExpectElementsEQ(Y, R);
}

TEST(DNNTest, Hardmax) {
    auto X = Tensor<float>({4, 4}, {
        3, 0, 1, 2, 2, 5, 1, 0, 0, 1, 3, 2, 0, 1, 2, 3
    });
    auto R = Tensor<float>({4, 4}, {
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1
    });

    auto Y = dnn::hardmax(X);
    EXPECT_EQ(Y, R);

    auto dev_Y = dnn::hardmax(dev(X));
    EXPECT_EQ(dev_Y.read(), R);
}

TEST(DNNTest, HardmaxOneHot) {
    // For multiple occurrances of the maximal calues, the first
    // occurrence is selected for one-hot output
    auto X = Tensor<float>({1, 4}, {3, 3, 3, 1});
    auto R = Tensor<float>({1, 4}, {1, 0, 0, 0});

    auto Y = dnn::hardmax(X);
    EXPECT_EQ(Y, R);

    auto dev_Y = dnn::hardmax(dev(X));
    EXPECT_EQ(dev_Y.read(), R);
}
