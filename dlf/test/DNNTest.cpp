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
    auto dev_B = DevTensor<T>(B);

    gpgpu::dnn::abs(dev_A.size(), dev_A.data(), dev_B.data());
    dev_B.read(B);
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
    auto dev_B = DevTensor<T>(B);

    gpgpu::dnn::neg(dev_A.size(), dev_A.data(), dev_B.data());
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
    auto dev_B = DevTensor<T>(B);

    gpgpu::dnn::sign(dev_A.size(), dev_A.data(), dev_B.data());
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
    auto dev_C = DevTensor<T>(C);

    gpgpu::dnn::transform2("add", N, dev_A.data(), N, dev_B.data(), dev_C.data());
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
    auto dev_C = DevTensor<T>(C);

    gpgpu::dnn::transform2("sub", N, dev_A.data(), N, dev_B.data(), dev_C.data());
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
    auto dev_C = DevTensor<T>(C);

    gpgpu::dnn::transform2("mul", N, dev_A.data(), N, dev_B.data(), dev_C.data());
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
    auto dev_C = DevTensor<T>(C);

    gpgpu::dnn::transform2("div", N, dev_A.data(), N, dev_B.data(), dev_C.data());
    dev_C.read(C);

    for (size_t i = 0; i < N; i++) {
        if (B(i) == T(0))
            continue;
        ExpectEQ(T(A(i) / B(i)), C(i));
    }
}

TEST(ActivationTest, Relu) {
    constexpr size_t N = 20;

    auto A = Tensor<int>::random({N}, -int(N), int(N)).template cast<float>();
    auto B = Tensor<float>({N});

    auto dev_A = DevTensor<float>(A);
    auto dev_B = DevTensor<float>({N});

    relu(A, B);
    relu(dev_A, dev_B);
    ExpectEQ(B, dev_B.read());
}

TEST(ActivationTest, PRelu) {
    constexpr size_t N = 20;

    auto A = Tensor<int>::random({N}, -int(N), int(N)).template cast<float>();
    auto B = Tensor<float>({N});

    auto dev_A = DevTensor<float>(A);
    auto dev_B = DevTensor<float>({N});

    prelu(A, scalar(0.01f), B);
    prelu(dev_A, dev(0.01f), dev_B);
    ExpectEQ(B, dev_B.read());
}
