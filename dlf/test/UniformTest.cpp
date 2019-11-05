#include "tensor.h"
#include "gtest/gtest.h"
#include "test_utility.h"

using namespace dlf;

template <typename T>
void ExpectElementsEQ(const Tensor<T>& a, const Tensor<T>& b) {
    ASSERT_EQ(a.shape(), b.shape());
    for (size_t i = 0; i < a.size(); i++) {
        ExpectEQ(a.data()[i], b.data()[i]);
    }
}

TEST(UniformTest, ScalarRobust) {
    auto V = Tensor<int>({5}).range(1);
    auto x = Tensor<int>::scalar(3);
    auto y = Tensor<int>::scalar(7);
    auto v = Tensor<int>({1}, 3); // a vector instead of scalar

    auto W = Tensor<int>({5}, {3, 6, 9, 12, 15});
    EXPECT_EQ(V * x, W);
    EXPECT_EQ(x * V, W);
    EXPECT_EQ(dot(V, x), W);
    EXPECT_EQ(dot(x, V), W);
    EXPECT_EQ(inner(V, x), W);
    EXPECT_EQ(inner(x, V), W);
    EXPECT_ANY_THROW(matmul(V, x));
    EXPECT_ANY_THROW(matmul(x, V));
    EXPECT_ANY_THROW(tensordot(V, x));
    EXPECT_ANY_THROW(tensordot(x, V));

    EXPECT_EQ(matmul(V, V), Tensor<int>::scalar(55));
    EXPECT_EQ(dot(V, V), Tensor<int>::scalar(55));
    EXPECT_EQ(inner(V, V), Tensor<int>::scalar(55));
    EXPECT_ANY_THROW(tensordot(V, V));

    EXPECT_EQ(V * v, W); // broadcast
    EXPECT_EQ(v * V, W);
    EXPECT_ANY_THROW(dot(v, V)); // incompatible shape
    EXPECT_ANY_THROW(dot(V, v));
    EXPECT_ANY_THROW(inner(v, V));
    EXPECT_ANY_THROW(inner(V, v));

    EXPECT_EQ(x * y, Tensor<int>::scalar(21));
    EXPECT_EQ(dot(x, y), Tensor<int>::scalar(21));
    EXPECT_EQ(inner(x, y), Tensor<int>::scalar(21));
    EXPECT_ANY_THROW(matmul(x, y));
    EXPECT_ANY_THROW(tensordot(x, y));

    EXPECT_EQ(x, x);
    EXPECT_NE(x, y);
    EXPECT_EQ(x.transpose(), x);

    EXPECT_EQ(squeeze(x), x);
    EXPECT_ANY_THROW(squeeze(x, 0));
    EXPECT_EQ(squeeze(v), x);
    EXPECT_EQ(unsqueeze(x, 0), v);

    EXPECT_TRUE(V[0].is_scalar());
    EXPECT_TRUE(V["0"].is_vector());
    EXPECT_ANY_THROW(x[0]);
    EXPECT_ANY_THROW(x["0"]);
    EXPECT_EQ(*x, 3);
    EXPECT_EQ(x(), 3);
    EXPECT_EQ(*V[3], 4);
    EXPECT_EQ(V[3](), 4);
    EXPECT_EQ(V["3"](0), 4);
}

TEST(UniformTest, BroadcastTransform) {
    auto X = Tensor<float>({2, 3, 2, 2}).range(1);
    auto Y = Tensor<float>({3, 1, 1}).range(1);
    auto Z = X + Y;
    EXPECT_EQ(Z, Tensor<float>({2, 3, 2, 2}, {
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 14, 15,

        14, 15, 16, 17,
        19, 20, 21, 22,
        24, 25, 26, 27
    }));

    auto dev_Z = dev(X) + dev(Y);
    EXPECT_EQ(dev_Z.read(), Z);
}

TEST(UniformTest, BroadcastTransformOnView) {
    auto X = Tensor<float>({2, 3, 2, 2}).range(1);
    auto Y = Tensor<float>({6, 1, 1}, {4, 5, 6, 1, 2, 3});
    auto Z = X + Y["3:6"];
    EXPECT_EQ(Z, Tensor<float>({2, 3, 2, 2}, {
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 14, 15,

        14, 15, 16, 17,
        19, 20, 21, 22,
        24, 25, 26, 27
    }));

    auto dev_Z = dev(X) + dev(Y)["3:6"];
    EXPECT_EQ(dev_Z.read(), Z);
}

TEST(UniformTest, TransformOnConstantElements) {
    auto A = Tensor<float>({2, 2}, 0.f);
    auto B = Tensor<const float>::wrap({2, 2}, A.data());
    A + B; matmul(A, B);
    B + A; matmul(B, A);
    B + B; matmul(B, B);
}

TEST(UniformTest, Dot) {
    auto A = Tensor<float>({2, 3, 4}).range(0);
    auto B = Tensor<float>({3, 4, 5}).range(0);
    auto C = Tensor<float>({2, 3, 3, 5}, {
          70, 76,  82,  88,   94,
         190, 196, 202, 208, 214,
         310, 316, 322, 328, 334,

         190, 212, 234, 256, 278,
         630, 652, 674, 696, 718,
        1070,1092,1114,1136,1158,

         310, 348, 386, 424, 462,
        1070,1108,1146,1184,1222,
        1830,1868,1906,1944,1982,

         430, 484, 538, 592, 646,
        1510,1564,1618,1672,1726,
        2590,2644,2698,2752,2806,

         550, 620, 690, 760, 830,
        1950,2020,2090,2160,2230,
        3350,3420,3490,3560,3630,

         670, 756, 842, 928,1014,
        2390,2476,2562,2648,2734,
        4110,4196,4282,4368,4454,
    });

    EXPECT_EQ(dot(A, B), C);
    EXPECT_EQ(dot(dev(A), dev(B)).read(), C);
}

TEST(UniformTest, MultiDot) {
    auto A = Tensor<int>({2, 3}).range();
    auto B = Tensor<int>({3, 4}).range();
    auto C = Tensor<int>({4, 5}).range();
    auto D = multi_dot(A, B, C);

    EXPECT_EQ(D, Tensor<int>({2, 5}, {
         810,  908, 1006, 1104, 1202,
        2520, 2816, 3112, 3408, 3704
    }));
    EXPECT_EQ(multi_dot(dev(A), dev(B), dev(C)).read(), D);
}

TEST(UniformTest, MultiDotVectorFirst) {
    auto A = Tensor<int>({3}).range();
    auto B = Tensor<int>({3, 4}).range();
    auto C = Tensor<int>({4, 5}).range();
    auto D = multi_dot(A, B, C);

    EXPECT_EQ(D, Tensor<int>({5}, {810, 908, 1006, 1104, 1202}));
    EXPECT_EQ(multi_dot(dev(A), dev(B), dev(C)).read(), D);
}

TEST(UniformTest, MultiDotVectorLast) {
    auto A = Tensor<int>({2, 3}).range();
    auto B = Tensor<int>({3, 4}).range();
    auto C = Tensor<int>({4}).range();
    auto D = multi_dot(A, B, C);

    EXPECT_EQ(D, Tensor<int>({2}, {162, 504}));
    EXPECT_EQ(multi_dot(dev(A), dev(B), dev(C)).read(), D);
}

TEST(UniformTest, MultiDotVectorFirstAndLast) {
    auto A = Tensor<int>({3}).range();
    auto B = Tensor<int>({3, 4}).range();
    auto C = Tensor<int>({4}).range();
    auto D = multi_dot(A, B, C);

    EXPECT_EQ(D, Tensor<int>::scalar(162));
    EXPECT_EQ(multi_dot(dev(A), dev(B), dev(C)).read(), D);
}

TEST(UniformTest, TensorDotSimple) {
    auto A = Tensor<float>({3,4,5}).range();
    auto B = Tensor<float>({4,3,2}).range();
    auto C = Tensor<float>({5, 2}, {
        4400, 4730,
        4532, 4874,
        4664, 5018,
        4796, 5162,
        4928, 5306
    });

    EXPECT_EQ(tensordot(A, B, {1,0}, {0,1}), C);
    EXPECT_EQ(tensordot(dev(A), dev(B), {1,0}, {0,1}).read(), C);
}

/**
 * An extended example taking advantage of the overloading of + and *.
 */
struct Expr {
    std::string s;
    Expr() = default;
    Expr(int i) : s(i == 0 ? "" : std::to_string(i)) {}
    Expr(const char* s) : s(s) {}
    Expr(std::string s) : s(std::move(s)) {}
};

inline Expr operator+(const Expr& x, const Expr& y) {
    if (x.s.empty())
        return y;
    if (y.s.empty())
        return x;
    return Expr(x.s + "+" + y.s);
}

inline Expr& operator+=(Expr& x, const Expr& y) {
    if (!x.s.empty() && !y.s.empty())
        x.s += "+";
    if (!y.s.empty())
        x.s += y.s;
    return x;
}

inline Expr operator*(const Expr& x, const Expr& y) {
    if (x.s == "1")
        return y;
    if (y.s == "1")
        return x;
    return Expr(x.s + y.s);
}

inline bool operator==(const Expr& a, const Expr& b) {
    return a.s == b.s;
}

inline bool operator!=(const Expr& a, const Expr& b) {
    return a.s != b.s;
}

inline std::ostream& operator<<(std::ostream& out, const Expr& x) {
    return out << x.s;
}

TEST(UniformTest, TensorDotExt) {
    auto a = Tensor<Expr>({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto A = Tensor<Expr>({2, 2}, {"a", "b", "c", "d"});

    EXPECT_EQ(tensordot(a, A), Tensor<Expr>({2}, {
        "a+2b+3c+4d", "5a+6b+7c+8d"
    }));

    EXPECT_EQ(tensordot(a, A, 1), Tensor<Expr>({2, 2, 2}, {
        "a+2c", "b+2d",
        "3a+4c", "3b+4d",
        "5a+6c", "5b+6d",
        "7a+8c", "7b+8d"
    }));
    EXPECT_EQ(tensordot(a, A, {0}, {1}), Tensor<Expr>({2, 2, 2}, {
        "a+5b", "c+5d",
        "2a+6b", "2c+6d",
        "3a+7b", "3c+7d",
        "4a+8b", "4c+8d"
    }));
    EXPECT_EQ(tensordot(a, A, {2}, {1}), Tensor<Expr>({2, 2, 2}, {
        "a+2b", "c+2d",
        "3a+4b", "3c+4d",
        "5a+6b", "5c+6d",
        "7a+8b", "7c+8d"
    }));
    EXPECT_EQ(tensordot(a, A, {0,1}, {0,1}), Tensor<Expr>({2}, {
        "a+3b+5c+7d", "2a+4b+6c+8d"
    }));
    EXPECT_EQ(tensordot(a, A, {2,1}, {1,0}), Tensor<Expr>({2}, {
        "a+3c+2b+4d", "5a+7c+6b+8d"
    }));

    EXPECT_EQ(tensordot(a, A, 0), outer(a, A));
    EXPECT_EQ(tensordot(a, A, 1), dot(a, A));
}

TEST(UniformTest, MatPow) {
    auto A = Tensor<int>({2, 2}, {1, 1, 1, 0});
    EXPECT_EQ(matpow(A, 0)(0, 0), 1);
    EXPECT_EQ(matpow(A, 11)(0, 0), 144);
    EXPECT_EQ(matpow(dev(A), 0).read()(0, 0), 1);
    EXPECT_EQ(matpow(dev(A), 11).read()(0, 0), 144);
}

TEST(UniformTest, Kronecker) {
    auto A = Tensor<int>({2, 2, 3}).range(1);
    auto B = Tensor<int>({2, 2, 2}).range(1);
    auto C = kronecker(A, B);

    EXPECT_EQ(C, Tensor<int>({4, 4, 6}, {
         1,  2,  2,  4,  3,  6,
         3,  4,  6,  8,  9, 12,
         4,  8,  5, 10,  6, 12,
        12, 16, 15, 20, 18, 24,

         5,  6, 10, 12, 15, 18,
         7,  8, 14, 16, 21, 24,
        20, 24, 25, 30, 30, 36,
        28, 32, 35, 40, 42, 48,

         7, 14,  8, 16,  9, 18,
        21, 28, 24, 32, 27, 36,
        10, 20, 11, 22, 12, 24,
        30, 40, 33, 44, 36, 48,

        35, 42, 40, 48, 45, 54,
        49, 56, 56, 64, 63, 72,
        50, 60, 55, 66, 60, 72,
        70, 80, 77, 88, 84, 96
    }));

    EXPECT_EQ(kronecker(dev(A), dev(B)).read(), C);
}

TEST(UniformTest, PowCPU) {
    auto A = Tensor<float>({4}, {1, 2, 3, 4});
    auto B = pow(A, 2);
    EXPECT_EQ(B, Tensor<double>({4}, {1, 4, 9, 16}));
}

TEST(UnifromTest, PowGPU) {
    auto A = Tensor<float>({4}, {1, 2, 3, 4});
    auto B = pow(dev(A), 2.f);
    EXPECT_EQ(B.read(), Tensor<float>({4}, {1, 4, 9, 16}));
}

TEST(UniformTest, ModCPU) {
    auto A = Tensor<int>({10}).range(1);
    auto B = A % 7;
    EXPECT_EQ(B, Tensor<int>({10}, {1, 2, 3, 4, 5, 6, 0, 1, 2, 3}));
}

TEST(UniformTest, ModGPU) {
    auto A = DevTensor<float>({10}).range(1);
    auto B = A % 7.f;
    EXPECT_EQ(B.read(), Tensor<float>({10}, {1, 2, 3, 4, 5, 6, 0, 1, 2, 3}));
}

TEST(UniformTest, NestedTensor) {
    auto A = Tensor<Tensor<int>>({2, 2}, {
        {{2, 2}, { 1,  2,  3,  4}},
        {{2, 2}, { 5,  6,  7,  8}},
        {{2, 2}, { 9, 10, 11, 12}},
        {{2, 2}, {13, 14, 15, 16}}
    });

    auto B = Tensor<float>({2, 2}, {1, 2, 3, 4});

    EXPECT_EQ(A * B, Tensor<Tensor<float>>({2, 2}, {
        {{2, 2}, { 1,  2,  3,  4}},
        {{2, 2}, {10, 12, 14, 16}},
        {{2, 2}, {27, 30, 33, 36}},
        {{2, 2}, {52, 56, 60, 64}}
    }));
}

TEST(UniformTest, AggregateCPU) {
    auto A = Tensor<int>({3, 4}).range(1);
    auto B = Tensor<int>({2, 3, 1}).range(1);
    auto C = Tensor<int>({4}, {1, 2, 3, 4});

    EXPECT_EQ(sum(A, B, C), Tensor<int>({2, 3, 4}, {
         3,  5,  7,  9,
         8, 10, 12, 14,
        13, 15, 17, 19,
         6,  8, 10, 12,
        11, 13, 15, 17,
        16, 18, 20, 22
    }));

    EXPECT_EQ(mean(A, B, C), Tensor<int>({2, 3, 4}, {
        1, 1, 2, 3,
        2, 3, 4, 4,
        4, 5, 5, 6,
        2, 2, 3, 4,
        3, 4, 5, 5,
        5, 6, 6, 7
    }));

    EXPECT_EQ(max(A, B, C), Tensor<int>({2, 3, 4}, {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
        4,  4,  4,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    }));

    EXPECT_EQ(min(A, B, C), Tensor<int>({2, 3, 4}, {
         1, 1, 1, 1,
         1, 2, 2, 2,
         1, 2, 3, 3,
         1, 2, 3, 4,
         1, 2, 3, 4,
         1, 2, 3, 4
     }));
}

TEST(UniformTest, AggregateGPU) {
    auto A = DevTensor<int>({3, 4}).range(1);
    auto B = DevTensor<int>({2, 3, 1}).range(1);
    auto C = dev(Tensor<int>({4}, {1, 2, 3, 4}));

    EXPECT_EQ(sum(A, B, C).read(), Tensor<int>({2, 3, 4}, {
         3,  5,  7,  9,
         8, 10, 12, 14,
        13, 15, 17, 19,
         6,  8, 10, 12,
        11, 13, 15, 17,
        16, 18, 20, 22
    }));

    EXPECT_EQ(mean(A, B, C).read(), Tensor<int>({2, 3, 4}, {
        1, 1, 2, 3,
        2, 3, 4, 4,
        4, 5, 5, 6,
        2, 2, 3, 4,
        3, 4, 5, 5,
        5, 6, 6, 7
    }));

    EXPECT_EQ(max(A, B, C).read(), Tensor<int>({2, 3, 4}, {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
        4,  4,  4,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    }));

    EXPECT_EQ(min(A, B, C).read(), Tensor<int>({2, 3, 4}, {
         1, 1, 1, 1,
         1, 2, 2, 2,
         1, 2, 3, 3,
         1, 2, 3, 4,
         1, 2, 3, 4,
         1, 2, 3, 4
     }));
}

TEST(UniformTest, MinMaxGPU) {
    auto A = dev(Tensor<float>({4}, {-2.718, 3.14, 5.25, 1.234}));
    auto B = dev(Tensor<float>({4}, {4.178, 1.412, 4.13, 2.913}));
    EXPECT_EQ(max(A, B).read(), Tensor<float>({4}, {4.178, 3.14, 5.25, 2.913}));
    EXPECT_EQ(min(A, B).read(), Tensor<float>({4}, {-2.718, 1.412, 4.13, 1.234}));
}

TEST(UniformTest, BitwiseCPU) {
    auto A = Tensor<short>({4}).range();
    EXPECT_EQ(A | 1, Tensor<int>({4}, {1, 1, 3, 3}));
    EXPECT_EQ(A & 1, Tensor<int>({4}, {0, 1, 0, 1}));
    EXPECT_EQ(A ^ 1, Tensor<int>({4}, {1, 0, 3, 2}));
}

TEST(UniformTest, BitwiseGPU) {
    auto A = DevTensor<int>({4}).range();
    EXPECT_EQ((A | 1).read(), Tensor<int>({4}, {1, 1, 3, 3}));
    EXPECT_EQ((A & 1).read(), Tensor<int>({4}, {0, 1, 0, 1}));
    EXPECT_EQ((A ^ 1).read(), Tensor<int>({4}, {1, 0, 3, 2}));
}

TEST(UniformTest, ReshapeCPU) {
    auto A = Tensor<int>({2, 3}, {1, 2, 3, 4, 5, 6});
    auto R = Tensor<int>({6}, {1, 2, 3, 4, 5, 6});

    auto B = reshape(A, {6});
    EXPECT_EQ(B, R);

    EXPECT_ANY_THROW(reshape(A, {7}));
}

TEST(UniformTest, ReshapeGPU) {
    auto A = dev(Tensor<int>({2, 3}, {1, 2, 3, 4, 5, 6}));
    auto R = Tensor<int>({6}, {1, 2, 3, 4, 5, 6});

    auto B = reshape(A, {6});
    EXPECT_EQ(B.read(), R);

    EXPECT_ANY_THROW(reshape(A, {7}));
}

TEST(UniformTest, FlattenCPU) {
    auto A = Tensor<int>({2, 3, 4}).range(1);
    EXPECT_EQ(flatten(A, 0), Tensor<int>({1, 24}).range(1));
    EXPECT_EQ(flatten(A, 1), Tensor<int>({2, 12}).range(1));
    EXPECT_EQ(flatten(A, 2), Tensor<int>({6, 4}).range(1));
    EXPECT_EQ(flatten(A, 3), Tensor<int>({24, 1}).range(1));
    EXPECT_ANY_THROW(flatten(A, 4));
}

TEST(UniformTest, FlattenGPU) {
    auto A = DevTensor<int>({2, 3, 4}).range(1);
    EXPECT_EQ(flatten(A, 0).read(), Tensor<int>({1, 24}).range(1));
    EXPECT_EQ(flatten(A, 1).read(), Tensor<int>({2, 12}).range(1));
    EXPECT_EQ(flatten(A, 2).read(), Tensor<int>({6, 4}).range(1));
    EXPECT_EQ(flatten(A, 3).read(), Tensor<int>({24, 1}).range(1));
    EXPECT_ANY_THROW(flatten(A, 4));
}

TEST(UniformTest, SqueezeCPU) {
    auto A = Tensor<int>({2, 1, 3, 1, 4}).range(1);
    EXPECT_EQ(squeeze(A), Tensor<int>({2, 3, 4}).range(1));
    EXPECT_EQ(squeeze(A, 1), Tensor<int>({2, 3, 1, 4}).range(1));
    EXPECT_EQ(squeeze(A, -2), Tensor<int>({2, 1, 3, 4}).range(1));
    EXPECT_ANY_THROW(squeeze(A, 0));
}

TEST(UniformTest, SqueezeToScalar) {
    auto A = Tensor<int>({1, 1}, 123);
    EXPECT_EQ(squeeze(A), Tensor<int>::scalar(123));
}

TEST(UniformTest, UnsqueezeCPU) {
    auto A = Tensor<int>({2, 3, 4}).range(1);
    EXPECT_EQ(unsqueeze(A, 1, -2), Tensor<int>({2, 1, 3, 1, 4}).range(1));
    EXPECT_ANY_THROW(unsqueeze(A, 1, 5));
}

TEST(UniformTest, ConcatCPU) {
    {
        auto A = Tensor<int>({2, 3, 2}).range(1);
        auto B = Tensor<int>({3, 3, 2}).range(-1, -1);
        auto C = Tensor<int>({4, 3, 2}).range(24, -1);
        auto D = concat(0, A, B, C);
        EXPECT_EQ(D, Tensor<int>({9, 3, 2}, {
               1,   2,   3,   4,   5,   6,
               7,   8,   9,  10,  11,  12,
              -1,  -2,  -3,  -4,  -5,  -6,
              -7,  -8,  -9, -10, -11, -12,
             -13, -14, -15, -16, -17, -18,
              24,  23,  22,  21,  20,  19,
              18,  17,  16,  15,  14,  13,
              12,  11,  10,   9,   8,   7,
               6,   5,   4,   3,   2,   1
        }));
    }

    {
        auto A = Tensor<int>({2, 2, 2}).range(1);
        auto B = Tensor<int>({2, 3, 2}).range(-1, -1);
        auto C = Tensor<int>({2, 4, 2}).range(16, -1);
        auto D = concat(1, A, B, C);
        EXPECT_EQ(D, Tensor<int>({2, 9, 2}, {
             1,   2,   3,   4,
            -1,  -2,  -3,  -4,  -5,  -6,
            16,  15,  14,  13,  12,  11,  10,  9,
             5,   6,   7,   8,
            -7,  -8,  -9, -10, -11, -12,
             8,   7,   6,   5,   4,   3,   2,  1
        }));
    }

    {
        auto A = Tensor<int>({2, 2, 2}).range(1);
        auto B = Tensor<int>({2, 2, 3}).range(-1, -1);
        auto C = Tensor<int>({2, 2, 4}).range(16, -1);
        auto D = concat(2, A, B, C);
        EXPECT_EQ(D, Tensor<int>({2, 2, 9}, {
             1,  2,  -1,  -2,  -3,  16,  15,  14,  13,
             3,  4,  -4,  -5,  -6,  12,  11,  10,   9,
             5,  6,  -7,  -8,  -9,   8,   7,   6,   5,
             7,  8, -10, -11, -12,   4,   3,   2,   1
        }));
    }

    {
        auto A = Tensor<int>({2, 3, 4}).range(1);
        auto B = Tensor<int>({4, 3, 2}).range(1);
        auto C = concat(1, A.transpose(), B);
        EXPECT_EQ(C, Tensor<int>({4, 6, 2}, {
            1, 13, 5, 17,  9, 21,  1,  2,  3,  4,  5,  6,
            2, 14, 6, 18, 10, 22,  7,  8,  9, 10, 11, 12,
            3, 15, 7, 19, 11, 23, 13, 14, 15, 16, 17, 18,
            4, 16, 8, 20, 12, 24, 19, 20, 21, 22, 23, 24
        }));
    }
}

TEST(UniformTest, ConcatGPU) {
    {
        auto A = DevTensor<int>({2, 3, 2}).range(1);
        auto B = DevTensor<int>({3, 3, 2}).range(-1, -1);
        auto C = DevTensor<int>({4, 3, 2}).range(24, -1);
        auto D = concat(0, A, B, C);
        EXPECT_EQ(D.read(), Tensor<int>({9, 3, 2}, {
               1,   2,   3,   4,   5,   6,
               7,   8,   9,  10,  11,  12,
              -1,  -2,  -3,  -4,  -5,  -6,
              -7,  -8,  -9, -10, -11, -12,
             -13, -14, -15, -16, -17, -18,
              24,  23,  22,  21,  20,  19,
              18,  17,  16,  15,  14,  13,
              12,  11,  10,   9,   8,   7,
               6,   5,   4,   3,   2,   1
        }));
    }

    {
        auto A = DevTensor<int>({2, 2, 2}).range(1);
        auto B = DevTensor<int>({2, 3, 2}).range(-1, -1);
        auto C = DevTensor<int>({2, 4, 2}).range(16, -1);
        auto D = concat(1, A, B, C);
        EXPECT_EQ(D.read(), Tensor<int>({2, 9, 2}, {
             1,   2,   3,   4,
            -1,  -2,  -3,  -4,  -5,  -6,
            16,  15,  14,  13,  12,  11,  10,  9,
             5,   6,   7,   8,
            -7,  -8,  -9, -10, -11, -12,
             8,   7,   6,   5,   4,   3,   2,  1
        }));
    }

    {
        auto A = DevTensor<int>({2, 2, 2}).range(1);
        auto B = DevTensor<int>({2, 2, 3}).range(-1, -1);
        auto C = DevTensor<int>({2, 2, 4}).range(16, -1);
        auto D = concat(2, A, B, C);
        EXPECT_EQ(D.read(), Tensor<int>({2, 2, 9}, {
             1,  2,  -1,  -2,  -3,  16,  15,  14,  13,
             3,  4,  -4,  -5,  -6,  12,  11,  10,   9,
             5,  6,  -7,  -8,  -9,   8,   7,   6,   5,
             7,  8, -10, -11, -12,   4,   3,   2,   1
        }));
    }

    {
        auto A = DevTensor<int>({2, 3, 4}).range(1);
        auto B = DevTensor<int>({4, 3, 2}).range(1);
        auto C = concat(1, A.transpose(), B);
        EXPECT_EQ(C.read(), Tensor<int>({4, 6, 2}, {
            1, 13, 5, 17,  9, 21,  1,  2,  3,  4,  5,  6,
            2, 14, 6, 18, 10, 22,  7,  8,  9, 10, 11, 12,
            3, 15, 7, 19, 11, 23, 13, 14, 15, 16, 17, 18,
            4, 16, 8, 20, 12, 24, 19, 20, 21, 22, 23, 24
        }));
    }
}

TEST(UniformTest, SplitCPU) {
    {
        auto X = Tensor<int>({9, 3, 2}).range(1);
        auto A = Tensor<int>({2, 3, 2});
        auto B = Tensor<int>({3, 3, 2});
        auto C = Tensor<int>({4, 3, 2});
        auto v = std::vector<TensorView<int>>{A.view(), B.view(), C.view()};
        split(X, 0, v.begin(), v.end());
        EXPECT_EQ(A, Tensor<int>({2, 3, 2}, {
             1,  2,  3,  4,  5,  6,
             7,  8,  9, 10, 11, 12}));
        EXPECT_EQ(B, Tensor<int>({3, 3, 2}, {
            13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30}));
        EXPECT_EQ(C, Tensor<int>({4, 3, 2}, {
            31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42,
            43, 44, 45, 46, 47, 48,
            49, 50, 51, 52, 53, 54
        }));

        auto splits = split(X, 0, {2, 3, 4});
        EXPECT_EQ(splits[0], A);
        EXPECT_EQ(splits[1], B);
        EXPECT_EQ(splits[2], C);
    }

    {
        /*
             1,  2,  3,
             4,  5,  6,
             7,  8,  9,
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
            19, 20, 21,
            22, 23, 24,
            25, 26, 27,
            28, 29, 30,
            31, 32, 33,
            34, 35, 36,
            37, 38, 39,
            40, 41, 42,
            43, 44, 45,
            46, 47, 48,
            49, 50, 51,
            52, 53, 54

        */
        auto X = Tensor<int>({2, 9, 3}).range(1);
        auto A = Tensor<int>({2, 2, 3});
        auto B = Tensor<int>({2, 3, 3});
        auto C = Tensor<int>({2, 4, 3});
        auto v = std::vector<TensorView<int>>{A.view(), B.view(), C.view()};
        split(X, 1, v.begin(), v.end());
        EXPECT_EQ(A, Tensor<int>({2, 2, 3}, {
             1,  2,  3,
             4,  5,  6,
            28, 29, 30,
            31, 32, 33
        }));
        EXPECT_EQ(B, Tensor<int>({2, 3, 3}, {
             7,  8,  9,
            10, 11, 12,
            13, 14, 15,
            34, 35, 36,
            37, 38, 39,
            40, 41, 42,
        }));
        EXPECT_EQ(C, Tensor<int>({2, 4, 3}, {
            16, 17, 18,
            19, 20, 21,
            22, 23, 24,
            25, 26, 27,
            43, 44, 45,
            46, 47, 48,
            49, 50, 51,
            52, 53, 54
        }));

        auto splits = split(X, 1, {2, 3, 4});
        EXPECT_EQ(splits[0], A);
        EXPECT_EQ(splits[1], B);
        EXPECT_EQ(splits[2], C);
    }

    {
        /*
             1,  2,  3,  4,  5,  6,  7,  8,  9,
            10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54
         */
        auto X = Tensor<int>({2, 3, 9}).range(1);
        auto A = Tensor<int>({2, 3, 2});
        auto B = Tensor<int>({2, 3, 3});
        auto C = Tensor<int>({2, 3, 4});
        auto v = std::vector<TensorView<int>>{A.view(), B.view(), C.view()};
        split(X, 2, v.begin(), v.end());
        EXPECT_EQ(A, Tensor<int>({2, 3, 2}, {
            1, 2, 10, 11, 19, 20, 28, 29, 37, 38, 46, 47
        }));
        EXPECT_EQ(B, Tensor<int>({2, 3, 3}, {
             3,  4,  5, 12, 13, 14, 21, 22, 23,
            30, 31, 32, 39, 40, 41, 48, 49, 50
        }));
        EXPECT_EQ(C, Tensor<int>({2, 3, 4}, {
             6,  7,  8,  9, 15, 16, 17, 18,
            24, 25, 26, 27, 33, 34, 35, 36,
            42, 43, 44, 45, 51, 52, 53, 54
        }));

        auto splits = split(X, 2, {2, 3, 4});
        EXPECT_EQ(splits[0], A);
        EXPECT_EQ(splits[1], B);
        EXPECT_EQ(splits[2], C);
    }
}

TEST(UniformTest, SplitGPU) {
    {
        auto X = DevTensor<int>({9, 3, 2}).range(1);
        auto A = DevTensor<int>({2, 3, 2});
        auto B = DevTensor<int>({3, 3, 2});
        auto C = DevTensor<int>({4, 3, 2});
        auto v = std::vector<DevTensorView<int>>{A.view(), B.view(), C.view()};
        split(X, 0, v.begin(), v.end());
        EXPECT_EQ(A.read(), Tensor<int>({2, 3, 2}, {
            1, 2, 3,  4,  5,  6,
            7, 8, 9, 10, 11, 12}));
        EXPECT_EQ(B.read(), Tensor<int>({3, 3, 2}, {
            13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30}));
        EXPECT_EQ(C.read(), Tensor<int>({4, 3, 2}, {
            31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42,
            43, 44, 45, 46, 47, 48,
            49, 50, 51, 52, 53, 54
        }));

        auto splits = split(X, 0, {2, 3, 4});
        EXPECT_EQ(splits[0].read(), A.read());
        EXPECT_EQ(splits[1].read(), B.read());
        EXPECT_EQ(splits[2].read(), C.read());
    }

    /*
         1,  2,  3,
         4,  5,  6,
         7,  8,  9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
        19, 20, 21,
        22, 23, 24,
        25, 26, 27,
        28, 29, 30,
        31, 32, 33,
        34, 35, 36,
        37, 38, 39,
        40, 41, 42,
        43, 44, 45,
        46, 47, 48,
        49, 50, 51,
        52, 53, 54
     */
    {
        auto X = DevTensor<int>({2, 9, 3}).range(1);
        auto A = DevTensor<int>({2, 2, 3});
        auto B = DevTensor<int>({2, 3, 3});
        auto C = DevTensor<int>({2, 4, 3});
        auto v = std::vector<DevTensorView<int>>{A.view(), B.view(), C.view()};
        split(X, 1, v.begin(), v.end());
        EXPECT_EQ(A.read(), Tensor<int>({2, 2, 3}, {
             1,  2,  3,
             4,  5,  6,
            28, 29, 30,
            31, 32, 33
        }));
        EXPECT_EQ(B.read(), Tensor<int>({2, 3, 3}, {
             7,  8,  9,
            10, 11, 12,
            13, 14, 15,
            34, 35, 36,
            37, 38, 39,
            40, 41, 42,
        }));
        EXPECT_EQ(C.read(), Tensor<int>({2, 4, 3}, {
            16, 17, 18,
            19, 20, 21,
            22, 23, 24,
            25, 26, 27,
            43, 44, 45,
            46, 47, 48,
            49, 50, 51,
            52, 53, 54
        }));

        auto splits = split(X, 1, {2, 3, 4});
        EXPECT_EQ(splits[0].read(), A.read());
        EXPECT_EQ(splits[1].read(), B.read());
        EXPECT_EQ(splits[2].read(), C.read());
    }

    /*
         1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, 52, 53, 54
    */
    {
        auto X = DevTensor<int>({2, 3, 9}).range(1);
        auto A = DevTensor<int>({2, 3, 2});
        auto B = DevTensor<int>({2, 3, 3});
        auto C = DevTensor<int>({2, 3, 4});
        auto v = std::vector<DevTensorView<int>>{A.view(), B.view(), C.view()};
        split(X, 2, v.begin(), v.end());
        EXPECT_EQ(A.read(), Tensor<int>({2, 3, 2}, {
            1, 2, 10, 11, 19, 20, 28, 29, 37, 38, 46, 47
        }));
        EXPECT_EQ(B.read(), Tensor<int>({2, 3, 3}, {
             3,  4,  5, 12, 13, 14, 21, 22, 23,
            30, 31, 32, 39, 40, 41, 48, 49, 50
        }));
        EXPECT_EQ(C.read(), Tensor<int>({2, 3, 4}, {
             6,  7,  8,  9, 15, 16, 17, 18,
            24, 25, 26, 27, 33, 34, 35, 36,
            42, 43, 44, 45, 51, 52, 53, 54
        }));

        auto splits = split(X, 2, {2, 3, 4});
        EXPECT_EQ(splits[0].read(), A.read());
        EXPECT_EQ(splits[1].read(), B.read());
        EXPECT_EQ(splits[2].read(), C.read());
    }
}

TEST(UnifromTest, Join) {
    auto A = Tensor<int>::identity({2, 2}, 2);
    auto B = Tensor<int>({2, 3}).fill(0);
    auto C = Tensor<int>({3, 2}).fill(1);
    auto D = Tensor<int>::identity({3, 3}, 3);

    auto R = join(Matrix({
        {A.view(), B.view()},
        {C.view(), D.view()}
    }));

    EXPECT_EQ(R, Matrix({
        {2, 0, 0, 0, 0},
        {0, 2, 0, 0, 0},
        {1, 1, 3, 0, 0},
        {1, 1, 0, 3, 0},
        {1, 1, 0, 0, 3}
    }));

    auto dev_R = join(Matrix({
        {dev(A), dev(B)},
        {dev(C), dev(D)}
    }));
    EXPECT_EQ(dev_R.read(), R);

    EXPECT_EQ(join(Vector({Matrix({{1, 2}}), Scalar(3)})),
              Matrix({{1, 2, 3}}));
}

TEST(UniformTest, Slice) {
    auto X = Tensor<float>({10, 10, 5}).range();
    auto Y = Tensor<float>({3, 4, 5}, {
        115, 116, 117, 118, 119,
        120, 121, 122, 123, 124,
        125, 126, 127, 128, 129,
        130, 131, 132, 133, 134,

        165, 166, 167, 168, 169,
        170, 171, 172, 173, 174,
        175, 176, 177, 178, 179,
        180, 181, 182, 183, 184,

        215, 216, 217, 218, 219,
        220, 221, 222, 223, 224,
        225, 226, 227, 228, 229,
        230, 231, 232, 233, 234,
    });

    EXPECT_EQ(X["2:5, 3:7, :"], Y);
    EXPECT_EQ(dev(X)["2:5, 3:7, :"].read(), Y);
}

TEST(UniformTest, SliceWithStep) {
    auto X = Tensor<float>({10, 10, 5}).range(0);
    auto Y2 = Tensor<float>({3, 2, 5}, {
        115, 116, 117, 118, 119,
        125, 126, 127, 128, 129,

        165, 166, 167, 168, 169,
        175, 176, 177, 178, 179,

        215, 216, 217, 218, 219,
        225, 226, 227, 228, 229,
    });
    auto Y3 = Tensor<float>({3, 2, 5}, {
        115, 116, 117, 118, 119,
        130, 131, 132, 133, 134,

        165, 166, 167, 168, 169,
        180, 181, 182, 183, 184,

        215, 216, 217, 218, 219,
        230, 231, 232, 233, 234,
    });
    auto Y4 = Tensor<float>({3, 1, 5}, {
        115, 116, 117, 118, 119,
        165, 166, 167, 168, 169,
        215, 216, 217, 218, 219,
    });

    EXPECT_EQ(X["2:5, 3:7:2"], Y2);
    EXPECT_EQ(dev(X)["2:5, 3:7:2"].read(), Y2);
    EXPECT_EQ(X["2:5, 3:7:3"], Y3);
    EXPECT_EQ(dev(X)["2:5, 3:7:3"].read(), Y3);
    EXPECT_EQ(X["2:5, 3:7:4"], Y4);
    EXPECT_EQ(dev(X)["2:5, 3:7:4"].read(), Y4);
    EXPECT_EQ(X["2:5, 3:7:5"], Y4);
    EXPECT_EQ(dev(X)["2:5, 3:7:5"].read(), Y4);
}

TEST(UniformTest, SliceWithNegativeStep) {
    auto X = Tensor<float>({10, 10, 5}).range(0);
    auto Y1 = Tensor<float>({3, 4, 5}, {
        135, 136, 137, 138, 139,
        130, 131, 132, 133, 134,
        125, 126, 127, 128, 129,
        120, 121, 122, 123, 124,

        185, 186, 187, 188, 189,
        180, 181, 182, 183, 184,
        175, 176, 177, 178, 179,
        170, 171, 172, 173, 174,

        235, 236, 237, 238, 239,
        230, 231, 232, 233, 234,
        225, 226, 227, 228, 229,
        220, 221, 222, 223, 224,
    });
    auto Y2 = Tensor<float>({3, 2, 5}, {
        135, 136, 137, 138, 139,
        125, 126, 127, 128, 129,

        185, 186, 187, 188, 189,
        175, 176, 177, 178, 179,

        235, 236, 237, 238, 239,
        225, 226, 227, 228, 229,
    });
    auto Y3 = Tensor<float>({3, 2, 5}, {
        135, 136, 137, 138, 139,
        120, 121, 122, 123, 124,

        185, 186, 187, 188, 189,
        170, 171, 172, 173, 174,

        235, 236, 237, 238, 239,
        220, 221, 222, 223, 224,
    });
    auto Y4 = Tensor<float>({3, 1, 5}, {
        135, 136, 137, 138, 139,
        185, 186, 187, 188, 189,
        235, 236, 237, 238, 239,
    });

    EXPECT_EQ(X["2:5, 7:3:-1"], Y1);
    EXPECT_EQ(dev(X)["2:5, 7:3:-1"].read(), Y1);
    EXPECT_EQ(X["2:5, 7:3:-2"], Y2);
    EXPECT_EQ(dev(X)["2:5, 7:3:-2"].read(), Y2);
    EXPECT_EQ(X["2:5, 7:3:-3"], Y3);
    EXPECT_EQ(dev(X)["2:5, 7:3:-3"].read(), Y3);
    EXPECT_EQ(X["2:5, 7:3:-4"], Y4);
    EXPECT_EQ(dev(X)["2:5, 7:3:-4"].read(), Y4);
    EXPECT_EQ(X["2:5, 7:3:-5"], Y4);
    EXPECT_EQ(dev(X)["2:5, 7:3:-5"].read(), Y4);
}

TEST(UniformTest, SliceOfSlice) {
    auto X = Tensor<float>({10, 10, 5}).range();
    auto Y = Tensor<float>({2, 2, 5});

    auto shape = X.shape().slice({{2, 5}, {3, 7}}).slice({{1, 3}, {1, 3}});
    reorder(X, shape, Y);
    EXPECT_EQ(Y, Tensor<float>({2, 2, 5}, {
        170, 171, 172, 173, 174,
        175, 176, 177, 178, 179,
        220, 221, 222, 223, 224,
        225, 226, 227, 228, 229
    }));

    EXPECT_EQ(X["2:5, 3:7"]["1:3, 1:3"], Y);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 2, 5});
    reorder(dev_X, shape, dev_Y);
    EXPECT_EQ(dev_Y.read(), Y);
    EXPECT_EQ(dev_X["2:5, 3:7"]["1:3, 1:3"].read(), Y);
}

TEST(UniformTest, SliceAndTranspose) {
    auto X = Tensor<float>({10, 10, 5}).range();
    auto Y = Tensor<float>({5, 4, 3});

    auto shape = X.shape().slice({{2, 5}, {3, 7}}).transpose();
    reorder(X, shape, Y);

    EXPECT_EQ(Y, Tensor<float>({5, 4, 3}, {
        115, 165, 215,
        120, 170, 220,
        125, 175, 225,
        130, 180, 230,

        116, 166, 216,
        121, 171, 221,
        126, 176, 226,
        131, 181, 231,

        117, 167, 217,
        122, 172, 222,
        127, 177, 227,
        132, 182, 232,

        118, 168, 218,
        123, 173, 223,
        128, 178, 228,
        133, 183, 233,

        119, 169, 219,
        124, 174, 224,
        129, 179, 229,
        134, 184, 234,
    }));

    EXPECT_EQ(X["2:5, 3:7"].transpose(), Y);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({5, 4, 3});
    reorder(dev_X, shape, dev_Y);
    EXPECT_EQ(dev_Y.read(), Y);
    EXPECT_EQ(dev_X["2:5, 3:7"].transpose().read(), Y);
}

TEST(UniformTest, CopyNonContiguousSlice) {
    auto X = Tensor<float>({2, 2, 10}).range();
    auto X1 = X;
    auto dev_X = dev(X);
    auto dev_X1 = dev(X1);

    auto src_shape = X.shape().slice({{}, {}, {1, 4}});
    auto dst_shape = X.shape().slice({{}, {}, {5, 8}});
    reorder(X, src_shape, X, dst_shape);

    EXPECT_EQ(X, Tensor<float>({2, 2, 10}, {
         0,  1,  2,  3,  4,  1,  2,  3,  8,  9,
        10, 11, 12, 13, 14, 11, 12, 13, 18, 19,

        20, 21, 22, 23, 24, 21, 22, 23, 28, 29,
        30, 31, 32, 33, 34, 31, 32, 33, 38, 39
    }));

    reorder(X1[":, :, 1:4"], X1[":, :, 5:8"]);
    EXPECT_EQ(X1, X);

    reorder(dev_X, src_shape, dev_X, dst_shape);
    EXPECT_EQ(dev_X.read(), X);
    reorder(dev_X1[":, :, 1:4"], dev_X1[":, :, 5:8"]);
    EXPECT_EQ(dev_X1.read(), X);
}

TEST(UniformTest, CopyContiguousSlice) {
    auto X = Tensor<float>({3, 2, 10}).range(0);
    auto X1 = X;
    auto dev_X = dev(X);
    auto dev_X1 = dev(X1);

    auto src_shape = X.shape().slice({{1, 2}});
    auto dst_shape = X.shape().slice({{2, 3}});
    reorder(X, src_shape, X, dst_shape);

    EXPECT_EQ(X, Tensor<float>({3, 2, 10}, {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,

        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,

        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    }));

    reorder(X1["1"], X1["2"]);
    EXPECT_EQ(X1, X);

    reorder(dev_X, src_shape, dev_X, dst_shape);
    EXPECT_EQ(dev_X.read(), X);

    reorder(dev_X1["1"], dev_X1["2"]);
    EXPECT_EQ(dev_X1.read(), X);
}

TEST(UniformTest, CopyToSlice) {
    auto X = Tensor<float>({3, 3, 5}).range();
    auto X1 = X;
    auto Y = Tensor<float>({2, 2, 2}, {1, 4, 2, 8, 5, 7, 6, 9});

    auto dev_X = dev(X);
    auto dev_X1 = dev(X1);
    auto dev_Y = dev(Y);

    auto R = Tensor<float>({3, 3, 5}, {
         0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,

        15, 16, 17, 18, 19,
        20,  1,  4, 23, 24,
        25,  2,  8, 28, 29,

        30, 31, 32, 33, 34,
        35,  5,  7, 38, 39,
        40,  6,  9, 43, 44
    });

    auto shape = X.shape().slice({{1, 3}, {1, 3}, {1, 3}});
    reorder(Y, Y.shape(), X, shape);
    EXPECT_EQ(X, R);

    reorder(Y, X1["1:3, 1:3, 1:3"]);
    EXPECT_EQ(X1, R);

    reorder(dev_Y, dev_Y.shape(), dev_X, shape);
    EXPECT_EQ(dev_X.read(), R);

    reorder(dev_Y, dev_X1["1:3, 1:3, 1:3"]);
    EXPECT_EQ(dev_X1.read(), R);
}

TEST(UniformTest, BroadcastCopyToSlice) {
    auto X = Tensor<float>({5}, {-1, -2, -3, -4, -5});
    auto Y = Tensor<float>({3, 3, 5}).range();
    auto X1 = X, Y1 = Y;

    auto dev_X = dev(X);
    auto dev_X1 = dev(X1);
    auto dev_Y = dev(Y);
    auto dev_Y1 = dev(Y1);

    auto src_shape = X.shape().broadcast({1, 3, 5});
    auto dst_shape = Y.shape().slice({{1, 2}});

    auto R = Tensor<float>({3, 3, 5}, {
         0,  1,  2,  3,  4,
         5,  6,  7,  8,  9,
        10, 11, 12, 13, 14,

        -1, -2, -3, -4, -5,
        -1, -2, -3, -4, -5,
        -1, -2, -3, -4, -5,

        30, 31, 32, 33, 34,
        35, 36, 37, 38, 39,
        40, 41, 42, 43, 44
    });

    reorder(X, src_shape, Y, dst_shape);
    EXPECT_EQ(Y, R);

    reorder(X1.broadcast({1, 3, 5}), Y1["1"]);
    EXPECT_EQ(Y1, R);

    reorder(dev_X, src_shape, dev_Y, dst_shape);
    EXPECT_EQ(dev_Y.read(), R);

    reorder(dev_X1.broadcast({1,3,5}), dev_Y1["1"]);
    EXPECT_EQ(dev_Y1.read(), R);
}

TEST(UniformTest, BroadcastSlice) {
    /*
     * 0 1 2 3
     * 0 1 2 3
     * 0 1 2 3
     *
     * 4 5 6 7
     * 4 5 6 7
     * 4 5 6 7
     */
    auto X = Tensor<float>({2, 1, 4}).range(0);
    auto Y = Tensor<float>({2, 2, 2});
    auto Y1 = Y;

    auto shape = X.shape().broadcast({2, 3, 4}).slice({{}, {1, 3}, {1, 3}});
    reorder(X, shape, Y);
    EXPECT_EQ(Y, Tensor<float>({2, 2, 2}, {
        1, 2, 1, 2, 5, 6, 5, 6
    }));

    reorder(X.broadcast({2,3,4})[":, 1:3, 1:3"], Y1);
    EXPECT_EQ(Y1, Y);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 2, 2});
    reorder(dev_X, shape, dev_Y);
    EXPECT_EQ(dev_Y.read(), Y);

    auto dev_Y1 = DevTensor<float>({2, 2, 2});
    reorder(dev_X.broadcast({2,3,4})[":, 1:3, 1:3"], dev_Y1);
    EXPECT_EQ(dev_Y1.read(), Y);
}

TEST(UniformTest, SliceBroadcast) {
    /*
     *  0  1  2  3
     *  4  5  6  7
     *  8  9 10 11
     *
     * 12 13 14 15
     * 16 17 18 19
     * 20 21 22 23
     */
    auto X = Tensor<float>({2, 3, 4}).range(0);
    auto Y = Tensor<float>({2, 3, 4});
    auto Y1 = Y;

    auto shape = X.shape().slice({{}, {}, {1, 2}}).broadcast({2, 3, 4});
    reorder(X, shape, Y);
    EXPECT_EQ(Y, Tensor<float>({2, 3, 4}, {
         1,  1,  1,  1,
         5,  5,  5,  5,
         9,  9,  9,  9,
        13, 13, 13, 13,
        17, 17, 17, 17,
        21, 21, 21, 21
    }));

    reorder(X[":, :, 1"].broadcast({2,3,4}), Y1);
    EXPECT_EQ(Y1, Y);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 3, 4});
    reorder(dev_X, shape, dev_Y);
    EXPECT_EQ(dev_Y.read(), Y);

    auto dev_Y1 = DevTensor<float>({2, 3, 4});
    reorder(dev_X[":, :, 1"].broadcast({2,3,4}), dev_Y1);
    EXPECT_EQ(dev_Y1.read(), Y);
}

template <typename T> struct TransposeTest : public testing::Test {};
using TransposeTestTypes = testing::Types<int, float>;
TYPED_TEST_CASE(TransposeTest, TransposeTestTypes);

TYPED_TEST(TransposeTest, Transpose1D_CPU) {
    auto A = Tensor<TypeParam>({4}, {1, 2, 3, 4});
    auto B = Tensor<TypeParam>({4, 1}, {1, 2, 3, 4});
    EXPECT_EQ(A.transpose(), B);
}

TYPED_TEST(TransposeTest, Transpose1D_GPU) {
    auto A = dev(Tensor<TypeParam>({4}, {1, 2, 3, 4}));
    auto B = Tensor<TypeParam>({4, 1}, {1, 2, 3, 4});
    EXPECT_EQ(A.transpose().read(), B);
}

TYPED_TEST(TransposeTest, TransposeSquare_CPU) {
    auto A = Tensor<TypeParam>({3, 3}).range(1);
    auto B = Tensor<TypeParam>({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9});
    EXPECT_EQ(A.transpose(), B);
}

TYPED_TEST(TransposeTest, TransposeSquare_GPU) {
    auto A = DevTensor<TypeParam>({3, 3}).range(1);
    auto B = Tensor<TypeParam>({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9});
    EXPECT_EQ(A.transpose().read(), B);
}

TYPED_TEST(TransposeTest, Transpose2D_CPU) {
    auto A = Tensor<TypeParam>({3, 4}).range(1);
    auto B = Tensor<TypeParam>({4, 3}, {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
    EXPECT_EQ(A.transpose(), B);
}

TYPED_TEST(TransposeTest, Transpose2D_GPU) {
    auto A = DevTensor<TypeParam>({3, 4}).range(1);
    auto B = Tensor<TypeParam>({4, 3}, {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
    EXPECT_EQ(A.transpose().read(), B);
}

TYPED_TEST(TransposeTest, Transpose3D_CPU) {
    auto A = Tensor<TypeParam>({2, 3, 4}).range(1);
    auto B = Tensor<TypeParam>({4, 3, 2}, {
        1, 13, 5, 17,  9, 21,
        2, 14, 6, 18, 10, 22,
        3, 15, 7, 19, 11, 23,
        4, 16, 8, 20, 12, 24
    });
    EXPECT_EQ(A.transpose(), B);
}

TYPED_TEST(TransposeTest, Transpose3D_GPU) {
    auto A = DevTensor<TypeParam>({2, 3, 4}).range(1);
    auto B = Tensor<TypeParam>({4, 3, 2}, {
        1, 13, 5, 17,  9, 21,
        2, 14, 6, 18, 10, 22,
        3, 15, 7, 19, 11, 23,
        4, 16, 8, 20, 12, 24
    });
    EXPECT_EQ(A.transpose().read(), B);
}

TYPED_TEST(TransposeTest, TransposePerm_CPU) {
    auto A = Tensor<TypeParam>({2, 3, 4}).range(1);
    auto B1 = Tensor<TypeParam>({3, 2, 4}, {
         1,  2,  3,  4,
        13, 14, 15, 16,
         5,  6,  7,  8,
        17, 18, 19, 20,
         9, 10, 11, 12,
        21, 22, 23, 24
    });
    auto B2 = Tensor<TypeParam>({2, 4, 3}, {
         1,  5,  9,
         2,  6, 10,
         3,  7, 11,
         4,  8, 12,
        13, 17, 21,
        14, 18, 22,
        15, 19, 23,
        16, 20, 24
    });
    EXPECT_EQ(A.transpose(1, 0, 2), B1);
    EXPECT_EQ(A.transpose(0, 2, 1), B2);
}

TYPED_TEST(TransposeTest, TransposePerm_GPU) {
    auto A = DevTensor<TypeParam>({2, 3, 4}).range(1);
    auto B1 = Tensor<TypeParam>({3, 2, 4}, {
         1,  2,  3,  4,
        13, 14, 15, 16,
         5,  6,  7,  8,
        17, 18, 19, 20,
         9, 10, 11, 12,
        21, 22, 23, 24
    });
    auto B2 = Tensor<TypeParam>({2, 4, 3}, {
         1,  5,  9,
         2,  6, 10,
         3,  7, 11,
         4,  8, 12,
        13, 17, 21,
        14, 18, 22,
        15, 19, 23,
        16, 20, 24
    });
    EXPECT_EQ(A.transpose(1, 0, 2).read(), B1);
    EXPECT_EQ(A.transpose(0, 2, 1).read(), B2);
}

TEST(UniformTest, MoveAxis) {
    auto X = Tensor<int>({3, 4, 5, 6});
    EXPECT_EQ(moveaxis(X, 0, -1).shape(), Shape(4, 5, 6, 3));
    EXPECT_EQ(moveaxis(X, -1, 0).shape(), Shape(6, 3, 4, 5));
    EXPECT_EQ(moveaxis(X, {0,-1}, {-1,0}).shape(), Shape({6, 4, 5, 3}));
    EXPECT_EQ(moveaxis(X, {0,1}, {-1,-2}).shape(), Shape(5, 6, 4, 3));
    EXPECT_EQ(moveaxis(X, {0,3}, {-1,2}).shape(), Shape(4, 5, 6, 3));
}

TYPED_TEST(TransposeTest, SwapAxes) {
    auto A = Tensor<TypeParam>({2, 2, 2}).range(0);
    auto B = Tensor<TypeParam>({2, 2, 2}, {0, 4, 2, 6, 1, 5, 3, 7});
    EXPECT_EQ(swapaxes(A, 0, 2), B);
    EXPECT_EQ(swapaxes(dev(A), 0, 2).read(), B);
}

TEST(UniformTest, Flip) {
    auto A = Tensor<float>({2, 2, 2}).range();

    EXPECT_EQ(flip(A, 0), Tensor<float>({2, 2, 2}, {
        4, 5, 6, 7, 0, 1, 2, 3
    }));
    EXPECT_EQ(flip(A, 1), Tensor<float>({2, 2, 2}, {
        2, 3, 0, 1, 6, 7, 4, 5
    }));
    EXPECT_EQ(flip(A), Tensor<float>({2, 2, 2}, {
        7, 6, 5, 4, 3, 2, 1, 0
    }));
    EXPECT_EQ(flip(A, {0, 2}), Tensor<float>({2, 2, 2}, {
        5, 4, 7, 6, 1, 0, 3, 2
    }));

    auto B = Tensor<int>({3, 4, 5}).random(0, 100);
    EXPECT_EQ(flip(B, 2), B[":,:,::-1"]);
}

TEST(UniformTest, Rot90) {
    auto A = Tensor<float>({2, 2}, {1, 2, 3, 4});
    EXPECT_EQ(rot90(A), Tensor<float>({2, 2}, {2, 4, 1, 3}));
    EXPECT_EQ(rot90(A, 2), Tensor<float>({2, 2}, {4, 3, 2, 1}));
    EXPECT_EQ(rot90(A, 3), Tensor<float>({2, 2}, {3, 1, 4, 2}));
    EXPECT_EQ(rot90(A, 4), Tensor<float>({2, 2}, {1, 2, 3, 4}));

    EXPECT_EQ(rot90(rot90(A)), rot90(A, 2));
    EXPECT_EQ(rot90(rot90(rot90(A))), rot90(A, 3));

    auto B = Tensor<float>({2, 3, 4}).range();
    EXPECT_EQ(rot90(B, 1, 0, 1), Tensor<float>({3, 2, 4}, {
        8,  9, 10, 11, 20, 21, 22, 23,
        4,  5,  6,  7, 16, 17, 18, 19,
        0,  1,  2,  3, 12, 13, 14, 15
    }));
    EXPECT_EQ(rot90(B, 1, 1, 0), Tensor<float>({3, 2, 4}, {
        12, 13, 14, 15,  0,  1,  2,  3,
        16, 17, 18, 19,  4,  5,  6,  7,
        20, 21, 22, 23,  8,  9, 10, 11
    }));
    EXPECT_EQ(rot90(B, 1, 0, 2), Tensor<float>({4, 3, 2}, {
         3, 15,  7, 19, 11, 23,  2, 14,  6, 18, 10, 22,
         1, 13,  5, 17,  9, 21,  0, 12,  4, 16,  8, 20,
    }));
    EXPECT_EQ(rot90(B, 1, 2, 0), Tensor<float>({4, 3, 2}, {
        12,  0, 16,  4, 20,  8, 13,  1, 17,  5, 21,  9,
        14,  2, 18,  6, 22, 10, 15,  3, 19,  7, 23, 11
    }));
    EXPECT_EQ(rot90(B, 1, 1, 2), Tensor<float>({2, 4, 3}, {
         3,  7, 11,  2,  6, 10,  1,  5,  9,  0,  4,  8,
        15, 19, 23, 14, 18, 22, 13, 17, 21, 12, 16, 20
    }));
    EXPECT_EQ(rot90(B, 1, 2, 1), Tensor<float>({2, 4, 3}, {
         8,  4,  0,  9,  5,  1, 10,  6,  2, 11,  7,  3,
        20, 16, 12, 21, 17, 13, 22, 18, 14, 23, 19, 15
    }));
}

TEST(UniformTest, Where_CPU) {
    auto condition = Tensor<bool>({2}, {true, false});
    auto X = Tensor<int>({2, 2}, {1, 2, 3, 4});
    auto Y = Tensor<int>({2, 2}, {5, 6, 7, 8});
    auto Z = where(condition, X, Y);
    EXPECT_EQ(Z, Tensor<int>({2, 2}, {1, 6, 3, 8}));
}

TEST(UniformTest, Where_GPU) {
    auto condition = dev(Tensor<bool>({2}, {true, false}));
    auto X = dev(Tensor<int>({2, 2}, {1, 2, 3, 4}));
    auto Y = dev(Tensor<int>({2, 2}, {5, 6, 7, 8}));
    auto Z = where(condition, X, Y);
    EXPECT_EQ(Z.read(), Tensor<int>({2, 2}, {1, 6, 3, 8}));
}

TEST(UniformTest, WhereExpr_CPU) {
    auto X = Tensor<int>({10}).range();
    auto Y = where(X<5, X, 10*X);
    EXPECT_EQ(Y, Tensor<int>({10}, {0, 1, 2, 3, 4, 50, 60, 70, 80, 90}));
}

TEST(UniformTest, WhereExpr_GPU) {
    auto X = DevTensor<int>({10}).range();
    auto Y = where(X<5, X, 10*X);
    EXPECT_EQ(Y.read(), Tensor<int>({10}, {0, 1, 2, 3, 4, 50, 60, 70, 80, 90}));
}

TEST(UniformTest, WhereView_CPU) {
    auto condition = Tensor<bool>({2}, {true, false});
    auto X = Tensor<int>({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto Z = where(condition["0:2"], X["0:2"], X["2:4"]);
    EXPECT_EQ(Z, Tensor<int>({2, 2}, {1, 6, 3, 8}));
}

TEST(UniformTest, WhereView_GPU) {
    auto condition = dev(Tensor<bool>({2}, {true, false}));
    auto X = dev(Tensor<int>({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));
    auto Z = where(condition["0:2"], X["0:2"], X["2:4"]);
    EXPECT_EQ(Z.read(), Tensor<int>({2, 2}, {1, 6, 3, 8}));
}

TEST(UniformTest, onehot_without_axis) {
    auto indices = Vector<int>({0, 7, 8});
    auto values  = Vector<float>({2, 5});
    auto output  = Tensor<float>();
    one_hot(indices, values, output, 12);
    EXPECT_EQ(output, Tensor<float>({3, 12}, {
        5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2,
    }));

    auto dev_out = DevTensor<float>();
    one_hot(dev(indices.cast<float>()), dev(values), dev_out, 12);
    EXPECT_EQ(dev_out.read(), output);
}

TEST(UniformTest, onehot_with_axis) {
    auto indices = Matrix<float>({{1, 9}, {2, 4}});
    auto values  = Vector<float>({1, 3});
    auto output  = Tensor<float>();
    one_hot(indices, values, output, 10, 1);
    EXPECT_EQ(output, Tensor<float>({2, 10, 2}, {
        1,1, 3,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,3,
        1,1, 1,1, 3,1, 1,1, 1,3, 1,1, 1,1, 1,1, 1,1, 1,1
    }));

    auto dev_out = DevTensor<float>();
    one_hot(dev(indices), dev(values), dev_out, 10, 1);
    EXPECT_EQ(dev_out.read(), output);
}

TEST(UniformTest, onehot_with_negative_indices) {
    auto indices = Vector<int>({0, -7, -8});
    auto values  = Vector<float>({1, 3});
    auto output  = Tensor<float>();
    one_hot(indices, values, output, 10);
    EXPECT_EQ(output, Tensor<float>({3, 10}, {
        3, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 3, 1, 1, 1, 1, 1, 1,
        1, 1, 3, 1, 1, 1, 1, 1, 1, 1
    }));

    auto dev_out = DevTensor<float>();
    one_hot(dev(indices.cast<float>()), dev(values), dev_out, 10);
    EXPECT_EQ(dev_out.read(), output);
}

TEST(UniformTest, onehot_with_negative_axis) {
    auto indices = Matrix<float>({{1, 9}, {2, 4}});
    auto values  = Vector<float>({1, 3});
    auto output  = Tensor<float>();
    one_hot(indices, values, output, 10, -2);
    EXPECT_EQ(output, Tensor<float>({2, 10, 2}, {
        1,1, 3,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,3,
        1,1, 1,1, 3,1, 1,1, 1,3, 1,1, 1,1, 1,1, 1,1, 1,1
    }));

    auto dev_out = DevTensor<float>();
    one_hot(dev(indices), dev(values), dev_out, 10, -2);
    EXPECT_EQ(dev_out.read(), output);
}

TEST(UniformTest, Gather_0) {
    auto A = Matrix<float>({{1, 2}, {3, 4}, {5, 6}});
    auto indices = Matrix<int>({{0, 1}, {1, 2}});
    auto B = Tensor<float>({2, 2, 2}, {1, 2, 3, 4, 3, 4, 5, 6});
    EXPECT_EQ(gather(A, indices), B);
    EXPECT_EQ(gather(dev(A), dev(indices)).read(), B);
}

TEST(UniformTest, Gather_1) {
    auto A = Matrix<float>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    auto indices = Matrix<int>({{0, 2}});
    auto B = Tensor<float>({3, 1, 2}, {1, 3, 4, 6, 7, 9});
    auto C = Tensor<float>({3, 1, 2}, {1, 7, 2, 8, 3, 9});
    EXPECT_EQ(gather(A, indices, 1), B);
    EXPECT_EQ(gather(A.transpose(), indices, 1), C);
    EXPECT_EQ(gather(dev(A), dev(indices), 1).read(), B);
    EXPECT_EQ(gather(dev(A).transpose(), dev(indices), 1).read(), C);
}

TEST(UniformTest, GatherElements_0) {
    auto A = Matrix<int>({{1, 2}, {3, 4}});
    auto indices = Matrix<int>({{0, 0}, {1, 0}});
    auto B = Matrix<int>({{1, 1}, {4, 3}});
    EXPECT_EQ(gather_elements(A, indices, 1), B);
    EXPECT_EQ(gather_elements(dev(A), dev(indices), 1).read(), B);
}

TEST(UniformTest, GatherElements_1) {
    auto A = Matrix<int>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    auto indices = Matrix<int>({{1, 2, 0}, {2, 0, 0}});
    auto B = Matrix<int>({{4, 8, 3}, {7, 2, 3}});
    auto C = Matrix<int>({{2, 6, 7}, {3, 4, 7}});
    EXPECT_EQ(gather_elements(A, indices, 0), B);
    EXPECT_EQ(gather_elements(A.transpose(), indices, 0), C);
    EXPECT_EQ(gather_elements(dev(A), dev(indices), 0).read(), B);
    EXPECT_EQ(gather_elements(dev(A).transpose(), dev(indices), 0).read(), C);
}

TEST(UniformTest, GatherElements_NegativeIndices) {
    auto A = Matrix<int>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    auto indices = Matrix<int>({{-1, -2, 0}, {-2, 0, 0}});
    auto B = Matrix<int>({{7, 5, 3}, {4, 2, 3}});
    auto C = Matrix<int>({{3, 5, 7}, {2, 4, 7}});
    EXPECT_EQ(gather_elements(A, indices, 0), B);
    EXPECT_EQ(gather_elements(A.transpose(), indices, 0), C);
    EXPECT_EQ(gather_elements(dev(A), dev(indices), 0).read(), B);
    EXPECT_EQ(gather_elements(dev(A).transpose(), dev(indices), 0).read(), C);
}

TEST(UniformTest, ScatterElements_0) {
    auto A = Matrix<float>({{1, 2, 3, 4, 5}});
    auto dev_A = dev(A);
    auto indices = Matrix<int>({{1, 3}});
    auto updates = Matrix<float>({{1.1, 2.1}});
    auto B = Matrix<float>({{1, 1.1, 3, 2.1, 5}});

    scatter_elements(A, indices, updates, 1);
    EXPECT_EQ(A, B);

    scatter_elements(dev_A, dev(indices), dev(updates), 1);
    EXPECT_EQ(dev_A.read(), B);
}

TEST(UniformTest, ScatterElements_1) {
    auto A = Tensor<float>({3, 3}).fill(0);
    auto dev_A = dev(A);
    auto indices = Matrix<int>({{1, 0, 2}, {0, 2, 1}});
    auto updates = Matrix<float>({{1.0, 1.1, 1.2}, {2.0, 2.1, 2.2}});
    auto B = Matrix<float>({
        {2.0, 1.1, 0.0},
        {1.0, 0.0, 2.2},
        {0.0, 2.1, 1.2}
    });

    scatter_elements(A, indices, updates);
    EXPECT_EQ(A, B);

    scatter_elements(dev_A, dev(indices), dev(updates));
    EXPECT_EQ(dev_A.read(), B);
}

TEST(UniformTest, ScatterElements_WithNegativeIndices) {
    auto A = Matrix<float>({{1, 2, 3, 4, 5}});
    auto dev_A = dev(A);
    auto indices = Matrix<int>({{1, -3}});
    auto updates = Matrix<float>({{1.1, 2.1}});
    auto B = Matrix<float>({{1.0, 1.1, 2.1, 4.0, 5.0}});

    scatter_elements(A, indices, updates, 1);
    EXPECT_EQ(A, B);

    scatter_elements(dev_A, dev(indices), dev(updates), 1);
    EXPECT_EQ(dev_A.read(), B);
}

TEST(UniformTest, GatherND_1) {
    auto A = Matrix({{0, 1}, {2, 3}});

    auto I1 = Matrix({{0, 0,}, {1, 1}});
    auto B1 = Vector({0, 3});
    EXPECT_EQ(gather_nd(A, I1), B1);
    EXPECT_EQ(gather_nd(dev(A), dev(I1)).read(), B1);

    auto I2 = Matrix({{1}, {0}});
    auto B2 = Matrix({{2, 3}, {0, 1}});
    EXPECT_EQ(gather_nd(A, I2), B2);
    EXPECT_EQ(gather_nd(dev(A), dev(I2)).read(), B2);
}

TEST(UniformTest, GatherND_2) {
    auto A = make_tensor<int, 3>({{{0,1}, {2,3}}, {{4,5}, {6,7}}});

    auto I1 = Matrix({{0,1}, {1,0}});
    auto B1 = Matrix({{2,3}, {4,5}});
    auto B1_t = Matrix({{1,3}, {4,6}});
    EXPECT_EQ(gather_nd(A, I1), B1);
    EXPECT_EQ(gather_nd(A.transpose(0,2,1), I1), B1_t);
    EXPECT_EQ(gather_nd(dev(A), dev(I1)).read(), B1);
    EXPECT_EQ(gather_nd(dev(A).transpose(0,2,1), dev(I1)).read(), B1_t);

    auto I2 = make_tensor<int, 3>({{{0,1}}, {{1,0}}});
    auto B2 = make_tensor<int, 3>({{{2,3}}, {{4,5}}});
    auto B2_t = make_tensor<int, 3>({{{1,3}}, {{4,6}}});
    EXPECT_EQ(gather_nd(A, I2), B2);
    EXPECT_EQ(gather_nd(A.transpose(0,2,1), I2), B2_t);
    EXPECT_EQ(gather_nd(dev(A), dev(I2)).read(), B2);
    EXPECT_EQ(gather_nd(dev(A).transpose(0,2,1), dev(I2)).read(), B2_t);
}

TEST(UniformTest, ScatterND_1) {
    auto A = Vector({1, 2, 3, 4, 5, 6, 7, 8});
    auto dev_A = dev(A);
    auto indices = Matrix({{4}, {3}, {1}, {7}});
    auto updates = Vector({9, 10, 11, 12});
    auto B = Vector({1, 11, 3, 10, 9, 6, 7, 12});

    scatter_nd(A, indices, updates);
    EXPECT_EQ(A, B);

    scatter_nd(dev_A, dev(indices), dev(updates));
    EXPECT_EQ(dev_A.read(), B);
}

TEST(UniformTest, ScatterND_2) {
    auto A = make_tensor<int, 3>({
        {{1, 2, 3, 4}, {5, 6, 7, 8}, {8, 7, 6, 5}, {4, 3, 2, 1}},
        {{1, 2, 3, 4}, {5, 6, 7, 8}, {8, 7, 6, 5}, {4, 3, 2, 1}},
        {{8, 7, 6, 5}, {4, 3, 2, 1}, {1, 2, 3, 4}, {5, 6, 7, 8}},
        {{8, 7, 6, 5}, {4, 3, 2, 1}, {1, 2, 3, 4}, {5, 6, 7, 8}}
    });
    auto dev_A = dev(A);
    auto indices = Matrix({{0}, {2}});
    auto updates = make_tensor<int, 3>({
        {{5, 5, 5, 5}, {6, 6, 6, 6}, {7, 7, 7, 7}, {8, 8, 8, 8}},
        {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}}
    });
    auto B = make_tensor<int, 3>({
        {{5, 5, 5, 5}, {6, 6, 6, 6}, {7, 7, 7, 7}, {8, 8, 8, 8}},
        {{1, 2, 3, 4}, {5, 6, 7, 8}, {8, 7, 6, 5}, {4, 3, 2, 1}},
        {{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}},
        {{8, 7, 6, 5}, {4, 3, 2, 1}, {1, 2, 3, 4}, {5, 6, 7, 8}}
    });

    scatter_nd(A, indices, updates);
    EXPECT_EQ(A, B);

    scatter_nd(dev_A, dev(indices), dev(updates));
    EXPECT_EQ(dev_A.read(), B);
}

TEST(UniformTest, ReduceMax) {
    auto X = Tensor<float>({3, 2, 2}, {
        5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2
    });

    EXPECT_EQ(reduce_max(X, {}, true), Tensor<float>({1,1,1}, {60}));
    EXPECT_EQ(reduce_max(X, {1}, false), Tensor<float>({3,2}, {20, 2, 40, 2, 60, 2}));
    EXPECT_EQ(reduce_max(X, {1}, true), Tensor<float>({3,1,2}, {20, 2, 40, 2, 60, 2}));
}

TEST(UniformTest, ReduceMax_GPU) {
    auto X = Tensor<float>({3, 2, 2}, {
        5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2
    });

    EXPECT_EQ(reduce_max(dev(X), {}).read(), reduce_max(X, {}));
    EXPECT_EQ(reduce_max(dev(X), {0}).read(), reduce_max(X, {0}));
    EXPECT_EQ(reduce_max(dev(X), {1}).read(), reduce_max(X, {1}));
    EXPECT_EQ(reduce_max(dev(X), {2}).read(), reduce_max(X, {2}));
    EXPECT_EQ(reduce_max(dev(X), {0,2}).read(), reduce_max(X, {0,2}));
}

TEST(UniformTest, ReduceMin) {
    auto X = Tensor<float>({3, 2, 2}, {
        5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2
    });

    EXPECT_EQ(reduce_min(X, {}, true), Tensor<float>({1,1,1}, {1}));
    EXPECT_EQ(reduce_min(X, {1}, false), Tensor<float>({3,2}, {5, 1, 30, 1, 55, 1}));
    EXPECT_EQ(reduce_min(X, {1}, true), Tensor<float>({3,1,2}, {5, 1, 30, 1, 55, 1}));
}

TEST(UniformTest, ReduceMin_GPU) {
    auto X = Tensor<float>({3, 2, 2}, {
        5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2
    });

    EXPECT_EQ(reduce_min(dev(X), {}).read(), reduce_min(X, {}));
    EXPECT_EQ(reduce_min(dev(X), {0}).read(), reduce_min(X, {0}));
    EXPECT_EQ(reduce_min(dev(X), {1}).read(), reduce_min(X, {1}));
    EXPECT_EQ(reduce_min(dev(X), {2}).read(), reduce_min(X, {2}));
    EXPECT_EQ(reduce_min(dev(X), {0,2}).read(), reduce_min(X, {0,2}));
}

TEST(UniformTest, ReduceSum) {
    auto X = Tensor<float>({3,3,3}).range();

    EXPECT_EQ(reduce_sum(X, {0}), Tensor<float>({3,3}, {
        27, 30, 33,
        36, 39, 42,
        45, 48, 51
    }));

    EXPECT_EQ(reduce_sum(X, {1}), Tensor<float>({3, 3}, {
         9, 12, 15,
        36, 39, 42,
        63, 66, 69
    }));

    EXPECT_EQ(reduce_sum(X, {2}), Tensor<float>({3, 3}, {
         3, 12, 21,
        30, 39, 48,
        57, 66, 75
    }));

    EXPECT_EQ(reduce_sum(X, {0,2}), Tensor<float>({3}, {90, 117, 144}));
}

TEST(UniformTest, ReduceSum_GPU) {
    auto X = Tensor<float>({3, 3, 3}).range();
    EXPECT_EQ(reduce_sum(dev(X), {}).read(), reduce_sum(X, {}));
    EXPECT_EQ(reduce_sum(dev(X), {0}).read(), reduce_sum(X, {0}));
    EXPECT_EQ(reduce_sum(dev(X), {1}).read(), reduce_sum(X, {1}));
    EXPECT_EQ(reduce_sum(dev(X), {2}).read(), reduce_sum(X, {2}));
    EXPECT_EQ(reduce_sum(dev(X), {0,2}).read(), reduce_sum(X, {0,2}));
}

TEST(UniformTest, ReduceMean) {
    auto X = Tensor<float>({3,3,3}).range();

    EXPECT_EQ(reduce_mean(X, {0}, true), Tensor<float>({1, 3, 3}, {
         9, 10, 11,
        12, 13, 14,
        15, 16, 17
    }));

    EXPECT_EQ(reduce_mean(X, {1}, true), Tensor<float>({3, 1, 3}, {
         3,  4,  5,
        12, 13, 14,
        21, 22, 23
    }));

    EXPECT_EQ(reduce_mean(X, {2}, true), Tensor<float>({3, 3, 1}, {
         1,  4,  7,
        10, 13, 16,
        19, 22, 25
    }));

    EXPECT_EQ(reduce_mean(X, {0,2}, true), Tensor<float>({1, 3, 1}, {10, 13, 16}));
}

TEST(UniformTest, ReduceMean_GPU) {
    auto X = Tensor<float>({3, 3, 3}).range();
    EXPECT_EQ(round(reduce_mean(dev(X), {}).read()), reduce_mean(X, {}));
    EXPECT_EQ(round(reduce_mean(dev(X), {0}).read()), reduce_mean(X, {0}));
    EXPECT_EQ(round(reduce_mean(dev(X), {1}).read()), reduce_mean(X, {1}));
    EXPECT_EQ(round(reduce_mean(dev(X), {2}).read()), reduce_mean(X, {2}));
    EXPECT_EQ(round(reduce_mean(dev(X), {0,2}).read()), reduce_mean(X, {0,2}));
}

TEST(UniformTest, ReduceSumSquare) {
    auto X = Tensor<float>({3, 2, 2}).range(1);
    EXPECT_EQ(reduce_sum_square(X, {1}), reduce_sum(X*X, {1}));
}

TEST(UniformTest, ReduceSumSquare_GPU) {
    auto X = Tensor<float>({3, 3, 3}).range();
    ExpectElementsEQ(reduce_sum_square(dev(X), {}).read(), reduce_sum_square(X, {}));
    ExpectElementsEQ(reduce_sum_square(dev(X), {0}).read(), reduce_sum_square(X, {0}));
    ExpectElementsEQ(reduce_sum_square(dev(X), {1}).read(), reduce_sum_square(X, {1}));
    ExpectElementsEQ(reduce_sum_square(dev(X), {2}).read(), reduce_sum_square(X, {2}));
    ExpectElementsEQ(reduce_sum_square(dev(X), {0,2}).read(), reduce_sum_square(X, {0,2}));
}

TEST(UniformTest, ReduceLogSum) {
    auto X = Tensor<float>({3, 2, 2}, {5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2});
    EXPECT_EQ(reduce_log_sum(X, {1}), log(reduce_sum(X, {1})));
}

TEST(UniformTest, ReduceLogSum_GPU) {
    auto X = Tensor<float>({3, 3, 3}).range();
    ExpectElementsEQ(reduce_log_sum(dev(X), {}).read(), reduce_log_sum(X, {}));
    ExpectElementsEQ(reduce_log_sum(dev(X), {0}).read(), reduce_log_sum(X, {0}));
    ExpectElementsEQ(reduce_log_sum(dev(X), {1}).read(), reduce_log_sum(X, {1}));
    ExpectElementsEQ(reduce_log_sum(dev(X), {2}).read(), reduce_log_sum(X, {2}));
    ExpectElementsEQ(reduce_log_sum(dev(X), {0,2}).read(), reduce_log_sum(X, {0,2}));
}

TEST(UniformTest, ReduceLogSumExp) {
    auto X = Tensor<float>({3, 2, 2}, {5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2});
    EXPECT_EQ(reduce_log_sum_exp(X, {1}), log(reduce_sum(exp(X), {1})));
}

TEST(UniformTest, ReduceLogSumExp_GPU) {
    auto X = Tensor<float>({3, 3, 3}).range();
    ExpectElementsEQ(reduce_log_sum_exp(dev(X), {}).read(), reduce_log_sum_exp(X, {}));
    ExpectElementsEQ(reduce_log_sum_exp(dev(X), {0}).read(), reduce_log_sum_exp(X, {0}));
    ExpectElementsEQ(reduce_log_sum_exp(dev(X), {1}).read(), reduce_log_sum_exp(X, {1}));
    ExpectElementsEQ(reduce_log_sum_exp(dev(X), {2}).read(), reduce_log_sum_exp(X, {2}));
    ExpectElementsEQ(reduce_log_sum_exp(dev(X), {0,2}).read(), reduce_log_sum_exp(X, {0,2}));
}

TEST(UniformTest, ReduceProd) {
    auto X = Tensor<float>({3, 2, 2}).range(1);

    ExpectElementsEQ(reduce_prod(X, {}, true), Tensor<float>({1,1,1}, {
        4.790016e+08
    }));
    ExpectElementsEQ(reduce_prod(X, {1}, false), Tensor<float>({3,2}, {
        3, 8, 35, 48, 99, 120
    }));
    ExpectElementsEQ(reduce_prod(X, {1}, true), Tensor<float>({3,1,2}, {
        3, 8, 35, 48, 99, 120
    }));
}

TEST(UniformTest, ReduceProd_GPU) {
    auto X = Tensor<float>({3, 3, 3}).range();
    ExpectElementsEQ(reduce_prod(dev(X), {}).read(), reduce_prod(X, {}));
    ExpectElementsEQ(reduce_prod(dev(X), {0}).read(), reduce_prod(X, {0}));
    ExpectElementsEQ(reduce_prod(dev(X), {1}).read(), reduce_prod(X, {1}));
    ExpectElementsEQ(reduce_prod(dev(X), {2}).read(), reduce_prod(X, {2}));
    ExpectElementsEQ(reduce_prod(dev(X), {0,2}).read(), reduce_prod(X, {0,2}));
}

TEST(UniformTest, ReduceL1) {
    auto X = Tensor<float>({3, 2, 2}).random(-10, 10);
    EXPECT_EQ(reduce_asum(X, {1}), reduce_sum(abs(X), {1}));
}

TEST(UniformTest, ReduceL1_GPU) {
    auto X = Tensor<float>({3, 3, 3}).range();
    ExpectElementsEQ(reduce_asum(dev(X), {}).read(), reduce_asum(X, {}));
    ExpectElementsEQ(reduce_asum(dev(X), {0}).read(), reduce_asum(X, {0}));
    ExpectElementsEQ(reduce_asum(dev(X), {1}).read(), reduce_asum(X, {1}));
    ExpectElementsEQ(reduce_asum(dev(X), {2}).read(), reduce_asum(X, {2}));
    ExpectElementsEQ(reduce_asum(dev(X), {0,2}).read(), reduce_asum(X, {0,2}));
}

TEST(UniformTest, ReduceL2) {
    auto X = Tensor<float>({3, 2, 2}).random(-10, 10);
    EXPECT_EQ(reduce_nrm2(X, {1}), sqrt(reduce_sum(X*X, {1})));
}

TEST(UniformTest, ReduceL2_GPU) {
    auto X = Tensor<float>({3, 3, 3}).range();
    ExpectElementsEQ(reduce_nrm2(dev(X), {}).read(), reduce_nrm2(X, {}));
    ExpectElementsEQ(reduce_nrm2(dev(X), {0}).read(), reduce_nrm2(X, {0}));
    ExpectElementsEQ(reduce_nrm2(dev(X), {1}).read(), reduce_nrm2(X, {1}));
    ExpectElementsEQ(reduce_nrm2(dev(X), {2}).read(), reduce_nrm2(X, {2}));
    ExpectElementsEQ(reduce_nrm2(dev(X), {0,2}).read(), reduce_nrm2(X, {0,2}));
}

TEST(UniformTest, ReduceL2_Complex) {
    auto X = Tensor<std::complex<float>>({2}, {{3,4}, {5,12}});
    EXPECT_EQ(reduce_nrm2(X, {0}), Scalar<std::complex<float>>(std::sqrt(194.f)));
}

TEST(UniformTest, Count) {
    auto A = Matrix<float>({
        {1, 0, 3},
        {0, 2, 4},
        {0, 0, 3}
    });

    EXPECT_EQ(count(A, 0, 0), Vector<int>({2, 2, 0}));
    EXPECT_EQ(count(A, 0, 1), Vector<int>({1, 1, 2}));
    EXPECT_EQ(count(A, 0), Scalar<int>(4));

    auto dev_A = dev(A);
    EXPECT_EQ(count(dev_A, 0, 0).read(), Vector<int>({2, 2, 0}));
    EXPECT_EQ(count(dev_A, 0, 1).read(), Vector<int>({1, 1, 2}));
    EXPECT_EQ(count(dev_A, 0).read(), Scalar<int>(4));
}

TEST(UniformTest, Norm) {
    auto a = Tensor<float>({9}).range() - 4;
    auto b = reshape(a, {3, 3});

    EXPECT_EQ(norm(a, std::numeric_limits<float>::infinity()), Scalar(4.0f));
    EXPECT_EQ(norm(b, std::numeric_limits<float>::infinity()), Scalar(9.0f));
    EXPECT_EQ(norm(a, 1), Scalar(20.0f));
    EXPECT_EQ(norm(b, 1), Scalar(7.0f));

    ExpectElementsEQ(norm(a, 3), Scalar(5.84804f));
}

TEST(UniformTest, Norm_GPU) {
    auto a = dev(Tensor<float>({9}).range() - 4);
    auto b = reshape(a, {3, 3});

    EXPECT_EQ(norm(a, std::numeric_limits<float>::infinity()).read(), Scalar(4.0f));
    EXPECT_EQ(norm(b, std::numeric_limits<float>::infinity()).read(), Scalar(9.0f));
    EXPECT_EQ(norm(a, 1).read(), Scalar(20.0f));
    EXPECT_EQ(norm(b, 1).read(), Scalar(7.0f));

    ExpectElementsEQ(norm(a, 3).read(), Scalar(5.84804f));
}

TEST(UniformTest, Scan) {
    auto A = Tensor<int>({5, 5}).range(1);
    EXPECT_EQ(cumsum(A, 0), Matrix({
        { 1,  2,  3,  4,  5},
        { 7,  9, 11, 13, 15},
        {18, 21, 24, 27, 30},
        {34, 38, 42, 46, 50},
        {55, 60, 65, 70, 75}
    }));
    EXPECT_EQ(cumsum(A, 1), Matrix({
        { 1,  3,  6, 10, 15},
        { 6, 13, 21, 30, 40},
        {11, 23, 36, 50, 65},
        {16, 33, 51, 70, 90},
        {21, 43, 66, 90,115}
    }));
    EXPECT_EQ(cumsum(A, 0, true), Matrix({
        { 0,  0,  0,  0,  0},
        { 1,  2,  3,  4,  5},
        { 7,  9, 11, 13, 15},
        {18, 21, 24, 27, 30},
        {34, 38, 42, 46, 50},
    }));
    EXPECT_EQ(cumsum(A, 1, true), Matrix({
        { 0,  1,  3,  6, 10},
        { 0,  6, 13, 21, 30},
        { 0, 11, 23, 36, 50},
        { 0, 16, 33, 51, 70},
        { 0, 21, 43, 66, 90}
    }));
    EXPECT_EQ(cumsum(A, 0, false, true), Matrix({
        {55, 60, 65, 70, 75},
        {54, 58, 62, 66, 70},
        {48, 51, 54, 57, 60},
        {37, 39, 41, 43, 45},
        {21, 22, 23, 24, 25},
    }));
    EXPECT_EQ(cumsum(A, 0, true, true), Matrix({
        {54, 58, 62, 66, 70},
        {48, 51, 54, 57, 60},
        {37, 39, 41, 43, 45},
        {21, 22, 23, 24, 25},
        { 0,  0,  0,  0,  0}
    }));
    EXPECT_EQ(cumsum(A, 1, false, true), Matrix({
        {15, 14, 12,  9,  5},
        {40, 34, 27, 19, 10},
        {65, 54, 42, 29, 15},
        {90, 74, 57, 39, 20},
        {115,94, 72, 49, 25},
    }));
    EXPECT_EQ(cumsum(A, 1, true, true), Matrix({
        {14, 12,  9,  5,  0},
        {34, 27, 19, 10,  0},
        {54, 42, 29, 15,  0},
        {74, 57, 39, 20,  0},
        {94, 72, 49, 25,  0},
    }));

    auto B = Tensor<int>({10}).range(1);
    EXPECT_EQ(scan(B, 0, 1, xfn::multiplies<>()), Vector({
        1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800
    }));
}

TEST(UniformTest, Scan_GPU) {
    auto A = Tensor<int>({5, 5}).range(1);
    auto dev_A = dev(A);

    EXPECT_EQ(cumsum(dev_A, 0).read(), cumsum(A, 0));
    EXPECT_EQ(cumsum(dev_A, 1).read(), cumsum(A, 1));
    EXPECT_EQ(cumsum(dev_A, 0, true).read(), cumsum(A, 0, true));
    EXPECT_EQ(cumsum(dev_A, 1, true).read(), cumsum(A, 1, true));
    EXPECT_EQ(cumsum(dev_A, 0, false, true).read(), cumsum(A, 0, false, true));
    EXPECT_EQ(cumsum(dev_A, 1, false, true).read(), cumsum(A, 1, false, true));
    EXPECT_EQ(cumsum(dev_A, 0, true, true).read(), cumsum(A, 0, true, true));
    EXPECT_EQ(cumsum(dev_A, 1, true, true).read(), cumsum(A, 1, true, true));

    auto B = Tensor<int>({10}).range(1);
    auto dev_B = dev(B);
    EXPECT_EQ(cumprod(dev_B, 0).read(), cumprod(B, 0));

    auto X = Tensor<int>({2, 100'000}).random();
    EXPECT_EQ(cumsum(X), cumsum(dev(X)).read());
}

TEST(UniformTest, Pad1D) {
    auto A = Vector({1, 2, 3, 4, 5});
    EXPECT_EQ(pad(A, {2, 3}), Vector({0, 0, 1, 2, 3, 4, 5, 0, 0, 0}));
    EXPECT_EQ(pad(A, {2, 3}, PadMode::Edge), Vector({1, 1, 1, 2, 3, 4, 5, 5, 5, 5}));
    EXPECT_EQ(pad(A, {2, 3}, PadMode::Reflect), Vector({3, 2, 1, 2, 3, 4, 5, 4, 3, 2}));
    EXPECT_EQ(pad(A, {2, 3}, PadMode::Symmetric), Vector({2, 1, 1, 2, 3, 4, 5, 5, 4, 3}));

    auto dev_A = dev(A);
    EXPECT_EQ(pad(dev_A, {2, 3}).read(), Vector({0, 0, 1, 2, 3, 4, 5, 0, 0, 0}));
    EXPECT_EQ(pad(dev_A, {2, 3}, PadMode::Edge).read(), Vector({1, 1, 1, 2, 3, 4, 5, 5, 5, 5}));
    EXPECT_EQ(pad(dev_A, {2, 3}, PadMode::Reflect).read(), Vector({3, 2, 1, 2, 3, 4, 5, 4, 3, 2}));
    EXPECT_EQ(pad(dev_A, {2, 3}, PadMode::Symmetric).read(), Vector({2, 1, 1, 2, 3, 4, 5, 5, 4, 3}));
}

TEST(UniformTest, Pad2D) {
    auto A = Tensor<int>({3, 3}).range(1);

    EXPECT_EQ(pad(A, {2, 2, 2, 2}, PadMode::Constant, 3), Matrix({
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 1, 2, 3, 3, 3},
        {3, 3, 4, 5, 6, 3, 3},
        {3, 3, 7, 8, 9, 3, 3},
        {3, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3}
    }));

    EXPECT_EQ(pad(A, {2, 2, 2, 2}, PadMode::Edge), Matrix({
        {1, 1, 1, 2, 3, 3, 3},
        {1, 1, 1, 2, 3, 3, 3},
        {1, 1, 1, 2, 3, 3, 3},
        {4, 4, 4, 5, 6, 6, 6},
        {7, 7, 7, 8, 9, 9, 9},
        {7, 7, 7, 8, 9, 9, 9},
        {7, 7, 7, 8, 9, 9, 9}
    }));

    EXPECT_EQ(pad(A, {2, 2, 2, 2}, PadMode::Reflect), Matrix({
        {9, 8, 7, 8, 9, 8, 7},
        {6, 5, 4, 5, 6, 5, 4},
        {3, 2, 1, 2, 3, 2, 1},
        {6, 5, 4, 5, 6, 5, 4},
        {9, 8, 7, 8, 9, 8, 7},
        {6, 5, 4, 5, 6, 5, 4},
        {3, 2, 1, 2, 3, 2, 1}
    }));

    EXPECT_EQ(pad(A, {2, 2, 2, 2}, PadMode::Symmetric), Matrix({
        {5, 4, 4, 5, 6, 6, 5},
        {2, 1, 1, 2, 3, 3, 2},
        {2, 1, 1, 2, 3, 3, 2},
        {5, 4, 4, 5, 6, 6, 5},
        {8, 7, 7, 8, 9, 9, 8},
        {8, 7, 7, 8, 9, 9, 8},
        {5, 4, 4, 5, 6, 6, 5}
    }));
}

TEST(UniformTest, Merge) {
    constexpr int M = 2000;
    constexpr int N = 3000;

    auto A = Tensor<int>({M}).random();
    auto B = Tensor<int>({N}).random();
    auto C = Tensor<int>();
    auto D = Tensor<int>({M+N}, 0);

    sort(A); sort(B);
    merge(A, B, C, -1);
    std::merge(A.begin(), A.end(), B.begin(), B.end(), D.begin());
    EXPECT_EQ(C, D);

    auto dev_A = dev(A);
    auto dev_B = dev(B);
    auto dev_C = DevTensor<int>();
    merge(dev_A, dev_B, dev_C, -1);
    EXPECT_EQ(dev_C.read(), D);

    // When fail, dump the test data for debugging
    if (C != D || dev_C.read() != D) {
        std::cout << A << B;
    }
}

TEST(UniformTest, MergeBatch) {
    auto A = Tensor<int>({10, 10}).random(0, 1000);
    auto B = Tensor<int>({10, 10}).random(0, 1000);
    sort(A, -1);
    sort(B, -1);
    EXPECT_EQ(merge(A, B), merge(dev(A), dev(B)).read());
}

TEST(UniformTest, Sort_CPU) {
    auto X = Tensor<int>({3, 4, 5}, {
         8, 19, 11, 41, 53,
        79,  2, 10,  0, 94,
        37, 30, 82, 81, 75,
        60, 18, 66, 47, 51,

        91, 83, 67, 88, 76,
        17, 17, 73, 16, 65,
        66, 62, 56, 90, 18,
        32, 65, 11, 46, 24,

        57, 38, 77, 56,  0,
        82, 69, 78,  3, 83,
        57, 37, 97, 93, 32,
        78, 96, 15, 80, 95
    });

    auto Y = Tensor<int>();

    sort(X, Y, 0);
    EXPECT_EQ(Y, Tensor<int>({3, 4, 5}, {
         8, 19, 11, 41,  0,
        17,  2, 10,  0, 65,
        37, 30, 56, 81, 18,
        32, 18, 11, 46, 24,

        57, 38, 67, 56, 53,
        79, 17, 73,  3, 83,
        57, 37, 82, 90, 32,
        60, 65, 15, 47, 51,

        91, 83, 77, 88, 76,
        82, 69, 78, 16, 94,
        66, 62, 97, 93, 75,
        78, 96, 66, 80, 95,
    }));

    sort(X, Y, 1);
    EXPECT_EQ(Y, Tensor<int>({3, 4, 5}, {
         8,  2, 10,  0, 51,
        37, 18, 11, 41, 53,
        60, 19, 66, 47, 75,
        79, 30, 82, 81, 94,

        17, 17, 11, 16, 18,
        32, 62, 56, 46, 24,
        66, 65, 67, 88, 65,
        91, 83, 73, 90, 76,

        57, 37, 15,  3,  0,
        57, 38, 77, 56, 32,
        78, 69, 78, 80, 83,
        82, 96, 97, 93, 95,
    }));

    sort(X, Y, 2);
    EXPECT_EQ(Y, Tensor<int>({3, 4, 5}, {
         8, 11, 19, 41, 53,
         0,  2, 10, 79, 94,
        30, 37, 75, 81, 82,
        18, 47, 51, 60, 66,

        67, 76, 83, 88, 91,
        16, 17, 17, 65, 73,
        18, 56, 62, 66, 90,
        11, 24, 32, 46, 65,

         0, 38, 56, 57, 77,
         3, 69, 78, 82, 83,
        32, 37, 57, 93, 97,
        15, 78, 80, 95, 96,
    }));
}

TEST(UniformTest, Sort_GPU) {
    auto X = Tensor<int>({3, 4, 5}).random(0, 1000);
    auto dev_X = dev(X);

    EXPECT_EQ(sorted(X), sorted(dev_X).read());
    EXPECT_EQ(sorted(X, 0), sorted(dev_X, 0).read());
    EXPECT_EQ(sorted(X, 1), sorted(dev_X, 1).read());
    EXPECT_EQ(sorted(X, 2), sorted(dev_X, 2).read());

    EXPECT_EQ(sorted(X, 0, xfn::greater<>()), sorted(dev_X, 0, xfn::greater<>()).read());
    EXPECT_EQ(sorted(X, 1, xfn::greater<>()), sorted(dev_X, 1, xfn::greater<>()).read());
    EXPECT_EQ(sorted(X, 2, xfn::greater<>()), sorted(dev_X, 2, xfn::greater<>()).read());

    for (size_t n = 1; n <= 1024; n++) {
        auto Y = Tensor<int>({2, n}).random(0, 2000);
        EXPECT_EQ(sorted(Y), sorted(dev(Y)).read());
    }

    for (size_t n = 1000; n <= 10000; n += 1000) {
        auto Y = Tensor<int>({2, n}).random(0, 100000);
        EXPECT_EQ(sorted(Y), sorted(dev(Y)).read());
    }
}

TEST(UniformTest, ArgSort) {
    auto X = Tensor<int>({3, 4, 5}).random(0, 1000);
    EXPECT_EQ(sorted(X, 0), gather_elements(X, argsort(X, 0), 0));
    EXPECT_EQ(sorted(X, 1), gather_elements(X, argsort(X, 1), 1));
    EXPECT_EQ(sorted(X, 2), gather_elements(X, argsort(X, 2), 2));

    auto dev_X = dev(X);
    EXPECT_EQ(sorted(X, 0), gather_elements(X, argsort(dev_X, 0).read(), 0));
    EXPECT_EQ(sorted(X, 1), gather_elements(X, argsort(dev_X, 1).read(), 1));
    EXPECT_EQ(sorted(X, 2), gather_elements(X, argsort(dev_X, 2).read(), 2));

    for (size_t n = 1; n <= 1024; n++) {
        auto Y = Tensor<int>({2, n}).random(0, 2000);
        EXPECT_EQ(sorted(Y), gather_elements(Y, argsort(dev(Y)).read(), -1));
    }

    for (size_t n = 1000; n <= 10000; n += 1000) {
        auto Y = Tensor<int>({2, n}).random(0, 100000);
        EXPECT_EQ(sorted(Y), gather_elements(Y, argsort(dev(Y)).read(), -1));
    }
}

TEST(UniformTest, resize_upsample_scales_linear) {
    auto X = Tensor<float>({1, 1, 2, 2}, {1, 2, 3, 4});
    auto Y = im::resize(X, std::vector<float>{1, 1, 2, 2});
    EXPECT_EQ(Y, Tensor<float>({1, 1, 4, 4}, {
        1.,  1.25, 1.75, 2.,
        1.5, 1.75, 2.25, 2.5,
        2.5, 2.75, 3.25, 3.5,
        3.,  3.25, 3.75, 4.
    }));
    EXPECT_EQ(im::resize(X, Shape{1, 1, 4, 4}), Y);

    auto dev_X = dev(X);
    auto dev_Y = im::resize(dev_X, std::vector<float>{1, 1, 2, 2});
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(UniformTest, NonZero) {
    auto X = Tensor<float>({2, 2, 2}, {1, 0, 0, 2, 5, 6, 0, 0});
    auto Y = Matrix<int>({{0, 0, 1, 1}, {0, 1, 0, 0}, {0, 1, 0, 1}});

    EXPECT_EQ(nonzero(X), Y);
    EXPECT_EQ(nonzero(dev(X)).read(), Y);
}
