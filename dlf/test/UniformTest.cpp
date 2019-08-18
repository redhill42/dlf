#include "tensor.h"
#include "gtest/gtest.h"

using namespace dlf;

TEST(UniformTest, BroadcastTransform) {
    auto X = Tensor<float>::range({2, 3, 2, 2}, 1);
    auto Y = Tensor<float>::range({3, 1, 1}, 1);
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
    auto X = Tensor<float>::range({2, 3, 2, 2}, 1);
    auto Y = Tensor<float>({6, 1, 1}, {4, 5, 6, 1, 2, 3});
    auto Z = X + Y.slice({{3,6}});
    EXPECT_EQ(Z, Tensor<float>({2, 3, 2, 2}, {
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 14, 15,

        14, 15, 16, 17,
        19, 20, 21, 22,
        24, 25, 26, 27
    }));

    auto dev_Z = dev(X) + dev(Y).slice({{3,6}});
    EXPECT_EQ(dev_Z.read(), Z);
}

TEST(UniformTest, MatMul) {
    auto A = Tensor<float>::range({2, 3, 4}, 1);
    auto B = Tensor<float>::range({2, 4, 5}, 1);
    auto C = Tensor<float>({2, 3, 5}, {
         110,  120,  130,  140,  150,
         246,  272,  298,  324,  350,
         382,  424,  466,  508,  550,

        1678, 1736, 1794, 1852, 1910,
        2134, 2208, 2282, 2356, 2430,
        2590, 2680, 2770, 2860, 2950
    });

    EXPECT_EQ(matmul(A, B), C);

    auto dev_A = dev(A);
    auto dev_B = dev(B);
    auto dev_C = matmul(dev_A, dev_B);
    EXPECT_EQ(dev_C.read(), C);
}

TEST(UniformTest, MatMulBroadcast) {
    auto A = Tensor<float>::range({2, 3, 4}, 1);
    auto B = Tensor<float>::range({1, 4, 5}, 1);
    auto C = Tensor<float>({2, 3, 5}, {
        110,  120,  130,  140,  150,
        246,  272,  298,  324,  350,
        382,  424,  466,  508,  550,

        518,  576,  634,  692,  750,
        654,  728,  802,  876,  950,
        790,  880,  970, 1060, 1150
    });

    EXPECT_EQ(matmul(A, B), C);

    auto dev_C = matmul(dev(A), dev(B));
    EXPECT_EQ(dev_C.read(), C);
}

TEST(Uniform, MatMulVectorL) {
    auto A = Tensor<float>::range({4}, 1);
    auto B = Tensor<float>::range({2, 4, 5}, 1);
    auto C = Tensor<float>({2, 5}, {
        110, 120, 130, 140, 150,
        310, 320, 330, 340, 350,
    });

    EXPECT_EQ(matmul(A, B), C);

    auto dev_C = matmul(dev(A), dev(B));
    EXPECT_EQ(dev_C.read(), C);
}

TEST(Uniform, MatMulVectorR) {
    auto A = Tensor<float>::range({2, 3, 4}, 1);
    auto B = Tensor<float>::range({4}, 1);
    auto C = Tensor<float>({2, 3}, {
         30,  70, 110,
        150, 190, 230
    });

    EXPECT_EQ(matmul(A, B), C);

    auto dev_C = matmul(dev(A), dev(B));
    EXPECT_EQ(dev_C.read(), C);
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

TEST(UniformTest, MatPowCPU) {
    auto A = Tensor<int>({2, 2}, {1, 1, 1, 0});
    EXPECT_EQ(matpow(A, 0)(0, 0), 1);
    EXPECT_EQ(matpow(A, 11)(0, 0), 144);
}

TEST(UniformTest, MatPowGPU) {
    auto A = dev(Tensor<int>({2, 2}, {1, 1, 1, 0}));
    EXPECT_EQ(matpow(A, 0).read()(0, 0), 1);
    EXPECT_EQ(matpow(A, 11).read()(0, 0), 144);
}

TEST(UniformTest, ModCPU) {
    auto A = Tensor<int>::range({10}, 1);
    auto B = A % 7;
    EXPECT_EQ(B, Tensor<int>({10}, {1, 2, 3, 4, 5, 6, 0, 1, 2, 3}));
}

TEST(UniformTest, ModGPU) {
    auto A = dev(Tensor<float>::range({10}, 1));
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
    auto A = Tensor<int>::range({3, 4}, 1);
    auto B = Tensor<int>::range({2, 3, 1}, 1);
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
    auto A = dev(Tensor<int>::range({3, 4}, 1));
    auto B = dev(Tensor<int>::range({2, 3, 1}, 1));
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
    auto A = Tensor<short>::range({4}, 0);
    EXPECT_EQ(A | 1, Tensor<int>({4}, {1, 1, 3, 3}));
    EXPECT_EQ(A & 1, Tensor<int>({4}, {0, 1, 0, 1}));
    EXPECT_EQ(A ^ 1, Tensor<int>({4}, {1, 0, 3, 2}));
}

TEST(UniformTest, BitwiseGPU) {
    auto A = dev(Tensor<int>::range({4}, 0));
    EXPECT_EQ((A | 1).read(), Tensor<int>({4}, {1, 1, 3, 3}));
    EXPECT_EQ((A & 1).read(), Tensor<int>({4}, {0, 1, 0, 1}));
    EXPECT_EQ((A ^ 1).read(), Tensor<int>({4}, {1, 0, 3, 2}));
}

TEST(UniformTest, ReshapeCPU) {
    auto A = Tensor<int>({2, 3}, {1, 2, 3, 4, 5, 6});
    auto R = Tensor<int>({6}, {1, 2, 3, 4, 5, 6});

    auto B = reshape(A, {6});
    EXPECT_EQ(B, R);

    auto C = Tensor<int>({6});
    reshape(A, C);
    EXPECT_EQ(C, R);

    auto D = Tensor<int>({7});
    EXPECT_ANY_THROW(reshape(A, {7}));
    EXPECT_ANY_THROW(reshape(A, D));
}

TEST(UniformTest, ReshapeGPU) {
    auto A = dev(Tensor<int>({2, 3}, {1, 2, 3, 4, 5, 6}));
    auto R = Tensor<int>({6}, {1, 2, 3, 4, 5, 6});

    auto B = reshape(A, {6});
    EXPECT_EQ(B.read(), R);

    auto C = DevTensor<int>({6});
    reshape(A, C);
    EXPECT_EQ(C.read(), R);

    auto D = DevTensor<int>({7});
    EXPECT_ANY_THROW(reshape(A, {7}));
    EXPECT_ANY_THROW(reshape(A, D));
}

TEST(UniformTest, FlattenCPU) {
    auto A = Tensor<int>::range({2, 3, 4}, 1);
    EXPECT_EQ(flatten(A, 0), Tensor<int>::range({1, 24}, 1));
    EXPECT_EQ(flatten(A, 1), Tensor<int>::range({2, 12}, 1));
    EXPECT_EQ(flatten(A, 2), Tensor<int>::range({6, 4}, 1));
    EXPECT_EQ(flatten(A, 3), Tensor<int>::range({24, 1}, 1));
    EXPECT_ANY_THROW(flatten(A, 4));
}

TEST(UniformTest, FlattenGPU) {
    auto A = dev(Tensor<int>::range({2, 3, 4}, 1));
    EXPECT_EQ(flatten(A, 0).read(), Tensor<int>::range({1, 24}, 1));
    EXPECT_EQ(flatten(A, 1).read(), Tensor<int>::range({2, 12}, 1));
    EXPECT_EQ(flatten(A, 2).read(), Tensor<int>::range({6, 4}, 1));
    EXPECT_EQ(flatten(A, 3).read(), Tensor<int>::range({24, 1}, 1));
    EXPECT_ANY_THROW(flatten(A, 4));
}

TEST(UniformTest, SqueezeCPU) {
    auto A = Tensor<int>::range({2, 1, 3, 1, 4}, 1);
    EXPECT_EQ(squeeze(A), Tensor<int>::range({2, 3, 4}, 1));
    EXPECT_EQ(squeeze(A, {1}), Tensor<int>::range({2, 3, 1, 4}, 1));
    EXPECT_EQ(squeeze(A, {-2}), Tensor<int>::range({2, 1, 3, 4}, 1));
    EXPECT_ANY_THROW(squeeze(A, {0}));
}

TEST(UniformTest, UnsqueezeCPU) {
    auto A = Tensor<int>::range({2, 3, 4}, 1);
    EXPECT_EQ(unsqueeze(A, {1, -2}), Tensor<int>::range({2, 1, 3, 1, 4}, 1));
    EXPECT_ANY_THROW(unsqueeze(A, {1, 5}));
}

TEST(UniformTest, ConcatCPU) {
    {
        auto A = Tensor<int>::range({2, 3, 2}, 1);
        auto B = Tensor<int>::range({3, 3, 2}, -1, -1);
        auto C = Tensor<int>::range({4, 3, 2}, 24, -1);
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
        auto A = Tensor<int>::range({2, 2, 2}, 1);
        auto B = Tensor<int>::range({2, 3, 2}, -1, -1);
        auto C = Tensor<int>::range({2, 4, 2}, 16, -1);
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
        auto A = Tensor<int>::range({2, 2, 2}, 1);
        auto B = Tensor<int>::range({2, 2, 3}, -1, -1);
        auto C = Tensor<int>::range({2, 2, 4}, 16, -1);
        auto D = concat(2, A, B, C);
        EXPECT_EQ(D, Tensor<int>({2, 2, 9}, {
             1,  2,  -1,  -2,  -3,  16,  15,  14,  13,
             3,  4,  -4,  -5,  -6,  12,  11,  10,   9,
             5,  6,  -7,  -8,  -9,   8,   7,   6,   5,
             7,  8, -10, -11, -12,   4,   3,   2,   1
        }));
    }
}

TEST(UniformTest, ConcatGPU) {
    {
        auto A = dev(Tensor<int>::range({2, 3, 2}, 1));
        auto B = dev(Tensor<int>::range({3, 3, 2}, -1, -1));
        auto C = dev(Tensor<int>::range({4, 3, 2}, 24, -1));
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
        auto A = dev(Tensor<int>::range({2, 2, 2}, 1));
        auto B = dev(Tensor<int>::range({2, 3, 2}, -1, -1));
        auto C = dev(Tensor<int>::range({2, 4, 2}, 16, -1));
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
        auto A = dev(Tensor<int>::range({2, 2, 2}, 1));
        auto B = dev(Tensor<int>::range({2, 2, 3}, -1, -1));
        auto C = dev(Tensor<int>::range({2, 2, 4}, 16, -1));
        auto D = concat(2, A, B, C);
        EXPECT_EQ(D.read(), Tensor<int>({2, 2, 9}, {
             1,  2,  -1,  -2,  -3,  16,  15,  14,  13,
             3,  4,  -4,  -5,  -6,  12,  11,  10,   9,
             5,  6,  -7,  -8,  -9,   8,   7,   6,   5,
             7,  8, -10, -11, -12,   4,   3,   2,   1
        }));
    }
}

TEST(UniformTest, SplitCPU) {
    {
        auto X = Tensor<int>::range({9, 3, 2}, 1);
        auto A = Tensor<int>({2, 3, 2});
        auto B = Tensor<int>({3, 3, 2});
        auto C = Tensor<int>({4, 3, 2});
        split(0, X, {&A, &B, &C});
        EXPECT_EQ(A, Tensor<int>({2, 3, 2}, {
            1, 2, 3,  4,  5,  6,
            7, 8, 9, 10, 11, 12}));
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
        auto X = Tensor<int>::range({2, 9, 3}, 1);
        auto A = Tensor<int>({2, 2, 3});
        auto B = Tensor<int>({2, 3, 3});
        auto C = Tensor<int>({2, 4, 3});
        split(1, X, {&A, &B, &C});
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
        auto X = Tensor<int>::range({2, 3, 9}, 1);
        auto A = Tensor<int>({2, 3, 2});
        auto B = Tensor<int>({2, 3, 3});
        auto C = Tensor<int>({2, 3, 4});
        split(2, X, {&A, &B, &C});
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
    }
}

TEST(UniformTest, SplitGPU) {
    {
        auto X = dev(Tensor<int>::range({9, 3, 2}, 1));
        auto A = dev(Tensor<int>({2, 3, 2}));
        auto B = dev(Tensor<int>({3, 3, 2}));
        auto C = dev(Tensor<int>({4, 3, 2}));
        split(0, X, {&A, &B, &C});
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
        auto X = dev(Tensor<int>::range({2, 9, 3}, 1));
        auto A = dev(Tensor<int>({2, 2, 3}));
        auto B = dev(Tensor<int>({2, 3, 3}));
        auto C = dev(Tensor<int>({2, 4, 3}));
        split(1, X, {&A, &B, &C});
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
        auto X = dev(Tensor<int>::range({2, 3, 9}, 1));
        auto A = dev(Tensor<int>({2, 3, 2}));
        auto B = dev(Tensor<int>({2, 3, 3}));
        auto C = dev(Tensor<int>({2, 3, 4}));
        split(2, X, {&A, &B, &C});
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
    }
}

TEST(UniformTest, Slice) {
    auto X = Tensor<float>::range({10, 10, 5}, 0);
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

    EXPECT_EQ(X.slice({{2, 5}, {3, 7}, {}}), Y);
    EXPECT_EQ(dev(X).slice({{2, 5}, {3, 7}, {}}).read(), Y);
}

TEST(UniformTest, SliceWithStep) {
    auto X = Tensor<float>::range({10, 10, 5}, 0);
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

    EXPECT_EQ(X.slice({{2, 5}, {3, 7, 2}}), Y2);
    EXPECT_EQ(dev(X).slice({{2, 5}, {3, 7, 2}}).read(), Y2);
    EXPECT_EQ(X.slice({{2, 5}, {3, 7, 3}}), Y3);
    EXPECT_EQ(dev(X).slice({{2, 5}, {3, 7, 3}}).read(), Y3);
    EXPECT_EQ(X.slice({{2, 5}, {3, 7, 4}}), Y4);
    EXPECT_EQ(dev(X).slice({{2, 5}, {3, 7, 4}}).read(), Y4);
    EXPECT_EQ(X.slice({{2, 5}, {3, 7, 5}}), Y4);
    EXPECT_EQ(dev(X).slice({{2, 5}, {3, 7, 5}}).read(), Y4);
}

TEST(UniformTest, SliceWithNegativeStep) {
    auto X = Tensor<float>::range({10, 10, 5}, 0);
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

    EXPECT_EQ(X.slice({{2, 5}, {7, 3, -1}}), Y1);
    EXPECT_EQ(dev(X).slice({{2, 5}, {7, 3, -1}}).read(), Y1);
    EXPECT_EQ(X.slice({{2, 5}, {7, 3, -2}}), Y2);
    EXPECT_EQ(dev(X).slice({{2, 5}, {7, 3, -2}}).read(), Y2);
    EXPECT_EQ(X.slice({{2, 5}, {7, 3, -3}}), Y3);
    EXPECT_EQ(dev(X).slice({{2, 5}, {7, 3, -3}}).read(), Y3);
    EXPECT_EQ(X.slice({{2, 5}, {7, 3, -4}}), Y4);
    EXPECT_EQ(dev(X).slice({{2, 5}, {7, 3, -4}}).read(), Y4);
    EXPECT_EQ(X.slice({{2, 5}, {7, 3, -5}}), Y4);
    EXPECT_EQ(dev(X).slice({{2, 5}, {7, 3, -5}}).read(), Y4);
}

TEST(UniformTest, SliceOfSlice) {
    auto X = Tensor<float>::range({10, 10, 5}, 0);
    auto Y = Tensor<float>({2, 2, 5});

    auto shape = X.shape().slice({{2, 5}, {3, 7}}).slice({{1, 3}, {1, 3}});
    reorder(X, shape, Y);
    EXPECT_EQ(Y, Tensor<float>({2, 2, 5}, {
        170, 171, 172, 173, 174,
        175, 176, 177, 178, 179,
        220, 221, 222, 223, 224,
        225, 226, 227, 228, 229
    }));

    EXPECT_EQ((X[{{2,5}, {3,7}}][{{1,3}, {1,3}}]), Y);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 2, 5});
    reorder(dev_X, shape, dev_Y);
    EXPECT_EQ(dev_Y.read(), Y);
    EXPECT_EQ((dev_X[{{2,5}, {3,7}}][{{1,3}, {1,3}}]).read(), Y);
}

TEST(UniformTest, SliceAndTranspose) {
    auto X = Tensor<float>::range({10, 10, 5}, 0);
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

    EXPECT_EQ(X.slice({{2,5}, {3,7}}).transpose(), Y);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({5, 4, 3});
    reorder(dev_X, shape, dev_Y);
    EXPECT_EQ(dev_Y.read(), Y);
    EXPECT_EQ(dev_X.slice({{2,5}, {3,7}}).transpose().read(), Y);
}

TEST(UniformTest, CopyNonContiguousSlice) {
    auto X = Tensor<float>::range({2, 2, 10}, 0);
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

    reorder(X1[{{}, {}, {1,4}}], X1[{{}, {}, {5,8}}]);
    EXPECT_EQ(X1, X);

    reorder(dev_X, src_shape, dev_X, dst_shape);
    EXPECT_EQ(dev_X.read(), X);
    reorder(dev_X1[{{}, {}, {1,4}}], dev_X1[{{}, {}, {5,8}}]);
    EXPECT_EQ(dev_X1.read(), X);
}

TEST(UniformTest, CopyContiguousSlice) {
    auto X = Tensor<float>::range({3, 2, 10}, 0);
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

    reorder(X1.slice({{1,2}}), X1.slice({{2,3}}));
    EXPECT_EQ(X1, X);

    reorder(dev_X, src_shape, dev_X, dst_shape);
    EXPECT_EQ(dev_X.read(), X);

    reorder(dev_X1[{{1,2}}], dev_X1[{{2,3}}]);
    EXPECT_EQ(dev_X1.read(), X);
}

TEST(UniformTest, CopyToSlice) {
    auto X = Tensor<float>::range({3, 3, 5}, 0);
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

    reorder(Y, X1[{{1,3}, {1,3}, {1,3}}]);
    EXPECT_EQ(X1, R);

    reorder(dev_Y, dev_Y.shape(), dev_X, shape);
    EXPECT_EQ(dev_X.read(), R);

    reorder(dev_Y, dev_X1[{{1,3}, {1,3}, {1,3}}]);
    EXPECT_EQ(dev_X1.read(), R);
}

TEST(UniformTest, BroadcastCopyToSlice) {
    auto X = Tensor<float>({5}, {-1, -2, -3, -4, -5});
    auto Y = Tensor<float>::range({3, 3, 5}, 0);
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

    reorder(X1.broadcast({1, 3, 5}), Y1.slice({{1,2}}));
    EXPECT_EQ(Y1, R);

    reorder(dev_X, src_shape, dev_Y, dst_shape);
    EXPECT_EQ(dev_Y.read(), R);

    reorder(dev_X1.broadcast({1,3,5}), dev_Y1.slice({{1,2}}));
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
    auto X = Tensor<float>::range({2, 1, 4}, 0);
    auto Y = Tensor<float>({2, 2, 2});
    auto Y1 = Y;

    auto shape = X.shape().broadcast({2, 3, 4}).slice({{}, {1, 3}, {1, 3}});
    reorder(X, shape, Y);
    EXPECT_EQ(Y, Tensor<float>({2, 2, 2}, {
        1, 2, 1, 2, 5, 6, 5, 6
    }));

    reorder(X.broadcast({2,3,4}).slice({{}, {1,3}, {1,3}}), Y1);
    EXPECT_EQ(Y1, Y);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 2, 2});
    reorder(dev_X, shape, dev_Y);
    EXPECT_EQ(dev_Y.read(), Y);

    auto dev_Y1 = DevTensor<float>({2, 2, 2});
    reorder(dev_X.broadcast({2,3,4}).slice({{}, {1,3}, {1,3}}), dev_Y1);
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
    auto X = Tensor<float>::range({2, 3, 4}, 0);
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

    reorder(X[{{}, {}, {1,2}}].broadcast({2,3,4}), Y1);
    EXPECT_EQ(Y1, Y);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 3, 4});
    reorder(dev_X, shape, dev_Y);
    EXPECT_EQ(dev_Y.read(), Y);

    auto dev_Y1 = DevTensor<float>({2, 3, 4});
    reorder(dev_X[{{}, {}, {1,2}}].broadcast({2,3,4}), dev_Y1);
    EXPECT_EQ(dev_Y1.read(), Y);
}

template <typename T> struct TransposeTest : public testing::Test {};
using TransposeTestTypes = testing::Types<int, float>;
TYPED_TEST_CASE(TransposeTest, TransposeTestTypes);

TYPED_TEST(TransposeTest, Transpose1D_CPU) {
    auto A = Tensor<TypeParam>({4}, {1, 2, 3, 4});
    auto B = Tensor<TypeParam>({4, 1}, {1, 2, 3, 4});
    EXPECT_EQ(A.transpose(), B);
    EXPECT_EQ(~A, B);
}

TYPED_TEST(TransposeTest, Transpose1D_GPU) {
    auto A = dev(Tensor<TypeParam>({4}, {1, 2, 3, 4}));
    auto B = Tensor<TypeParam>({4, 1}, {1, 2, 3, 4});
    EXPECT_EQ(A.transpose().read(), B);
    EXPECT_EQ((~A).read(), B);
}

TYPED_TEST(TransposeTest, TransposeSquare_CPU) {
    auto A = Tensor<TypeParam>::range({3, 3}, 1);
    auto B = Tensor<TypeParam>({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9});
    EXPECT_EQ(A.transpose(), B);
    EXPECT_EQ(~A, B);
}

TYPED_TEST(TransposeTest, TransposeSquare_GPU) {
    auto A = dev(Tensor<TypeParam>::range({3, 3}, 1));
    auto B = Tensor<TypeParam>({3, 3}, {1, 4, 7, 2, 5, 8, 3, 6, 9});
    EXPECT_EQ(A.transpose().read(), B);
    EXPECT_EQ((~A).read(), B);
}

TYPED_TEST(TransposeTest, Transpose2D_CPU) {
    auto A = Tensor<TypeParam>::range({3, 4}, 1);
    auto B = Tensor<TypeParam>({4, 3}, {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
    EXPECT_EQ(A.transpose(), B);
    EXPECT_EQ(~A, B);
}

TYPED_TEST(TransposeTest, Transpose2D_GPU) {
    auto A = dev(Tensor<TypeParam>::range({3, 4}, 1));
    auto B = Tensor<TypeParam>({4, 3}, {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
    EXPECT_EQ(A.transpose().read(), B);
    EXPECT_EQ((~A).read(), B);
}

TYPED_TEST(TransposeTest, Transpose3D_CPU) {
    auto A = Tensor<TypeParam>::range({2, 3, 4}, 1);
    auto B = Tensor<TypeParam>({4, 3, 2}, {
        1, 13, 5, 17,  9, 21,
        2, 14, 6, 18, 10, 22,
        3, 15, 7, 19, 11, 23,
        4, 16, 8, 20, 12, 24
    });
    EXPECT_EQ(A.transpose(), B);
    EXPECT_EQ(~A, B);
}

TYPED_TEST(TransposeTest, Transpose3D_GPU) {
    auto A = dev(Tensor<TypeParam>::range({2, 3, 4}, 1));
    auto B = Tensor<TypeParam>({4, 3, 2}, {
        1, 13, 5, 17,  9, 21,
        2, 14, 6, 18, 10, 22,
        3, 15, 7, 19, 11, 23,
        4, 16, 8, 20, 12, 24
    });
    EXPECT_EQ(A.transpose().read(), B);
    EXPECT_EQ((~A).read(), B);
}

TYPED_TEST(TransposeTest, TransposePerm_CPU) {
    auto A = Tensor<TypeParam>::range({2, 3, 4}, 1);
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
    auto A = dev(Tensor<TypeParam>::range({2, 3, 4}, 1));
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
    auto X = Tensor<int>::range({10}, 0);
    auto Y = where(X<5, X, 10*X);
    EXPECT_EQ(Y, Tensor<int>({10}, {0, 1, 2, 3, 4, 50, 60, 70, 80, 90}));
}

TEST(UniformTest, WhereExpr_GPU) {
    auto X = dev(Tensor<int>::range({10}, 0));
    auto Y = where(X<5, X, 10*X);
    EXPECT_EQ(Y.read(), Tensor<int>({10}, {0, 1, 2, 3, 4, 50, 60, 70, 80, 90}));
}

TEST(UniformTest, WhereView_CPU) {
    auto condition = Tensor<bool>({2}, {true, false});
    auto X = Tensor<int>({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto Z = where(condition, X[{{0,2}}], X[{{2,4}}]);
    EXPECT_EQ(Z, Tensor<int>({2, 2}, {1, 6, 3, 8}));
}

TEST(UniformTest, WhereView_GPU) {
    auto condition = dev(Tensor<bool>({2}, {true, false}));
    auto X = dev(Tensor<int>({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));
    auto Z = where(condition, X[{{0,2}}], X[{{2,4}}]);
    EXPECT_EQ(Z.read(), Tensor<int>({2, 2}, {1, 6, 3, 8}));
}
