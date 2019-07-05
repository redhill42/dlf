#include "tensor.h"
#include "gtest/gtest.h"

using namespace dlf;

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
