#include "tensor.h"
#include "gtest/gtest.h"

using namespace dlf;

template <typename T> struct MatMulTest : public testing::Test {};
using MatMulTestTypes = testing::Types<float, int>;
TYPED_TEST_CASE(MatMulTest, MatMulTestTypes);

TYPED_TEST(MatMulTest, MatMul) {
    auto A = Tensor<TypeParam>({2, 3, 4}).range(1);
    auto B = Tensor<TypeParam>({2, 4, 5}).range(1);
    auto C = Tensor<TypeParam>({2, 3, 5}, {
         110,  120,  130,  140,  150,
         246,  272,  298,  324,  350,
         382,  424,  466,  508,  550,

        1678, 1736, 1794, 1852, 1910,
        2134, 2208, 2282, 2356, 2430,
        2590, 2680, 2770, 2860, 2950
    });

    EXPECT_EQ(matmul(A, B), C);
    EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, LeftHandSideIsVector) {
    auto A = Tensor<TypeParam>({4}).range(1);
    auto B = Tensor<TypeParam>({2, 4, 5}).range(1);
    auto C = Tensor<TypeParam>({2, 5}, {
        110, 120, 130, 140, 150,
        310, 320, 330, 340, 350,
    });

    EXPECT_EQ(matmul(A, B), C);
    EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, RightHandSideIsVector) {
    auto A = Tensor<TypeParam>({2, 3, 4}).range(1);
    auto B = Tensor<TypeParam>({4}).range(1);
    auto C = Tensor<TypeParam>({2, 3}, {
         30,  70, 110,
        150, 190, 230
    });

    EXPECT_EQ(matmul(A, B), C);
    EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, VectorLike) {
    auto A = Tensor<TypeParam>({1, 8}).range(1);
    auto B = Tensor<TypeParam>({8, 1}).range(100);
    auto C = Tensor<TypeParam>({1, 1}, {3768});
    EXPECT_EQ(matmul(A, B), C);
    EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, Broadcast3D) {
    auto A = Tensor<TypeParam>({2, 3, 4}).range(1);
    auto B = Tensor<TypeParam>({4, 5}).range(1);
    auto C = Tensor<TypeParam>({2, 3, 5}, {
        110,  120,  130,  140,  150,
        246,  272,  298,  324,  350,
        382,  424,  466,  508,  550,

        518,  576,  634,  692,  750,
        654,  728,  802,  876,  950,
        790,  880,  970, 1060, 1150
    });

    EXPECT_EQ(matmul(A, B), C);
    EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);

    auto B1 = unsqueeze(B, 0);
    EXPECT_EQ(matmul(A, B1), C);
    EXPECT_EQ(matmul(dev(A), dev(B1)).read(), C);
}

TYPED_TEST(MatMulTest, Broadcast4DLeft) {
    {
        auto A = Tensor<TypeParam>({3, 4}).range(1);
        auto B = Tensor<TypeParam>({2, 2, 4, 5}).range(1);
        auto C = Tensor<TypeParam>({2, 2, 3, 5}, {
             110,  120,  130,  140,  150,
             246,  272,  298,  324,  350,
             382,  424,  466,  508,  550,

             310,  320,  330,  340,  350,
             766,  792,  818,  844,  870,
            1222, 1264, 1306, 1348, 1390,

             510,  520,  530,  540,  550,
            1286, 1312, 1338, 1364, 1390,
            2062, 2104, 2146, 2188, 2230,

             710,  720,  730,  740,  750,
            1806, 1832, 1858, 1884, 1910,
            2902, 2944, 2986, 3028, 3070
        });

        EXPECT_EQ(matmul(A, B), C);
        EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);

        auto A1 = unsqueeze(A, 0, 1);
        EXPECT_EQ(matmul(A1, B), C);
        EXPECT_EQ(matmul(dev(A1), dev(B)).read(), C);
    }

    {
        auto A = Tensor<TypeParam>({1, 2, 3, 4}).range(1);
        auto B = Tensor<TypeParam>({2, 2, 4, 5}).range(1);
        auto C = Tensor<TypeParam>({2, 2, 3, 5}, {
             110,  120,  130,  140,  150,
             246,  272,  298,  324,  350,
             382,  424,  466,  508,  550,

            1678, 1736, 1794, 1852, 1910,
            2134, 2208, 2282, 2356, 2430,
            2590, 2680, 2770, 2860, 2950,

             510,  520,  530,  540,  550,
            1286, 1312, 1338, 1364, 1390,
            2062, 2104, 2146, 2188, 2230,

            3998, 4056, 4114, 4172, 4230,
            5094, 5168, 5242, 5316, 5390,
            6190, 6280, 6370, 6460, 6550
        });

        EXPECT_EQ(matmul(A, B), C);
        EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
    }

    {
        auto A = Tensor<TypeParam>({2, 1, 3, 4}).range(1);
        auto B = Tensor<TypeParam>({2, 2, 4, 5}).range(1);
        auto C = Tensor<TypeParam>({2, 2, 3, 5}, {
             110,  120,  130,  140,  150,
             246,  272,  298,  324,  350,
             382,  424,  466,  508,  550,

             310,  320,  330,  340,  350,
             766,  792,  818,  844,  870,
            1222, 1264, 1306, 1348, 1390,

            2838, 2896, 2954, 3012, 3070,
            3614, 3688, 3762, 3836, 3910,
            4390, 4480, 4570, 4660, 4750,

            3998, 4056, 4114, 4172, 4230,
            5094, 5168, 5242, 5316, 5390,
            6190, 6280, 6370, 6460, 6550
        });

        EXPECT_EQ(matmul(A, B), C);
        EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
    }
}

TYPED_TEST(MatMulTest, Broadcast4DRight) {
    {
        auto A = Tensor<TypeParam>({2, 2, 3, 4}).range(1);
        auto B = Tensor<TypeParam>({4, 5}).range(1);
        auto C = Tensor<TypeParam>({2, 2, 3, 5}, {
             110,  120,  130,  140,  150,
             246,  272,  298,  324,  350,
             382,  424,  466,  508,  550,

             518,  576,  634,  692,  750,
             654,  728,  802,  876,  950,
             790,  880,  970, 1060, 1150,

             926, 1032, 1138, 1244, 1350,
            1062, 1184, 1306, 1428, 1550,
            1198, 1336, 1474, 1612, 1750,

            1334, 1488, 1642, 1796, 1950,
            1470, 1640, 1810, 1980, 2150,
            1606, 1792, 1978, 2164, 2350
        });

        EXPECT_EQ(matmul(A, B), C);
        EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);

        auto B1 = unsqueeze(B, 0, 1);
        EXPECT_EQ(matmul(A, B1), C);
        EXPECT_EQ(matmul(dev(A), dev(B1)).read(), C);
    }

    {
        auto A = Tensor<TypeParam>({2, 2, 3, 4}).range(1);
        auto B = Tensor<TypeParam>({1, 2, 4, 5}).range(1);
        auto C = Tensor<TypeParam>({2, 2, 3, 5}, {
             110,  120,  130,  140,  150,
             246,  272,  298,  324,  350,
             382,  424,  466,  508,  550,

            1678, 1736, 1794, 1852, 1910,
            2134, 2208, 2282, 2356, 2430,
            2590, 2680, 2770, 2860, 2950,

             926, 1032, 1138, 1244, 1350,
            1062, 1184, 1306, 1428, 1550,
            1198, 1336, 1474, 1612, 1750,

            4414, 4568, 4722, 4876, 5030,
            4870, 5040, 5210, 5380, 5550,
            5326, 5512, 5698, 5884, 6070
        });

        EXPECT_EQ(matmul(A, B), C);
        EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
    }

    {
        auto A = Tensor<TypeParam>({2, 2, 3, 4}).range(1);
        auto B = Tensor<TypeParam>({2, 1, 4, 5}).range(1);
        auto C = Tensor<TypeParam>({2, 2, 3, 5}, {
             110,  120,  130,  140,  150,
             246,  272,  298,  324,  350,
             382,  424,  466,  508,  550,

             518,  576,  634,  692,  750,
             654,  728,  802,  876,  950,
             790,  880,  970, 1060, 1150,

            3046, 3152, 3258, 3364, 3470,
            3502, 3624, 3746, 3868, 3990,
            3958, 4096, 4234, 4372, 4510,

            4414, 4568, 4722, 4876, 5030,
            4870, 5040, 5210, 5380, 5550,
            5326, 5512, 5698, 5884, 6070
        });

        EXPECT_EQ(matmul(A, B), C);
        EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
    }
}

TYPED_TEST(MatMulTest, Broadcast4DBothSide) {
    {
        auto A = Tensor<TypeParam>({1, 2, 3, 4}).range(1);
        auto B = Tensor<TypeParam>({2, 1, 4, 5}).range(1);
        auto C = Tensor<TypeParam>({2, 2, 3, 5}, {
             110,  120,  130,  140,  150,
             246,  272,  298,  324,  350,
             382,  424,  466,  508,  550,

             518,  576,  634,  692,  750,
             654,  728,  802,  876,  950,
             790,  880,  970, 1060, 1150,

             310,  320,  330,  340,  350,
             766,  792,  818,  844,  870,
            1222, 1264, 1306, 1348, 1390,

            1678, 1736, 1794, 1852, 1910,
            2134, 2208, 2282, 2356, 2430,
            2590, 2680, 2770, 2860, 2950
        });

        EXPECT_EQ(matmul(A, B), C);
        EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
    }

    {
        auto A = Tensor<TypeParam>({2, 1, 3, 4}).range(1);
        auto B = Tensor<TypeParam>({1, 2, 4, 5}).range(1);
        auto C = Tensor<TypeParam>({2, 2, 3, 5}, {
             110,  120,  130,  140,  150,
             246,  272,  298,  324,  350,
             382,  424,  466,  508,  550,

             310,  320,  330,  340,  350,
             766,  792,  818,  844,  870,
            1222, 1264, 1306, 1348, 1390,

             518,  576,  634,  692,  750,
             654,  728,  802,  876,  950,
             790,  880,  970, 1060, 1150,

            1678, 1736, 1794, 1852, 1910,
            2134, 2208, 2282, 2356, 2430,
            2590, 2680, 2770, 2860, 2950
        });

        EXPECT_EQ(matmul(A, B), C);
        EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
    }
}

TYPED_TEST(MatMulTest, Slice) {
    auto X = Tensor<TypeParam>({7, 9}).range(1);
    auto A = X["0:3, 0:4"]; // 2x3x4
    auto B = X["3:7, 4:9"]; // 2x4x5
    auto C = Tensor<TypeParam>({3, 5}, {
         500,  510,  520,  530,  540,
        2138, 2184, 2230, 2276, 2322,
        3776, 3858, 3940, 4022, 4104,
    });

    EXPECT_EQ(matmul(A, B), C);

    auto dev_X = dev(X);
    auto dev_A = dev_X["0:3, 0:4"]; // 2x3x4
    auto dev_B = dev_X["3:7, 4:9"]; // 2x4x5
    EXPECT_EQ(matmul(dev_A, dev_B).read(), C);
}

TYPED_TEST(MatMulTest, BatchedSlice) {
    auto X = Tensor<TypeParam>({2, 7, 9}).range(1);
    auto A = X[":, 0:3, 0:4"]; // 2x3x4
    auto B = X[":, 3:7, 4:9"]; // 2x4x5
    auto C = Tensor<TypeParam>({2, 3, 5}, {
         500,  510,  520,  530,  540,
        2138, 2184, 2230, 2276, 2322,
        3776, 3858, 3940, 4022, 4104,

        28472, 28734, 28996, 29258, 29520,
        32378, 32676, 32974, 33272, 33570,
        36284, 36618, 36952, 37286, 37620
    });

    EXPECT_EQ(matmul(A, B), C);

    auto dev_X = dev(X);
    auto dev_A = dev_X[":, 0:3, 0:4"]; // 2x3x4
    auto dev_B = dev_X[":, 3:7, 4:9"]; // 2x4x5
    EXPECT_EQ(matmul(dev_A, dev_B).read(), C);
}

TYPED_TEST(MatMulTest, NonContiguousSlice) {
    auto X = Tensor<TypeParam>({2, 8, 8}).range(1);
    auto A = X[":, 0:7:2, 0:7:2"];
    auto B = X[":, 1:8:2, 1:8:2"];
    auto C = Tensor<TypeParam>({2, 4, 4}, {
         704,   736,   768,   800,
        2880,  3040,  3200,  3360,
        5056,  5344,  5632,  5920,
        7232,  7648,  8064,  8480,

        26816, 27360, 27904, 28448,
        33088, 33760, 34432, 35104,
        39360, 40160, 40960, 41760,
        45632, 46560, 47488, 48416
    });

    EXPECT_EQ(matmul(A, B), C);

    auto dev_X = dev(X);
    auto dev_A = dev_X[":, 0:7:2, 0:7:2"];
    auto dev_B = dev_X[":, 1:8:2, 1:8:2"];
    EXPECT_EQ(matmul(dev_A, dev_B).read(), C);
}

TYPED_TEST(MatMulTest, NonContiguousSliceWithBroadcast) {
    auto X = Tensor<TypeParam>({2, 8, 8}).range(1);
    auto A = X[":, 0:7:2, 0:7:2"];
    auto B = X["1, 1:8:2, 1:8:2"];
    auto C = Tensor<TypeParam>({2, 4, 4}, {
         1728,  1760,  1792,  1824,
         8000,  8160,  8320,  8480,
        14272, 14560, 14848, 15136,
        20544, 20960, 21376, 21792,

        26816, 27360, 27904, 28448,
        33088, 33760, 34432, 35104,
        39360, 40160, 40960, 41760,
        45632, 46560, 47488, 48416
    });

    EXPECT_EQ(matmul(A, B), C);

    auto dev_X = dev(X);
    auto dev_A = dev_X[":, 0:7:2, 0:7:2"];
    auto dev_B = dev_X["1, 1:8:2, 1:8:2"];
    EXPECT_EQ(matmul(dev_A, dev_B).read(), C);
}

TYPED_TEST(MatMulTest, NonContiguousNonSquareSlice) {
    auto X = Tensor<TypeParam>({2, 6, 8}).range(1);
    auto Y = Tensor<TypeParam>({2, 8, 10}).range(1);
    auto A = X[":, 0:6:2, 0:8:2"];  // 2x3x4
    auto B = Y[":, 0:8:2, 0:10:2"]; // 2x4x5
    auto C = Tensor<TypeParam>({2, 3, 5}, {
          696,   728,   760,   792,   824,
         2680,  2840,  3000,  3160,  3320,
         4664,  4952,  5240,  5528,  5816,

        23288, 23704, 24120, 24536, 24952,
        30392, 30936, 31480, 32024, 32568,
        37496, 38168, 38840, 39512, 40184
    });

    EXPECT_EQ(matmul(A, B), C);

    auto dev_X = dev(X);
    auto dev_Y = dev(Y);
    auto dev_A = dev_X[":, 0:6:2, 0:8:2"];  // 2x3x4
    auto dev_B = dev_Y[":, 0:8:2, 0:10:2"]; // 2x4x5
    EXPECT_EQ(matmul(dev_A, dev_B).read(), C);
}

TYPED_TEST(MatMulTest, NonContiguousVector) {
    auto X = Tensor<TypeParam>({8}).range(1);
    auto Y = Tensor<TypeParam>::scalar(100);
    EXPECT_EQ(matmul(X["::2"], X["1::2"]), Y);

    auto dev_X = dev(X);
    EXPECT_EQ(matmul(dev_X["::2"], dev_X["1::2"]).read(), Y);
}

TYPED_TEST(MatMulTest, NonContiguousVectorWithNegativeStride) {
    auto X = Tensor<TypeParam>({8}).range(1);
    auto Y = Tensor<TypeParam>::scalar(100);
    EXPECT_EQ(matmul(X["::-2"], flip(X["::2"])), Y);

    auto dev_X = dev(X);
    EXPECT_EQ(matmul(dev_X["::-2"], flip(dev_X["::2"])).read(), Y);
}

TYPED_TEST(MatMulTest, MatrixAndNonContiguousVector) {
    auto X = Tensor<TypeParam>({2, 4, 4}).range(1);
    auto Y = Tensor<TypeParam>({8}).range(1);
    auto Z = Matrix<TypeParam>({{50, 114, 178, 242}, {306, 370, 434, 498}});
    auto W = Matrix<TypeParam>({{152, 168, 184, 200}, {408, 424, 440, 456}});
    EXPECT_EQ(matmul(X, Y["::2"]), Z);
    EXPECT_EQ(matmul(Y["::2"], X), W);
    EXPECT_EQ(matmul(X[0], Y["::2"]), Z[0]);
    EXPECT_EQ(matmul(Y["::2"], X[0]), W[0]);

    auto dev_X = dev(X);
    auto dev_Y = dev(Y);
    EXPECT_EQ(matmul(dev_X, dev_Y["::2"]).read(), Z);
    EXPECT_EQ(matmul(dev_Y["::2"], dev_X).read(), W);
    EXPECT_EQ(matmul(dev_X[0], dev_Y["::2"]).read(), Z[0]);
    EXPECT_EQ(matmul(dev_Y["::2"], dev_X[0]).read(), W[0]);
}

TYPED_TEST(MatMulTest, MatrixAndNonContiguousVectorWithNegativeStride) {
    auto X = Tensor<TypeParam>({2, 4, 4}).range(1);
    auto Y = Tensor<TypeParam>({8}).range(1);
    auto Z = Matrix<TypeParam>({{40, 120, 200, 280}, {360, 440, 520, 600}});
    auto W = Matrix<TypeParam>({{100, 120, 140, 160}, {420, 440, 460, 480}});
    EXPECT_EQ(matmul(X, Y["::-2"]), Z);
    EXPECT_EQ(matmul(Y["::-2"], X), W);
    EXPECT_EQ(matmul(X[0], Y["::-2"]), Z[0]);
    EXPECT_EQ(matmul(Y["::-2"], X[0]), W[0]);

    auto dev_X = dev(X);
    auto dev_Y = dev(Y);
    EXPECT_EQ(matmul(dev_X, dev_Y["::-2"]).read(), Z);
    EXPECT_EQ(matmul(dev_Y["::-2"], dev_X).read(), W);
    EXPECT_EQ(matmul(dev_X[0], dev_Y["::-2"]).read(), Z[0]);
    EXPECT_EQ(matmul(dev_Y["::-2"], dev_X[0]).read(), W[0]);
}

TYPED_TEST(MatMulTest, OverlappedStride) {
    auto A = Tensor<TypeParam>({2, 5}).range(1);
    auto B = Vector<TypeParam>({1, 2, 3});
    auto C = Matrix<TypeParam>({{14, 20, 26}, {44, 50, 56}});
    EXPECT_EQ(matmul(partition(A, 1, 3, 1, 1), B), C);
    EXPECT_EQ(matmul(B, partition(A, 1, 3, 1, 1)), C);

    auto dev_A = dev(A);
    auto dev_B = dev(B);
    EXPECT_EQ(matmul(partition(dev_A, 1, 3, 1, 1), dev_B).read(), C);
    EXPECT_EQ(matmul(dev_B, partition(dev_A, 1, 3, 1, 1)).read(), C);
}

TYPED_TEST(MatMulTest, BroadcastedVectors) {
    auto X = Scalar<TypeParam>(2);
    auto Y = Scalar<TypeParam>(3);
    auto Z = Scalar<TypeParam>(24);
    EXPECT_EQ(matmul(X.broadcast({4}), Y.broadcast({4})), Z);

    auto dev_X = dev(X);
    auto dev_Y = dev(Y);
    EXPECT_EQ(matmul(dev_X.broadcast({4}), dev_Y.broadcast({4})).read(), Z);
}

TYPED_TEST(MatMulTest, MatrixAndBroadcastedVector) {
    auto X = Tensor<TypeParam>({4, 4}).range(1);
    auto Y = Scalar<TypeParam>(3);
    auto Z = Tensor<TypeParam>({4}, {30, 78, 126, 174});
    auto W = Tensor<TypeParam>({4}, {84, 96, 108, 120});
    EXPECT_EQ(matmul(X, Y.broadcast({4})), Z);
    EXPECT_EQ(matmul(Y.broadcast({4}), X), W);

    auto dev_X = dev(X);
    auto dev_Y = dev(Y);
    EXPECT_EQ(matmul(dev_X, dev_Y.broadcast({4})).read(), Z);
    EXPECT_EQ(matmul(dev_Y.broadcast({4}), dev_X).read(), W);
}

TYPED_TEST(MatMulTest, Transpose) {
    auto A = Tensor<TypeParam>({4, 3}).range(1);
    auto B = Tensor<TypeParam>({4, 5}).range(1);
    auto C = Tensor<TypeParam>({3, 5}, {
        262,  284,  306,  328,  350,
        296,  322,  348,  374,  400,
        330,  360,  390,  420,  450,
    });

    EXPECT_EQ(matmul(A.transpose(), B), C);
    EXPECT_EQ(matmul(dev(A).transpose(), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, TransposeLast) {
    auto A = Tensor<TypeParam>({2, 4, 3}).range(1);
    auto B = Tensor<TypeParam>({2, 4, 5}).range(1);
    auto C = Tensor<TypeParam>({2, 3, 5}, {
        262,  284,  306,  328,  350,
        296,  322,  348,  374,  400,
        330,  360,  390,  420,  450,

        2070, 2140, 2210, 2280, 2350,
        2184, 2258, 2332, 2406, 2480,
        2298, 2376, 2454, 2532, 2610
    });

    EXPECT_EQ(matmul(A.transpose(0, 2, 1), B), C);
    EXPECT_EQ(matmul(dev(A).transpose(0, 2, 1), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, TransposePrefix) {
    auto A = Tensor<TypeParam>({2, 3, 2, 3}).range(1);
    auto B = Tensor<TypeParam>({3, 1, 3, 2}).range(1);
    auto C = Tensor<TypeParam>({3, 2, 2, 2}, {
          22,   28,   49,   64,
         184,  244,  211,  280,
         220,  244,  301,  334,
         706,  784,  787,  874,
         634,  676,  769,  820,
        1444, 1540, 1579, 1684
    });

    EXPECT_EQ(matmul(A.transpose(1, 0, 2, 3), B), C);
    EXPECT_EQ(matmul(dev(A).transpose(1, 0, 2, 3), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, TransposeAll) {
    auto A = Tensor<TypeParam>({4, 3, 2}).range(1);
    auto B = Tensor<TypeParam>({2, 4, 5}).range(1);
    auto C = Tensor<TypeParam>({2, 3, 5}, {
         490,  530,  570,  610,  650,
         558,  606,  654,  702,  750,
         626,  682,  738,  794,  850,

        1404, 1448, 1492, 1536, 1580,
        1632, 1684, 1736, 1788, 1840,
        1860, 1920, 1980, 2040, 2100
    });

    EXPECT_EQ(matmul(A.transpose(), B), C);
    EXPECT_EQ(matmul(dev(A).transpose(), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, MatrixAndVectorView) {
    auto A = Tensor<TypeParam>({3, 4}).range(1);
    auto B = Tensor<TypeParam>({8, 2}).range(1);
    auto C = Tensor<TypeParam>({3, 1}, {
         60, 140, 220,
    });

    EXPECT_EQ(matmul(A, B["0:4, 1"]), C);
    EXPECT_EQ(matmul(dev(A), dev(B)["0:4, 1"]).read(), C);
}

TYPED_TEST(MatMulTest, BatchedMatrixAndVectorView) {
    auto A = Tensor<TypeParam>({2, 3, 4}).range(1);
    auto B = Tensor<TypeParam>({8, 2}).range(1);
    auto C = Tensor<TypeParam>({2, 3, 1}, {
         60, 140, 220,
        300, 380, 460
    });

    EXPECT_EQ(matmul(A, B["0:4, 1"]), C);
    EXPECT_EQ(matmul(dev(A), dev(B)["0:4, 1"]).read(), C);
}

TYPED_TEST(MatMulTest, TransposedMaxtrixAndVectorView) {
    auto A = Tensor<TypeParam>({4, 3}).range(1);
    auto B = Tensor<TypeParam>({8, 2}).range(1);
    auto C = Tensor<TypeParam>({3, 1}, {
        140, 160, 180,
    });

    EXPECT_EQ(matmul(A.transpose(), B["0:4, 1"]), C);
    EXPECT_EQ(matmul(dev(A).transpose(), dev(B)["0:4, 1"]).read(), C);
}

TYPED_TEST(MatMulTest, BatchedTransposedMaxtrixAndVectorView) {
    auto A = Tensor<TypeParam>({2, 4, 3}).range(1);
    auto B = Tensor<TypeParam>({8, 2}).range(1);
    auto C = Tensor<TypeParam>({2, 3, 1}, {
        140, 160, 180,
        380, 400, 420
    });

    EXPECT_EQ(matmul(A.transpose(0,2,1), B["0:4, 1"]), C);
    EXPECT_EQ(matmul(dev(A).transpose(0,2,1), dev(B)["0:4, 1"]).read(), C);
}

TYPED_TEST(MatMulTest, ScalarBroadcast) {
    auto A = Tensor<TypeParam>({2, 3, 4}).range(1);
    auto B = Tensor<TypeParam>::scalar(3);
    auto C = Tensor<TypeParam>({2, 3, 5}, {
         30,  30,  30,  30,  30,
         78,  78,  78,  78,  78,
        126, 126, 126, 126, 126,

        174, 174, 174, 174, 174,
        222, 222, 222, 222, 222,
        270, 270, 270, 270, 270
    });

    EXPECT_EQ(matmul(A, B.broadcast({2, 4, 5})), C);
    EXPECT_EQ(matmul(dev(A), dev(B).broadcast({2,4,5})).read(), C);
}

TYPED_TEST(MatMulTest, AsStrided) {
    auto X = Tensor<TypeParam>({4, 4}).range(0);
    auto A = as_strided(X, {2,2,3,3}, {4,1,4,1});
    auto B = Tensor<TypeParam>({3, 3}).range(1);
    auto C = Tensor<TypeParam>({2, 2, 3, 3}, {
         18,  21,  24,  66,  81,  96, 114, 141, 168,
         30,  36,  42,  78,  96, 114, 126, 156, 186,
         66,  81,  96, 114, 141, 168, 162, 201, 240,
         78,  96, 114, 126, 156, 186, 174, 216, 258
    });

    EXPECT_EQ(matmul(A, B), C);

    auto dev_X = dev(X);
    auto dev_A = as_strided(dev_X, {2,2,3,3}, {4,1,4,1});
    auto dev_B = dev(B);
    EXPECT_EQ(matmul(dev_A, dev_B).read(), C);
}

TYPED_TEST(MatMulTest, Tile) {
    auto A = Matrix<TypeParam>({
        {3, 3, 7, 2, 8},
        {5, 9, 1, 2, 1},
        {1, 2, 7, 4, 8},
        {1, 7, 0, 7, 7},
        {1, 9, 8, 2, 4}
    });
    auto B = Matrix<TypeParam>({
        {6, 7, 2, 2, 10},
        {0, 1, 8, 0, 8},
        {6, 4, 0, 9, 1},
        {7, 6, 9, 1, 8},
        {8, 0, 8, 8, 3}
    });
    auto C = Tensor<TypeParam>({5, 5});

    // https://www.maths.manchester.ac.uk/~higham/papers/high90s.pdf
    matmul(1, A[":,-1"],      B["-1,:"],      0, C);
    matmul(1, A["0:-1,0:-1"], B["0:-1,0:-1"], 1, C["0:-1,0:-1"]);
    matmul(1, A["0:-1,0:-1"], B["0:-1,-1"],   1, C["0:-1,-1"]);
    matmul(1, A["-1,0:-1"],   B["0:-1,0:-1"], 1, C["-1,0:-1"]);
    matmul(1, A["-1,0:-1"],   B["0:-1,-1"],   1, C["-1,-1"]);
    EXPECT_EQ(C, matmul(A, B));

    auto dev_A = dev(A);
    auto dev_B = dev(B);
    auto dev_C = DevTensor<TypeParam>({5, 5});

    matmul(1, dev_A[":,-1"],      dev_B["-1,:"],      0, dev_C);
    matmul(1, dev_A["0:-1,0:-1"], dev_B["0:-1,0:-1"], 1, dev_C["0:-1,0:-1"]);
    matmul(1, dev_A["0:-1,0:-1"], dev_B["0:-1,-1"],   1, dev_C["0:-1,-1"]);
    matmul(1, dev_A["-1,0:-1"],   dev_B["0:-1,0:-1"], 1, dev_C["-1,0:-1"]);
    matmul(1, dev_A["-1,0:-1"],   dev_B["0:-1,-1"],   1, dev_C["-1,-1"]);
    EXPECT_EQ(dev_C.read(), matmul(dev_A, dev_B).read());
}
