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

template <typename T> struct MatMulTest : public testing::Test {};
using MatMulTestTypes = testing::Types<float, int>;
TYPED_TEST_CASE(MatMulTest, MatMulTestTypes);

TYPED_TEST(MatMulTest, MatMul) {
    auto A = Tensor<TypeParam>::range({2, 3, 4}, 1);
    auto B = Tensor<TypeParam>::range({2, 4, 5}, 1);
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
    auto A = Tensor<TypeParam>::range({4}, 1);
    auto B = Tensor<TypeParam>::range({2, 4, 5}, 1);
    auto C = Tensor<TypeParam>({2, 5}, {
        110, 120, 130, 140, 150,
        310, 320, 330, 340, 350,
    });

    EXPECT_EQ(matmul(A, B), C);
    EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, RightHandSideIsVector) {
    auto A = Tensor<TypeParam>::range({2, 3, 4}, 1);
    auto B = Tensor<TypeParam>::range({4}, 1);
    auto C = Tensor<TypeParam>({2, 3}, {
         30,  70, 110,
        150, 190, 230
    });

    EXPECT_EQ(matmul(A, B), C);
    EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, VectorLike) {
    auto A = Tensor<TypeParam>::range({1, 8}, 1);
    auto B = Tensor<TypeParam>::range({8, 1}, 100);
    auto C = Tensor<TypeParam>({1, 1}, {3768});
    EXPECT_EQ(matmul(A, B), C);
    EXPECT_EQ(matmul(dev(A), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, Broadcast3D) {
    auto A = Tensor<TypeParam>::range({2, 3, 4}, 1);
    auto B = Tensor<TypeParam>::range({4, 5}, 1);
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

    auto B1 = unsqueeze(B, {0});
    EXPECT_EQ(matmul(A, B1), C);
    EXPECT_EQ(matmul(dev(A), dev(B1)).read(), C);
}

TYPED_TEST(MatMulTest, Broadcast4DLeft) {
    {
        auto A = Tensor<TypeParam>::range({3, 4}, 1);
        auto B = Tensor<TypeParam>::range({2, 2, 4, 5}, 1);
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

        auto A1 = unsqueeze(A, {0, 1});
        EXPECT_EQ(matmul(A1, B), C);
        EXPECT_EQ(matmul(dev(A1), dev(B)).read(), C);
    }

    {
        auto A = Tensor<TypeParam>::range({1, 2, 3, 4}, 1);
        auto B = Tensor<TypeParam>::range({2, 2, 4, 5}, 1);
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
        auto A = Tensor<TypeParam>::range({2, 1, 3, 4}, 1);
        auto B = Tensor<TypeParam>::range({2, 2, 4, 5}, 1);
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
        auto A = Tensor<TypeParam>::range({2, 2, 3, 4}, 1);
        auto B = Tensor<TypeParam>::range({4, 5}, 1);
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

        auto B1 = unsqueeze(B, {0, 1});
        EXPECT_EQ(matmul(A, B1), C);
        EXPECT_EQ(matmul(dev(A), dev(B1)).read(), C);
    }

    {
        auto A = Tensor<TypeParam>::range({2, 2, 3, 4}, 1);
        auto B = Tensor<TypeParam>::range({1, 2, 4, 5}, 1);
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
        auto A = Tensor<TypeParam>::range({2, 2, 3, 4}, 1);
        auto B = Tensor<TypeParam>::range({2, 1, 4, 5}, 1);
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
        auto A = Tensor<TypeParam>::range({1, 2, 3, 4}, 1);
        auto B = Tensor<TypeParam>::range({2, 1, 4, 5}, 1);
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
        auto A = Tensor<TypeParam>::range({2, 1, 3, 4}, 1);
        auto B = Tensor<TypeParam>::range({1, 2, 4, 5}, 1);
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
    auto X = Tensor<TypeParam>::range({7, 9}, 1);
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
    auto X = Tensor<TypeParam>::range({2, 7, 9}, 1);
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
    auto X = Tensor<TypeParam>::range({2, 8, 8}, 1);
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
    auto X = Tensor<TypeParam>::range({2, 8, 8}, 1);
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
    auto X = Tensor<TypeParam>::range({2, 6, 8}, 1);
    auto Y = Tensor<TypeParam>::range({2, 8, 10}, 1);
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
    auto X = Tensor<TypeParam>::range({8}, 1);
    auto Y = Tensor<TypeParam>({1}, {100});
    EXPECT_EQ(matmul(X["0:7:2"], X["1:8:2"]), Y);

    auto dev_X = dev(X);
    EXPECT_EQ(matmul(dev_X["0:7:2"], dev_X["1:8:2"]).read(), Y);
}

TYPED_TEST(MatMulTest, Transpose) {
    auto A = Tensor<TypeParam>::range({4, 3}, 1);
    auto B = Tensor<TypeParam>::range({4, 5}, 1);
    auto C = Tensor<TypeParam>({3, 5}, {
        262,  284,  306,  328,  350,
        296,  322,  348,  374,  400,
        330,  360,  390,  420,  450,
    });

    EXPECT_EQ(matmul(A.transpose(), B), C);
    EXPECT_EQ(matmul(dev(A).transpose(), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, TransposeLast) {
    auto A = Tensor<TypeParam>::range({2, 4, 3}, 1);
    auto B = Tensor<TypeParam>::range({2, 4, 5}, 1);
    auto C = Tensor<TypeParam>({2, 3, 5}, {
        262,  284,  306,  328,  350,
        296,  322,  348,  374,  400,
        330,  360,  390,  420,  450,

        2070, 2140, 2210, 2280, 2350,
        2184, 2258, 2332, 2406, 2480,
        2298, 2376, 2454, 2532, 2610
    });

    EXPECT_EQ(matmul(A.transpose({0, 2, 1}), B), C);
    EXPECT_EQ(matmul(dev(A).transpose({0, 2, 1}), dev(B)).read(), C);
}

TYPED_TEST(MatMulTest, TransposePrefix) {
    auto A = Tensor<TypeParam>::range({2, 3, 2, 3}, 1);
    auto B = Tensor<TypeParam>::range({3, 1, 3, 2}, 1);
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
    auto A = Tensor<TypeParam>::range({4, 3, 2}, 1);
    auto B = Tensor<TypeParam>::range({2, 4, 5}, 1);
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
    auto A = Tensor<TypeParam>::range({3, 4}, 1);
    auto B = Tensor<TypeParam>::range({8, 2}, 1);
    auto C = Tensor<TypeParam>({3, 1}, {
         60, 140, 220,
    });

    EXPECT_EQ(matmul(A, B["0:4, 1"]), C);
    EXPECT_EQ(matmul(dev(A), dev(B)["0:4, 1"]).read(), C);
}

TYPED_TEST(MatMulTest, BatchedMatrixAndVectorView) {
    auto A = Tensor<TypeParam>::range({2, 3, 4}, 1);
    auto B = Tensor<TypeParam>::range({8, 2}, 1);
    auto C = Tensor<TypeParam>({2, 3, 1}, {
         60, 140, 220,
        300, 380, 460
    });

    EXPECT_EQ(matmul(A, B["0:4, 1"]), C);
    EXPECT_EQ(matmul(dev(A), dev(B)["0:4, 1"]).read(), C);
}

TYPED_TEST(MatMulTest, TransposedMaxtrixAndVectorView) {
    auto A = Tensor<TypeParam>::range({4, 3}, 1);
    auto B = Tensor<TypeParam>::range({8, 2}, 1);
    auto C = Tensor<TypeParam>({3, 1}, {
        140, 160, 180,
    });

    EXPECT_EQ(matmul(A.transpose(), B["0:4, 1"]), C);
    EXPECT_EQ(matmul(dev(A).transpose(), dev(B)["0:4, 1"]).read(), C);
}

TYPED_TEST(MatMulTest, BatchedTransposedMaxtrixAndVectorView) {
    auto A = Tensor<TypeParam>::range({2, 4, 3}, 1);
    auto B = Tensor<TypeParam>::range({8, 2}, 1);
    auto C = Tensor<TypeParam>({2, 3, 1}, {
        140, 160, 180,
        380, 400, 420
    });

    EXPECT_EQ(matmul(A.transpose(0,2,1), B["0:4, 1"]), C);
    EXPECT_EQ(matmul(dev(A).transpose(0,2,1), dev(B)["0:4, 1"]).read(), C);
}

TYPED_TEST(MatMulTest, ScalarBroadcast) {
    auto A = Tensor<TypeParam>::range({2, 3, 4}, 1);
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
    auto X = Tensor<TypeParam>::range({4, 4}, 0);
    auto A = as_strided(X, {2,2,3,3}, {4,1,4,1});
    auto B = Tensor<TypeParam>::range({3, 3}, 1);
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

TEST(UniformTest, Dot) {
    auto A = Tensor<float>::range({2, 3, 4}, 0);
    auto B = Tensor<float>::range({3, 4, 5}, 0);
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
    auto A = Tensor<int>::range({2, 3});
    auto B = Tensor<int>::range({3, 4});
    auto C = Tensor<int>::range({4, 5});
    auto D = multi_dot(A, B, C);

    EXPECT_EQ(D, Tensor<int>({2, 5}, {
         810,  908, 1006, 1104, 1202,
        2520, 2816, 3112, 3408, 3704
    }));
    EXPECT_EQ(multi_dot(dev(A), dev(B), dev(C)).read(), D);
}

TEST(UniformTest, MultiDotVectorFirst) {
    auto A = Tensor<int>::range({3});
    auto B = Tensor<int>::range({3, 4});
    auto C = Tensor<int>::range({4, 5});
    auto D = multi_dot(A, B, C);

    EXPECT_EQ(D, Tensor<int>({5}, {810, 908, 1006, 1104, 1202}));
    EXPECT_EQ(multi_dot(dev(A), dev(B), dev(C)).read(), D);
}

TEST(UniformTest, MultiDotVectorLast) {
    auto A = Tensor<int>::range({2, 3});
    auto B = Tensor<int>::range({3, 4});
    auto C = Tensor<int>::range({4});
    auto D = multi_dot(A, B, C);

    EXPECT_EQ(D, Tensor<int>({2}, {162, 504}));
    EXPECT_EQ(multi_dot(dev(A), dev(B), dev(C)).read(), D);
}

TEST(UniformTest, MultiDotVectorFirstAndLast) {
    auto A = Tensor<int>::range({3});
    auto B = Tensor<int>::range({3, 4});
    auto C = Tensor<int>::range({4});
    auto D = multi_dot(A, B, C);

    EXPECT_EQ(D, Tensor<int>({1}, {162}));
    EXPECT_EQ(multi_dot(dev(A), dev(B), dev(C)).read(), D);
}

TEST(UniformTest, TensorDotSimple) {
    auto A = Tensor<float>::range({3,4,5}, 0);
    auto B = Tensor<float>::range({4,3,2}, 0);
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
struct Value {
    int n; char c;
    std::string s;

    Value() = default;
    Value(int n) : n(n) {}
    Value(char c) : c(c) {}
    Value(std::string s) : s(std::move(s)) {}
    Value(const char* s) : s(s) {}
};

inline Value operator*(const Value& n, const Value& c) {
    return std::string(n.n, c.c);
}

inline Value operator+(const Value& a, const Value& b) {
    return a.s + b.s;
}

inline Value& operator+=(Value& a, const Value& b) {
    a.s += b.s;
    return a;
}

inline bool operator==(const Value& a, const Value& b) {
    return a.s == b.s;
}

inline std::ostream& operator<<(std::ostream& out, const Value& s) {
    return out << s.s;
}

TEST(UniformTest, TensorDotExt) {
    auto a = Tensor<Value>({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto A = Tensor<Value>({2, 2}, {'a', 'b', 'c', 'd'});

    EXPECT_EQ(tensordot(a, A), Tensor<Value>({2}, {
        "abbcccdddd", "aaaaabbbbbbcccccccdddddddd"
    }));
    EXPECT_EQ(tensordot(a, A, 1), Tensor<Value>({2, 2, 2}, {
        "acc", "bdd",
        "aaacccc", "bbbdddd",
        "aaaaacccccc", "bbbbbdddddd",
        "aaaaaaacccccccc", "bbbbbbbdddddddd"
    }));
    EXPECT_EQ(tensordot(a, A, {0}, {1}), Tensor<Value>({2, 2, 2}, {
        "abbbbb", "cddddd",
        "aabbbbbb", "ccdddddd",
        "aaabbbbbbb", "cccddddddd",
        "aaaabbbbbbbb", "ccccdddddddd"
    }));
    EXPECT_EQ(tensordot(a, A, {2}, {1}), Tensor<Value>({2, 2, 2}, {
        "abb", "cdd",
        "aaabbbb", "cccdddd",
        "aaaaabbbbbb", "cccccdddddd",
        "aaaaaaabbbbbbbb", "cccccccdddddddd"
    }));
    EXPECT_EQ(tensordot(a, A, {0,1}, {0,1}), Tensor<Value>({2}, {
        "abbbcccccddddddd", "aabbbbccccccdddddddd",
    }));
    EXPECT_EQ(tensordot(a, A, {2,1}, {1,0}), Tensor<Value>({2}, {
        "acccbbdddd", "aaaaacccccccbbbbbbdddddddd",
    }));

    EXPECT_EQ(tensordot(a, A, 0), cross(a, A));
    EXPECT_EQ(tensordot(a, A, 1), dot(a, A));
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

TEST(UniformTest, SqueezeToScalar) {
    auto A = Tensor<int>({1, 1}, 123);
    EXPECT_EQ(squeeze(A), Tensor<int>({1}, 123));
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

    {
        auto A = Tensor<int>::range({2, 3, 4}, 1);
        auto B = Tensor<int>::range({4, 3, 2}, 1);
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

    {
        auto A = dev(Tensor<int>::range({2, 3, 4}, 1));
        auto B = dev(Tensor<int>::range({4, 3, 2}, 1));
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

        auto splits = split(X, 0, {2, 3, 4});
        EXPECT_EQ(splits[0], A);
        EXPECT_EQ(splits[1], B);
        EXPECT_EQ(splits[2], C);
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

        auto splits = split(X, 1, {2, 3, 4});
        EXPECT_EQ(splits[0], A);
        EXPECT_EQ(splits[1], B);
        EXPECT_EQ(splits[2], C);
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

        auto splits = split(X, 2, {2, 3, 4});
        EXPECT_EQ(splits[0], A);
        EXPECT_EQ(splits[1], B);
        EXPECT_EQ(splits[2], C);
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

        auto splits = split(X, 2, {2, 3, 4});
        EXPECT_EQ(splits[0].read(), A.read());
        EXPECT_EQ(splits[1].read(), B.read());
        EXPECT_EQ(splits[2].read(), C.read());
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

    EXPECT_EQ(X["2:5, 3:7, :"], Y);
    EXPECT_EQ(dev(X)["2:5, 3:7, :"].read(), Y);
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

    EXPECT_EQ(X["2:5, 3:7"]["1:3, 1:3"], Y);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({2, 2, 5});
    reorder(dev_X, shape, dev_Y);
    EXPECT_EQ(dev_Y.read(), Y);
    EXPECT_EQ(dev_X["2:5, 3:7"]["1:3, 1:3"].read(), Y);
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

    EXPECT_EQ(X["2:5, 3:7"].transpose(), Y);

    auto dev_X = dev(X);
    auto dev_Y = DevTensor<float>({5, 4, 3});
    reorder(dev_X, shape, dev_Y);
    EXPECT_EQ(dev_Y.read(), Y);
    EXPECT_EQ(dev_X["2:5, 3:7"].transpose().read(), Y);
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

    reorder(X1[":, :, 1:4"], X1[":, :, 5:8"]);
    EXPECT_EQ(X1, X);

    reorder(dev_X, src_shape, dev_X, dst_shape);
    EXPECT_EQ(dev_X.read(), X);
    reorder(dev_X1[":, :, 1:4"], dev_X1[":, :, 5:8"]);
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

    reorder(X1["1"], X1["2"]);
    EXPECT_EQ(X1, X);

    reorder(dev_X, src_shape, dev_X, dst_shape);
    EXPECT_EQ(dev_X.read(), X);

    reorder(dev_X1["1"], dev_X1["2"]);
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

    reorder(Y, X1["1:3, 1:3, 1:3"]);
    EXPECT_EQ(X1, R);

    reorder(dev_Y, dev_Y.shape(), dev_X, shape);
    EXPECT_EQ(dev_X.read(), R);

    reorder(dev_Y, dev_X1["1:3, 1:3, 1:3"]);
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
    auto X = Tensor<float>::range({2, 1, 4}, 0);
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

TEST(UniformTest, MoveAxis) {
    auto X = Tensor<int>({3, 4, 5, 6});
    EXPECT_EQ(moveaxis(X, 0, -1).shape(), Shape(4, 5, 6, 3));
    EXPECT_EQ(moveaxis(X, -1, 0).shape(), Shape(6, 3, 4, 5));
    EXPECT_EQ(moveaxis(X, {0,-1}, {-1,0}).shape(), Shape({6, 4, 5, 3}));
    EXPECT_EQ(moveaxis(X, {0,1}, {-1,-2}).shape(), Shape(5, 6, 4, 3));
    EXPECT_EQ(moveaxis(X, {0,3}, {-1,2}).shape(), Shape(4, 5, 6, 3));
}

TYPED_TEST(TransposeTest, SwapAxes) {
    auto A = Tensor<TypeParam>::range({2, 2, 2}, 0);
    auto B = Tensor<TypeParam>({2, 2, 2}, {0, 4, 2, 6, 1, 5, 3, 7});
    EXPECT_EQ(swapaxes(A, 0, 2), B);
    EXPECT_EQ(swapaxes(dev(A), 0, 2).read(), B);
}

TEST(UniformTest, Flip) {
    auto A = Tensor<float>::range({2, 2, 2}, 0);

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

    auto B = Tensor<float>::range({2, 3, 4}, 0);
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
    auto Z = where(condition["0:2"], X["0:2"], X["2:4"]);
    EXPECT_EQ(Z, Tensor<int>({2, 2}, {1, 6, 3, 8}));
}

TEST(UniformTest, WhereView_GPU) {
    auto condition = dev(Tensor<bool>({2}, {true, false}));
    auto X = dev(Tensor<int>({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));
    auto Z = where(condition["0:2"], X["0:2"], X["2:4"]);
    EXPECT_EQ(Z.read(), Tensor<int>({2, 2}, {1, 6, 3, 8}));
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
    auto X = Tensor<float>::range({3,3,3}, 0);

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
    auto X = Tensor<float>::range({3, 3, 3}, 0);
    EXPECT_EQ(reduce_sum(dev(X), {}).read(), reduce_sum(X, {}));
    EXPECT_EQ(reduce_sum(dev(X), {0}).read(), reduce_sum(X, {0}));
    EXPECT_EQ(reduce_sum(dev(X), {1}).read(), reduce_sum(X, {1}));
    EXPECT_EQ(reduce_sum(dev(X), {2}).read(), reduce_sum(X, {2}));
    EXPECT_EQ(reduce_sum(dev(X), {0,2}).read(), reduce_sum(X, {0,2}));
}

TEST(UniformTest, ReduceMean) {
    auto X = Tensor<float>::range({3,3,3}, 0);

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
    auto X = Tensor<float>::range({3, 3, 3}, 0);
    EXPECT_EQ(round(reduce_mean(dev(X), {}).read()), reduce_mean(X, {}));
    EXPECT_EQ(round(reduce_mean(dev(X), {0}).read()), reduce_mean(X, {0}));
    EXPECT_EQ(round(reduce_mean(dev(X), {1}).read()), reduce_mean(X, {1}));
    EXPECT_EQ(round(reduce_mean(dev(X), {2}).read()), reduce_mean(X, {2}));
    EXPECT_EQ(round(reduce_mean(dev(X), {0,2}).read()), reduce_mean(X, {0,2}));
}

TEST(UniformTest, ReduceSumSquare) {
    auto X = Tensor<float>::range({3, 2, 2}, 1);
    EXPECT_EQ(reduce_sum_square(X, {1}), reduce_sum(X*X, {1}));
}

TEST(UniformTest, ReduceSumSquare_GPU) {
    auto X = Tensor<float>::range({3, 3, 3}, 0);
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
    auto X = Tensor<float>::range({3, 3, 3}, 0);
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
    auto X = Tensor<float>::range({3, 3, 3}, 0);
    ExpectElementsEQ(reduce_log_sum_exp(dev(X), {}).read(), reduce_log_sum_exp(X, {}));
    ExpectElementsEQ(reduce_log_sum_exp(dev(X), {0}).read(), reduce_log_sum_exp(X, {0}));
    ExpectElementsEQ(reduce_log_sum_exp(dev(X), {1}).read(), reduce_log_sum_exp(X, {1}));
    ExpectElementsEQ(reduce_log_sum_exp(dev(X), {2}).read(), reduce_log_sum_exp(X, {2}));
    ExpectElementsEQ(reduce_log_sum_exp(dev(X), {0,2}).read(), reduce_log_sum_exp(X, {0,2}));
}

TEST(UniformTest, ReduceProd) {
    auto X = Tensor<float>::range({3, 2, 2}, 1);

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
    auto X = Tensor<float>::range({3, 3, 3}, 0);
    ExpectElementsEQ(reduce_prod(dev(X), {}).read(), reduce_prod(X, {}));
    ExpectElementsEQ(reduce_prod(dev(X), {0}).read(), reduce_prod(X, {0}));
    ExpectElementsEQ(reduce_prod(dev(X), {1}).read(), reduce_prod(X, {1}));
    ExpectElementsEQ(reduce_prod(dev(X), {2}).read(), reduce_prod(X, {2}));
    ExpectElementsEQ(reduce_prod(dev(X), {0,2}).read(), reduce_prod(X, {0,2}));
}

TEST(UniformTest, ReduceL1) {
    auto X = Tensor<float>({3, 2, 2}).random(-10, 10);
    EXPECT_EQ(reduce_l1(X, {1}), reduce_sum(abs(X), {1}));
}

TEST(UniformTest, ReduceL1_GPU) {
    auto X = Tensor<float>::range({3, 3, 3}, 0);
    ExpectElementsEQ(reduce_l1(dev(X), {}).read(), reduce_l1(X, {}));
    ExpectElementsEQ(reduce_l1(dev(X), {0}).read(), reduce_l1(X, {0}));
    ExpectElementsEQ(reduce_l1(dev(X), {1}).read(), reduce_l1(X, {1}));
    ExpectElementsEQ(reduce_l1(dev(X), {2}).read(), reduce_l1(X, {2}));
    ExpectElementsEQ(reduce_l1(dev(X), {0,2}).read(), reduce_l1(X, {0,2}));
}

TEST(UniformTest, ReduceL2) {
    auto X = Tensor<float>({3, 2, 2}).random(-10, 10);
    EXPECT_EQ(reduce_l2(X, {1}), sqrt(reduce_sum(X*X, {1})));
}

TEST(UniformTest, ReduceL2_GPU) {
    auto X = Tensor<float>::range({3, 3, 3}, 0);
    ExpectElementsEQ(reduce_l2(dev(X), {}).read(), reduce_l2(X, {}));
    ExpectElementsEQ(reduce_l2(dev(X), {0}).read(), reduce_l2(X, {0}));
    ExpectElementsEQ(reduce_l2(dev(X), {1}).read(), reduce_l2(X, {1}));
    ExpectElementsEQ(reduce_l2(dev(X), {2}).read(), reduce_l2(X, {2}));
    ExpectElementsEQ(reduce_l2(dev(X), {0,2}).read(), reduce_l2(X, {0,2}));
}
