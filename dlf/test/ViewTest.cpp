#include "gtest/gtest.h"
#include "tensor.h"

using namespace dlf;

TEST(ViewTest, TransformTensorToView) {
    auto X = Tensor<int>::range({2, 2}, 1);
    auto Y = Tensor<int>::range({4, 4}, 1);

    auto dev_X = dev(X);
    auto dev_Y = dev(Y);

    transformTo(X, Y["1:3, 1:3"].transpose(), xfn::negate<>());
    EXPECT_EQ(Y, Tensor<int>({4, 4}, {
         1,  2,  3,  4,
         5, -1, -3,  8,
         9, -2, -4, 12,
        13, 14, 15, 16
    }));

    transformTo(dev_X, dev_Y["1:3, 1:3"].transpose(), xfn::negate<>());
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(ViewTest, TransformViewToTensor) {
    auto X = Tensor<int>::range({4, 4}, 1);
    auto Y = Tensor<int>::range({2, 2}, 1);

    auto dev_X = dev(X);
    auto dev_Y = dev(Y);

    transformTo(X["1:3, 1:3"].transpose(), Y, xfn::negate<>());
    EXPECT_EQ(Y, Tensor<int>({2, 2}, {
        -6, -10,
        -7, -11
    }));

    transformTo(dev_X["1:3, 1:3"].transpose(), dev_Y, xfn::negate<>());
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(ViewTest, TransformViewToView) {
    auto X = Tensor<int>::range({4, 4}, 1);
    auto Y = Tensor<int>::range({4, 4}, 1);

    auto dev_X = dev(X);
    auto dev_Y = dev(Y);

    transformTo(X["1:3, 1:3"].transpose(), Y["0:2, 0:2"], xfn::negate<>());
    EXPECT_EQ(Y, Tensor<int>({4, 4}, {
        -6, -10,  3,  4,
        -7, -11,  7,  8,
         9,  10, 11, 12,
        13,  14, 15, 16
    }));

    transformTo(dev_X["1:3, 1:3"].transpose(), dev_Y["0:2, 0:2"], xfn::negate<>());
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(ViewTest, CalculateOnView) {
    auto X = Tensor<int>::range({4, 4}, 1);
    auto Y = X["0:2, 0:2"] + X["0:2, 2:4"] + X["2:4, 0:2"] + X["2:4, 2:4"];
    EXPECT_EQ(Y, Tensor<int>({2, 2}, {
        24, 28,
        40, 44
    }));

    auto dev_X = dev(X);
    auto dev_Y = dev_X["0:2, 0:2"] + dev_X["0:2, 2:4"] + dev_X["2:4, 0:2"] + dev_X["2:4, 2:4"];
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(ViewTest, AggregateOnView) {
    auto X = Tensor<int>::range({4, 4}, 1);
    auto Y = sum(X["0:2, 0:2"], X["0:2, 2:4"], X["2:4, 0:2"], X["2:4, 2:4"]);
    EXPECT_EQ(Y, Tensor<int>({2, 2}, {
        24, 28,
        40, 44
    }));

    auto dev_X = dev(X);
    auto dev_Y = sum(dev_X["0:2, 0:2"], dev_X["0:2, 2:4"], dev_X["2:4, 0:2"], dev_X["2:4, 2:4"]);
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(ViewTest, UpdateOnView) {
    auto X = Tensor<int>::range({4, 4}, 1);
    auto dev_X = dev(X);

    X["1:3, 1:3"] *= 2;
    EXPECT_EQ(X, Tensor<int>({4, 4}, {
         1,  2,  3,  4,
         5, 12, 14,  8,
         9, 20, 22, 12,
        13, 14, 15, 16
    }));

    dev_X["1:3, 1:3"] *= 2;
    EXPECT_EQ(dev_X.read(), X);
}

TEST(ViewTest, Diagonal) {
    auto X = Tensor<int>::range({2, 2, 2}, 0);
    auto Y = Tensor<int>({2, 2}, {0, 6, 1, 7});
    EXPECT_EQ(X.diagonal(0, 0, 1), Y);
    EXPECT_EQ(dev(X).diagonal(0, 0, 1).read(), Y);
}

TEST(ViewTest, DiagonalWithPositiveOffset) {
    auto X = Tensor<int>::range({8, 8}, 0);
    auto Y = Tensor<int>({6}, {2, 11, 20, 29, 38, 47});
    EXPECT_EQ(X.diagonal(2), Y);
    EXPECT_EQ(dev(X).diagonal(2).read(), Y);
}

TEST(ViewTest, DiagonalWithNegativeOffset) {
    auto X = Tensor<int>::range({8, 8}, 0);
    auto Y = Tensor<int>({5}, {24, 33, 42, 51, 60});
    EXPECT_EQ(X.diagonal(-3), Y);
    EXPECT_EQ(dev(X).diagonal(-3).read(), Y);
}

TEST(ViewTest, DiagonalOfNonSquareMatrix) {
    auto X = Tensor<int>::range({4, 5}, 0);
    auto Y = Tensor<int>({4}, {0, 6, 12, 18});
    EXPECT_EQ(X.diagonal(), Y);
    EXPECT_EQ(dev(X).diagonal().read(), Y);
}

TEST(ViewTest, DiagonalOfNonSquareMatrixWithPositiveOffset) {
    auto X = Tensor<int>::range({4, 5}, 0);
    auto Y = Tensor<int>({3}, {1, 7, 13});
    EXPECT_EQ(X.diagonal(1), Y);
    EXPECT_EQ(dev(X).diagonal(1).read(), Y);
}

TEST(ViewTest, DiagonalOfNonSquareMatrixWithNegativeOffset) {
    auto X = Tensor<int>::range({4, 5}, 0);
    auto Y = Tensor<int>({3}, {5, 11, 17});
    EXPECT_EQ(X.diagonal(-1), Y);
    EXPECT_EQ(dev(X).diagonal(-1).read(), Y);
}

TEST(ViewTest, FillDiagonal) {
    auto X = Tensor<int>::range({4, 4}, 1);
    X.diagonal().fill(0);
    EXPECT_EQ(X, Tensor<int>({4, 4}, {
         0,  2,  3,  4,
         5,  0,  7,  8,
         9, 10,  0, 12,
        13, 14, 15,  0,
    }));
}

TEST(ViewTest, Diag) {
    auto diagonal = Tensor<int>::range({2, 4}, 1);
    auto Y = Tensor<int>({2, 4, 4}, {
        1, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 3, 0,
        0, 0, 0, 4,

        5, 0, 0, 0,
        0, 6, 0, 0,
        0, 0, 7, 0,
        0, 0, 0, 8
    });

    EXPECT_EQ(diag(diagonal), Y);
    EXPECT_EQ(diag(dev(diagonal)).read(), Y);
}

TEST(ViewTest, Trace) {
    auto X = Tensor<float>::range({2, 2, 2, 3}, 0);
    auto Y = Tensor<float>({2, 3}, {18, 20, 22, 24, 26, 28});
    EXPECT_EQ(trace(X, 0, 0, 1), Y);
    EXPECT_EQ(trace(dev(X), 0, 0, 1).read(), Y);
}

TEST(ViewTest, AsStrided) {
    auto sudoku = Tensor<int>({9, 9}, {
        2, 8, 7, 1, 6, 5, 9, 4, 3,
        9, 5, 4, 7, 3, 2, 1, 6, 8,
        6, 1, 3, 8, 4, 9, 7, 5, 2,
        8, 7, 9, 6, 5, 1, 2, 3, 4,
        4, 2, 1, 3, 9, 8, 6, 7, 5,
        3, 6, 5, 4, 2, 7, 8, 9, 1,
        1, 9, 8, 5, 7, 3, 4, 2, 6,
        5, 4, 2, 9, 1, 6, 3, 8, 7,
        7, 3, 6, 2, 8, 4, 5, 1, 9
    });

    auto squares = as_strided(sudoku, {3, 3, 3, 3}, {27, 3, 9, 1});
    EXPECT_EQ(squares, Tensor<int>({3, 3, 3, 3}, {
        2, 8, 7, 9, 5, 4, 6, 1, 3,
        1, 6, 5, 7, 3, 2, 8, 4, 9,
        9, 4, 3, 1, 6, 8, 7, 5, 2,
        8, 7, 9, 4, 2, 1, 3, 6, 5,
        6, 5, 1, 3, 9, 8, 4, 2, 7,
        2, 3, 4, 6, 7, 5, 8, 9, 1,
        1, 9, 8, 5, 4, 2, 7, 3, 6,
        5, 7, 3, 9, 1, 6, 2, 8, 4,
        4, 2, 6, 3, 8, 7, 5, 1, 9
    }));

    EXPECT_EQ(as_strided(dev(sudoku), {3, 3, 3, 3}, {27, 3, 9, 1}).read(), squares);
}

TEST(ViewTest, SlidingWindow) {
    auto X = Tensor<int>::range({1, 1, 4, 4}, 0);
    auto Y = as_strided(X, {2,2,3,3}, {4,1,4,1});

    EXPECT_EQ(Y, Tensor<int>({2,2,3,3}, {
        0,  1,  2,  4,  5,  6,  8,  9, 10,
        1,  2,  3,  5,  6,  7,  9, 10, 11,
        4,  5,  6,  8,  9, 10, 12, 13, 14,
        5,  6,  7,  9, 10, 11, 13, 14, 15
    }));

    EXPECT_EQ(as_strided(dev(X), {2,2,3,3}, {4,1,4,1}).read(), Y);

    EXPECT_EQ(reduce_mean(Y, {2,3}, true).transpose(2,3,0,1),
              dnn::average_pooling(X, dnn::Filter2D(X.shape(), {1,1,3,3}), false));
}

TEST(ViewTest, Partition1D) {
    auto A = Tensor<int>::range({6}, 1);
    EXPECT_EQ(partition(A, 0, 2), Tensor<int>({3, 2}, {
        1, 2, 3, 4, 5, 6
    }));
    EXPECT_EQ(partition(A, 0, 2, 1), Tensor<int>({5, 2}, {
        1, 2, 2, 3, 3, 4, 4, 5, 5, 6
    }));
    EXPECT_EQ(partition(A, 0, 3, 1, 2), Tensor<int>({2, 3}, {
        1, 3, 5, 2, 4, 6
    }));
}

TEST(ViewTest, Partition2D) {
    auto A = Tensor<int>({3, 4}, {
        11, 12, 13, 14,
        21, 22, 23, 24,
        31, 32, 33, 34
    });

    EXPECT_EQ(partition(A, 0, 2, 1), Tensor<int>({2, 4, 2}, {
        11, 21, 12, 22,
        13, 23, 14, 24,
        21, 31, 22, 32,
        23, 33, 24, 34
    }));

    EXPECT_EQ(partition(A, 1, 2, 1), Tensor<int>({3, 3, 2}, {
        11, 12, 12, 13, 13, 14,
        21, 22, 22, 23, 23, 24,
        31, 32, 32, 33, 33, 34
    }));

    EXPECT_EQ(partition(A, {0, 1}, {2, 2}, {1, 1}), Tensor<int>({2, 3, 2, 2}, {
        11, 12, 21, 22,
        12, 13, 22, 23,
        13, 14, 23, 24,
        21, 22, 31, 32,
        22, 23, 32, 33,
        23, 24, 33, 34
    }));

    EXPECT_EQ(partition(A, {0, 1}, {2, 2}, {1, 2}), Tensor<int>({2, 2, 2, 2}, {
        11, 12, 21, 22,
        13, 14, 23, 24,
        21, 22, 31, 32,
        23, 24, 33, 34
    }));

    EXPECT_EQ(partition(A, {0, 1}, {2, 2}, {2, 1}), Tensor<int>({1, 3, 2, 2}, {
        11, 12, 21, 22,
        12, 13, 22, 23,
        13, 14, 23, 24
    }));

    EXPECT_EQ(partition(A, {0, 1}, {2, 2}, {2, 2}), Tensor<int>({1, 2, 2, 2}, {
        11, 12, 21, 22,
        13, 14, 23, 24
    }));

    auto B = Tensor<int>({4, 3}, {
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43
    });

    EXPECT_EQ(partition(B, 0, 2, 1), Tensor<int>({3, 3, 2}, {
        11, 21, 12, 22, 13, 23,
        21, 31, 22, 32, 23, 33,
        31, 41, 32, 42, 33, 43
    }));

    EXPECT_EQ(partition(B, 1, 2, 1), Tensor<int>({4, 2, 2}, {
        11, 12, 12, 13,
        21, 22, 22, 23,
        31, 32, 32, 33,
        41, 42, 42, 43
    }));

    EXPECT_EQ(partition(B, {0, 1}, {2, 2}, {1, 1}), Tensor<int>({3, 2, 2, 2}, {
        11, 12, 21, 22,
        12, 13, 22, 23,
        21, 22, 31, 32,
        22, 23, 32, 33,
        31, 32, 41, 42,
        32, 33, 42, 43
    }));

    EXPECT_EQ(partition(B, {0, 1}, {2, 2}, {1, 2}), Tensor<int>({3, 1, 2, 2}, {
        11, 12, 21, 22,
        21, 22, 31, 32,
        31, 32, 41, 42
    }));

    EXPECT_EQ(partition(B, {0, 1}, {2, 2}, {2, 1}), Tensor<int>({2, 2, 2, 2}, {
        11, 12, 21, 22,
        12, 13, 22, 23,
        31, 32, 41, 42,
        32, 33, 42, 43
    }));

    EXPECT_EQ(partition(B, {0, 1}, {2, 2}, {2, 2}), Tensor<int>({2, 1, 2, 2}, {
        11, 12, 21, 22,
        31, 32, 41, 42
    }));
}

TEST(ViewTest, Parition3D) {
    auto A = Tensor<int>({2, 3, 4}, {
        111, 112, 113, 114,
        121, 122, 123, 124,
        131, 132, 133, 134,

        211, 212, 213, 214,
        221, 222, 223, 224,
        231, 232, 233, 234
    });

    EXPECT_EQ(partition(A, {1, 2}, {2, 2}, {1, 1}), Tensor<int>({2, 2, 3, 2, 2}, {
        111, 112, 121, 122,
        112, 113, 122, 123,
        113, 114, 123, 124,
        121, 122, 131, 132,
        122, 123, 132, 133,
        123, 124, 133, 134,
        211, 212, 221, 222,
        212, 213, 222, 223,
        213, 214, 223, 224,
        221, 222, 231, 232,
        222, 223, 232, 233,
        223, 224, 233, 234
    }));

    EXPECT_EQ(partition(A, {0, 2}, {2, 2}, {1, 1}), Tensor<int>({1, 3, 3, 2, 2}, {
        111, 112, 211, 212,
        112, 113, 212, 213,
        113, 114, 213, 214,
        121, 122, 221, 222,
        122, 123, 222, 223,
        123, 124, 223, 224,
        131, 132, 231, 232,
        132, 133, 232, 233,
        133, 134, 233, 234
    }));

    EXPECT_EQ(partition(A, {0, 1}, {2, 2}, {1, 1}), Tensor<int>({1, 2, 4, 2, 2}, {
        111, 121, 211, 221,
        112, 122, 212, 222,
        113, 123, 213, 223,
        114, 124, 214, 224,
        121, 131, 221, 231,
        122, 132, 222, 232,
        123, 133, 223, 233,
        124, 134, 224, 234
    }));

    EXPECT_EQ(partition(A, {0, 1, 2}, {2, 2, 2}, {1, 1, 1}), Tensor<int>({1, 2, 3, 2, 2, 2}, {
        111, 112, 121, 122,
        211, 212, 221, 222,
        112, 113, 122, 123,
        212, 213, 222, 223,
        113, 114, 123, 124,
        213, 214, 223, 224,
        121, 122, 131, 132,
        221, 222, 231, 232,
        122, 123, 132, 133,
        222, 223, 232, 233,
        123, 124, 133, 134,
        223, 224, 233, 234
    }));
}

TEST(ViewTest, PartitionOnView) {
    auto A = Tensor<int>({4, 3}, {
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43
    });

    EXPECT_EQ(partition(A.transpose(), 0, 2, 1), Tensor<int>({2, 4, 2}, {
        11, 12, 21, 22,
        31, 32, 41, 42,
        12, 13, 22, 23,
        32, 33, 42, 43
    }));

    EXPECT_EQ(partition(A.transpose(), 1, 2, 1), Tensor<int>({3, 3, 2}, {
        11, 21, 21, 31, 31, 41,
        12, 22, 22, 32, 32, 42,
        13, 23, 23, 33, 33, 43
    }));

    EXPECT_EQ(partition(A.transpose(), {0, 1}, {2, 2}, {1, 1}), Tensor<int>({2, 3, 2, 2}, {
        11, 21, 12, 22,
        21, 31, 22, 32,
        31, 41, 32, 42,
        12, 22, 13, 23,
        22, 32, 23, 33,
        32, 42, 33, 43
    }));

    EXPECT_EQ(partition(A.transpose(), {0, 1}, {2, 2}, {1, 2}), Tensor<int>({2, 2, 2, 2}, {
        11, 21, 12, 22,
        31, 41, 32, 42,
        12, 22, 13, 23,
        32, 42, 33, 43
    }));

    EXPECT_EQ(partition(A.transpose(), {0, 1}, {2, 2}, {2, 1}), Tensor<int>({1, 3, 2, 2}, {
        11, 21, 12, 22,
        21, 31, 22, 32,
        31, 41, 32, 42
    }));

    EXPECT_EQ(partition(A.transpose(), {0, 1}, {2, 2}, {2, 2}), Tensor<int>({1, 2, 2, 2}, {
        11, 21, 12, 22,
        31, 41, 32, 42
    }));
}

TEST(ViewTest, PartitionAndReduceMeanEquivalentToAveragePool) {
    auto A = Tensor<float>({3, 3, 8, 10}).random(0, 1);
    auto B = reduce_mean(partition(A, {-2, -1}, {3, 3}, {1, 1}), {-2, -1});
    auto C = dnn::average_pooling(A, dnn::Filter2D(A.shape(), 3, 3), false);
    EXPECT_EQ(B, C);
}
