#include "gtest/gtest.h"
#include "tensor.h"

using namespace dlf;

TEST(ViewTest, TransformTensorToView) {
    auto X = Tensor<int>::range({2, 2}, 1);
    auto Y = Tensor<int>::range({4, 4}, 1);

    auto dev_X = dev(X);
    auto dev_Y = dev(Y);

    transformTo(X, Y.slice({{1,3}, {1,3}}).transpose(), xfn::negate<>());
    EXPECT_EQ(Y, Tensor<int>({4, 4}, {
         1,  2,  3,  4,
         5, -1, -3,  8,
         9, -2, -4, 12,
        13, 14, 15, 16
    }));

    transformTo(dev_X, dev_Y.slice({{1,3}, {1,3}}).transpose(), xfn::negate<>());
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(ViewTest, TransformViewToTensor) {
    auto X = Tensor<int>::range({4, 4}, 1);
    auto Y = Tensor<int>::range({2, 2}, 1);

    auto dev_X = dev(X);
    auto dev_Y = dev(Y);

    transformTo(X.slice({{1,3}, {1,3}}).transpose(), Y, xfn::negate<>());
    EXPECT_EQ(Y, Tensor<int>({2, 2}, {
        -6, -10,
        -7, -11
    }));

    transformTo(dev_X.slice({{1,3}, {1,3}}).transpose(), dev_Y, xfn::negate<>());
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(ViewTest, TransformViewToView) {
    auto X = Tensor<int>::range({4, 4}, 1);
    auto Y = Tensor<int>::range({4, 4}, 1);

    auto dev_X = dev(X);
    auto dev_Y = dev(Y);

    transformTo(X.slice({{1,3}, {1,3}}).transpose(), Y.slice({{0,2}, {0,2}}), xfn::negate<>());
    EXPECT_EQ(Y, Tensor<int>({4, 4}, {
        -6, -10,  3,  4,
        -7, -11,  7,  8,
         9,  10, 11, 12,
        13,  14, 15, 16
    }));

    transformTo(dev_X.slice({{1,3}, {1,3}}).transpose(), dev_Y.slice({{0,2}, {0,2}}), xfn::negate<>());
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(ViewTest, CalculateOnView) {
    auto X = Tensor<int>::range({4, 4}, 1);
    auto Y = X.slice({{0,2}, {0,2}}) +
             X.slice({{0,2}, {2,4}}) +
             X.slice({{2,4}, {0,2}}) +
             X.slice({{2,4}, {2,4}});
    EXPECT_EQ(Y, Tensor<int>({2, 2}, {
        24, 28,
        40, 44
    }));

    auto dev_X = dev(X);
    auto dev_Y = dev_X.slice({{0,2}, {0,2}}) +
                 dev_X.slice({{0,2}, {2,4}}) +
                 dev_X.slice({{2,4}, {0,2}}) +
                 dev_X.slice({{2,4}, {2,4}});
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(ViewTest, AggregateOnView) {
    auto X = Tensor<int>::range({4, 4}, 1);
    auto Y = sum(X.slice({{0,2}, {0,2}}),
                 X.slice({{0,2}, {2,4}}),
                 X.slice({{2,4}, {0,2}}),
                 X.slice({{2,4}, {2,4}}));
    EXPECT_EQ(Y, Tensor<int>({2, 2}, {
        24, 28,
        40, 44
    }));

    auto dev_X = dev(X);
    auto dev_Y = sum(dev_X.slice({{0,2}, {0,2}}),
                     dev_X.slice({{0,2}, {2,4}}),
                     dev_X.slice({{2,4}, {0,2}}),
                     dev_X.slice({{2,4}, {2,4}}));
    EXPECT_EQ(dev_Y.read(), Y);
}

TEST(ViewTest, UpdateView) {
    auto X = Tensor<int>::range({4, 4}, 1);
    auto Y = X.slice({{1,3}, {1,3}});
    auto dev_X = dev(X);
    auto dev_Y = dev_X.slice({{1,3}, {1,3}});

    Y *= 2;
    EXPECT_EQ(X, Tensor<int>({4, 4}, {
         1,  2,  3,  4,
         5, 12, 14,  8,
         9, 20, 22, 12,
        13, 14, 15, 16
    }));

    dev_Y *= 2;
    EXPECT_EQ(dev_X.read(), X);
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
