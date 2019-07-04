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
