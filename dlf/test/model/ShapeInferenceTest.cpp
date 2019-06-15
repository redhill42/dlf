#include "model.h"
#include "model/operators.h"
#include "gtest/gtest.h"

using namespace dlf::model;

TEST(ShapeInference, Conv) {
    Graph g;

    Value* X = g.addInput();
    X->set_type(DataType::FLOAT);
    X->set_dims({1, 3, 224, 224});

    Value* W = g.addInput();
    W->set_type(DataType::FLOAT);
    W->set_dims({64, 3, 7, 7});

    Node* conv = g.createNode(kConv);
    conv->addInput(X);
    conv->addInput(W);
    Value* Y = conv->addOutput();

    conv->set_is(kdilations, {1, 1});
    conv->set_i(kgroup, 1);
    conv->set_is(kkernel_shape, {7, 7});
    conv->set_is(kpads, {3, 3, 3, 3});
    conv->set_is(kstrides, {2, 2});

    ShapeInference::Instance().infer(conv);

    EXPECT_EQ(Y->type(), DataType::FLOAT);
    EXPECT_EQ(Y->dims(), std::vector<size_t>({1, 64, 112, 112}));
}

TEST(ShapeInference, Gemm) {
    Graph g;

    Value* A = g.addInput();
    A->set_type(DataType::FLOAT);
    A->set_dims({3, 7});

    Value* B = g.addInput();
    B->set_type(DataType::FLOAT);
    B->set_dims({7, 4});

    Value* C = g.addInput();
    C->set_type(DataType::FLOAT);
    C->set_dims({3, 4});

    Node* gemm = g.createNode(kGemm);
    gemm->addInput(A);
    gemm->addInput(B);
    gemm->addInput(C);
    Value* Y = gemm->addOutput();

    ShapeInference::Instance().infer(gemm);

    EXPECT_EQ(Y->type(), DataType::FLOAT);
    EXPECT_EQ(Y->dims(), std::vector<size_t>({3, 4}));
}
