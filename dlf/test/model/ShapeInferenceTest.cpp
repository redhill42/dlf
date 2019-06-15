#include "model.h"
#include "model/operators.h"
#include "gtest/gtest.h"

using namespace dlf::model;

TEST(ShapeInference, Conv) {
    Graph g;

    Node* conv = g.createNode(kConv);
    conv->addInput(g.addInput("X", DataType::FLOAT, {1, 3, 224, 224}));
    conv->addInput(g.addInput("W", DataType::FLOAT, {64, 3, 7, 7}));
    Value* Y = conv->addOutput("Y");

    conv->set_is(kdilations, {1, 1});
    conv->set_i(kgroup, 1);
    conv->set_is(kkernel_shape, {7, 7});
    conv->set_is(kpads, {3, 3, 3, 3});
    conv->set_is(kstrides, {2, 2});

    ShapeInference::Instance().infer(conv);

    EXPECT_EQ(Y->type(), DataType::FLOAT);
    EXPECT_EQ(Y->dims(), Dims({1, 64, 112, 112}));
}

TEST(ShapeInference, Gemm) {
    Graph g;

    Node* gemm = g.createNode(kGemm);
    gemm->addInput(g.addInput("A", DataType::FLOAT, {3, 7}));
    gemm->addInput(g.addInput("B", DataType::FLOAT, {7, 4}));
    gemm->addInput(g.addInput("C", DataType::FLOAT, {3, 4}));
    Value* Y = gemm->addOutput("Y");

    ShapeInference::Instance().infer(gemm);

    EXPECT_EQ(Y->type(), DataType::FLOAT);
    EXPECT_EQ(Y->dims(), Dims({3, 4}));
}
