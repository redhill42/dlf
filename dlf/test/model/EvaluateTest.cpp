#include "eval.h"
#include "gtest/gtest.h"

using namespace dlf;
using namespace dlf::model;
using namespace dlf::eval;

template <typename Context> struct EvaluateTest : public testing::Test {};
using EvaluateTestTypes = testing::Types<CPU, GPU>;
TYPED_TEST_CASE(EvaluateTest, EvaluateTestTypes);

TYPED_TEST(EvaluateTest, Simple) {
    using Context = TypeParam;
    Graph g;

    auto x = g.append<Add>();
    x->addInput(g.addInput("A", DataType::FLOAT, {2, 3}));
    x->addInput(g.addInput("B", DataType::FLOAT, {1}));
    x->addOutput("X");

    auto y = g.append<Mul>();
    y->addInput(x->output());
    y->addInput(g.addInput("C", DataType::FLOAT, {1}));
    g.addOutput(y->addOutput("Y"));

    auto o1 = g.append<Clip>();
    o1->set_min(10)->set_max(15);
    o1->addInput(y->output());
    g.addOutput(o1->addOutput("Z"));

    auto o2 = g.append<Reshape>();
    o2->addInput(y->output());
    o2->addInput(g.addInitializer(TensorData("shape", DataType::INT64, {2}, {3, 2})));
    g.addOutput(o2->addOutput("reshaped"));

    auto o3 = g.append<Flatten>();
    o3->set_axis(0);
    o3->addInput(y->output());
    g.addOutput(o3->addOutput("flatten"));

    Evaluator<Context, float> eval(g);
    eval.set(0, Tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6}));
    eval.set(1, scalar<float>(3));
    eval.set(2, scalar<float>(2));
    eval.evaluate();

    EXPECT_EQ(eval.get(0), Tensor<float>({2, 3}, {8, 10, 12, 14, 16, 18}));
    EXPECT_EQ(eval.get(1), Tensor<float>({2, 3}, {10, 10, 12, 14, 15, 15}));
    EXPECT_EQ(eval.get(2), Tensor<float>({3, 2}, {8, 10, 12, 14, 16, 18}));
    EXPECT_EQ(eval.get(3), Tensor<float>({1, 6}, {8, 10, 12, 14, 16, 18}));
}

TYPED_TEST(EvaluateTest, Gemm) {
    using Context = TypeParam;
    Graph g;

    auto x = g.append<Gemm>();
    x->addInput(g.addInput("A", DataType::FLOAT, {2, 2}));
    x->addInput(g.addInput("B", DataType::FLOAT, {2, 2}));
    x->addInput(g.addInput("C", DataType::FLOAT, {2}));
    x->set_alpha(1.0f);
    x->set_beta(1.0f);
    g.addOutput(x->addOutput("Y"));

    Evaluator<Context, float> eval(g);
    eval.set(0, Tensor<float>({2, 2}, {1, 2, 3, 4}));
    eval.set(1, Tensor<float>({2, 2}, {5, 6, 7, 8}));
    eval.set(2, Tensor<float>({2}, {9, 10}));
    eval.evaluate();
    EXPECT_EQ(eval.get(0), Tensor<float>({2,2}, {28, 32, 52, 60}));
}
