#include "eval.h"
#include "gtest/gtest.h"

using namespace dlf;
using namespace dlf::model;
using namespace dlf::eval;

TEST(Evaluate, CPU) {
    Graph g;

    auto x = g.append<Add>();
    x->addInput(g.addInput("A", DataType::FLOAT, {2, 3}));
    x->addInput(g.addInput("B", DataType::FLOAT, {1}));
    x->addOutput("T");

    auto y = g.append<Mul>();
    y->addInput(x->output());
    y->addInput(g.addInput("C", DataType::FLOAT, {1}));
    g.addOutput(y->addOutput("Y"));

    Evaluator<Tensor<float >> eval;
    eval.load(g);

    eval.input(0) = Tensor<float >({2, 3}, {1, 2, 3, 4, 5, 6});
    eval.input(1) = scalar<float>(3);
    eval.input(2) = scalar<float>(2);
    eval.evaluate();

    EXPECT_EQ(eval.output(0), Tensor<float >({2, 3}, {8, 10, 12, 14, 16, 18}));
}

TEST(Evaluate, GPU) {
    Graph g;

    auto x = g.append<Add>();
    x->addInput(g.addInput("A", DataType::FLOAT, {2, 3}));
    x->addInput(g.addInput("B", DataType::FLOAT, {1}));
    x->addOutput("T");

    auto y = g.append<Mul>();
    y->addInput(x->output());
    y->addInput(g.addInput("C", DataType::FLOAT, {1}));
    g.addOutput(y->addOutput("Y"));

    Evaluator<DevTensor<float>> eval;
    eval.load(g);

    eval.input(0).write(Tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6}));
    eval.input(1).write(scalar<float>(3));
    eval.input(2).write(scalar<float>(2));
    eval.evaluate();

    EXPECT_EQ(eval.output(0).read(), Tensor<float>({2, 3}, {8, 10, 12, 14, 16, 18}));
}

TEST(Evaluate, GemmCPU) {
    Graph g;

    auto x = g.append<Gemm>();
    x->addInput(g.addInput("A", DataType::FLOAT, {2, 2}));
    x->addInput(g.addInput("B", DataType::FLOAT, {2, 2}));
    x->addInput(g.addInput("C", DataType::FLOAT, {2, 2}));
    x->set_alpha(1.0f);
    x->set_beta(1.0f);
    g.addOutput(x->addOutput("Y"));

    Evaluator<Tensor<float>> eval;
    eval.load(g);

    eval.input(0) = Tensor<float>({2, 2}, {1, 2, 3, 4});
    eval.input(1) = Tensor<float>({2, 2}, {5, 6, 7, 8});
    eval.input(2) = Tensor<float>({2}, {9, 10});
    eval.evaluate();
    EXPECT_EQ(eval.output(0), Tensor<float>({2,2}, {28, 32, 52, 60}));
}

TEST(Evaluate, GemmGPU) {
    Graph g;

    auto x = g.append<Gemm>();
    x->addInput(g.addInput("A", DataType::FLOAT, {2, 2}));
    x->addInput(g.addInput("B", DataType::FLOAT, {2, 2}));
    x->addInput(g.addInput("C", DataType::FLOAT, {2, 2}));
    x->set_alpha(1.0f);
    x->set_beta(1.0f);
    g.addOutput(x->addOutput("Y"));

    Evaluator<DevTensor<float>> eval;
    eval.load(g);

    eval.input(0) = Tensor<float>({2, 2}, {1, 2, 3, 4});
    eval.input(1) = Tensor<float>({2, 2}, {5, 6, 7, 8});
    eval.input(2) = Tensor<float>({2, 2}, {9, 10, 11, 12});
    eval.evaluate();
    EXPECT_EQ(eval.output(0).read(), Tensor<float>({2,2}, {28, 32, 54, 62}));
}
