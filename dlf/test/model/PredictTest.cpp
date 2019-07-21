#include <fstream>
#include "predict.h"
#include "gtest/gtest.h"
#include "../test_utility.h"

using namespace dlf;
using namespace dlf::model;
using namespace dlf::predict;

template <typename Context> struct PredictTest : public testing::Test {};
using PredictTestTypes = testing::Types<CPU, GPU>;
TYPED_TEST_CASE(PredictTest, PredictTestTypes);

TYPED_TEST(PredictTest, Simple) {
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

    auto o4 = g.append<Concat>();
    o4->set_axis(-1);
    o4->addInput(y->output());
    o4->addInput(o1->output());
    g.addOutput(o4->addOutput("concat"));

    auto o5 = g.append<Split>();
    o5->set_axis(-1);
    o5->addInput(o4->output());
    g.addOutput(o5->addOutput("split1"));
    g.addOutput(o5->addOutput("split2"));
    g.addOutput(o5->addOutput("split3"));

    auto o6 = g.append<Transpose>();
    o6->addInput(y->output());
    g.addOutput(o6->addOutput("transpose"));

    Predictor<Context, float> predictor(g);
    predictor.set(0, Tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6}));
    predictor.set(1, Tensor<float>::scalar(3));
    predictor.set(2, Tensor<float>::scalar(2));
    predictor.predict();

    EXPECT_EQ(predictor.get(0), Tensor<float>({2, 3}, {8, 10, 12, 14, 16, 18}));
    EXPECT_EQ(predictor.get(1), Tensor<float>({2, 3}, {10, 10, 12, 14, 15, 15}));
    EXPECT_EQ(predictor.get(2), Tensor<float>({3, 2}, {8, 10, 12, 14, 16, 18}));
    EXPECT_EQ(predictor.get(3), Tensor<float>({1, 6}, {8, 10, 12, 14, 16, 18}));
    EXPECT_EQ(predictor.get(4), Tensor<float>({2, 6}, {8, 10, 12, 10, 10, 12,
                                                       14, 16, 18, 14, 15, 15}));
    EXPECT_EQ(predictor.get(5), Tensor<float>({2, 2}, {8, 10, 14, 16}));
    EXPECT_EQ(predictor.get(6), Tensor<float>({2, 2}, {12, 10, 18, 14}));
    EXPECT_EQ(predictor.get(7), Tensor<float>({2, 2}, {10, 12, 15, 15}));
    EXPECT_EQ(predictor.get(8), Tensor<float>({3, 2}, {8, 14, 10, 16, 12, 18}));
}

TYPED_TEST(PredictTest, Gemm) {
    using Context = TypeParam;
    Graph g;

    auto x = g.append<Gemm>();
    x->addInput(g.addInput("A", DataType::FLOAT, {2, 2}));
    x->addInput(g.addInput("B", DataType::FLOAT, {2, 2}));
    x->addInput(g.addInput("C", DataType::FLOAT, {2}));
    x->set_alpha(1.0f);
    x->set_beta(1.0f);
    g.addOutput(x->addOutput("Y"));

    Predictor<Context, float> predictor(g);
    predictor.set(0, Tensor<float>({2, 2}, {1, 2, 3, 4}));
    predictor.set(1, Tensor<float>({2, 2}, {5, 6, 7, 8}));
    predictor.set(2, Tensor<float>({2}, {9, 10}));
    predictor.predict();
    EXPECT_EQ(predictor.get(0), Tensor<float>({2,2}, {28, 32, 52, 60}));
}

TYPED_TEST(PredictTest, Conv) {
    using Context = TypeParam;
    Graph g;

    auto x = g.append<Conv>();
    x->addInput(g.addInput("X", DataType::FLOAT, {1, 1, 5, 5}));
    x->addInput(g.addInput("W", DataType::FLOAT, {1, 1, 3, 3}));
    x->addInput(g.addInitializer(TensorData("B", DataType::FLOAT, {1}, {1})));
    x->set_pads({1, 1, 1, 1});
    g.addOutput(x->addOutput("Y"));

    auto X = Tensor<float>::range({1, 1, 5, 5}, 0);
    auto W = Tensor<float>({1, 1, 3, 3});
    std::fill(W.begin(), W.end(), 1);

    Predictor<Context, float> predictor(g);
    predictor.set(0, X);
    predictor.set(1, W);
    predictor.predict();

    EXPECT_EQ(predictor.get(0), Tensor<float>({1, 1, 5, 5}, {
        13, 22, 28, 34, 25,
        34, 55, 64, 73, 52,
        64, 100, 109, 118, 82,
        94, 145, 154, 163, 112,
        73, 112, 118, 124, 85
    }));
}

TYPED_PERFORMANCE_TEST(PredictTest, Performance) {
    std::fstream fs("data/resnet18v1.onnx", std::ios::in | std::ios::binary);
    auto g = import_model(fs);
    fs.close();

    Predictor<TypeParam, float> pred(std::move(g));

    auto input = Tensor<float>::random({1, 3, 224, 224}, 0, 1);
    pred.set(0, input);
    pred.predict(); // warm up
    pred.get(0);

    std::string name = std::is_same<TypeParam, CPU>::value ? "CPU" : "GPU";
    timing("Predict " + name, 100, [&]() {
        pred.set(0, input);
        pred.predict();
        pred.get(0);
    });
}
