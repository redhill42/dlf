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
    x->addInput(g.addInput("B", DataType::FLOAT, {}));
    x->addOutput("X");

    auto y = g.append<Mul>();
    y->addInput(x->output());
    y->addInput(g.addInput("C", DataType::FLOAT, {}));
    g.addOutput(y->addOutput("Y"));

    auto o1 = g.append<Clip>()->min(10)->max(15);
    o1->addInput(y->output());
    g.addOutput(o1->addOutput("Z"));

    auto o2 = g.append<Reshape>();
    o2->addInput(y->output());
    o2->addInput(g.addInitializer(TensorData("shape", DataType::INT64, {2}, {3, 2})));
    g.addOutput(o2->addOutput("reshaped"));

    auto o3 = g.append<Flatten>()->axis(0);
    o3->addInput(y->output());
    g.addOutput(o3->addOutput("flatten"));

    auto o4 = g.append<Concat>()->axis(-1);
    o4->addInput(y->output());
    o4->addInput(o1->output());
    g.addOutput(o4->addOutput("concat"));

    auto o5 = g.append<Split>()->axis(-1);
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

    auto x = g.append<Gemm>()->alpha(1.f)->beta(1.f);
    x->addInput(g.addInput("A", DataType::FLOAT, {2, 2}));
    x->addInput(g.addInput("B", DataType::FLOAT, {2, 2}));
    x->addInput(g.addInput("C", DataType::FLOAT, {2}));
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

    auto x = g.append<Conv>()->pads({1, 1, 1, 1});
    x->addInput(g.addInput("X", DataType::FLOAT, {1, 1, 5, 5}));
    x->addInput(g.addInput("W", DataType::FLOAT, {1, 1, 3, 3}));
    x->addInput(g.addInitializer(TensorData("B", DataType::FLOAT, {1}, {1})));
    g.addOutput(x->addOutput("Y"));

    auto X = Tensor<float>({1, 1, 5, 5}).range(0);
    auto W = Tensor<float>({1, 1, 3, 3}, 1);

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

TYPED_TEST(PredictTest, Slice) {
    using Context = TypeParam;
    Graph g;

    auto x = g.append<Slice>();
    x->addInput(g.addInput("X", DataType::FLOAT, {10, 10, 5}));
    x->addInput(g.addInitializer(TensorData("starts", DataType::INT32, {2}, {2, 3})));
    x->addInput(g.addInitializer(TensorData("ends", DataType::INT32, {2}, {5, 7})));
    g.addOutput(x->addOutput("Y"));

    auto X = Tensor<float>({10, 10, 5}).range(0);

    Predictor<Context, float> predictor(g);
    predictor.set(0, X);
    predictor.predict();

    EXPECT_EQ(predictor.get(0), Tensor<float>({3, 4, 5}, {
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
    }));
}

TYPED_TEST(PredictTest, Where) {
    using Context = TypeParam;
    Graph g;

    auto x = g.addInput("X", DataType::FLOAT, {10});

    auto a = g.append<Less>();
    a->addInput(x);
    a->addInput(g.addInitializer(TensorData("c5", DataType::FLOAT, {1}, {5})));
    a->addOutput("cond");

    auto b = g.append<Mul>();
    b->addInput(x);
    b->addInput(g.addInitializer(TensorData("c10", DataType::FLOAT, {1}, {10})));
    b->addOutput("Y");

    // where(X < 5, X, X*10)
    auto c = g.append<Where>();
    c->addInput(a->output());
    c->addInput(x);
    c->addInput(b->output());
    g.addOutput(c->addOutput("Z"));

    Predictor<Context, float> predictor(g);
    predictor.set(0, Tensor<float>({10}).range(0));
    predictor.predict();

    EXPECT_EQ(predictor.get(0), Tensor<float>({10}, {
        0, 1, 2, 3, 4, 50, 60, 70, 80, 90
    }));
}

TYPED_TEST(PredictTest, Reduce) {
    using Context = TypeParam;
    Graph g;

    auto x = g.addInput("X", DataType::FLOAT, {3, 2, 2});

    auto a = g.append<ReduceSum>();
    a->addInput(x);
    g.addOutput(a->addOutput("A"));

    auto b = g.append<ReduceSum>()->axes({1});
    b->addInput(x);
    g.addOutput(b->addOutput("B"));

    auto c = g.append<ReduceSum>()->axes({1})->keepdims(false);
    c->addInput(x);
    g.addOutput(c->addOutput("C"));

    Predictor<Context, float> predictor(g);
    predictor.set(0, Tensor<float>({3, 2, 2}).range(1));
    predictor.predict();

    EXPECT_EQ(predictor.get(0), Tensor<float>({1,1,1}, {78}));
    EXPECT_EQ(predictor.get(1), Tensor<float>({3,1,2}, {
        4, 6, 12, 14, 20, 22
    }));
    EXPECT_EQ(predictor.get(2), Tensor<float>({3,2}, {
        4, 6, 12, 14, 20, 22
    }));
}

TYPED_TEST(PredictTest, Pad) {
    using Context = TypeParam;
    Graph g;

    auto x = g.addInput("X", DataType::FLOAT, {3, 3});

    auto a = g.append<Pad>()->pads({2, 2, 2, 2});
    a->addInput(x);
    g.addOutput(a->addOutput("A"));

    Predictor<Context, float> predictor(g);
    predictor.set(0, Tensor<float>({3, 3}).range(1));
    predictor.predict();

    EXPECT_EQ(predictor.get(0), Matrix<float>({
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 2, 3, 0, 0},
        {0, 0, 4, 5, 6, 0, 0},
        {0, 0, 7, 8, 9, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
    }));
}

TYPED_TEST(PredictTest, If) {
    using Context = TypeParam;
    Graph g;

    auto x = g.addInput("X", DataType::FLOAT, {5});

    auto then_branch = std::make_shared<Graph>();
    then_branch->set_name("then");
    auto a = then_branch->append<Add>();
    a->addInput(x);
    a->addInput(then_branch->addInitializer(TensorData("cXp2", Scalar<float>(2))));
    then_branch->addOutput(a->addOutput("Xp2"));

    auto else_branch = std::make_shared<Graph>();
    else_branch->set_name("else");
    auto b = else_branch->append<Mul>();
    b->addInput(x);
    b->addInput(else_branch->addInitializer(TensorData("cXx2", Scalar<float>(2))));
    else_branch->addOutput(b->addOutput("Xx2"));

    auto c = g.append<If>();
    c->addInput(g.addInput("cond", DataType::BOOL, {}));
    c->then_branch(then_branch);
    c->else_branch(else_branch);
    g.addOutput(c->addOutput("Y"));

    Predictor<Context, float> predictor(g);
    predictor.set(0, Vector<float>({1, 2, 3, 4, 5}));

    predictor.set(1, Scalar<bool>(true));
    predictor.predict();
    EXPECT_EQ(predictor.get(0), Vector<float>({3, 4, 5, 6, 7}));

    predictor.set(1, Scalar<bool>(false));
    predictor.predict();
    EXPECT_EQ(predictor.get(0), Vector<float>({2, 4, 6, 8, 10}));
}

TYPED_PERFORMANCE_TEST(PredictTest, Performance) {
    std::fstream fs("data/resnet18v1.onnx", std::ios::in | std::ios::binary);
    auto g = import_model(fs);
    fs.close();

    Predictor<TypeParam, float> pred(std::move(g));

    auto input = Tensor<float>({1, 3, 224, 224}).random(0, 1);
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
