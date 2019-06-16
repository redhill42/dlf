#include "model.h"
#include "model/operators.h"
#include "gtest/gtest.h"

using namespace dlf::model;

static void shapeBroadcastTest(const std::vector<Dims>& shapes, const Dims& expected) {
    Graph g;

    auto node = g.create<Sum>();
    for (size_t i = 0; i < shapes.size(); i++)
        node->addInput(g.addInput("input"+std::to_string(i), DataType::FLOAT, shapes[i]));
    node->addOutput("output");

    EXPECT_NO_THROW(ShapeInference::Instance().infer(node));
    EXPECT_EQ(node->output()->dims(), expected);
}

static void invalidShapeBroadcast(const std::vector<Dims>& shapes) {
    Graph g;

    auto node = g.create<Sum>();
    for (size_t i = 0; i < shapes.size(); i++)
        node->addInput(g.addInput("input"+std::to_string(i), DataType::FLOAT, shapes[i]));
    node->addOutput("output");

    EXPECT_ANY_THROW(ShapeInference::Instance().infer(node));
}

TEST(ShapeInference, shapeBroadCast) {
    shapeBroadcastTest({{5, 4}, {1}}, {5, 4});
    shapeBroadcastTest({{5, 4}, {4}}, {5, 4});
    shapeBroadcastTest({{15, 3, 5}, {15, 1, 5}}, {15, 3, 5});
    shapeBroadcastTest({{15, 3, 5}, {3, 5}}, {15, 3, 5});
    shapeBroadcastTest({{15, 3, 5}, {3, 1}}, {15, 3, 5});
    shapeBroadcastTest({{8, 1, 6, 1}, {7, 1, 5}}, {8, 7, 6, 5});
}

TEST(ShapeInference, invalidShapeBroadcast) {
    invalidShapeBroadcast({{3}, {4}});
    invalidShapeBroadcast({{2, 1}, {8, 4, 3}});
}

TEST(ShapeInference, Conv) {
    Graph g;

    auto node = g.create<Conv>();
    node->addInput(g.addInput("X", DataType::FLOAT, {1, 3, 224, 224}));
    node->addInput(g.addInput("W", DataType::FLOAT, {64, 3, 7, 7}));
    node->addOutput("Y");

    node->set_dilations({1, 1});
    node->set_group(1);
    node->set_kernel_shape({7, 7});
    node->set_pads({3, 3, 3, 3});
    node->set_strides({2, 2});

    ShapeInference::Instance().infer(node);

    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), Dims({1, 64, 112, 112}));
}

static void max_pool_test(Dims input_shape, Dims output_shape,
                          std::vector<int64_t> kernel_shape,
                          std::vector<int64_t> strides = {},
                          std::vector<int64_t> pads = {},
                          std::vector<int64_t> dilations = {},
                          const std::string& auto_pad = "VALID",
                          bool ceil_mode = false)
{
    Graph g;

    auto node = g.create<MaxPool>();
    node->addInput(g.addInput("input", DataType::FLOAT, input_shape));
    node->set_kernel_shape(kernel_shape);
    if (!strides.empty())
        node->set_strides(strides);
    if (!pads.empty())
        node->set_pads(pads);
    if (!dilations.empty())
        node->set_dilations(dilations);
    node->set_auto_pad(auto_pad);
    node->set_ceil_mode(ceil_mode);
    node->addOutput("output");

    EXPECT_NO_THROW(ShapeInference::Instance().infer(node));
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), output_shape);

}

TEST(ShapeInference, MaxPool) {
    // 1d default
    max_pool_test({1, 3, 32}, {1, 3, 31}, {2});
    // 2d default
    max_pool_test({1, 3, 32, 32}, {1, 3, 31, 31}, {2, 2});
    // 2d strides
    max_pool_test({1, 3, 32, 32}, {1, 3, 10, 10}, {5, 5}, {3, 3});
    // 2d pads
    max_pool_test({1, 3, 28, 28}, {1, 3, 30, 30}, {3, 3}, {}, {2, 2, 2, 2});
    // 2d dilations
    max_pool_test({1, 1, 4, 4}, {1, 1, 2, 2}, {2, 2}, {}, {}, {2, 2});
    // 2d same upper
    max_pool_test({1, 3, 32, 32}, {1, 3, 32, 32}, {2, 2}, {}, {}, {}, "SAME_UPPER");
    // 2d same lower
    max_pool_test({1, 3, 32, 32}, {1, 3, 32, 32}, {2, 2}, {}, {}, {}, "SAME_LOWER");
    // 2d ceil
    max_pool_test({1, 1, 4, 4}, {1, 1, 2, 2}, {3, 3}, {2, 2}, {}, {}, "VALID", true);
    // 2d precomputed strides
    max_pool_test({1, 1, 5, 5}, {1, 1, 2, 2}, {2, 2}, {2, 2});
    // 2d precomputed pads
    max_pool_test({1, 1, 5, 5}, {1, 1, 5, 5}, {5, 5}, {}, {2, 2, 2, 2});
    // 2d precomputed same upper
    max_pool_test({1, 1, 5, 5}, {1, 1, 3, 3}, {3, 3}, {2, 2}, {}, {}, "SAME_UPPER");
    // 3d default
    max_pool_test({1, 3, 32, 32, 32}, {1, 3, 31, 31, 31}, {2, 2, 2});
}

TEST(ShapeInference, MaxUnpool) {
    Graph g;

    auto node = g.create<MaxUnpool>();
    node->addInput(g.addInput("X", DataType::FLOAT, {1, 1, 2, 2}));
    node->addInput(g.addInput("I", DataType::INT64, {1, 1, 2, 2}));
    node->set_kernel_shape({2, 2});
    node->set_strides({2, 2});
    node->addOutput("Y");

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), Dims({1, 1, 4, 4}));
}

TEST(ShapeInference, Gemm) {
    Graph g;

    auto node = g.create<Gemm>();
    node->addInput(g.addInput("A", DataType::FLOAT, {3, 7}));
    node->addInput(g.addInput("B", DataType::FLOAT, {7, 4}));
    node->addInput(g.addInput("C", DataType::FLOAT, {3, 4}));
    node->addOutput("Y");

    ShapeInference::Instance().infer(node);

    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), Dims({3, 4}));
}

template <typename T>
static Value* addInitializer(Node* n, const std::string& name, std::vector<T> data) {
    if (data.size() > 0) {
        auto v = n->owningGraph()->addInput(name, DataTypeTrait<T>, {data.size()});
        auto t = dlf::Tensor<T>({data.size()}, data.begin(), data.end());
        v->set_initializer(TensorData(t));
        n->addInput(v);
        return v;
    } else {
        return n->addInput(n->owningGraph()->undefinedValue());
    }
}

static void slice_test(Dims input_shape,
                       std::vector<int32_t> starts,
                       std::vector<int32_t> ends,
                       std::vector<int32_t> axes,
                       std::vector<int32_t> steps,
                       Dims output_shape)
{
    Graph g;

    auto node = g.create<Slice>();
    node->addInput(g.addInput("data", DataType::FLOAT, input_shape));
    addInitializer(node, "starts", starts);
    addInitializer(node, "ends", ends);
    addInitializer(node, "axes", axes);
    addInitializer(node, "steps", steps);
    node->addOutput("output");

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), output_shape);
}

TEST(ShapeInference, Slice) {
    slice_test({20, 10, 5}, {0, 0}, {3, 10}, {0, 1}, {1, 1}, {3, 10, 5});
    slice_test({20, 10, 5}, {0, 0, 3}, {20, 10, 4}, {}, {}, {20, 10, 1});
    slice_test({20, 10, 5}, {0, 0, 3}, {20, 10, 4}, {0, 1, 2}, {}, {20, 10, 1});
    slice_test({20, 10, 5}, {1}, {1000}, {1}, {1}, {20, 9, 5});
    slice_test({20, 10, 5}, {0}, {-1}, {1}, {1}, {20, 9, 5});
    slice_test({20, 10, 5}, {20, 10, 4}, {0, 0, 1}, {0, 1, 2}, {-1, -3, -2}, {19, 3, 2});
    slice_test({20, 10, 5}, {1000}, {1000}, {1}, {1}, {20, 0, 5});
}

TEST(ShapeInference, Split1D) {
    Graph g;

    auto node = g.create<Split>();
    node->addInput(g.addInput("input", DataType::FLOAT, {6}));
    node->set_axis(0);

    Value* o1 = node->addOutput("output_1");
    Value* o2 = node->addOutput("output_2");
    Value* o3 = node->addOutput("output_3");

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(o1->type(), DataType::FLOAT);
    EXPECT_EQ(o1->dims(), Dims({2}));
    EXPECT_EQ(o2->type(), DataType::FLOAT);
    EXPECT_EQ(o2->dims(), Dims({2}));
    EXPECT_EQ(o3->type(), DataType::FLOAT);
    EXPECT_EQ(o3->dims(), Dims({2}));
}

TEST(ShapeInference, Split2D) {
    Graph g;

    auto node = g.create<Split>();
    node->addInput(g.addInput("input", DataType::FLOAT, {2, 6}));
    node->set_axis(1);

    Value* o1 = node->addOutput("output_1");
    Value* o2 = node->addOutput("output_2");

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(o1->type(), DataType::FLOAT);
    EXPECT_EQ(o1->dims(), Dims({2, 3}));
    EXPECT_EQ(o2->type(), DataType::FLOAT);
    EXPECT_EQ(o2->dims(), Dims({2, 3}));
}

TEST(ShapeInference, SplitExplicit) {
    Graph g;

    auto node = g.create<Split>();
    node->addInput(g.addInput("input", DataType::FLOAT, {2, 6}));
    node->set_axis(1);
    node->set_split({2, 4});

    Value* o1 = node->addOutput("output_1");
    Value* o2 = node->addOutput("output_2");

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(o1->type(), DataType::FLOAT);
    EXPECT_EQ(o1->dims(), Dims({2, 2}));
    EXPECT_EQ(o2->type(), DataType::FLOAT);
    EXPECT_EQ(o2->dims(), Dims({2, 4}));
}

TEST(ShapeInference, SplitNonEqualSize) {
    Graph g;

    auto node = g.create<Split>();
    node->addInput(g.addInput("input", DataType::FLOAT, {2, 8}));
    node->set_axis(1);

    Value* o1 = node->addOutput("output_1");
    Value* o2 = node->addOutput("output_2");
    Value* o3 = node->addOutput("output_3");

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(o1->type(), DataType::FLOAT);
    EXPECT_EQ(o1->dims(), Dims({2, 3}));
    EXPECT_EQ(o2->type(), DataType::FLOAT);
    EXPECT_EQ(o2->dims(), Dims({2, 3}));
    EXPECT_EQ(o3->type(), DataType::FLOAT);
    EXPECT_EQ(o3->dims(), Dims({2, 2}));
}

static void concat_test(const std::vector<Dims>& shapes, int axis, const Dims& expected) {
    Graph g;

    auto node = g.create<Concat>();
    node->set_axis(axis);
    for (size_t i = 0; i < shapes.size(); i++)
        node->addInput(g.addInput("input_"+std::to_string(i), DataType::FLOAT, shapes[i]));
    node->addOutput("output");

    if (expected.empty()) {
        EXPECT_ANY_THROW(ShapeInference::Instance().infer(node));
    } else {
        EXPECT_NO_THROW(ShapeInference::Instance().infer(node));
        EXPECT_EQ(node->output()->type(), DataType::FLOAT);
        EXPECT_EQ(node->output()->dims(), expected);
    }
}

TEST(ShapeInference, Concat) {
    concat_test({{3, 2, 4}, {3, 5, 4}}, 1, {3, 7, 4});
    concat_test({{3, 2, 4}, {3, 5, 5}}, 1, {});
}

TEST(ShapeInference, TransposePermutation) {
    Graph g;

    auto node = g.create<Transpose>();
    node->addInput(g.addInput("input", DataType::FLOAT, {2, 3, 4}));
    node->set_perm({2, 0, 1});
    node->addOutput("output");

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), Dims({4, 2, 3}));
}

TEST(ShapeInference, TransposeDefault) {
    Graph g;

    auto node = g.create<Transpose>();
    node->addInput(g.addInput("input", DataType::FLOAT, {2, 3, 4}));
    node->addOutput("output");

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), Dims({4, 3, 2}));
}

TEST(ShapeInference, TransposeInvalidPerm) {
    Graph g;

    auto node = g.create<Transpose>();
    node->addInput(g.addInput("input", DataType::FLOAT, {2, 3, 4}));
    node->addOutput("output");

    node->set_perm({3, 0, 1});
    EXPECT_ANY_THROW(ShapeInference::Instance().infer(node));

    node->set_perm({1, 1, 0});
    EXPECT_ANY_THROW(ShapeInference::Instance().infer(node));

    node->set_perm({0, 1});
    EXPECT_ANY_THROW(ShapeInference::Instance().infer(node));
}

TEST(ShapeInference, SqueezeAll) {
    Graph g;

    auto node = g.create<Squeeze>();
    node->addInput(g.addInput("input", DataType::FLOAT, {1, 3, 1, 5}));
    node->addOutput("output");

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), Dims({3, 5}));
}

TEST(ShapeInference, SqueezeSelectedAxis) {
    Graph g;

    auto node = g.create<Squeeze>();
    node->addInput(g.addInput("input", DataType::FLOAT, {1, 3, 1, 5, 1, 6}));
    node->addOutput("output");
    node->set_axes({0, 2});

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), Dims({3, 5, 1, 6}));
}

TEST(ShapeInference, SqueezeNegativeAxis) {
    Graph g;

    auto node = g.create<Squeeze>();
    node->addInput(g.addInput("input", DataType::FLOAT, {1, 3, 1, 5, 1, 6}));
    node->addOutput("output");
    node->set_axes({-2});

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), Dims({1, 3, 1, 5, 6}));
}

TEST(ShapeInference, SequenceNoneZeroSizeAxisMustFail) {
    Graph g;

    auto node = g.create<Squeeze>();
    node->addInput(g.addInput("input", DataType::FLOAT, {1, 3, 1, 5}));
    node->addOutput("output");
    node->set_axes({0, 1, 2});
    EXPECT_ANY_THROW(ShapeInference::Instance().infer(node));
}

TEST(ShapeInference, Unsqueeze) {
    Graph g;

    auto node = g.create<Unsqueeze>();
    node->addInput(g.addInput("input", DataType::FLOAT, {3, 4, 5}));
    node->addOutput("output");
    node->set_axes({0, 2, 5});

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), Dims({1, 3, 1, 4, 5, 1}));
}

TEST(ShapeInference, UnsqueezeWithNegativeAxis) {
    Graph g;

    auto node = g.create<Unsqueeze>();
    node->addInput(g.addInput("input", DataType::FLOAT, {3, 4, 5}));
    node->addOutput("output");
    node->set_axes({-1, 0, 2});

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), Dims({1, 3, 1, 4, 5, 1}));
}

TEST(ShapeInference, UnsqueezeWithInvalidAxis) {
    Graph g;

    auto node = g.create<Unsqueeze>();
    node->addInput(g.addInput("input", DataType::FLOAT, {3, 4, 5}));
    node->addOutput("output");

    node->set_axes({4});
    EXPECT_ANY_THROW(ShapeInference::Instance().infer(node));

    node->set_axes({0, 0});
    EXPECT_ANY_THROW(ShapeInference::Instance().infer(node));
}

TEST(ShapeInference, Pad) {
    Graph g;

    auto node = g.create<Pad>();
    node->addInput(g.addInput("input", DataType::FLOAT, {1, 3, 4, 5}));
    node->addOutput("output");
    node->set_pads({0, 0, 1, 3, 0, 0, 2, 4});
    node->set_mode("constant");
    node->set_value(1.2f);

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), Dims({1, 3, 7, 12}));
}

TEST(ShapeInference, PadWithNegativeValue) {
    Graph g;

    auto node = g.create<Pad>();
    node->addInput(g.addInput("input", DataType::FLOAT, {1, 3, 7, 12}));
    node->addOutput("output");
    node->set_pads({0, 0, -1, -3, 0, 0, -2, -4});

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), Dims({1, 3, 4, 5}));
}
