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

TEST(ShapeInference, TfIdfVectorizer_1D) {
    Graph g;

    auto n = g.create<TfIdfVectorizer>();
    n->addInput(g.addInput("input", DataType::FLOAT, {12}));
    n->set_min_gram_length(2);
    n->set_max_gram_length(2);
    n->set_max_skip_count(0);
    n->set_ngram_counts({0, 4});
    n->set_ngram_indexes({0, 1, 2, 3, 4, 5, 6});
    n->set_pool_int64s({2, 3, 5, 4, 5, 6, 7, 8, 6, 7});
    n->addOutput("output");

    ShapeInference::Instance().infer(n);
    EXPECT_EQ(n->output()->type(), DataType::FLOAT);
    EXPECT_EQ(n->output()->dims(), Dims({7}));
}

TEST(ShapeInference, TfIdfVectorizer_2D) {
    Graph g;

    auto n = g.create<TfIdfVectorizer>();
    n->addInput(g.addInput("input", DataType::FLOAT, {2, 6}));
    n->set_min_gram_length(2);
    n->set_max_gram_length(2);
    n->set_max_skip_count(0);
    n->set_ngram_counts({0, 4});
    n->set_ngram_indexes({0, 1, 2, 3, 4, 5, 6});
    n->set_pool_int64s({2, 3, 5, 4, 5, 6, 7, 8, 6, 7});
    n->addOutput("output");

    ShapeInference::Instance().infer(n);
    EXPECT_EQ(n->output()->type(), DataType::FLOAT);
    EXPECT_EQ(n->output()->dims(), Dims({2, 7}));
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

static void matmul_test(Dims A, Dims B, Dims C) {
    Graph g;

    auto node = g.create<MatMul>();
    node->addInput(g.addInput("A", DataType::FLOAT, A));
    node->addInput(g.addInput("B", DataType::FLOAT, B));
    node->addOutput("C");

    ShapeInference::Instance().infer(node);
    EXPECT_EQ(node->output()->type(), DataType::FLOAT);
    EXPECT_EQ(node->output()->dims(), C);
}

static void matmul_failed_test(Dims A, Dims B) {
    Graph g;

    auto node = g.create<MatMul>();
    node->addInput(g.addInput("A", DataType::FLOAT, A));
    node->addInput(g.addInput("B", DataType::FLOAT, B));
    node->addOutput("C");

    EXPECT_ANY_THROW(ShapeInference::Instance().infer(node));
}

TEST(ShapeInference, MatMul) {
    matmul_test({3, 4}, {4, 3}, {3, 3});
    matmul_test({3, 4}, {4}, {3});
    matmul_test({4}, {4, 3}, {3});
    matmul_test({2, 3, 4}, {2, 4, 3}, {2, 3, 3});
    matmul_test({1, 3, 4}, {2, 4, 3}, {2, 3, 3});
    matmul_test({2, 3, 4}, {1, 4, 3}, {2, 3, 3});
    matmul_test({2, 3, 4}, {4, 3}, {2, 3, 3});
    matmul_test({3, 4}, {2, 4, 3}, {2, 3, 3});
    matmul_test({2, 3, 4}, {4}, {2, 3});
    matmul_test({4}, {2, 4, 3}, {2, 3});

    matmul_failed_test({3, 4}, {5, 4});
    matmul_failed_test({3, 4}, {5});
    matmul_failed_test({4}, {3, 4});
    matmul_failed_test({2, 3, 4}, {3, 4, 3});
    matmul_failed_test({2, 3, 4}, {2, 5, 4});
    matmul_failed_test({2, 3, 4}, {5, 4});
    matmul_failed_test({3, 5}, {2, 4, 3});
    matmul_failed_test({2, 3, 4}, {5});
    matmul_failed_test({3}, {2, 4, 3});
}

TEST(ShapeInference, TopK) {
    Graph g;

    auto n = g.create<TopK>();
    n->addInput(g.addInput("input", DataType::FLOAT, {3, 4, 5}));
    n->addInput(g.addInitializer({"K", DataType::INT64, {1}, {3}}));
    n->set_axis(1);
    n->addOutput("output");
    n->addOutput("indices");

    ShapeInference::Instance().infer(n);
    EXPECT_EQ(n->output()->type(), DataType::FLOAT);
    EXPECT_EQ(n->output()->dims(), Dims({3, 3, 5}));
    EXPECT_EQ(n->indices()->type(), DataType::INT64);
    EXPECT_EQ(n->indices()->dims(), Dims({3, 3, 5}));
}

static void expand_test(Dims input_shape, Dims shape, Dims expected) {
    Graph g;

    auto shape_data = TensorData("shape", DataType::INT64, {shape.size()});
    for (auto d : shape) {
        shape_data.int64_data().push_back(d);
    }

    auto n = g.create<Expand>();
    n->addInput(g.addInput("input", DataType::FLOAT, input_shape));
    n->addInput(g.addInitializer(shape_data));
    n->addOutput("output");

    ShapeInference::Instance().infer(n);
    EXPECT_EQ(n->output()->type(), DataType::FLOAT);
    EXPECT_EQ(n->output()->dims(), expected);
}

TEST(ShapeInference, Expand) {
    expand_test({1}, {5, 4}, {5, 4});
    expand_test({4}, {5, 4}, {5, 4});
    expand_test({15, 1, 5}, {15, 3, 5}, {15, 3, 5});
    expand_test({3, 5}, {15, 3, 5}, {15, 3, 5});
    expand_test({3, 1}, {2, 1, 6}, {2, 3, 6});
    expand_test({7, 1, 5}, {8, 1, 6, 1}, {8, 7, 6, 5});
}

static void compress_test(Dims input_shape, int axis, std::vector<bool> condition, Dims expected) {
    Graph g;

    auto cond_data = TensorData("condition", DataType::BOOL, {condition.size()});
    for (auto b : condition) {
        cond_data.int32_data().push_back(b ? 1 : 0);
    }

    auto n = g.create<Compress>();
    if (axis >= 0)
        n->set_axis(axis);
    n->addInput(g.addInput("input", DataType::FLOAT, input_shape));
    n->addInput(g.addInitializer(cond_data));
    n->addOutput("output");

    ShapeInference::Instance().infer(n);
    EXPECT_EQ(n->output()->type(), DataType::FLOAT);
    EXPECT_EQ(n->output()->dims(), expected);
}

TEST(ShapeInference, Compress) {
    compress_test({2, 5}, 0, {false, true}, {1, 5});
    compress_test({2, 5}, 0, {true, true}, {2, 5});
    compress_test({2, 5}, 0, {true, true, true}, {2, 5});
    compress_test({2, 5}, 1, {false, true, true}, {2, 2});
    compress_test({2, 5}, -1, {false, true, true}, {2});
}

template <typename T>
static Value* addInitializer(Node* n, const std::string& name, std::vector<T> data) {
    if (data.size() > 0) {
        auto t = dlf::Tensor<T>({data.size()}, data.begin(), data.end());
        return n->addInput(n->owningGraph()->addInitializer(TensorData(name, t)));
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

TEST(ShapeInference, Tile) {
    Graph g;

    auto n = g.create<Tile>();
    n->addInput(g.addInput("input", DataType::FLOAT, {2, 3, 4}));
    n->addInput(g.addInitializer({"repeats", DataType::INT64, {3}, {1, 2, 3}}));
    n->addOutput("output");

    ShapeInference::Instance().infer(n);
    EXPECT_EQ(n->output()->type(), DataType::FLOAT);
    EXPECT_EQ(n->output()->dims(), Dims({2, 6, 12}));
}

TEST(ShapeInference, SpaceToDepth) {
    Graph g;

    auto n = g.create<SpaceToDepth>();
    n->addInput(g.addInput("input", DataType::FLOAT, {1, 3, 64, 64}));
    n->set_blocksize(4);
    n->addOutput("output");

    ShapeInference::Instance().infer(n);
    EXPECT_EQ(n->output()->type(), DataType::FLOAT);
    EXPECT_EQ(n->output()->dims(), Dims({1, 48, 16, 16}));
}

TEST(ShapeInference, DepthToSpace) {
    Graph g;

    auto n = g.create<DepthToSpace>();
    n->addInput(g.addInput("input", DataType::FLOAT, {1, 48, 16, 16}));
    n->set_blocksize(4);
    n->addOutput("output");

    ShapeInference::Instance().infer(n);
    EXPECT_EQ(n->output()->type(), DataType::FLOAT);
    EXPECT_EQ(n->output()->dims(), Dims({1, 3, 64, 64}));
}
