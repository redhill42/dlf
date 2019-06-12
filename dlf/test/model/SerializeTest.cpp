#include <fstream>

#include "model/serialize.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(Serialize, Parse) {
    using namespace dlf::model;

    std::fstream input("data/resnet18v1.onnx", std::ios::in | std::ios::binary);
    auto g = importModel<ModelFormat::ONNX>(input);
    auto it = g->nodes().begin();

    auto n = *it++;
    EXPECT_EQ(n->kind(), kConv);

    EXPECT_EQ(n->get_is(kdilations), std::vector<int64_t>({1, 1}));
    EXPECT_EQ(n->get_i(kgroup), 1);
    EXPECT_EQ(n->get_is(kkernel_shape), std::vector<int64_t>({7, 7}));
    EXPECT_EQ(n->get_is(kpads), std::vector<int64_t>({3, 3, 3, 3}));
    EXPECT_EQ(n->get_is(kstrides), std::vector<int64_t>({2, 2}));

    ASSERT_EQ(n->inputs().size(), 2);
    EXPECT_EQ(n->inputs()[0]->name(), "data");
    EXPECT_EQ(n->inputs()[0]->type(), DataType::FLOAT);
    EXPECT_EQ(n->inputs()[0]->dims(), std::vector<size_t>({1, 3, 224, 224}));
    EXPECT_EQ(n->inputs()[1]->type(), DataType::FLOAT);
    EXPECT_EQ(n->inputs()[1]->dims(), std::vector<size_t>({64, 3, 7, 7}));
    EXPECT_TRUE(n->inputs()[1]->has_initializer());

    EXPECT_EQ(n->outputs().size(), 1);
    EXPECT_EQ(n->output()->type(), DataType::UNDEFINED);
    EXPECT_EQ(n->output()->node(), n);

    auto t = *it++;
    EXPECT_EQ(t->kind(), kBatchNormalization);
    EXPECT_EQ(t->inputs()[0], n->output());
}
