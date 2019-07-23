#include <fstream>
#include <memory>

#include "model/serialize.h"
#include "model/operators.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace dlf::model;

class SerializeTest : public testing::Test {
protected:
    std::unique_ptr<Graph> g;

    SerializeTest() {
        g = import_model("data/resnet18v1.onnx");
    }
};

TEST_F(SerializeTest, Parse) {
    auto it = g->nodes().begin();

    auto n = *it++;
    ASSERT_EQ(n->kind(), kConv);
    auto x = n->cast<Conv>();

    EXPECT_EQ(x->dilations(), std::vector<int64_t>({1, 1}));
    EXPECT_EQ(x->group(), 1);
    EXPECT_EQ(x->kernel_shape(), std::vector<int64_t>({7, 7}));
    EXPECT_EQ(x->pads(), std::vector<int64_t>({3, 3, 3, 3}));
    EXPECT_EQ(x->strides(), std::vector<int64_t>({2, 2}));

    ASSERT_EQ(x->inputs().size(), 2);
    EXPECT_EQ(x->X()->name(), "data");
    EXPECT_EQ(x->X()->type(), DataType::FLOAT);
    EXPECT_EQ(x->X()->dims(), Dims({1, 3, 224, 224}));
    EXPECT_EQ(x->W()->type(), DataType::FLOAT);
    EXPECT_EQ(x->W()->dims(), Dims({64, 3, 7, 7}));
    EXPECT_TRUE(x->W()->has_initializer());

    EXPECT_EQ(x->outputs().size(), 1);
    EXPECT_EQ(x->Y()->node(), x);

    n = *it++;
    ASSERT_EQ(n->kind(), kBatchNormalization);
    auto y = n->cast<BatchNormalization>();
    EXPECT_EQ(y->X(), x->Y());
}

TEST_F(SerializeTest, GraphHierachy) {
    auto it = g->nodes().begin();
    auto n1 = *it++;
    auto n2 = *it++;
    auto n3 = *it++;
    auto n4 = *it++;

    // The Graph:
    //                                                  +-> Conv ... -+
    //  Conv -> BatchNormalization -> Relu -> MaxPool --|             |-> Add ...
    //                                                  +-------------+

    EXPECT_EQ(n1->kind(), kConv);
    EXPECT_EQ(n1->inputs().size(), 2);
    EXPECT_EQ(n1->input(0)->name(), "data");
    EXPECT_TRUE(n1->input(1)->has_initializer());
    EXPECT_EQ(n1->output()->uses().size(), 1);
    EXPECT_EQ(n1->output()->uses()[0], Use(n2, 0));

    EXPECT_EQ(n2->kind(), kBatchNormalization);
    EXPECT_EQ(n2->inputs().size(), 5);
    EXPECT_EQ(n2->input(0)->node(), n1);
    EXPECT_EQ(n2->output()->uses().size(), 1);
    EXPECT_EQ(n2->output()->uses()[0], Use(n3, 0));

    EXPECT_EQ(n3->kind(), kRelu);
    EXPECT_EQ(n3->input()->node(), n2);
    EXPECT_EQ(n3->output()->uses().size(), 1);
    EXPECT_EQ(n3->output()->uses()[0], Use(n4, 0));

    EXPECT_EQ(n4->kind(), kMaxPool);
    EXPECT_EQ(n4->input()->node(), n3);
    EXPECT_EQ(n4->output()->uses().size(), 2);
    EXPECT_EQ(n4->output()->uses()[0].user->kind(), kConv);
    EXPECT_EQ(n4->output()->uses()[1].user->kind(), kAdd);
}
