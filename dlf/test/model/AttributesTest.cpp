#include <iterator>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "model.h"
#include "model/serialize.h"

using namespace dlf::model;

class TestNode: public Attributes<TestNode> {};

TEST(Attributes, General) {
    auto n = TestNode();

    n.set_i(kOffset, 3).set_f(kvalue, 3.5).set_is(kdim, {1, 2, 3});

    EXPECT_TRUE(n.hasAttribute(kOffset));
    EXPECT_TRUE(n.hasAttribute(kvalue));
    EXPECT_TRUE(n.hasAttribute(kdim));

    EXPECT_EQ(n.kindOf(kOffset), AttributeKind::INT);
    EXPECT_EQ(n.kindOf(kvalue), AttributeKind::FLOAT);
    EXPECT_EQ(n.kindOf(kdim), AttributeKind::INTS);

    EXPECT_EQ(n.get_i(kOffset), 3);
    EXPECT_EQ(n.get_f(kvalue), 3.5);
    EXPECT_EQ(n.get_is(kdim), std::vector<int64_t>({1, 2, 3}));

    EXPECT_THAT(n.attributeNames(), testing::ElementsAre(kOffset, kvalue, kdim));
    EXPECT_TRUE(n.removeAttribute(kdim));
    EXPECT_FALSE(n.removeAttribute(kdim)); // should success
    EXPECT_THAT(n.attributeNames(), testing::ElementsAre(kOffset, kvalue));
    EXPECT_FALSE(n.hasAttribute(kdim));
}

TEST(Attributes, BadVariantAccess) {
    auto n = TestNode();

    n.set_f(kvalue, 3.5);
    EXPECT_EQ(n.get_f(kvalue), 3.5);
    n.set_f(kvalue, 4.5);
    EXPECT_EQ(n.get_f(kvalue), 4.5);
    n.set_i(kvalue, 5);
    EXPECT_EQ(n.get_i(kvalue), 5);

    EXPECT_THROW(n.get_is(kvalue), bad_variant_access);
}

TEST(Attributes, DefaultValue) {
    auto n = TestNode();

    n.set_i(kvalue, 4);

    EXPECT_EQ(n.get_i(kvalue), 4);
    EXPECT_EQ(n.get_i(kvalue, 5), 4);
    EXPECT_EQ(n.get_f(kalpha, 3.5), 3.5);
    EXPECT_EQ(n.get_f("beta", 4.5), 4.5);

    auto v = n.get_is(kdim, {1,2,3});
    EXPECT_EQ(v, std::vector<int64_t>({1,2,3}));

    EXPECT_EQ(n.kindOf(kvalue), AttributeKind::INT);
    EXPECT_EQ(n.kindOf(kalpha), AttributeKind::UNDEFINED);
    EXPECT_EQ(n.kindOf("beta"), AttributeKind::UNDEFINED);
}

TEST(Attributes, Serialize) {
    auto np = onnx::NodeProto();

    auto ap = np.add_attribute();
    ap->set_name("alpha");
    ap->set_type(onnx::AttributeProto::FLOAT);
    ap->set_f(3.5f);

    ap = np.add_attribute();
    ap->set_name("beta");
    ap->set_type(onnx::AttributeProto::FLOAT);
    ap->set_f(7.2f);

    ap = np.add_attribute();
    ap->set_name("transA");
    ap->set_type(onnx::AttributeProto::INT);
    ap->set_i(1);

    ap = np.add_attribute();
    ap->set_name("transB");
    ap->set_type(onnx::AttributeProto::INT);
    ap->set_i(0);

    ap = np.add_attribute();
    ap->set_name("pads");
    ap->set_type(onnx::AttributeProto::INTS);
    ap->add_ints(1);
    ap->add_ints(2);
    ap->add_ints(3);

    ap = np.add_attribute();
    ap->set_name("messages");
    ap->set_type(onnx::AttributeProto::STRINGS);
    ap->add_strings("hello");
    ap->add_strings("world");

    auto n = TestNode();
    Serializer().load(np, n);

    EXPECT_EQ(n.get_f(kalpha), 3.5f);
    EXPECT_EQ(n.get_f(kbeta), 7.2f);
    EXPECT_EQ(n.get_i(ktransA), 1);
    EXPECT_EQ(n.get_i(ktransB), 0);
    EXPECT_EQ(n.get_is(kpads), std::vector<int64_t>({1, 2, 3}));
    EXPECT_THAT(n.get_ss("messages"), std::vector<std::string>({"hello", "world"}));
}
