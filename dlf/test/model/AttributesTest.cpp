#include "model.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <iterator>

using namespace dlf::model;

class TestNode: public Attributes<TestNode> {};

TEST(Attributes, General) {
    auto n = std::make_unique<TestNode>();

    n->set_i(kOffset, 3)->set_f(kvalue, 3.5);
    n->set_is(kdim, {1, 2, 3});

    EXPECT_TRUE(n->hasAttribute(kOffset));
    EXPECT_TRUE(n->hasAttribute(kvalue));
    EXPECT_TRUE(n->hasAttribute(kdim));

    EXPECT_EQ(n->kindOf(kOffset), AttributeKind::INT);
    EXPECT_EQ(n->kindOf(kvalue), AttributeKind::FLOAT);
    EXPECT_EQ(n->kindOf(kdim), AttributeKind::INTS);

    EXPECT_EQ(n->get_i(kOffset), 3);
    EXPECT_EQ(n->get_f(kvalue), 3.5);
    EXPECT_EQ(n->get_is(kdim), std::vector<int64_t>({1, 2, 3}));

    EXPECT_THAT(n->attributeNames(), testing::ElementsAre(kOffset, kvalue, kdim));
    n->removeAttribute(kdim);
    EXPECT_THAT(n->attributeNames(), testing::ElementsAre(kOffset, kvalue));
    EXPECT_FALSE(n->hasAttribute(kdim));
}

TEST(Attributes, BadVariantAccess) {
    auto n = std::make_unique<TestNode>();

    n->set_f(kvalue, 3.5);
    EXPECT_EQ(n->get_f(kvalue), 3.5);
    n->set_f(kvalue, 4.5);
    EXPECT_EQ(n->get_f(kvalue), 4.5);
    n->set_i(kvalue, 5);
    EXPECT_EQ(n->get_i(kvalue), 5);

    EXPECT_THROW(n->get_is(kvalue), bad_variant_access);
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
}
