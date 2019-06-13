#include <iterator>
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "model.h"
#include "model/serialize.h"

using namespace dlf::model;

class TestNode: public Attributes<TestNode> {};

TEST(Attributes, General) {
    auto n = TestNode();

    n.set_i(kgroup, 3).set_f(kvalue, 3.5).set_is(kdim, {1, 2, 3});

    EXPECT_TRUE(n.hasAttribute(kgroup));
    EXPECT_TRUE(n.hasAttribute(kvalue));
    EXPECT_TRUE(n.hasAttribute(kdim));

    EXPECT_EQ(n.attributeKind(kgroup), AttributeKind::INT);
    EXPECT_EQ(n.attributeKind(kvalue), AttributeKind::FLOAT);
    EXPECT_EQ(n.attributeKind(kdim), AttributeKind::INTS);

    EXPECT_EQ(n.get_i(kgroup), 3);
    EXPECT_EQ(n.get_f(kvalue), 3.5);
    EXPECT_EQ(n.get_is(kdim), std::vector<int64_t>({1, 2, 3}));

    EXPECT_THAT(n.attributeNames(), testing::ElementsAre(kgroup, kvalue, kdim));
    EXPECT_TRUE(n.removeAttribute(kdim));
    EXPECT_FALSE(n.removeAttribute(kdim)); // should success
    EXPECT_THAT(n.attributeNames(), testing::ElementsAre(kgroup, kvalue));
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

    EXPECT_EQ(n.attributeKind(kvalue), AttributeKind::INT);
    EXPECT_EQ(n.attributeKind(kalpha), AttributeKind::UNDEFINED);
    EXPECT_EQ(n.attributeKind("beta"), AttributeKind::UNDEFINED);
}
