#include <model/intern.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using namespace dlf::model;

TEST(Intern, BuiltinSymbol) {
    EXPECT_EQ(Symbol("Conv"), kConv);
    EXPECT_EQ(Symbol("abs"), kabs);

    EXPECT_EQ(Symbol("Conv").val(), kConv);
    EXPECT_EQ(Symbol("abs").val(), kabs);

    EXPECT_STREQ(Symbol(kConv).str(), "Conv");
    EXPECT_STREQ(Symbol(kabs).str(), "abs");

    EXPECT_EQ("Conv"_sym, Symbol("Conv"));
    EXPECT_EQ("Conv"_sym, kConv);
    EXPECT_EQ("Conv"_sym, Symbol(kConv));
}

TEST(Intern, CustomSymbol) {
    const char* TEST_SYM = "the_test_symbol";

    Symbol s1 = Symbol(TEST_SYM);
    Symbol s2 = Symbol(TEST_SYM);
    Symbol s3 = Symbol("another_test_symbol");

    EXPECT_EQ(s1, s2);
    EXPECT_EQ(s1.val(), s2.val());
    EXPECT_STREQ(s1.str(), TEST_SYM);
    EXPECT_STREQ(s2.str(), TEST_SYM);
    EXPECT_EQ(s1.str(), s2.str()); // address eq
    EXPECT_THAT(s1.val(), testing::Ge(kLastSymbol));

    EXPECT_NE(s1, s3);
    EXPECT_NE(s1.val(), s3.val());
    EXPECT_STRNE(s1.str(), s3.str());
}
