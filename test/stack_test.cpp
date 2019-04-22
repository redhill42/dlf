#include "gtest/gtest.h"
#include "stack.h"

class StackTest : public ::testing::Test
{
protected:
    static constexpr int values[] = {3, 2, 7, 6, 8};
    static constexpr auto n_values = std::size(values);

    Stack<int,n_values> stack;

    void push_all() {
        for (auto x : values) {
            stack.push(x);
        }
    }
};

TEST_F(StackTest, PushOrPopElementsShouldChangeStackSize) {
    EXPECT_EQ(stack.size(), 0);

    for (int i = 0; i < n_values; i++) {
        EXPECT_EQ(stack.size(), i);
        stack.push(values[i]);
    }

    for (int i = n_values; i > 0; i--) {
        EXPECT_EQ(stack.size(), i);
        stack.pop();
    }

    EXPECT_EQ(stack.size(), 0);
}

TEST_F(StackTest, PushOnFullStackShouldThrowException) {
    push_all();
    EXPECT_THROW(stack.push(100), StackOverflow);
}

TEST_F(StackTest, PopFromEmptyStackShouldThrowException) {
    EXPECT_THROW(stack.pop(), StackUnderflow);
}

TEST_F(StackTest, PeekFromEmptyStackShouldThrowException) {
    EXPECT_THROW(stack.peek(), StackUnderflow);
}

TEST_F(StackTest, PeekShouldNotChangeStackContents) {
    push_all();

    auto last_pushed = values[n_values-1];

    EXPECT_EQ(stack.size(), n_values);
    EXPECT_EQ(stack.peek(), last_pushed);
    EXPECT_EQ(stack.peek(), last_pushed);
    EXPECT_EQ(stack.size(), n_values);
}

TEST_F(StackTest, StackShouldInLIFOOrder) {
    push_all();

    for (int i = n_values; --i >= 0; ) {
        EXPECT_EQ(stack.pop(), values[i]);
    }
}
