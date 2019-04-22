#include "gtest/gtest.h"
#include "fibonacci.h"

TEST(fibonacci, ForPositiveArgs) {
    int expect[] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144};
    for (int i = 0; i < std::size(expect); i++) {
        EXPECT_EQ(fibonacci(i), expect[i]);
    }
}

TEST(fibonacci, ForNegativeArgs) {
    int expect[] = {0, 1, -1, 2, -3, 5, -8, 13, -21, 34, -55, 89, -144};
    for (int i = 0; i < std::size(expect); i++) {
        EXPECT_EQ(fibonacci(-i), expect[i]);
    }
}

TEST(fibonacci, ForBigNumbers) {
    EXPECT_EQ(fibonacci<long long>(90), 2880067194370816120);
    EXPECT_EQ(fibonacci<long long>(91), 4660046610375530309);

    EXPECT_EQ(fibonacci<long long>(-90), -2880067194370816120);
    EXPECT_EQ(fibonacci<long long>(-91), 4660046610375530309);
}
