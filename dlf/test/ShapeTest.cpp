#include "tensor.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test_utility.h"

using namespace dlf;

TEST(Shape, ShapeBroadcast) {
    EXPECT_EQ(Shape::broadcast(Shape({5, 4}), Shape({1})), Shape({5, 4}));
    EXPECT_EQ(Shape::broadcast(Tensor<int>({5, 4}), Shape({1})), Shape({5, 4}));

    EXPECT_EQ(Shape::broadcast(Shape({5, 4}), Shape({4})), Shape({5, 4}));
    EXPECT_EQ(Shape::broadcast(Shape({15, 3, 5}), Shape({15, 1, 5})), Shape({15, 3, 5}));
    EXPECT_EQ(Shape::broadcast(Shape({8, 1, 6, 1}), Shape({7, 1, 5})), Shape({8, 7, 6, 5}));

    EXPECT_ANY_THROW(Shape::broadcast(Shape({2, 1}), Shape({8, 4, 3})));
}

static void broadcast_stride_test(Shape from, Shape to, std::vector<size_t> expected) {
    Shape shape = from.broadcast(to);
    EXPECT_EQ(shape.strides(), expected);
}

TEST(Shape, ShapeBroadcastStride) {
    broadcast_stride_test({4}, {3, 4}, {0, 1});
    broadcast_stride_test({3, 1}, {3, 4}, {1, 0});
    broadcast_stride_test({1, 4}, {3, 4}, {0, 1});
    broadcast_stride_test({3, 4}, {3, 4}, {4, 1});
    broadcast_stride_test({5, 1, 4}, {5, 3, 4}, {4, 0, 1});
}

TEST(ShapeIterator, Forward) {
    auto shape = Shape({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    auto i = 0;
    for (auto it = begin; it != end; ) {
        EXPECT_EQ(it.offset(), i);
        ++it, ++i;
    }
}

TEST(ShapeIterator, Backward) {
    auto shape = Shape({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    auto i = static_cast<int>(shape.size());
    for (auto it = end; it != begin; ) {
        --it, --i;
        EXPECT_EQ(it.offset(), i);
    }
}

TEST(ShapeIterator, BroadcastScalarForward) {
    auto shape = Shape({1}).broadcast({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    for (auto it = begin; it != end; ) {
        EXPECT_EQ(it.offset(), 0);
        ++it;
    }
}

TEST(ShapeIterator, BroadcastScalarBackward) {
    auto shape = Shape({1}).broadcast({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    for (auto it = end; it != begin; ) {
        --it;
        EXPECT_EQ(it.offset(), 0);
    }
}

TEST(ShapeIterator, BroadcastRowForward) {
    auto shape = Shape({3, 4}).broadcast({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    auto i = 0;
    for (auto it = begin; it != end; ) {
        EXPECT_EQ(it.offset(), i % 12);
        ++it, ++i;
    }
}

TEST(ShapeIterator, BroadcastRowBackward) {
    auto shape = Shape({3, 4}).broadcast({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    auto i = static_cast<int>(shape.size());
    for (auto it = end; it != begin; ) {
        --it, --i;
        EXPECT_EQ(it.offset(), i % 12);
    }
}

TEST(ShapeIterator, BroadcastColumnForward) {
    auto shape = Shape({3, 1}).broadcast({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    std::vector<size_t> indices{0, 0, 0};
    for (auto it = begin; it != end; ) {
        EXPECT_EQ(it.offset(), indices[1]);
        ++it;
        shape.next(indices);
    }
}

TEST(ShapeIterator, BroadcastColumnBackward) {
    auto shape = Shape({3, 1}).broadcast({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    std::vector<size_t> indices{0, 0, 0};
    for (auto it = end; it != begin; ) {
        --it;
        shape.previous(indices);
        EXPECT_EQ(it.offset(), indices[1]);
    }
}

#ifdef NDEBUG
TEST(Shape, BroadcastPerformance) {
    auto A1 = Tensor<int>::range({1024, 1024}, 1);
    auto A2 = Tensor<int>::range({1024, 1}, 1);
    auto B1 = Tensor<int>::range({1024, 1024}, 1);
    auto B2 = Tensor<int>::range({1024}, 1);

    timing("No broadcast", 100, [&]() {
        A1 + B1;
    });

    timing("Broadcast Right", 100, [&]() {
        A1 + B2;
    });

    timing("Broadcast Left", 100, [&]() {
        A2 + B1;
    });

    timing("Broadcast Both", 100, [&]() {
        A2 + B2;
    });
}
#endif
