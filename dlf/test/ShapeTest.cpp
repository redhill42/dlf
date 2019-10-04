#include "tensor.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test_utility.h"

using namespace dlf;

TEST(Shape, ShapeBroadcast) {
    EXPECT_EQ(Shape::broadcast(Shape(5, 4), Shape(1)), Shape(5, 4));
    EXPECT_EQ(Shape::broadcast(Tensor<int>({5, 4}), Shape(1)), Shape(5, 4));

    EXPECT_EQ(Shape::broadcast(Shape(5, 4), Shape(4)), Shape(5, 4));
    EXPECT_EQ(Shape::broadcast(Shape(15, 3, 5), Shape(15, 1, 5)), Shape(15, 3, 5));
    EXPECT_EQ(Shape::broadcast(Shape(8, 1, 6, 1), Shape(7, 1, 5)), Shape(8, 7, 6, 5));

    EXPECT_EQ(Shape::broadcast(Shape(), Shape(2, 3, 4)), Shape(2, 3, 4));
    EXPECT_EQ(Shape::broadcast(Shape(2, 3, 4), Shape()), Shape(2, 3, 4));
    
    EXPECT_ANY_THROW(Shape::broadcast(Shape(2, 1), Shape(8, 4, 3)));
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
    auto shape = Shape(2, 3, 4);
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    auto i = 0;
    for (auto it = begin; it != end; ) {
        EXPECT_EQ(it.offset(), i);
        ++it, ++i;
    }
}

TEST(ShapeIterator, Backward) {
    auto shape = Shape(2, 3, 4);
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    auto i = static_cast<int>(shape.size());
    for (auto it = end; it != begin; ) {
        --it, --i;
        EXPECT_EQ(it.offset(), i);
    }
}

TEST(ShapeIterator, BroadcastScalarForward) {
    auto shape = Shape(1).broadcast({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    for (auto it = begin; it != end; ) {
        EXPECT_EQ(it.offset(), 0);
        ++it;
    }
}

TEST(ShapeIterator, BroadcastScalarBackward) {
    auto shape = Shape(1).broadcast({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    for (auto it = end; it != begin; ) {
        --it;
        EXPECT_EQ(it.offset(), 0);
    }
}

TEST(ShapeIterator, BroadcastRowForward) {
    auto shape = Shape(3, 4).broadcast({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    auto i = 0;
    for (auto it = begin; it != end; ) {
        EXPECT_EQ(it.offset(), i % 12);
        ++it, ++i;
    }
}

TEST(ShapeIterator, BroadcastRowBackward) {
    auto shape = Shape(3, 4).broadcast({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    auto i = static_cast<int>(shape.size());
    for (auto it = end; it != begin; ) {
        --it, --i;
        EXPECT_EQ(it.offset(), i % 12);
    }
}

TEST(ShapeIterator, BroadcastColumnForward) {
    auto shape = Shape(3, 1).broadcast({2, 3, 4});
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
    auto shape = Shape(3, 1).broadcast({2, 3, 4});
    auto begin = shaped_iterator<int>(shape, nullptr, 0);
    auto end = shaped_iterator<int>(shape, nullptr, shape.size());
    std::vector<size_t> indices{0, 0, 0};
    for (auto it = end; it != begin; ) {
        --it;
        shape.previous(indices);
        EXPECT_EQ(it.offset(), indices[1]);
    }
}

PERFORMANCE_TEST(Shape, BroadcastPerformance) {
    auto A1 = Tensor<int>({1024, 1024}).range(1);
    auto A2 = Tensor<int>({1024, 1}).range(1);
    auto B1 = Tensor<int>({1024, 1024}).range(1);
    auto B2 = Tensor<int>({1024}).range(1);

    for (int i = 0; i < 3; i++) {
        timing("No broadcast", 100, [&]() {
            A1 + B1;
        });

        timing("Broadcast right", 100, [&]() {
            A1 + B2;
        });

        timing("Broadcast left", 100, [&]() {
            A2 + B1;
        });

        timing("Broadcast both", 100, [&]() {
            A2 + B2;
        });

        timing("Broadcast scalar right", 100, [&]() {
            A1 + Tensor<int>::scalar(5);
        });

        timing("Broadcast scalar left", 100, [&]() {
            Tensor<int>::scalar(5) + A1;
        });

        std::cout << std::endl;
    }
}

PERFORMANCE_TEST(Shape, GPUBroadcastPerformance) {
    auto A1 = DevTensor<float>({1024, 1024}).range(1);
    auto A2 = DevTensor<float>({1024, 1}).range(1);
    auto B1 = DevTensor<float>({1024, 1024}).range(1);
    auto B2 = DevTensor<float>({1024}).range(1);

    for (int i = 0; i < 3; i++) {
        timing("GPU no broadcast", 1, [&]() {
            auto C = DevTensor<float>({1024, 1024});
            for (int i = 0; i < 100; i++)
                transformTo(A1, B1, C, xfn::plus<>());
            C.read();
        });

        timing("GPU broadcast right", 1, [&]() {
            auto C = DevTensor<float>({1024, 1024});
            for (int i = 0; i < 100; i++)
                transformTo(A1, B2, C, xfn::plus<>());
            C.read();
        });

        timing("GPU broadcast left", 1, [&]() {
            auto C = DevTensor<float>({1024, 1024});
            for (int i = 0; i < 100; i++)
                transformTo(A2, B1, C, xfn::plus<>());
            C.read();
        });

        timing("GPU broadcast both", 1, [&]() {
            auto C = DevTensor<float>({1024, 1024});
            for (int i = 0; i < 100; i++)
                transformTo(A2, B2, C, xfn::plus<>());
            C.read();
        });

        timing("GPU broadcast scalar right", 1, [&]() {
            auto C = DevTensor<float>({1024, 1024});
            auto S = DevTensor<float>::scalar(5.f);
            for (int i = 0; i < 100; i++)
                transformTo(A1, S, C, xfn::plus<>());
            C.read();
        });

        timing("GPU broadcast scalar left", 1, [&]() {
            auto C = DevTensor<float>({1024, 1024});
            auto S = DevTensor<float>::scalar(5.f);
            for (int i = 0; i < 100; i++)
                transformTo(S, A1, C, xfn::plus<>());
            C.read();
        });

        std::cout << std::endl;
    }
}

TEST(Filter2D, AutoPad) {
    for (size_t n = 3; n < 20; n++)
    for (size_t k = 1; k <= std::min<size_t>(7, n); k++)
    for (size_t s = 1; s <= 3; s++)
    for (size_t d = 1; d <= 3; d++) {
        auto filter = dnn::Filter2D({1, 1, n, n}, k, k)
            .strides(s, s)
            .dilations(d, d);
        filter.auto_pad("SAME_UPPER");
        EXPECT_EQ(filter.output_h(), (filter.height()-1)/s+1);
        filter.auto_pad("SAME_LOWER");
        EXPECT_EQ(filter.output_h(), (filter.height()-1)/s+1);
    }
}
