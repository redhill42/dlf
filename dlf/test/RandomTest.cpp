#include "tensor.h"
#include "gtest/gtest.h"

using namespace dlf;

/*
 * Note: The following tests are highly relevant to implementation and there is
 * no guarantee that the test will be passed when the implementation is changed.
 *
 * We use leap-frog method to distribute random streams to multiple threads.
 */

template <class IntType>
class bypass_distribution {
public:
    using result_type = IntType;
    struct param_type {
        using distribution_type = bypass_distribution;
    };
    template <class Engine> result_type operator()(Engine&& g)
        { return g(); }
};

TEST(RandomTest, minstd) {
    const size_t n = tbb::this_task_arena::max_concurrency();

    auto d = bypass_distribution<uint32_t>();
    auto X = Tensor<uint32_t>({n, 10000}).random(std::minstd_rand(), d);

    auto r = std::minstd_rand();
    auto Y = Tensor<uint32_t>({10000, n});
    r.discard(n);
    Y.generate(r);

    EXPECT_EQ(X, Y.transpose());
}

TEST(RandomTest, mt19937) {
    const size_t n = tbb::this_task_arena::max_concurrency();

    auto d = bypass_distribution<uint32_t>();
    auto X = Tensor<uint32_t>({n, 10000}).random(std::mt19937(), d);

    auto r = std::mt19937(std::mt19937()());
    auto Y = Tensor<uint32_t>({10000, n});
    Y.generate(r);

    EXPECT_EQ(X, Y.transpose());
}
