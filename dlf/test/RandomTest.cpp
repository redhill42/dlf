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

TEST(RandomTest, pcg32) {
    const size_t n = tbb::this_task_arena::max_concurrency();

    auto r1 = pcg32();
    auto d = bypass_distribution<uint32_t>();
    auto X = Tensor<uint32_t>({n, 10000}).random(r1, d);

    auto r2 = pcg32();
    auto Y = Tensor<uint32_t>({10000, n});
    Y.generate(std::ref(r2));

    EXPECT_EQ(X, Y.transpose());

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(r1(), r2());
    }
}

TEST(RandomTest, minstd) {
    const size_t n = tbb::this_task_arena::max_concurrency();

    auto r1 = std::minstd_rand();
    auto d = bypass_distribution<uint32_t>();
    auto X = Tensor<uint32_t>({n, 10000}).random(r1, d);

    auto r2 = std::minstd_rand();
    auto Y = Tensor<uint32_t>({10000, n});
    r2.discard(n);
    Y.generate(std::ref(r2));

    EXPECT_EQ(X, Y.transpose());

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(r1(), r2());
    }
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

TEST(RandomTest, pcg32_gpu) {
    const uint64_t seed = 0xcafef00dd15ea5e5ULL;
    const uint64_t stream = 1442695040888963407ULL;
    const size_t   data_size = 1'000'000;

    auto dev_X = DevTensor<int>({data_size});
    gpgpu::dnn::random(dev_X.size(), dev_X.shape().extents(), dev_X.shape().strides(),
                       dev_X.data(), dev_X.shape().offset(),
                       seed, stream,
                       std::numeric_limits<int>::lowest(),
                       std::numeric_limits<int>::max());

    auto X = Tensor<int>({data_size});
    auto rg = pcg32(seed, stream);
    std::generate(X.begin(), X.end(), rg);

    EXPECT_EQ(X, dev_X.read());
}
