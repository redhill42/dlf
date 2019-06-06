// Use parallel algorithms to solve Euler Project problems.

#include <tbb/tbb.h>
#include "routine.hpp"
#include "test_utility.h"
#include "GPGPUTest.h"

class EulerTest : public GPGPUTest {};

class Problem : public gpgpu::blas::Routine {
public:
    Problem(const gpgpu::Queue& queue, gpgpu::Event* event, const std::string& name)
        : Routine(queue, event, name, {"Xdot"}, gpgpu::blas::PrecisionValue<float>(), {}, {
        #include "euler.cl"
        }) {}

    template <typename... Args>
    long solve(const char* name, Args... args) {
        auto ans_buffer = context_.createBuffer<long>(1);

        auto kernel1 = program_.getKernel(name);
        auto kernel2 = program_.getKernel("Epilogue");

        auto temp_size = 2*db_["WGS2"];
        auto temp_buffer = context_.createBuffer<long>(temp_size);

        auto global1 = std::vector<size_t>{db_["WGS1"]*temp_size};
        auto local1 = std::vector<size_t>{db_["WGS1"]};
        kernel1.setArguments(args..., temp_buffer);
        gpgpu::blas::RunKernel(kernel1, queue_, device_, global1, local1, nullptr);

        auto global2 = std::vector<size_t>{db_["WGS2"]};
        auto local2 = std::vector<size_t>{db_["WGS2"]};
        kernel2.setArguments(temp_buffer, ans_buffer);
        gpgpu::blas::RunKernel(kernel2, queue_, device_, global2, local2, event_);

        long ans = 0;
        ans_buffer.read(queue_, &ans, 1);
        return ans;
    }
};

//==-------------------------------------------------------------------------
// Problem 210: Obtuse Angled Triangles
//
// Consider the set S(r) of points (x,y) with integer coordinates satisfying
// |x| + |y| ≤ r.
//
// Let O be the point (0,0) and C the point (r/4,r/4).
//
// Let N(r) be the number of points B in S(r), so that the triangle OBC has
// an obtuse angle, i.e. the largest angle α satisfies 90°<α<180°.
//
// So, for example, N(4)=24 and N(8)=100.
//
// What is N(1,000,000,000)?
//==-------------------------------------------------------------------------

inline long psi(long n) {
    return ((n & 3) == 1 || (n & 3) == 2) ? 1 : 0;
}

static long Problem210_seq(long r) {
    if (r % 8 != 0)
        throw std::logic_error("Unsolvable with current algorithm");

    auto n = r * r / 32 - 1;
    auto u = static_cast<long>(std::sqrt(n));
    long sum = 0;

    for (long a = 1; a <= u; a++) {
        long b = n / a;
        sum += psi(b);
        if ((a & 3) == 1)
            sum += b;
        if ((a & 3) == 3)
            sum -= b;
    }

    sum *= 4;
    sum -= 4 * u * psi(u);
    sum -= r / 4 - 2;
    sum += 3 * r * r / 2;
    return sum;
}

static long Problem210_par(long r) {
    if (r % 8 != 0)
        throw std::runtime_error("Unsolvable with current algorithm");

    auto n = r * r / 32 - 1;
    auto u = static_cast<long>(std::sqrt(n));
    auto ans = tbb::parallel_reduce(tbb::blocked_range<long>(1, u+1, 10000),
        0L,
        [=](auto&& r, long sum) {
            for (long a = r.begin(); a != r.end(); a++) {
                long b = n / a;
                sum += psi(b);
                if ((a & 3) == 1)
                    sum += b;
                if ((a & 3) == 3)
                    sum -= b;
            }
            return sum;
        },
        std::plus<>());

    ans *= 4;
    ans -= 4 * u * psi(u);
    ans -= r / 4 - 2;
    ans += 3 * r * r / 2;
    return ans;
}

static long Problem210_gpu(long r, const gpgpu::Queue& queue) {
    if (r % 8 != 0)
        throw std::runtime_error("Unsolvable with current algorithm");

    auto n = r * r / 32 - 1;
    auto u = static_cast<long>(std::sqrt(n));
    auto routine = Problem(queue, nullptr, "EULER210");
    auto ans = routine.solve("Euler210", n, u);

    ans *= 4;
    ans -= 4 * u * psi(u);
    ans -= r / 4 - 2;
    ans += 3 * r * r / 2;
    return ans;
}

TEST_F(EulerTest, Problem210) {
    constexpr auto n = 1'000'000'000L;
    constexpr auto solution = 1598174770174689458L;

    timing("Problem 210 CPU sequential", 1, [=]() {
        EXPECT_EQ(Problem210_seq(n), solution);
    });

    timing("Problem 210 CPU parallel", 1, [=]() {
        EXPECT_EQ(Problem210_par(n), solution);
    });

    doTest([=](auto const& queue) {
        timing("Problem 210 GPU " + queue.context().device().name(), 1, [&]() {
            EXPECT_EQ(Problem210_gpu(n, queue), solution);
        });
    });
}