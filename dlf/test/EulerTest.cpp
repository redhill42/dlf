// Use parallel algorithms to solve Euler Project problems.

#include <tbb/tbb.h>
#include "tensor.h"
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

static long Problem210_gpu(long r) {
    if (r % 8 != 0)
        throw std::runtime_error("Unsolvable with current algorithm");

    auto n = r * r / 32 - 1;
    auto u = static_cast<long>(std::sqrt(n));
    auto routine = Problem(gpgpu::current::queue(), nullptr, "EULER210");
    auto ans = routine.solve("Euler210", n, u);

    ans *= 4;
    ans -= 4 * u * psi(u);
    ans -= r / 4 - 2;
    ans += 3 * r * r / 2;
    return ans;
}

PERFORMANCE_TEST_F(EulerTest, Problem210) {
    constexpr auto n = 1'000'000'000L;
    constexpr auto solution = 1598174770174689458L;

    timing("Problem 210 CPU parallel", 1, [=]() {
        EXPECT_EQ(Problem210_par(n), solution);
    });

    doTest([=]() {
        timing("Problem 210 GPU " + gpgpu::current::context().device().name(), 1, [&]() {
            EXPECT_EQ(Problem210_gpu(n), solution);
        });
    });
}

//==-------------------------------------------------------------------------
// Problem 220: Heighway Dragon
//
// Let D0 be the two-letter string "Fa". For n≥1, derive Dn from Dn-1 by the
// string-rewriting rules:
//
// "a" → "aRbFR"
// "b" → "LFaLb"
//
// Thus, D0 = "Fa", D1 = "FaRbFR", D2 = "FaRbFRRLFaLbFR", and so on.
//
// These strings can be interpreted as instructions to a computer graphics program,
// with "F" meaning "draw forward one unit", "L" meaning "turn left 90 degrees",
// "R" meaning "turn right 90 degrees", and "a" and "b" being ignored. The initial
// position of the computer cursor is (0,0), pointing up towards (0,1).
//
// Then Dn is an exotic drawing known as the Heighway Dragon of order n. For example,
// D10 is shown below; counting each "F" as one step, the highlighted spot at (18,16)
// is the position reached after 500 steps.
//
//
// What is the position of the cursor after 1012 steps in D50 ?
// Give your answer in the form x,y with no spaces.

using Matrix = dlf::Tensor<long>;
using Vector = dlf::Tensor<long>;

static Matrix matrix(std::initializer_list<std::initializer_list<long>> data) {
    size_t rows = data.size();
    size_t cols = 0;
    for (auto col : data) {
        cols = std::max(cols, col.size());
    }

    Matrix mat({rows, cols});
    long* p = mat.data();
    for (auto col : data) {
        std::copy(col.begin(), col.end(), p);
        if (col.size() < cols)
            std::fill(p+col.size(), p+cols, 0);
        p += cols;
    }
    return mat;
}

static Vector vector(std::initializer_list<long> data) {
    return Vector({data.size()}, data);
}

static std::string Problem220(long steps) {
    using namespace dlf::dot_product;

    struct Rule {
        long x; Matrix r, l;
        Rule(long x, Matrix r, Matrix l)
            : x(x), r(std::move(r)), l(std::move(l)) {}
    };

    // matrices transform the vector (x, y, dx,dy)
    const Matrix F = matrix({{1, 0, 1, 0}, {0, 1, 0, 1}, {0, 0, 1, 0}, {0, 0, 0, 1}});
    const Matrix R = matrix({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, -1, 0}});
    const Matrix L = matrix({{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, -1}, {0, 0, 1, 0}});
    const Matrix R1 = (F , R , F);
    const Matrix L1 = (F , L , F);

    long x = 0;
    Matrix r = R, l = L;
    std::vector<Rule> rules;

    while (x < steps) {
        auto r1 = (l , R1 , r);
        auto l1 = (l , L1 , r);
        r = std::move(r1);
        l = std::move(l1);
        x = 2 * x + 2;
        rules.emplace_back(x, r, l);
    }

    Vector v = vector({0, 0, 0, 1});
    v = (F , v);
    steps--;

    bool in_r = true;
    for (int i = rules.size()-1; i >= 0 && steps > 1; i--) {
        Rule& rule = rules[i];
        if (rule.x > steps) {
            in_r = true;
            continue;
        }

        v = (rule.r , v);
        steps -= rule.x;
        if (steps >= 2) {
            v = in_r ? (R1 , v) : (L1 , v);
            steps -= 2;
        }
        in_r = false;
    }

    if (steps == 1) v = (F , v);
    return cxx::string_concat(v(0), ',', v(1));
}

PERFORMANCE_TEST(Euler, Problem220) {
    EXPECT_EQ(Problem220(1e12), "139776,963904");
}

//==-------------------------------------------------------------------------
// Problem 577: Counting hexagons

namespace Problem577 {
inline bool inside(int n, Vector p) {
    auto x = p(0), y = p(1);
    return x >= 0 && y >= 0 && x + y <= n;
}

bool inside(int n, int x1, int y1, int x2, int y2) {
    using namespace dlf::dot_product;

    if (x1 == x2 && y1 == y2)
        return false;

    Matrix m = matrix({{0, -1}, {1, 1}});
    Vector p1 = vector({x1, y1});
    Vector p2 = vector({x2, y2});
    Vector p3 = p2 + (m , (p2 - p1));
    Vector p4 = p3 + (m , (p3 - p2));
    Vector p5 = p4 + (m , (p4 - p3));
    Vector p6 = p5 + (m , (p5 - p4));
    return inside(n, p3) && inside(n, p4) && inside(n, p5) && inside(n, p6);
}

int H(int n) {
    int s = 0;
    for (int x1 = 0; x1 <= n; x1++)
    for (int y1 = 0; y1 <= n - x1; y1++)
    for (int x2 = 0; x2 <= n; x2++)
    for (int y2 = 0; y2 <= n - x2; y2++)
        if (inside(n, x1, y1, x2, y2))
            s++;
    return s / 6;
}

long solve(int n) {
    long sum = 0;
    for (long k = 3; k <= n; k += 3)
        sum += k * (n-k+1) * (n-k+2) * (n-k+3) / 18;
    return sum;
}
} // namespace Problem577

PERFORMANCE_TEST(Euler, Problem577) {
    EXPECT_EQ(Problem577::H(3), 1);
    EXPECT_EQ(Problem577::H(6), 12);
    EXPECT_EQ(Problem577::H(20), 966);
    EXPECT_EQ(Problem577::solve(12345), 265695031399260211L);
}
