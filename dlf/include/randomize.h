#pragma once

#include <random>
#include <algorithm>
#include <type_traits>
#include <atomic>
#include <tbb/tbb.h>
#include "pcg_random.h"
#include "pcg_extras.h"

#ifndef GRAINSIZE
#define GRAINSIZE 1024
#endif

namespace dlf {

template <typename Range, typename Engine, typename Body>
inline void parallel_randomize(Range range, Engine&& engine, Body&& body) {
    body(range, engine);
}

namespace detail {
template <typename Range>
Range split_range(Range& range, size_t len) {
    auto left = Range(range.begin(), range.begin() + len);
    range = Range(range.begin() + len, range.end());
    return left;
}
} // namespace detail

template <typename Range, typename xtype, typename itype,
          class output_mixin, bool output_previous,
          class stream_mixin, class multiplier_mixin,
          typename Body>
void parallel_randomize(
    Range range,
    pcg_detail::engine<xtype, itype, output_mixin, output_previous,
                       stream_mixin, multiplier_mixin>& eng,
    Body&& body)
{
    auto n_size     = range.size();
    auto n_split    = tbb::this_task_arena::max_concurrency();
    auto chunk_size = n_size / n_split;
    auto left_over  = n_size - chunk_size * n_split;

    if (n_size < GRAINSIZE || n_split < 2) {
        body(range, eng);
        return;
    }

    tbb::task_group tg;
    std::atomic<size_t> consumed{0};

    for (int i = 0; i < n_split; ++i) {
        auto len = chunk_size + (i < left_over);
        auto left = detail::split_range(range, len);
        tg.run([eng, n_split, &consumed, left, &body] {
            auto rg = eng.leapfrog(n_split);
            body(left, rg);
            consumed += eng.leapfrog_distance(rg) / n_split;
        });
        eng();
    }
    tg.wait();
    eng.advance(consumed - n_split);
}

template <typename Range,
          typename xtype, typename itype, typename output_mixin, bool output_previous,
          typename stream_mixin, typename multiplier_mixin,
          pcg_detail::bitcount_t table_pow2, pcg_detail::bitcount_t advance_pow2,
          typename extvalclass, bool kdd,
          typename Body>
std::enable_if_t<
    stream_mixin::can_specify_stream ||
    std::is_same<stream_mixin, pcg_detail::unique_stream<itype>>::value>
parallel_randomize(
    Range range,
    pcg_detail::extended<
        table_pow2, advance_pow2,
        pcg_detail::engine<xtype, itype, output_mixin, output_previous,
                           stream_mixin, multiplier_mixin>,
        extvalclass, kdd>& eng,
    Body&& body)
{
    auto n_size     = range.size();
    auto n_split    = tbb::this_task_arena::max_concurrency();
    auto chunk_size = n_size / n_split;
    auto left_over  = n_size - chunk_size * n_split;

    if (n_size < GRAINSIZE || n_split < 2) {
        body(range, eng);
        return;
    }

    using Engine = std::remove_reference_t<decltype(eng)>;
    auto seed_seq = pcg_extras::seed_seq_from<decltype(std::ref(eng))>(std::ref(eng));

    tbb::task_group tg;
    for (int i = 0; i < n_split; ++i) {
        auto len = chunk_size + (i < left_over);
        auto left = detail::split_range(range, len);
        tg.run([stream = Engine(seed_seq), left, &body] {
            auto rg = stream;
            body(left, rg);
        });
    }
    tg.wait();
}

/**
 * A generator of uniform pseudorandom values applicable for use in isolated
 * parallel computations that may generate subtasks.
 */
class splitmix64 {
public:
    using result_type = uint64_t;

private:
    static constexpr uint64_t GOLDEN_GAMMA = 0x9e3779b97f4a7c15ull;

    uint64_t seed_;
    uint64_t gamma_;

    explicit splitmix64(uint64_t seed, uint64_t gamma)
        : seed_(seed), gamma_(gamma) {}

    static uint64_t mix64(uint64_t z) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    static uint64_t mix_gamma(uint64_t z) {
        z = (z ^ (z >> 33)) * 0xff51afd7ed558ccdull;
        z = (z ^ (z >> 33)) * 0xc4ceb9fe1a85ec53ull;
        z = (z ^ (z >> 33)) | 1ull;
        int n = __builtin_popcountll(z ^ (z >> 1)); // FIXME
        return (n < 24) ? z ^ 0xaaaaaaaaaaaaaaaaull : z;
    }

    uint64_t next_seed() {
        return seed_ += gamma_;
    }

public:
    static constexpr result_type min() { return result_type(0); }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }
    static constexpr result_type default_seed = result_type(1);

    explicit splitmix64(result_type sd = default_seed) : splitmix64(sd, GOLDEN_GAMMA) {}

    template <class SeedSeq>
    explicit splitmix64(SeedSeq&& q,
        std::enable_if_t<!std::is_convertible<SeedSeq, result_type>::value &&
                         !std::is_same<std::decay_t<SeedSeq>, splitmix64>::value>* = 0)
        : splitmix64(pcg_extras::generate_one<uint64_t>(std::forward<SeedSeq>(q))) {}

    template <typename T>
    void seed(T&& sd) { new (this) splitmix64(std::forward<T>(sd)); }
    void seed() { seed(default_seed); }

    result_type operator()() {
        return mix64(next_seed());
    }

    void discard(unsigned long long z) {
        seed_ += gamma_ * z;
    }

    splitmix64 split() {
        return splitmix64(mix64(next_seed()), mix_gamma(next_seed()));
    }

    splitmix64 leapfrog(size_t step) {
        uint64_t leap = gamma_ * step;
        return splitmix64(next_seed() - leap, leap);
    }

    friend bool operator==(const splitmix64& x, const splitmix64& y)
        { return x.seed_ == y.seed_ && x.gamma_ == y.gamma_; }
    friend bool operator!=(const splitmix64& x, const splitmix64& y)
        { return !(x == y); }

    template <typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits>&
    operator<<(std::basic_ostream<CharT, Traits>& os, const splitmix64& x);

    template <typename CharT, typename Traits>
    friend std::basic_istream<CharT, Traits>&
    operator>>(std::basic_istream<CharT, Traits>& is, splitmix64& x);
};

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& out, const splitmix64& x) {
    auto orig_flags = out.flags(std::ios_base::dec | std::ios_base::left);
    auto space = out.widen(' ');
    auto orig_fill = out.fill();

    out << x.seed_ << space << x.gamma_;
    out.flags(orig_flags);
    out.fill(orig_fill);
    return out;
}

template <typename CharT, typename Traits>
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits>& in, splitmix64& x) {
    auto orig_flags = in.flags(std::ios_base::dec | std::ios_base::skipws);
    uint64_t seed, gamma;
    in >> seed >> gamma;
    if (!in.fail()) {
        x.seed_ = seed;
        x.gamma_ = gamma;
    }
    in.flags(orig_flags);
    return in;
}

namespace detail {
template <typename Range, typename Engine, typename Body>
class random_task : tbb::task {
    Range   m_range;
    Engine  m_engine;
    Body    m_body;

    random_task(Range range, Engine engine, Body body)
        : m_range(range), m_engine(std::move(engine)), m_body(body) {}

    task* execute() override;

public:
    static void run(Range range, Engine& engine, Body body) {
        if (!range.empty()) {
            auto& root = *new(allocate_root()) random_task(range, engine, body);
            spawn_root_and_wait(root);
            engine = root.m_engine;
        }
    }
};

template <typename Range, typename Engine, typename Body>
tbb::task* random_task<Range, Engine, Body>::execute() {
    if (!m_range.is_divisible()) {
        m_body(m_range, m_engine);
        return nullptr;
    } else {
        auto& right = *new(allocate_additional_child_of(*parent()))
            random_task(Range(m_range, tbb::split()), m_engine.split(), m_body);
        spawn(right);
        recycle_as_continuation();
        return this;
    }
}
} // namespace detail

template <typename Range, typename Body>
inline void parallel_randomize(Range range, splitmix64& eng, Body body) {
    detail::random_task<Range, splitmix64, Body>::run(range, eng, body);
}

namespace detail {
template <typename UIntType, UIntType m>
class skipable_lcg {
public:
    using result_type = UIntType;

private:
    result_type a, x;

    static inline UIntType modmul(UIntType a, UIntType x) {
        return static_cast<UIntType>(uint64_t(a) * x % m);
    }

    static UIntType modpow(UIntType a, UIntType x, size_t n) {
        while (n > 0) {
            if (n & 1)
                x = modmul(a, x);
            a = modmul(a, a);
            n >>= 1;
        }
        return x;
    }

public:
    static constexpr result_type min() { return 1u;     }
    static constexpr result_type max() { return m - 1u; }
    static constexpr result_type default_seed = 1u;

    explicit skipable_lcg(result_type a, result_type s = default_seed)
        : a(a) { seed(s); }

    void seed(result_type s) { x = s % m == 0 ? 1 : s % m; }
    result_type seed() const { return x;                   }
    result_type operator()() { return x = modmul(a, x);    }
    void discard(size_t n)   { x = modpow(a, x, n);        }

    static inline result_type stride(result_type a, size_t n) {
        return modpow(a, result_type(1), n);
    }
};
} // namespace detail

template <typename Range, typename UIntType, UIntType a, UIntType m, typename Body>
void parallel_randomize(Range range, std::linear_congruential_engine<UIntType, a, 0, m>& eng, Body&& body) {
    auto n_size     = range.size();
    auto n_split    = tbb::this_task_arena::max_concurrency();
    auto chunk_size = n_size / n_split;
    auto left_over  = n_size - chunk_size * n_split;

    if (n_size < GRAINSIZE || n_split < 2) {
        body(range, eng);
        return;
    }

    tbb::task_group tg;
    auto stride = detail::skipable_lcg<UIntType, m>::stride(a, n_split);
    auto new_seed = UIntType(0);

    for (int i = 0; i < n_split; ++i) {
        auto len = chunk_size + (i < left_over);
        auto left = detail::split_range(range, len);
        tg.run([stride, seed = eng(), &new_seed, last = (i==n_split-1), left, &body] {
            auto rg = detail::skipable_lcg<UIntType, m>(stride, seed);
            body(left, rg);
            if (last) new_seed = rg.seed();
        });
    }
    tg.wait();
    eng.seed(new_seed);
}

namespace detail {
template <typename UIntType, size_t w, size_t n, size_t m, size_t r,
          UIntType a, size_t u, UIntType d, size_t s,
          UIntType b, size_t t, UIntType c, size_t l, UIntType f>
class skipable_mt {
public:
    using result_type = UIntType;

private:
    result_type state[n];
    size_t offset, skip;
    size_t current = n;

    static constexpr result_type D   = std::numeric_limits<result_type>::digits;
    static constexpr result_type Min = 0;
    static constexpr result_type Max = w == D ? result_type(~0) :
                                       (result_type(1) << w) - result_type(1);

public:
    static constexpr result_type min() { return Min; }
    static constexpr result_type max() { return Max; }

    explicit skipable_mt(size_t offset, size_t skip, result_type sd)
        : offset(offset), skip(skip) { seed(sd); }

    void seed(result_type sd);
    result_type operator()();
    void twist();

private:
    template <size_t count>
    static inline std::enable_if_t<count < w, result_type>
    lshift(result_type x) { return (x << count) & Max; }

    template <size_t count>
    static inline std::enable_if_t<count >= w, result_type>
    lshift(result_type) { return result_type(0); }

    template <size_t count>
    static inline std::enable_if_t<count < D, result_type>
    rshift(result_type x) { return x >> count; }

    template <size_t count>
    static inline std::enable_if_t<count >= D, result_type>
    rshift(result_type) { return result_type(0); }
};

template <typename UIntType, size_t w, size_t n, size_t m, size_t r,
          UIntType a, size_t u, UIntType d, size_t s,
          UIntType b, size_t t, UIntType c, size_t l, UIntType f>
void skipable_mt<UIntType, w, n, m, r, a, u, d, s, b, t, c, l, f>::seed(result_type sd) {
    state[0] = sd & Max;
    for (size_t i = 1; i < n; ++i)
        state[i] = (f * (state[i-1] ^ rshift<w-2>(state[i-1])) + i) & Max;
    current = n + offset;
}

template <typename UIntType, size_t w, size_t n, size_t m, size_t r,
          UIntType a, size_t u, UIntType d, size_t s,
          UIntType b, size_t t, UIntType c, size_t l, UIntType f>
UIntType skipable_mt<UIntType, w, n, m, r, a, u, d, s, b, t, c, l, f>::operator()() {
    if (current >= n) {
        twist();
        current -= n;
    }

    result_type y = state[current];
    y ^= rshift<u>(y) & d;
    y ^= lshift<s>(y) & b;
    y ^= lshift<t>(y) & c;
    y ^= rshift<l>(y);
    current += skip;
    return y;
}

template <typename UIntType, size_t w, size_t n, size_t m, size_t r,
          UIntType a, size_t u, UIntType d, size_t s,
          UIntType b, size_t t, UIntType c, size_t l, UIntType f>
void skipable_mt<UIntType, w, n, m, r, a, u, d, s, b, t, c, l, f>::twist() {
    const result_type mask = r == D ? result_type(~0) : (result_type(1) << r) - result_type(1);
    for (size_t i = 0; i < n - m; ++i) {
        result_type x = (state[i] & ~mask) | (state[i+1] & mask);
        state[i] = state[i + m] ^ rshift<1>(x) ^ (a * (x & 1));
    }
    for (size_t i = n - m; i < n - 1; ++i) {
        result_type x = (state[i] & ~mask) | (state[i+1] & mask);
        state[i] = state[i + (m - n)] ^ rshift<1>(x) ^ (a * (x & 1));
    }
    result_type x = (state[n-1] & ~mask) | (state[0] & mask);
    state[n-1] = state[m-1] ^ rshift<1>(x) ^ (a * (x & 1));
}
} // namespace detail

template <typename Range, typename UIntType,
          size_t w, size_t n, size_t m, size_t r,
          UIntType a, size_t u, UIntType d, size_t s,
          UIntType b, size_t t, UIntType c, size_t l, UIntType f,
          typename Body>
void parallel_randomize(Range range,
    std::mersenne_twister_engine<UIntType,w,n,m,r,a,u,d,s,b,t,c,l,f>& eng,
    Body&& body)
{
    auto n_size     = range.size();
    auto n_split    = std::min(static_cast<size_t>(tbb::this_task_arena::max_concurrency()), n/8);
    auto chunk_size = n_size / n_split;
    auto left_over  = n_size - chunk_size * n_split;

    if (n_size < GRAINSIZE || n_split < 2) {
        body(range, eng);
        return;
    }

    tbb::task_group tg;
    auto seed = eng();

    for (int i = 0; i < n_split; ++i) {
        auto len = chunk_size + (i < left_over);
        auto left = detail::split_range(range, len);
        tg.run([left, i, n_split, seed, &body] {
            auto rg = detail::skipable_mt<UIntType,w,n,m,r,a,u,d,s,b,t,c,l,f>(i, n_split, seed);
            body(left, rg);
        });
    }
    tg.wait();
}

} // namespace dlf
