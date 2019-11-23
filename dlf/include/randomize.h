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
