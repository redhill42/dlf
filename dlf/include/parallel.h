#pragma once

#include <tbb/tbb.h>

namespace dlf { namespace par {

#ifndef GRAINSIZE
#define GRAINSIZE 1000
#endif

template <size_t grainsize = GRAINSIZE, class InputIterator, class OutputIterator>
inline void
copy(InputIterator first, InputIterator last, OutputIterator result) {
    tbb::parallel_for(tbb::blocked_range<InputIterator>(first, last, grainsize),
        [&](const auto& r) {
            auto offset = std::distance(first, r.begin());
            std::copy(r.begin(), r.end(), std::next(result, offset));
        });
}

template <size_t grainsize = GRAINSIZE, class ForwardIterator, typename T>
inline void fill(ForwardIterator first, ForwardIterator last, const T& value) {
    tbb::parallel_for(tbb::blocked_range<ForwardIterator>(first, last, grainsize),
        [&](auto r) {
            std::fill(r.begin(), r.end(), value);
        });
}

template <size_t grainsize = GRAINSIZE, class InputIterator, class OutputIterator, class UnaryOperation>
inline void
transform(InputIterator first, InputIterator last, OutputIterator result, UnaryOperation op) {
    tbb::parallel_for(tbb::blocked_range<InputIterator>(first, last, grainsize),
        [&](const auto& r) {
            auto offset = std::distance(first, r.begin());
            std::transform(r.begin(), r.end(), std::next(result, offset), op);
        });
}

template <size_t grainsize = GRAINSIZE,
          typename InputIterator1, typename InputIterator2,
          typename OutputIterator,
          typename BinaryOperation>
inline void
transform(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
          OutputIterator result, BinaryOperation op)
{
    tbb::parallel_for(tbb::blocked_range<InputIterator1>(first1, last1, grainsize),
        [&](const auto& r) {
            auto offset = std::distance(first1, r.begin());
            std::transform(r.begin(), r.end(),
                           std::next(first2, offset),
                           std::next(result, offset),
                           op);
        });
}

}} // namespace dlf::par
