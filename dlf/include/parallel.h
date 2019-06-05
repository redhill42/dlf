#pragma once

#include <tbb/tbb.h>

namespace dlf {

#ifndef GRAINSIZE
#define GRAINSIZE 1000
#endif

template <class InputIterator, class OutputIterator>
inline void
parallel_copy(size_t grainsize, InputIterator first, InputIterator last, OutputIterator result) {
    tbb::parallel_for(tbb::blocked_range(first, last, grainsize), [&](const auto& r) {
        auto offset = std::distance(first, r.begin());
        std::copy(r.begin(), r.end(), std::next(result, offset));
    });
}

template <class InputIterator, class OutputIterator>
inline void
parallel_copy(InputIterator first, InputIterator last, OutputIterator result) {
    parallel_copy(GRAINSIZE, first, last, result);
}

template <class InputIterator, class OutputIterator, class UnaryOperation>
inline void
parallel_transform(size_t grainsize, InputIterator first, InputIterator last,
                   OutputIterator result, UnaryOperation op)
{
    tbb::parallel_for(tbb::blocked_range(first, last, grainsize), [&](const auto& r) {
        auto offset = std::distance(first, r.begin());
        std::transform(r.begin(), r.end(), std::next(result, offset), op);
    });
}

template <class InputIterator, class OutputIterator, class UnaryOperation>
inline void
parallel_transform(InputIterator first, InputIterator last,
                   OutputIterator result, UnaryOperation op)
{
    parallel_transform(GRAINSIZE, first, last, result, op);
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryOperation>
inline void
parallel_transform(size_t grainsize, InputIterator1 first1, InputIterator1 last1,
                   InputIterator2 first2, OutputIterator result, BinaryOperation op)
{
    tbb::parallel_for(tbb::blocked_range(first1, last1, grainsize), [&](const auto& r) {
        auto offset = std::distance(first1, r.begin());
        std::transform(r.begin(), r.end(), std::next(first2, offset), std::next(result, offset), op);
    });
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryOperation>
inline void
parallel_transform(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                   OutputIterator result, BinaryOperation op)
{
    parallel_transform(GRAINSIZE, first1, last1, first2, result, op);
}

} // namespace dlf
