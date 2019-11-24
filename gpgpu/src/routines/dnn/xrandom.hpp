#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xrandom : public blas::Routine {
public:
    Xrandom(const Queue& queue, Event* event, const std::string& name = "RANDOM");

    void DoRandom(
        const size_t n, const std::vector<size_t>& dims, const std::vector<size_t>& strides,
        Buffer<T>& x_buffer, const size_t x_offset,
        const uint64_t seed, const uint64_t stream, const T low, const T high);

    void DoRandomNormal(
        const size_t n, const std::vector<size_t>& dims, const std::vector<size_t>& strides,
        Buffer<T>& x_buffer, const size_t x_offset,
        const uint64_t seed, const uint64_t stream, const T mean, const T stdev);
};

}} // namespace gpgpu::dnn
