#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xscan : public blas::Routine {
public:
    Xscan(const Queue& queue, Event* event, const std::string& name = "SCAN");

    void DoScan(const std::string& name, const size_t m, const size_t n,
                const bool exclusive, const std::vector<size_t>& dims,
                const Buffer<T>& x_buffer, const size_t x_offset,  const std::vector<size_t>& x_strides,
                Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides);
};

}} // namespace gpgpu::dnn