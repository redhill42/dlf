#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xfill : public blas::Routine {
public:
    Xfill(const Queue& queue, Event* event, const std::string& name = "DNN_FILL");

    void DoFill(const size_t n, Buffer<T>& x_buffer, const size_t x_offset, const T value);

    void DoFillStrided(
                const size_t n, const std::vector<size_t>& dims, const std::vector<size_t>& strides,
                Buffer<T>& x_buffer, const size_t x_offset,
                const T value);
};

}} // namespace gpgpu::dnn
