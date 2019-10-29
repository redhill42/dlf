#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xcopy: public blas::Routine {
public:
    Xcopy(const Queue &queue, Event* event, const std::string &name = "DNN_COPY");

    void DoCopy(const size_t x_size, const Buffer<T> &x_buffer, const size_t x_offset,
                const size_t y_size, Buffer<T> &y_buffer, const size_t y_offset);

    void DoCopyStrided(
                const size_t n, const std::vector<size_t>& dims,
                const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
                Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride);
};

}} // namespace gpgpu::dnn
