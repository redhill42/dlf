#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T, typename R = T>
class Xreduce : public blas::Routine {
public:
    Xreduce(const Queue& queue, Event* event, const std::string& name);

    void DoReduce(
        const size_t m, const size_t n, const T value,
        const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
        const Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
        Buffer<R>& y_buffer, const size_t y_offset);

    void DoReduce(
        const size_t m, const size_t n,
        const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
        const Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
        Buffer<R>& y_buffer, const size_t y_offset)
    {
        DoReduce(m, n, T{},
                 x_dims, x_strides, x_buffer, x_offset,
                 y_dims, y_strides, y_buffer, y_offset);
    }

private:
    void DoReduceDirect(
        const size_t m, const size_t n, const T value,
        const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
        const Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
        Buffer<R>& y_buffer, const size_t y_offset);

    void DoReduceIndirect(
        const size_t m, const size_t n, const T value,
        const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
        const Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
        Buffer<R>& y_buffer, const size_t y_offset);
};

}} // namespace gpgpu::dnn
