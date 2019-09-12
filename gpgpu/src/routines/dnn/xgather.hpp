#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xgather : public blas::Routine {
public:
    Xgather(const Queue& queue, Event* event, const std::string& name = "GATHER");

    void DoGather(
        const size_t m, const size_t n, const size_t chunk, const size_t max_item,
        const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
        const Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& i_dim, const std::vector<size_t>& i_stride,
        const Buffer<int>& i_buffer, const size_t i_offset,
        const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride,
        Buffer<T>& y_buffer, const size_t y_offset);

    void DoGatherElements(
        const size_t n, const int axis,
        const std::vector<size_t>& x_shape, const std::vector<size_t>& x_strides,
        const Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& i_shape, const std::vector<size_t>& i_strides,
        const Buffer<int>& i_buffer, const size_t i_offset,
        const std::vector<size_t>& y_shape, const std::vector<size_t>& y_strides,
        Buffer<T>& y_buffer, const size_t y_offset);

    void DoScatterElements(
        const size_t n, const int axis,
        const std::vector<size_t>& x_shape, const std::vector<size_t>& x_strides,
        Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& i_shape, const std::vector<size_t>& i_strides,
        const Buffer<int>& i_buffer, const size_t i_offset,
        const std::vector<size_t>& y_shape, const std::vector<size_t>& y_strides,
        const Buffer<T>& y_buffer, const size_t y_offset);

    void DoGatherND(
        const size_t n, const size_t k, const size_t chunk,
        const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
        const Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& i_dim, const std::vector<size_t>& i_stride,
        const Buffer<int>& i_buffer, const size_t i_offset,
        const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride,
        Buffer<T>& y_buffer, const size_t y_offset);

    void DoScatterND(
        const size_t n, const size_t k, const size_t chunk,
        const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
        Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& i_dim, const std::vector<size_t>& i_stride,
        const Buffer<int>& i_buffer, const size_t i_offset,
        const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride,
        const Buffer<T>& y_buffer, const size_t y_offset);
};

}} // namespace gpgpu::dnn
