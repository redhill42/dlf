#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xmerge : public blas::Routine {
public:
    Xmerge(const Queue& queue, Event* event, const std::string& name = "MERGE");

    void DoMerge(
        const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
        const Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
        const Buffer<T>& y_buffer, const size_t y_offset,
        const std::vector<size_t>& z_dims, const std::vector<size_t>& z_strides,
        Buffer<T>& z_buffer, const size_t z_offset);

private:
    void DoMergeDirect(
        const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
        const Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
        const Buffer<T>& y_buffer, const size_t y_offset,
        const std::vector<size_t>& z_dims, const std::vector<size_t>& z_strides,
        Buffer<T>& z_buffer, const size_t z_offset);

    void DoMergeIndirect(
        const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
        const Buffer<T>& x_buffer, const size_t x_offset,
        const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
        const Buffer<T>& y_buffer, const size_t y_offset,
        const std::vector<size_t>& z_dims, const std::vector<size_t>& z_strides,
        Buffer<T>& z_buffer, const size_t z_offset);
};

}} // namespace gpgpu::dnn
