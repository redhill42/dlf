#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xsort : public blas::Routine {
public:
    Xsort(const Queue& queue, Event* event, const std::string& name = "SORT");

    void DoSort(
        const int dir, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides);

private:
    void DoDirectSort(
        const int dir, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides);

    void DoIndirectSort(
        const int dir, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides);

    void DoBlockSort(
        const size_t n, const int dir, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides);

    void DoMerge(
        const size_t n, const size_t k, const int dir, const std::vector<size_t>& dims,
        Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
        Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides);

    void DoDirectMerge(
        const size_t n, const size_t k, const int dir, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& z_buffer, const size_t z_offset, const std::vector<size_t>& z_strides);

    void DoIndirectMerge(
        const size_t n, const size_t k, const int dir, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& z_buffer, const size_t z_offset, const std::vector<size_t>& z_strides);
};

}} // namespace gpgpu::dnn
