#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xsort : public blas::Routine {
public:
    Xsort(const Queue& queue, Event* event, const std::string& name = "SORT");

    void DoSort(const int dir, const std::vector<size_t>& dims,
                const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
                Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides);

    void DoArgSort(const int dir, const std::vector<size_t>& dims,
                   const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
                   Buffer<int32_t>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides);
};

}} // namespace gpgpu::dnn
