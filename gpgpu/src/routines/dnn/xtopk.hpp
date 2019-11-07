#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xtopk : public blas::Routine {
public:
    Xtopk(const Queue& queue, Event* event, const std::string& name = "TOPK");

    void DoTopK(
        const size_t limit, const int dir,
        const std::vector<size_t>& x_dims, const std::vector<size_t>& y_dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
        Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides);

private:
    void DoDirectTopK(
        const size_t n, const size_t limit, const int dir,
        const std::vector<size_t>& x_dims, const std::vector<size_t>& y_dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
        Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides);

    void DoBlockCompactTopK(
        const size_t n, const size_t limit, const int dir,
        const std::vector<size_t>& x_dims, const std::vector<size_t>& y_dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
        Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides);

    void DoSortedTopK(
        const size_t n, const size_t limit, const int dir,
        const std::vector<size_t>& x_dims, const std::vector<size_t>& y_dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
        Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides);

    void DoBlockTopK(
        const size_t n, const size_t limit, const int dir,
        const size_t blocks, const size_t block_size,
        const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
        const Buffer<T>& x_buffer, const size_t x_offset,
              Buffer<T>& y_buffer, const size_t y_offset,
        Buffer<int32_t>& i_buffer, const size_t i_offset);

    void DoCompactTopK(
        const size_t n, const size_t limit, const int dir, const std::vector<size_t>& y_dims,
        Buffer<T>&       x_buffer, const size_t x_offset,
        Buffer<int32_t>& i_buffer, const size_t i_offset,
        Buffer<T>&       y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
        Buffer<int32_t>& j_buffer, const size_t j_offset, const std::vector<size_t>& j_strides);

    void DoSelectTopK(
        const size_t n, const size_t limit, const std::vector<size_t>& y_dims,
        Buffer<T>&       x_buffer, const size_t x_offset,
        Buffer<int32_t>& i_buffer, const size_t i_offset,
        Buffer<T>&       y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
        Buffer<int32_t>& j_buffer, const size_t j_offset, const std::vector<size_t>& j_strides);

    size_t split(size_t n, size_t limit);
};

}} // namespace gpgpu::dnn
