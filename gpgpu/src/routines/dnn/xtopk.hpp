#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xtopk : public blas::Routine {
public:
    Xtopk(const Queue& queue, Event* event, const std::string& name = "TOPK");

    void DoTopK(
        const size_t limit, const int dir, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
        Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides);

private:
    void DoDirectTopK(
        const size_t n, const size_t limit, const int dir, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
        Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides);

    void DoIndirectTopK(
        const size_t n, const size_t limit, const int dir, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
        Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides);

    void DoCompactTopK(
        const size_t n, const size_t limit, const int dir, const std::vector<size_t>& dims,
        Buffer<T>&        x_buffer, const size_t  x_offset,
        Buffer<int32_t>& ix_buffer, const size_t ix_offset,
        Buffer<T>&        y_buffer, const size_t  y_offset, const std::vector<size_t>&  y_strides,
        Buffer<int32_t>& iy_buffer, const size_t iy_offset, const std::vector<size_t>& iy_strides);

};

}} // namespace gpgpu::dnn
