#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xargsort : public blas::Routine {
public:
    Xargsort(const Queue& queue, Event* event, const std::string& name = "ARGSORT");

    void DoArgSort(
        const int dir, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
        Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides);

    void DoArgSort(
        const int dir, const std::vector<size_t>& dims,
        const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
        Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides);

    void DoArgSort(
        const int dir, const std::vector<size_t>& dims,
        const Buffer<T>&       x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
        const Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides,
              Buffer<T>&       y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
              Buffer<int32_t>& j_buffer, const size_t j_offset, const std::vector<size_t>& j_strides);

private:
    void DoDirectArgSort(
        const int dir, const std::vector<size_t>& dims,
        const Buffer<T>&       x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
        const Buffer<int32_t>* i_buffer, const size_t i_offset, const std::vector<size_t>* i_strides,
              Buffer<T>&       y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
              Buffer<int32_t>& j_buffer, const size_t j_offset, const std::vector<size_t>& j_strides);

    void DoIndirectArgSort(
        const int dir, const std::vector<size_t>& dims,
        const Buffer<T>&       x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
        const Buffer<int32_t>* i_buffer, const size_t i_offset, const std::vector<size_t>* i_strides,
              Buffer<T>&       y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
              Buffer<int32_t>& j_buffer, const size_t j_offset, const std::vector<size_t>& j_strides);

    void DoBlockArgSort(
        const size_t n, const int dir, const std::vector<size_t>& dims,
        const Buffer<T>&       x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
        const Buffer<int32_t>* i_buffer, const size_t i_offset, const std::vector<size_t>* i_strides,
              Buffer<T>&       y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
              Buffer<int32_t>& j_buffer, const size_t j_offset, const std::vector<size_t>& j_strides);

    void DoArgMerge(
        const size_t n, const size_t k, const int dir, const std::vector<size_t>& dims,
        Buffer<T>&        x_buffer, const size_t  x_offset, const std::vector<size_t>&  x_strides,
        Buffer<int32_t>& ix_buffer, const size_t ix_offset, const std::vector<size_t>& ix_strides,
        Buffer<T>&        y_buffer, const size_t  y_offset, const std::vector<size_t>&  y_strides,
        Buffer<int32_t>& iy_buffer, const size_t iy_offset, const std::vector<size_t>& iy_strides);

    void DoDirectArgMerge(
        const size_t n, const size_t k, const int dir, const std::vector<size_t>& dims,
        Buffer<T>&        x_buffer, const size_t  x_offset, const std::vector<size_t>&  x_strides,
        Buffer<int32_t>& ix_buffer, const size_t ix_offset, const std::vector<size_t>& ix_strides,
        Buffer<T>&        z_buffer, const size_t  z_offset, const std::vector<size_t>&  y_strides,
        Buffer<int32_t>& iz_buffer, const size_t iz_offset, const std::vector<size_t>& iz_strides);

    void DoIndirectArgMerge(
        const size_t n, const size_t k, const int dir, const std::vector<size_t>& dims,
        Buffer<T>&        x_buffer, const size_t  x_offset, const std::vector<size_t>&  x_strides,
        Buffer<int32_t>& ix_buffer, const size_t ix_offset, const std::vector<size_t>& ix_strides,
        Buffer<T>&        z_buffer, const size_t  z_offset, const std::vector<size_t>&  y_strides,
        Buffer<int32_t>& iz_buffer, const size_t iz_offset, const std::vector<size_t>& iz_strides);
};

}} // namespace gpgpu::dnn
