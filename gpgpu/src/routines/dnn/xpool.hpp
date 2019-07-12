#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xpool : public blas::Routine {
public:
    Xpool(const Queue& queue, Event* event, const std::string& name = "POOL");

    void DoMaxPool(const size_t batches, const size_t channels,
                   const size_t height, const size_t width,
                   const size_t output_h, const size_t output_w,
                   const size_t kernel_h, const size_t kernel_w,
                   const size_t pad_h, const size_t pad_w,
                   const size_t stride_h, const size_t stride_w,
                   const size_t dilation_h, const size_t dilation_w,
                   const Buffer<T>& x_buffer, const size_t x_offset,
                   Buffer<T>& y_buffer, const size_t y_offset);

    void DoAvgPool(const size_t batches, const size_t channels,
                   const size_t height, const size_t width,
                   const size_t output_h, const size_t output_w,
                   const size_t kernel_h, const size_t kernel_w,
                   const size_t pad_h, const size_t pad_w,
                   const size_t stride_h, const size_t stride_w,
                   const size_t dilation_h, const size_t dilation_w,
                   bool  count_include_pad,
                   const Buffer<T>& x_buffer, const size_t x_offset,
                   Buffer<T>& y_buffer, const size_t y_offset);
};

}} // namespace gpgpu::dnn
