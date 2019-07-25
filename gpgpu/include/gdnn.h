#ifndef GPGPU_DNN_H_
#define GPGPU_DNN_H_

#include "gpgpu.h"

namespace gpgpu { namespace dnn {

template <typename T>
void copy(const size_t x_size, const Buffer<T>& x_buffer,
          const size_t y_size, Buffer<T>& y_buffer,
          const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void copy(const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
          const std::vector<size_t>& stride, const std::vector<size_t>& shape,
          const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void concat_copy(const size_t n, const size_t offset, const size_t block, const size_t stride,
                 const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                 const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void split_copy(const size_t n, const size_t offset, const size_t block, const size_t stride,
                const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void transform(const std::string& name, const size_t n,
               const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void transform(const std::string& name, const size_t n, const T alpha, const T beta,
               const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void transform(const std::string& name,
               const size_t x_size, const Buffer<T>& x_buffer,
               const size_t y_size, const Buffer<T>& y_buffer,
               Buffer<T>& z_buffer,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void transform(const std::string& name, const size_t n,
               const Buffer<T>& x_buffer, const Buffer<T>& y_buffer, Buffer<T>& z_buffer,
               const std::vector<size_t>& lstride, const std::vector<size_t>& rstride,
               const std::vector<size_t>& oshape,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void transform(const std::string& name,
               const size_t m, const size_t n, const size_t channels,
               const Buffer<T>& x_buffer, const Buffer<T>& y_buffer, Buffer<T>& z_buffer,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void batch_norm(const std::vector<size_t>& dims,
                const Buffer<T>& x_buffer,
                      Buffer<T>& y_buffer,
                const Buffer<T>& scale_buffer,
                const Buffer<T>& bias_buffer,
                const Buffer<T>& mean_buffer,
                const Buffer<T>& var_buffer,
                const T epsilon,
                const Queue& queue = gpgpu::current::queue(),
                Event* event = nullptr);

template <typename T>
void lrn(const std::vector<size_t>& dims, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
         const int nsize, const T alpha, const T beta, const T bias,
         const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void conv2d(const size_t batches, const size_t channels,
            const size_t height, const size_t width,
            const size_t output_h, const size_t output_w,
            const size_t num_kernels, const size_t group,
            const size_t kernel_h, const size_t kernel_w,
            const size_t pad_top, const size_t pad_left,
            const size_t pad_bottom, const size_t pad_right,
            const size_t stride_h, const size_t stride_w,
            const size_t dilation_h, const size_t dilation_w,
            const Buffer<T>& im_buffer, const Buffer<T>& kernel_buffer,
            Buffer<T>& result_buffer,
            const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void maxpool(const size_t batches, const size_t channels,
             const size_t height, const size_t width,
             const size_t kernel_h, const size_t kernel_w,
             const size_t output_h, const size_t output_w,
             const size_t pad_top, const size_t pad_left,
             const size_t pad_bottom, const size_t pad_right,
             const size_t stride_h, const size_t stride_w,
             const size_t dilation_h, const size_t dilation_w,
             const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
             const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void avgpool(const size_t batches, const size_t channels,
             const size_t height, const size_t width,
             const size_t output_h, const size_t output_w,
             const size_t kernel_h, const size_t kernel_w,
             const size_t pad_top, const size_t pad_left,
             const size_t pad_bottom, const size_t pad_right,
             const size_t stride_h, const size_t stride_w,
             const size_t dilation_h, const size_t dilation_w,
             const bool count_include_pad,
             const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
             const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void lppool (const size_t batches, const size_t channels,
             const size_t height, const size_t width,
             const size_t output_h, const size_t output_w,
             const size_t kernel_h, const size_t kernel_w,
             const size_t pad_top, const size_t pad_left,
             const size_t pad_bottom, const size_t pad_right,
             const size_t stride_h, const size_t stride_w,
             const size_t dilation_h, const size_t dilation_w,
             const int p,
             const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
             const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void softmax(const size_t m, const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
             const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void logsoftmax(const size_t m, const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void hardmax(const size_t m, const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
             const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void argmax(const size_t m, const size_t k, const size_t n,
            const Buffer<T>& x_buffer, Buffer<int>& y_buffer,
            const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void argmin(const size_t m, const size_t k, const size_t n,
            const Buffer<T>& x_buffer, Buffer<int>& y_buffer,
            const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

}} // namespace gpgpu::dnn

#endif //GPGPU_DNN_H_
