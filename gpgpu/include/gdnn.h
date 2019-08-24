#ifndef GPGPU_DNN_H_
#define GPGPU_DNN_H_

#include "gpgpu.h"

namespace gpgpu { namespace dnn {

template <typename T>
void copy(const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset,
          const size_t y_size, Buffer<T>& y_buffer, const size_t y_offset,
          const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void copy(const size_t n, const std::vector<size_t>& dims,
          const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride,
          const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void fill(const size_t n, Buffer<T>& x_buffer, const size_t x_offset, const T value,
          const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void fill(const size_t n, const std::vector<size_t>& dims, const std::vector<size_t>& strides,
          Buffer<T>& x_buffer, const size_t x_offset, const T value,
          const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void transform(const std::string& name, const size_t n,
               const Buffer<T>& x_buffer, const size_t x_offset,
               Buffer<T>& y_buffer, const size_t y_offset,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void transform(const std::string& name, const size_t n, const std::vector<size_t>& dims,
               const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
               Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void transform(const std::string& name, const T alpha, const T beta,
               const size_t n,
               const Buffer<T>& x_buffer, const size_t x_offset,
               Buffer<T>& y_buffer, const size_t y_offset,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void transform(const std::string& name, const T alpha, const T beta,
               const size_t n, const std::vector<size_t>& dims,
               const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
               Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T, typename R>
void transform(const std::string& name,
               const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset,
               const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset,
               Buffer<R>& z_buffer, const size_t z_offset,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T, typename R>
void transform(const std::string& name, const size_t n, const std::vector<size_t>& dims,
               const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
               const Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride,
               Buffer<R>& z_buffer, const size_t z_offset, const std::vector<size_t>& z_stride,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T, typename R>
void transform(const std::string& name,
               const size_t m, const size_t n, const size_t channels,
               const Buffer<T>& x_buffer, const size_t x_offset,
               const Buffer<T>& y_buffer, const size_t y_offset,
               Buffer<R>& z_buffer, const size_t z_offset,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void reduce(const std::string& name, const size_t m, const size_t n,
            const std::vector<size_t>& dims, const std::vector<size_t>& strides,
            const Buffer<T>& x_buffer, const size_t x_offset,
            Buffer<T>& y_buffer, const size_t y_offset,
            const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void arg_reduce(const std::string& name, const size_t n, const size_t k,
                const std::vector<size_t>& dims, const std::vector<size_t>& strides,
                const Buffer<T>& x_buffer, const size_t x_offset, Buffer<int>& y_buffer,
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
            Buffer<T>& result_buffer, Buffer<T>* work_buffer = nullptr,
            const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
size_t conv2dWorkspaceSize(const size_t batches, const size_t channels,
                           const size_t height, const size_t width,
                           const size_t output_h, const size_t output_w,
                           const size_t num_kernels, const size_t group,
                           const size_t kernel_h, const size_t kernel_w,
                           const size_t pad_top, const size_t pad_left,
                           const size_t pad_bottom, const size_t pad_right,
                           const size_t stride_h, const size_t stride_w,
                           const size_t dilation_h, const size_t dilation_w,
                           const Queue& queue = gpgpu::current::queue());

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
void where(const size_t n, const size_t rank,
           const Buffer<bool>& c_buffer, const size_t c_offset,
           const std::vector<size_t>& c_dim, const std::vector<size_t>& c_stride,
           const Buffer<T>& x_buffer, const size_t x_offset,
           const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
           const Buffer<T>& y_buffer, const size_t y_offset,
           const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride,
           Buffer<T>& z_buffer, const size_t z_offset,
           const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

}} // namespace gpgpu::dnn

#endif //GPGPU_DNN_H_
