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
void reverse(const size_t m, const size_t n, 
             const std::vector<size_t>& dims, const std::vector<size_t>& strides,
             Buffer<T>& x_buffer, const size_t x_offset,
             const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void range(const size_t n, const T start, const T delta, Buffer<T>& x_buffer, const size_t x_offset,
           const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void range(const size_t n, const T start, const T delta,
           const std::vector<size_t>& dims, const std::vector<size_t>& strides,
           Buffer<T>& x_buffer, const size_t x_offset,
           const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void random(
    const size_t n, const std::vector<size_t>& dims, const std::vector<size_t>& strides,
    Buffer<T>& x_buffer, const size_t x_offset,
    const uint64_t seed, const uint64_t stream,
    const T low, const T high,
    const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void random_normal(
    const size_t n, const std::vector<size_t>& dims, const std::vector<size_t>& strides,
    Buffer<T>& x_buffer, const size_t x_offset,
    const uint64_t seed, const uint64_t stream,
    const T mean, const T stdev,
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
            const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
            const Buffer<T>& x_buffer, const size_t x_offset,
            const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
            Buffer<T>& y_buffer, const size_t y_offset,
            const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void count(const size_t m, const size_t n, const T value,
           const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
           const Buffer<T>& x_buffer, const size_t x_offset,
           const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
           Buffer<int>& y_buffer, const size_t y_offset,
           const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void scan(const std::string& name, const size_t m, const size_t n,
          const bool exclusive, const std::vector<size_t>& dims,
          const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
          const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void scan_nonzero(const size_t m, const size_t n,
                  const bool exclusive, const std::vector<size_t>& dims,
                  const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
                  Buffer<int32_t>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
                  const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void arg_reduce(const std::string& name, const size_t n, const size_t k,
                const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
                const Buffer<T>& x_buffer, const size_t x_offset,
                const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
                Buffer<int>& y_buffer, const size_t y_offset,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);


template <typename T>
void merge(const int dir,
           const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
           const Buffer<T>& x_buffer, const size_t x_offset,
           const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
           const Buffer<T>& y_buffer, const size_t y_offset,
           const std::vector<size_t>& z_dims, const std::vector<size_t>& z_strides,
           Buffer<T>& z_buffer, const size_t z_offset,
           const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void sort(const int dir, const std::vector<size_t>& dims,
          const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
          const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void argsort(const int dir, const std::vector<size_t>& dims,
             const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
             Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides,
             const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void argsort(const int dir, const std::vector<size_t>& dims,
             const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
             Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
             Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides,
             const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void top_k(const size_t limit, const int dir,
           const std::vector<size_t>& x_dims, const std::vector<size_t>& y_dims,
           const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
           Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
           Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides,
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
void softmax(const size_t m, const size_t n, Buffer<T>& x_buffer, const size_t x_offset,
             const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void logsoftmax(const size_t m, const size_t n, Buffer<T>& x_buffer, const size_t x_offset,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void hardmax(const size_t m, const size_t n, Buffer<T>& x_buffer, const size_t x_offset,
             const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void where(const size_t n, const std::vector<size_t>& dim,
           const Buffer<bool>& c_buffer, const size_t c_offset, const std::vector<size_t>& c_stride,
           const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
           const Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride,
           Buffer<T>& z_buffer, const size_t z_offset,
           const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void onehot(const size_t n, const size_t d, const size_t k,
            const Buffer<T>& indices, const Buffer<T>& values, Buffer<T>& output,
            const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void gather(const size_t m, const size_t n, const size_t chunk, const size_t max_item,
            const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
            const Buffer<T>& x_buffer, const size_t x_offset,
            const std::vector<size_t>& i_dim, const std::vector<size_t>& i_stride,
            const Buffer<int>& i_buffer, const size_t i_offset,
            const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride,
            Buffer<T>& y_buffer, const size_t y_offset,
            const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void gather_elements(const size_t n, const int axis,
                     const std::vector<size_t>& x_shape, const std::vector<size_t>& x_strides,
                     const Buffer<T>& x_buffer, const size_t x_offset,
                     const std::vector<size_t>& i_shape, const std::vector<size_t>& i_strides,
                     const Buffer<int>& i_buffer, const size_t i_offset,
                     const std::vector<size_t>& y_shape, const std::vector<size_t>& y_strides,
                     Buffer<T>& y_buffer, const size_t y_offset,
                     const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void scatter_elements(const size_t n, const int axis,
                      const std::vector<size_t>& x_shape, const std::vector<size_t>& x_strides,
                      Buffer<T>& x_buffer, const size_t x_offset,
                      const std::vector<size_t>& i_shape, const std::vector<size_t>& i_strides,
                      const Buffer<int>& i_buffer, const size_t i_offset,
                      const std::vector<size_t>& y_shape, const std::vector<size_t>& y_strides,
                      const Buffer<T>& y_buffer, const size_t y_offset,
                      const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void gather_nd(const size_t n, const size_t k, const size_t chunk,
               const std::vector<size_t>& x_shape, const std::vector<size_t>& x_strides,
               const Buffer<T>& x_buffer, const size_t x_offset,
               const std::vector<size_t>& i_shape, const std::vector<size_t>& i_strides,
               const Buffer<int>& i_buffer, const size_t i_offset,
               const std::vector<size_t>& y_shape, const std::vector<size_t>& y_strides,
               Buffer<T>& y_buffer, const size_t y_offset,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void scatter_nd(const size_t n, const size_t k, const size_t chunk,
                const std::vector<size_t>& x_shape, const std::vector<size_t>& x_strides,
                Buffer<T>& x_buffer, const size_t x_offset,
                const std::vector<size_t>& i_shape, const std::vector<size_t>& i_strides,
                const Buffer<int>& i_buffer, const size_t i_offset,
                const std::vector<size_t>& y_shape, const std::vector<size_t>& y_strides,
                const Buffer<T>& y_buffer, const size_t y_offset,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

void gather_indices(const size_t m, const size_t n, const bool row_major,
                    const std::vector<size_t>& dims,
                    const Buffer<int32_t>& indices, const size_t indices_offset,
                    Buffer<int32_t>& output, const size_t output_offset,
                    const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void resize1d(const size_t batch_count,
              const Buffer<T>& x_buffer, const size_t x_offset,
              const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset,
              const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
              const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
void resize2d(const size_t batch_count,
              const Buffer<T>& x_buffer, const size_t x_offset,
              const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
              Buffer<T>& y_buffer, const size_t y_offset,
              const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
              const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

}} // namespace gpgpu::dnn

#endif //GPGPU_DNN_H_
