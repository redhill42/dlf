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

}} // namespace gpgpu::dnn

#endif //GPGPU_DNN_H_
