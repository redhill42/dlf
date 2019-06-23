#ifndef GPGPU_DNN_H_
#define GPGPU_DNN_H_

#include "gpgpu.h"

namespace gpgpu { namespace dnn {

template <typename T>
void transform(const std::string& name, const size_t n,
               const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
               Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
               const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void transform(const std::string& name, const size_t n,
                      const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                      const Queue& queue = gpgpu::current::queue())
{
    transform(name, n, x_buffer, 0, 1, y_buffer, 0, 1, queue);
}

template <typename T>
void abs(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void abs(const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                const Queue& queue = gpgpu::current::queue())
{
    abs(n, x_buffer, 0, 1, y_buffer, 0, 1, queue);
}

template <typename T>
void neg(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void neg(const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                const Queue& queue = gpgpu::current::queue())
{
    neg(n, x_buffer, 0, 1, y_buffer, 0, 1, queue);
}

template <typename T>
void sign(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void sign(const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                 const Queue& queue = gpgpu::current::queue())
{
    sign(n, x_buffer, 0, 1, y_buffer, 0, 1, queue);
}

}} // namespace gpgpu::dnn

#endif //GPGPU_DNN_H_
