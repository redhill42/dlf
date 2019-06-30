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
                      const Queue& queue = gpgpu::current::queue(), Event* event = nullptr)
{
    transform(name, n, x_buffer, 0, 1, y_buffer, 0, 1, queue, event);
}

template <typename T>
void activation(const std::string& name, const size_t n, const T alpha, const T beta,
                const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
                Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void activation(const std::string& name, const size_t n, const T alpha, const T beta,
                       const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                       const Queue& queue = gpgpu::current::queue(), Event* event = nullptr)
{
    activation(name, n, alpha, beta, x_buffer, 0, 1, y_buffer, 0, 1, queue, event);
}

template <typename T>
void activation(const std::string& name,
                const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
                const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
                Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void activation(const std::string& name,
                       const size_t x_size, const Buffer<T>& x_buffer,
                       const size_t y_size, const Buffer<T>& y_buffer,
                       Buffer<T>& z_buffer,
                       const Queue& queue = gpgpu::current::queue(), Event* event = nullptr)
{
    activation(name, x_size, x_buffer, 0, 1, y_size, y_buffer, 0, 1, z_buffer, 0, 1, queue, event);
}

template <typename T>
void abs(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void abs(const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr)
{
    abs(n, x_buffer, 0, 1, y_buffer, 0, 1, queue, event);
}

template <typename T>
void neg(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void neg(const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr)
{
    neg(n, x_buffer, 0, 1, y_buffer, 0, 1, queue, event);
}

template <typename T>
void sign(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void sign(const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                 const Queue& queue = gpgpu::current::queue(), Event* event = nullptr)
{
    sign(n, x_buffer, 0, 1, y_buffer, 0, 1, queue, event);
}

template <typename T>
void add(const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc,
         const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void add(const size_t x_size, const Buffer<T>& x_buffer,
                const size_t y_size, const Buffer<T>& y_buffer,
                Buffer<T>& z_buffer,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr)
{
    add(x_size, x_buffer, 0, 1, y_size, y_buffer, 0, 1, z_buffer, 0, 1, queue, event);
}

template <typename T>
void sub(const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc,
         const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void sub(const size_t x_size, const Buffer<T>& x_buffer,
                const size_t y_size, const Buffer<T>& y_buffer,
                Buffer<T>& z_buffer,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr)
{
    sub(x_size, x_buffer, 0, 1, y_size, y_buffer, 0, 1, z_buffer, 0, 1, queue, event);
}

template <typename T>
void mul(const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc,
         const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void mul(const size_t x_size, const Buffer<T>& x_buffer,
                const size_t y_size, const Buffer<T>& y_buffer,
                Buffer<T>& z_buffer,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr)
{
    mul(x_size, x_buffer, 0, 1, y_size, y_buffer, 0, 1, z_buffer, 0, 1, queue, event);
}

template <typename T>
void div(const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc,
         const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void div(const size_t x_size, const Buffer<T>& x_buffer,
                const size_t y_size, const Buffer<T>& y_buffer,
                Buffer<T>& z_buffer,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr)
{
    div(x_size, x_buffer, 0, 1, y_size, y_buffer, 0, 1, z_buffer, 0, 1, queue, event);
}

template <typename T>
void pow(const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc,
         const Queue& queue = gpgpu::current::queue(), Event* event = nullptr);

template <typename T>
inline void pow(const size_t x_size, const Buffer<T>& x_buffer,
                const size_t y_size, const Buffer<T>& y_buffer,
                Buffer<T>& z_buffer,
                const Queue& queue = gpgpu::current::queue(), Event* event = nullptr)
{
    pow(x_size, x_buffer, 0, 1, y_size, y_buffer, 0, 1, z_buffer, 0, 1, queue, event);
}

}} // namespace gpgpu::dnn

#endif //GPGPU_DNN_H_
