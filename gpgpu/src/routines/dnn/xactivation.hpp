#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xactivation : public blas::Routine {
public:
    Xactivation(const Queue& queue, Event* event, const std::string& name = "ACTIVATION");

    void DoActivation(const std::string& name, const size_t n, const T alpha, const T beta,
                      const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
                      Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc);

    void DoActivation(const std::string& name,
                      const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
                      const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
                      Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc);
};

}} // namespace gpgpu::dnn
