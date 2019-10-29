#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xhardmax : public blas::Routine {
public:
    Xhardmax(const Queue& queue, Event* event, const std::string& name = "hardmax");

    void DoHardmax(const size_t m, const size_t n, Buffer<T>& x_buffer, const size_t x_offset);
};

}} // namespace gpgpu::dnn
