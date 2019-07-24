#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xsoftmax : public blas::Routine {
public:
    Xsoftmax(const Queue& queue, Event* event, const std::string& name = "SOFTMAX");

    void DoSoftmax(const size_t m, const size_t n,
                   const Buffer<T>& x_buffer,
                   Buffer<T>& y_buffer);

    void DoLogSoftmax(const size_t m, const size_t n,
                      const Buffer<T>& x_buffer,
                      Buffer<T>& y_buffer);

    void DoHardmax(const size_t m, const size_t n,
                   const Buffer<T>& x_buffer,
                   Buffer<T>& y_buffer);
};

}}