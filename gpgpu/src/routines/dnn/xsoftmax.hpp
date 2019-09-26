#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xsoftmax : public blas::Routine {
public:
    Xsoftmax(const Queue& queue, Event* event, const std::string& name);

    void DoSoftmax(const size_t m, const size_t n, Buffer<T>& x_buffer, const size_t x_offset);
};

}}
