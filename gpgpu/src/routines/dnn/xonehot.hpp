#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xonehot : public blas::Routine {
public:
    Xonehot(const Queue& queue, Event* event, const std::string& name = "ONEHOT");

    void DoOneHot(const size_t n, const size_t d, const size_t k,
                  const Buffer<T>& indices, const Buffer<T>& values,
                  Buffer<T>& output);
};

}} // namespace gpgpu::dnn
