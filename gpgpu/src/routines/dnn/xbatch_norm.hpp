#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xbatch_norm : public blas::Routine {
public:
    Xbatch_norm(const Queue& queue, Event* event, const std::string& name = "BATCH_NORM");

    void DoBatchNorm(const size_t batches,
                     const size_t channels,
                     const size_t spatial,
                     const Buffer<T>& x_buffer,
                           Buffer<T>& y_buffer,
                     const Buffer<T>& scale_buffer,
                     const Buffer<T>& bias_buffer,
                     const Buffer<T>& mean_buffer,
                     const Buffer<T>& var_buffer,
                     const T epsilon);
};

}} // namespace gpgpu::dnn
