#ifndef GPGPU_BLAS_ROUTINES_XCLAMP_HPP
#define GPGPU_BLAS_ROUTINES_XCLAMP_HPP

#include "routine.hpp"

namespace gpgpu::blas {

template <typename T>
class Xclamp : public Routine {
public:
    // Constructor
    Xclamp(const Queue& queue, Event* event, const std::string& name = "CLAMP");

    // Templated-precision implementation of the routine
    void DoClamp(const size_t n, const T min, const T max,
                 Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc);
};

} // namespace gpgpu::blas

#endif //GPGPU_BLAS_ROUTINES_XCLAMP_HPP
