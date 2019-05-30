#include "routines/levelx/xclamp.hpp"

#include "string"
#include "vector"

namespace gpgpu::blas {

// Constructor: forwards to base class constructor
template <typename T>
Xclamp<T>::Xclamp(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
      #include "../../kernels/level1/level1.cl"
      #include "../../kernels/levelx/xclamp.cl"
    }) {
}

// The main routine
template <typename T>
void Xclamp<T>::DoClamp(const size_t n, const T minval, const T maxval,
                        Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc) {
    // Make sure all dimensions are larger than zero
    if (n == 0) {
        throw BLASError(StatusCode::kInvalidDimension);
    }

    // Tests the vector for validity
    TestVectorX(n, x_buffer, x_offset, x_inc);

    // Determines whether or not the fast-version can be used
    bool use_fast_kernel = (x_offset == 0) && (x_inc == 1) &&
                           IsMultiple(n, db_["WGS"]*db_["WPT"]*db_["VW"]);

    // Sets the kernel arguments and launch the kernel
    if (use_fast_kernel) {
        auto kernel = program_.getKernel("XclampFast");
        kernel.setArguments(static_cast<int>(n),
                            GetRealArg(minval),
                            GetRealArg(maxval),
                            x_buffer);

        auto global = std::vector<size_t>{CeilDiv(n, db_["WPT"]*db_["VW"])};
        auto local = std::vector<size_t>{db_["WGS"]};
        RunKernel(kernel, queue_, device_, global, local, event_);
    } else {
        auto kernel = program_.getKernel("Xclamp");
        kernel.setArguments(static_cast<int>(n),
                            GetRealArg(minval),
                            GetRealArg(maxval),
                            x_buffer,
                            static_cast<int>(x_offset),
                            static_cast<int>(x_inc));

        auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
        auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
        auto local = std::vector<size_t>{db_["WGS"]};
        RunKernel(kernel, queue_, device_, global, local, event_);
    }
}

// Compiles the templated class
template class Xclamp<half>;
template class Xclamp<float>;
template class Xclamp<double>;

} // namespace gpgpu::blas
