#include "routines/dnn/xpow.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xpow<T>::Xpow(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/xpow.cl"
    }) {
}

template <typename T>
void Xpow<T>::DoPow(const size_t n,
                    const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
                    const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
                    Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc)
{
    // Make sure all dimensions are larger than zero
    if (n == 0) throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vectors for validity
    if (x_inc != 0)
        TestVectorX(n, x_buffer, x_offset, x_inc);
    if (y_inc != 0)
        TestVectorY(n, y_buffer, y_offset, y_inc);
    TestVectorY(n, z_buffer, z_offset, z_inc); // TODO: Make a TestVectorZ function with error codes

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("Xpow");

    // Sets the kernel arguments
    kernel.setArgument(0, static_cast<int>(n));
    kernel.setArgument(1, x_buffer);
    kernel.setArgument(2, static_cast<int>(x_offset));
    kernel.setArgument(3, static_cast<int>(x_inc));
    kernel.setArgument(4, y_buffer);
    kernel.setArgument(5, static_cast<int>(y_offset));
    kernel.setArgument(6, static_cast<int>(y_inc));
    kernel.setArgument(7, z_buffer);
    kernel.setArgument(8, static_cast<int>(z_offset));
    kernel.setArgument(9, static_cast<int>(z_inc));

    // Launches the kernel
    const auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xpow<half>;
template class Xpow<float>;
template class Xpow<double>;

}} // namespace gpgpu::dnn
