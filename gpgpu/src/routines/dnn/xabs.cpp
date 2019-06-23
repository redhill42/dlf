#include "xabs.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xabs<T>::Xabs(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/xabs.cl"
    }) {
}

template <typename T>
void Xabs<T>::DoAbs(
    const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
    Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc)
{
    // Make sure all dimensions are larger than zero
    if (n == 0) throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vector for validity
    TestVectorX(n, x_buffer, x_offset, x_inc);
    TestVectorY(n, y_buffer, y_offset, y_inc);

    // Retrieves the transform kernel from the compiled binary
    auto kernel = program_.getKernel("Xabs");

    // Sets the kernel arguments
    kernel.setArgument(0, static_cast<int>(n));
    kernel.setArgument(1, x_buffer);
    kernel.setArgument(2, static_cast<int>(x_offset));
    kernel.setArgument(3, static_cast<int>(x_inc));
    kernel.setArgument(4, y_buffer);
    kernel.setArgument(5, static_cast<int>(y_offset));
    kernel.setArgument(6, static_cast<int>(y_inc));

    // Launches the kernel
    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xabs<int16_t>;
template class Xabs<int32_t>;
template class Xabs<int64_t>;
template class Xabs<half>;
template class Xabs<float>;
template class Xabs<double>;

}} // namespace gpgpu::dnn
