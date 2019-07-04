#include "xtransform.hpp"

namespace gpgpu { namespace dnn {

using namespace gpgpu::blas;

template <typename T>
Xtransform<T>::Xtransform(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/xtransform.cl"
    }) {
}

template <typename T>
void Xtransform<T>::DoTransform(
    const std::string& name, const size_t n,
    const Buffer<T>& x_buffer,Buffer<T>& y_buffer)
{
    // Make sure all dimensions are larger than zero
    if (n == 0) throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vector for validity
    TestVectorX(n, x_buffer, 0, 1);
    TestVectorY(n, y_buffer, 0, 1);

    // Retrieves the transform kernel from the compiled binary
    auto kernel = program_.getKernel("X" + name);

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(n), x_buffer, y_buffer);

    // Launches the kernel
    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xtransform<int16_t>;
template class Xtransform<int32_t>;
template class Xtransform<int64_t>;
template class Xtransform<half>;
template class Xtransform<float>;
template class Xtransform<double>;

}} // namespace gpgpu::dnn
