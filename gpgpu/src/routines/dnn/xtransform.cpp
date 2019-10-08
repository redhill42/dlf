#include "xtransform.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {

using namespace gpgpu::blas;

template <typename T>
Xtransform<T>::Xtransform(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/complex.cl"
    #include "../../kernels/dnn/xtransform.cl"
    }) {
}

template <typename T>
void Xtransform<T>::DoTransform(const size_t n,
    const Buffer<T>& x_buffer, const size_t x_offset,
    Buffer<T>& y_buffer, const size_t y_offset)
{
    // Make sure all dimensions are larger than zero
    if (n == 0) throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vector for validity
    TestVectorX(n, x_buffer, x_offset, 1);
    TestVectorY(n, y_buffer, y_offset, 1);

    // Retrieves the transform kernel from the compiled binary
    auto kernel = program_.getKernel("Xtransform");

    // Sets the kernel arguments
    kernel.setArguments(
        static_cast<int>(n),
        x_buffer, static_cast<int>(x_offset),
        y_buffer, static_cast<int>(y_offset));

    // Launches the kernel
    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xtransform<T>::DoTransform(
    const size_t n, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
    Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride)
{
    // Create compact buffer to hold strides and dims
    auto shape_buffer = PackShape(dims, x_stride, y_stride, context_, queue_);

    // Retrieve the transform kernel from the compiled binary
    auto kernel = program_.getKernel("XtransformStrided");

    // Sets the kernel arguments
    kernel.setArguments(
        static_cast<int>(n),
        static_cast<int>(dims.size()), shape_buffer,
        x_buffer, static_cast<int>(x_offset),
        y_buffer, static_cast<int>(y_offset));

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
template class Xtransform<float2>;
template class Xtransform<double2>;

}} // namespace gpgpu::dnn
