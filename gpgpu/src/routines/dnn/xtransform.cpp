#include "xtransform.hpp"
#include <cassert>

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
    const Buffer<T>& x_buffer, const size_t x_offset,
    Buffer<T>& y_buffer, const size_t y_offset)
{
    // Make sure all dimensions are larger than zero
    if (n == 0) throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vector for validity
    TestVectorX(n, x_buffer, x_offset, 1);
    TestVectorY(n, y_buffer, y_offset, 1);

    // Retrieves the transform kernel from the compiled binary
    auto kernel = program_.getKernel("X" + name);

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
    const std::string& name, const size_t n, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
    Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride)
{
    // Create compact buffer to hold strides and dims
    auto rank = dims.size();
    assert(x_stride.size() == rank && y_stride.size() == rank);
    std::vector<int> shape_data(rank * 3);
    std::copy(dims.begin(), dims.end(), shape_data.begin());
    std::copy(x_stride.begin(), x_stride.end(), shape_data.begin() + rank);
    std::copy(y_stride.begin(), y_stride.end(), shape_data.begin() + rank*2);
    auto shape_buffer = context_.getSharedBuffer<int>(shape_data.data(), shape_data.size(), queue_);

    // Retrieve the transform kernel from the compiled binary
    auto kernel = program_.getKernel("X" + name + "Strided");

    // Sets the kernel arguments
    kernel.setArguments(
        static_cast<int>(n), static_cast<int>(rank),
        shape_buffer,
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

}} // namespace gpgpu::dnn
