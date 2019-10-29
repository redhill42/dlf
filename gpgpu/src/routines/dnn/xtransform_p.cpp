#include "xtransform_p.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xtransform_p<T>::Xtransform_p(const gpgpu::Queue& queue, gpgpu::Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/xtransform_p.cl"
}) {}

template <typename T>
void Xtransform_p<T>::DoTransform(
    const std::string& name, const T alpha, const T beta, const size_t n,
    const Buffer<T>& x_buffer, const size_t x_offset,
    Buffer<T>& y_buffer, const size_t y_offset)
{
    // Make sure all dimensions are larger than zero
    if (n == 0) throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vector for validity
    TestVectorX(n, x_buffer, x_offset, 1);
    TestVectorY(n, y_buffer, y_offset, 1);

    // Retrieves the activation kernel from the compiled binary
    auto kernel = program_.getKernel("X" + name);

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(n),
                        GetRealArg(alpha), GetRealArg(beta),
                        x_buffer, static_cast<int>(x_offset),
                        y_buffer, static_cast<int>(y_offset));

    // Launches the kernel
    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xtransform_p<T>::DoTransform(
    const std::string& name, const T alpha, const T beta,
    const size_t n, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
    Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride)
{
    // Create compact buffer to hold strides and dims
    auto shape_buffer = PackShape(dims, x_stride, y_stride, context_, queue_);

    // Retrieve the transform kernel from the compiled binary
    auto kernel = program_.getKernel("X" + name + "Strided");

    // Sets the kernel arguments
    kernel.setArguments(
        GetRealArg(alpha), GetRealArg(beta),
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

template class Xtransform_p<half>;
template class Xtransform_p<float>;
template class Xtransform_p<double>;

}} // namespace gpgpu::dnn
