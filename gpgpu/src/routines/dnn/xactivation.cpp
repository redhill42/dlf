#include "xactivation.hpp"

namespace gpgpu { namespace dnn {

using namespace gpgpu::blas;

template <typename T>
Xactivation<T>::Xactivation(const gpgpu::Queue& queue, gpgpu::Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/xactivation.cl"
    }) {
}

template <typename T>
void Xactivation<T>::DoActivation(
    const std::string& name, const size_t n, const T alpha, const T beta,
    const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
    Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc)
{
    // Make sure all dimensions are larger than zero
    if (n == 0) throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vector for validity
    TestVectorX(n, x_buffer, x_offset, x_inc);
    TestVectorY(n, y_buffer, y_offset, y_inc);

    // Retrieves the activation kernel from the compiled binary
    auto kernel = program_.getKernel("X" + name);

    // Sets the kernel arguments
    kernel.setArgument(0, static_cast<int>(n));
    kernel.setArgument(1, GetRealArg(alpha));
    kernel.setArgument(2, GetRealArg(beta));
    kernel.setArgument(3, x_buffer);
    kernel.setArgument(4, static_cast<int>(x_offset));
    kernel.setArgument(5, static_cast<int>(x_inc));
    kernel.setArgument(6, y_buffer);
    kernel.setArgument(7, static_cast<int>(y_offset));
    kernel.setArgument(8, static_cast<int>(y_inc));

    // Launches the kernel
    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xactivation<T>::DoActivation(
    const std::string& name,
    const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
    const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
    Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc)
{
    // Make sure all dimensions are larger than zero
    if (x_size == 0 || y_size == 0)
        throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vector for validity
    TestVectorX(x_size, x_buffer, x_offset, x_inc);
    TestVectorY(y_size, y_buffer, y_offset, y_inc);
    TestVectorY(x_size, z_buffer, z_offset, z_inc);

    // Retrieves the activation kernel from the compiled binary
    auto kernel = this->program_.getKernel("X" + name);

    // Sets the kernel arguments
    kernel.setArgument(0, static_cast<int>(x_size));
    kernel.setArgument(1, x_buffer);
    kernel.setArgument(2, static_cast<int>(x_offset));
    kernel.setArgument(3, static_cast<int>(x_inc));
    kernel.setArgument(4, static_cast<int>(y_size));
    kernel.setArgument(5, y_buffer);
    kernel.setArgument(6, static_cast<int>(y_offset));
    kernel.setArgument(7, static_cast<int>(y_inc));
    kernel.setArgument(8, z_buffer);
    kernel.setArgument(9, static_cast<int>(z_offset));
    kernel.setArgument(10, static_cast<int>(z_inc));

    // Launches the kernel
    auto n_ceiled = Ceil(x_size, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xactivation<half>;
template class Xactivation<float>;
template class Xactivation<double>;

}} // namespace gpgpu::dnn
