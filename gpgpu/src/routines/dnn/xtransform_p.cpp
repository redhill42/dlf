#include "xtransform_p.hpp"

namespace gpgpu { namespace dnn {

using namespace gpgpu::blas;

template <typename T>
Xtransform_p<T>::Xtransform_p(const gpgpu::Queue& queue, gpgpu::Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "kernels/dnn/xtransform_p.cl"
}) {
}

template <typename T>
void Xtransform_p<T>::DoTransform(
    const std::string& name, const size_t n, const T alpha, const T beta,
    const Buffer<T>& x_buffer, Buffer<T>& y_buffer)
{
    // Make sure all dimensions are larger than zero
    if (n == 0) throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vector for validity
    TestVectorX(n, x_buffer, 0, 1);
    TestVectorY(n, y_buffer, 0, 1);

    // Retrieves the activation kernel from the compiled binary
    auto kernel = program_.getKernel("X" + name);

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(n),
                        GetRealArg(alpha), GetRealArg(beta),
                        x_buffer, y_buffer);

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
