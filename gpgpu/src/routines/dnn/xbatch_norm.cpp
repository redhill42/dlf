#include "xbatch_norm.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xbatch_norm<T>::Xbatch_norm(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/xbatch_norm.cl"
    }) {
}

template <typename T>
void Xbatch_norm<T>::DoBatchNorm(const size_t batches,
                                 const size_t channels,
                                 const size_t spatial,
                                 const Buffer<T>& x_buffer,
                                       Buffer<T>& y_buffer,
                                 const Buffer<T>& scale_buffer,
                                 const Buffer<T>& bias_buffer,
                                 const Buffer<T>& mean_buffer,
                                 const Buffer<T>& var_buffer,
                                 const T epsilon)
{
    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("Xbatch_norm");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(batches),
                        static_cast<int>(channels),
                        static_cast<int>(spatial),
                        x_buffer, y_buffer, scale_buffer, bias_buffer,
                        mean_buffer, var_buffer, GetRealArg(epsilon));

    // Launches the kernel
    auto n_ceiled = Ceil(spatial, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xbatch_norm<half>;
template class Xbatch_norm<float>;
template class Xbatch_norm<double>;

}} // namespace gpgpu::dnn
