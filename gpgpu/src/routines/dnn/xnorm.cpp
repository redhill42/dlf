#include "xnorm.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xnormalization<T>::Xnormalization(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/xnorm.cl"
}) {}

template <typename T>
void Xnormalization<T>::DoBatchNorm(const size_t batches,
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

template <typename T>
void Xnormalization<T>::DoLRN(const size_t batches, const size_t channels, const size_t spatial,
                              const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                              const int nsize, const T alpha, const T beta, const T bias)
{
    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("Xlrn");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(batches),
                        static_cast<int>(channels),
                        static_cast<int>(spatial),
                        x_buffer, y_buffer,
                        nsize,
                        GetRealArg(alpha),
                        GetRealArg(beta),
                        GetRealArg(bias));

    // Launches the kernel
    auto n_ceiled = Ceil(spatial, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xnormalization<half>;
template class Xnormalization<float>;
template class Xnormalization<double>;

}} // namespace gpgpu::dnn
