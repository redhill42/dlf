#include "xsoftmax.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xsoftmax<T>::Xsoftmax(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xsoftmax.cl"
    }) {
}

template <typename T>
void Xsoftmax<T>::DoSoftmax(const size_t m, const size_t n,
                            const Buffer<T>& x_buffer,
                            Buffer<T>& y_buffer)
{
    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("Xsoftmax");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(n), x_buffer, y_buffer);

    // Launches the kernel
    auto global = std::vector<size_t>{m};
    auto local = std::vector<size_t>{1};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xsoftmax<T>::DoLogSoftmax(const size_t m, const size_t n,
                               const Buffer<T>& x_buffer,
                               Buffer<T>& y_buffer)
{
    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("Xlogsoftmax");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(n), x_buffer, y_buffer);

    // Launches the kernel
    auto global = std::vector<size_t>{m};
    auto local = std::vector<size_t>{1};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xsoftmax<T>::DoHardmax(const size_t m, const size_t n,
                            const Buffer<T>& x_buffer,
                            Buffer<T>& y_buffer)
{
    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("Xhardmax");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(n), x_buffer, y_buffer);

    // Launches the kernel
    auto global = std::vector<size_t>{m};
    auto local = std::vector<size_t>{1};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xsoftmax<half>;
template class Xsoftmax<float>;
template class Xsoftmax<double>;

}}
