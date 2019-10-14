#include "xonehot.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xonehot<T>::Xonehot(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xonehot.cl"
    }) {}

template <typename T>
void Xonehot<T>::DoOneHot(const size_t n, const size_t d, const size_t k,
                          const Buffer<T>& indices, const Buffer<T>& values,
                          gpgpu::Buffer<T>& output)
{
    auto kernel = program_.getKernel("Xonehot");
    kernel.setArguments(
        static_cast<int>(n),
        static_cast<int>(d),
        static_cast<int>(k),
        indices, values, output);

    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xonehot<int16_t>;
template class Xonehot<int32_t>;
template class Xonehot<int64_t>;
template class Xonehot<half>;
template class Xonehot<float>;
template class Xonehot<double>;

}} // namespace gpgpu::dnn
