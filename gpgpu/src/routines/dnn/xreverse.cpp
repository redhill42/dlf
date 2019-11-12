#include "xreverse.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xreverse<T>::Xreverse(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xreverse.cl"
}){}

template <typename T>
void Xreverse<T>::DoReverse(
    const size_t m, const size_t n,
    const std::vector<size_t>& dims, const std::vector<size_t>& strides,
    gpgpu::Buffer<T>& x_buffer, const size_t x_offset)
{
    auto shape = PackShape(dims, strides, context_, queue_);

    if (n < db_["WGS"]) {
        auto kernel = program_.getKernel("DirectReverse");
        kernel.setArguments(
            static_cast<int>(n), static_cast<int>(dims.size()), shape,
            x_buffer, static_cast<int>(x_offset), static_cast<int>(strides.back()));

        auto global = std::vector<size_t>{m * db_["WGS"]};
        auto local  = std::vector<size_t>{db_["WGS"]};
        RunKernel(kernel, queue_, device_, global, local, event_);
    } else {
        auto block_size = CeilDiv(n/2, db_["WGS"]);
        auto kernel = program_.getKernel("BatchedReverse");
        kernel.setArguments(
            static_cast<int>(n), static_cast<int>(block_size),
            static_cast<int>(dims.size()), shape,
            x_buffer, static_cast<int>(x_offset), static_cast<int>(strides.back()));

        auto global = std::vector<size_t>{m * block_size * db_["WGS"]};
        auto local  = std::vector<size_t>{db_["WGS"]};
        RunKernel(kernel, queue_, device_, global, local, event_);
    }
}

template class Xreverse<int16_t>;
template class Xreverse<int32_t>;
template class Xreverse<int64_t>;
template class Xreverse<half>;
template class Xreverse<float>;
template class Xreverse<double>;
template class Xreverse<float2>;
template class Xreverse<double2>;

}} // namespace gpgpu::dnn
