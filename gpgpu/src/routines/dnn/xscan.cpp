#include "xscan.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xscan<T>::Xscan(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xscan.cl"
    }) {}

template <typename T>
void Xscan<T>::DoScan(
    const std::string& name, const size_t m, const size_t n,
    const bool exclusive, const std::vector<size_t>& dims,
    const gpgpu::Buffer<T>& x_buffer, const size_t x_offset,  const std::vector<size_t>& x_strides,
    gpgpu::Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides)
{
    auto shape_buffer = PackShape(dims, x_strides, y_strides, context_, queue_);
    auto incX = x_strides.back();
    auto incY = y_strides.back();

    auto kernel = program_.getKernel("X" + name);
    kernel.setArguments(
        static_cast<int>(n), static_cast<int>(exclusive),
        static_cast<int>(dims.size()), shape_buffer,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(incX),
        y_buffer, static_cast<int>(y_offset), static_cast<int>(incY));

    auto global = std::vector<size_t>{m};
    auto local = std::vector<size_t>{1};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xscan<half>;
template class Xscan<float>;
template class Xscan<double>;
template class Xscan<float2>;
template class Xscan<double2>;
template class Xscan<int32_t>;
template class Xscan<int64_t>;

}} // namespace gpgpu::dnn
