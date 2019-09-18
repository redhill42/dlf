#include "xargreduce.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xargreduce<T>::Xargreduce(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xargreduce.cl"
    }) {
}

template <typename T>
void Xargreduce<T>::DoArgReduce(
    const std::string& name, const size_t n, const size_t k,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
    Buffer<int>& y_buffer, const size_t y_offset)
{
    auto x_shape_buffer = PackShape(x_dims, x_strides, context_, queue_);
    auto y_shape_buffer = PackShape(y_dims, y_strides, context_, queue_);
    auto kernel = program_.getKernel("X" + name);
    kernel.setArguments(
        static_cast<int>(n), static_cast<int>(k),
        static_cast<int>(x_dims.size()), x_shape_buffer,
        x_buffer, static_cast<int>(x_offset),
        static_cast<int>(y_dims.size()), y_shape_buffer,
        y_buffer, y_offset);

    auto global = std::vector<size_t>{Ceil(n, db_["WGS"])};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xargreduce<int16_t>;
template class Xargreduce<int32_t>;
template class Xargreduce<int64_t>;
template class Xargreduce<half>;
template class Xargreduce<float>;
template class Xargreduce<double>;

}} // namespace gpgpu::dnn
