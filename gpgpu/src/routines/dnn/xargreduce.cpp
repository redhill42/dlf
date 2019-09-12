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
    const std::vector<size_t>& dims, const std::vector<size_t>& strides,
    const Buffer<T>& x_buffer, const size_t x_offset,
    Buffer<int>& y_buffer)
{
    auto shape_buffer = PackShape(dims, strides, context_, queue_);
    auto kernel = program_.getKernel("X" + name);
    kernel.setArguments(
        static_cast<int>(n), static_cast<int>(k),
        static_cast<int>(dims.size()), shape_buffer,
        x_buffer, static_cast<int>(x_offset),
        y_buffer);

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
