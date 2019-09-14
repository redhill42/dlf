#include "xreduce.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xreduce<T>::Xreduce(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xreduce.cl"
    }) {
}

template <typename T>
void Xreduce<T>::DoReduce(
    const std::string& name, const size_t m, const size_t n,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    const gpgpu::Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
    gpgpu::Buffer<T>& y_buffer, const size_t y_offset)
{
    auto temp_size = 2*db_["WGS2"];
    auto temp_buffer = context_.getTemporaryBuffer<T>(temp_size * m);

    if (IsContiguous(x_dims, x_strides)) {
        auto kernel1 = program_.getKernel("X" + name);
        kernel1.setArguments(
            static_cast<int>(n),
            x_buffer,
            static_cast<int>(x_offset),
            temp_buffer,
            static_cast<int>(temp_buffer.offset()));

        auto global1 = std::vector<size_t>{db_["WGS1"]*temp_size, m};
        auto local1 = std::vector<size_t>{db_["WGS1"], 1};
        RunKernel(kernel1, queue_, device_, global1, local1, nullptr);
    } else {
        auto shape_buffer = PackShape(x_dims, x_strides, context_, queue_);
        auto kernel1 = program_.getKernel("X" + name + "Strided");
        kernel1.setArguments(
            static_cast<int>(n),
            static_cast<int>(x_dims.size()),
            shape_buffer,
            x_buffer,
            static_cast<int>(x_offset),
            temp_buffer,
            static_cast<int>(temp_buffer.offset()));

        auto global1 = std::vector<size_t>{db_["WGS1"]*temp_size, m};
        auto local1 = std::vector<size_t>{db_["WGS1"], 1};
        RunKernel(kernel1, queue_, device_, global1, local1, nullptr);
    }

    if (IsContiguous(y_dims, y_strides)) {
        auto kernel2 = program_.getKernel("X" + name + "Epilogue");
        kernel2.setArguments(
            static_cast<int>(n),
            temp_buffer,
            static_cast<int>(temp_buffer.offset()),
            y_buffer,
            static_cast<int>(y_offset));

        auto global2 = std::vector<size_t>{db_["WGS2"], m};
        auto local2 = std::vector<size_t>{db_["WGS2"], 1};
        RunKernel(kernel2, queue_, device_, global2, local2, event_);
    } else {
        auto shape_buffer = PackShape(y_dims, y_strides, context_, queue_);
        auto kernel2 = program_.getKernel("X" + name + "StridedEpilogue");
        kernel2.setArguments(
            static_cast<int>(n),
            static_cast<int>(y_dims.size()),
            shape_buffer,
            temp_buffer,
            static_cast<int>(temp_buffer.offset()),
            y_buffer,
            static_cast<int>(y_offset));

        auto global2 = std::vector<size_t>{db_["WGS2"], m};
        auto local2 = std::vector<size_t>{db_["WGS2"], 1};
        RunKernel(kernel2, queue_, device_, global2, local2, event_);
    }
}

template class Xreduce<half>;
template class Xreduce<float>;
template class Xreduce<double>;
template class Xreduce<float2>;
template class Xreduce<double2>;
template class Xreduce<int32_t>;
template class Xreduce<int64_t>;

}} // namespace gpgpu::dnn
