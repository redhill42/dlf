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
        auto rank = x_dims.size();
        assert(x_strides.size() == rank);
        std::vector<int> shape_data(rank * 2);
        std::copy(x_dims.begin(), x_dims.end(), shape_data.begin());
        std::copy(x_strides.begin(), x_strides.end(), shape_data.begin() + rank);
        auto shape_buffer = context_.getSharedBuffer<int>(
            shape_data.data(), shape_data.size(), queue_);

        auto kernel1 = program_.getKernel("X" + name + "Strided");
        kernel1.setArguments(
            static_cast<int>(n),
            static_cast<int>(rank),
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
        auto rank =  y_dims.size();
        assert(y_strides.size() == rank);
        std::vector<int> shape_data(rank * 2);
        std::copy(y_dims.begin(), y_dims.end(), shape_data.begin());
        std::copy(y_strides.begin(), y_strides.end(), shape_data.begin() + rank);
        auto shape_buffer = context_.getSharedBuffer<int>(
            shape_data.data(), shape_data.size(), queue_);

        auto kernel2 = program_.getKernel("X" + name + "StridedEpilogue");
        kernel2.setArguments(
            static_cast<int>(n),
            static_cast<int>(rank),
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
template class Xreduce<int32_t>;
template class Xreduce<int64_t>;

}} // namespace gpgpu::dnn
