#include "xargreduce.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xargreduce<T>::Xargreduce(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xargreduce.cl"
}) {}

template <typename T>
void Xargreduce<T>::DoArgReduce(
    const size_t m, const size_t n,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
    Buffer<int>& y_buffer, const size_t y_offset)
{
    const size_t WGS1 = (m == 1) ? db_["WGS1"] : 32;
    const size_t WGS2 = db_["WGS2"];

    auto temp_size = 2*WGS2;
    auto max_buffer = context_.getTemporaryBuffer<T>(temp_size * m);
    auto imax_buffer = context_.getTemporaryBuffer<int>(temp_size * m);

    if (IsContiguous(x_dims, x_strides)) {
        auto kernel1 = program_.getKernel(m == 1 ? "Xargreduce" : "XargreduceBatched");

        kernel1.setArguments(
            static_cast<int>(n),
            x_buffer, static_cast<int>(x_offset),
            max_buffer, static_cast<int>(max_buffer.offset()),
            imax_buffer, static_cast<int>(imax_buffer.offset()));

        auto global1 = std::vector<size_t>{WGS1*temp_size, m};
        auto local1 = std::vector<size_t>{WGS1, 1};
        RunKernel(kernel1, queue_, device_, global1, local1, nullptr);
    } else {
        auto shape_buffer = PackShape(x_dims, x_strides, context_, queue_);
        auto kernel1 = program_.getKernel(m == 1 ? "XargreduceStrided" : "XargreduceStridedBatched");

        kernel1.setArguments(
            static_cast<int>(n), static_cast<int>(x_dims.size()), shape_buffer,
            x_buffer, static_cast<int>(x_offset),
            max_buffer, static_cast<int>(max_buffer.offset()),
            imax_buffer, static_cast<int>(imax_buffer.offset()));

        auto global1 = std::vector<size_t>{WGS1*temp_size, m};
        auto local1 = std::vector<size_t>{WGS1, 1};
        RunKernel(kernel1, queue_, device_, global1, local1, nullptr);
    }

    if (IsContiguous(y_dims, y_strides)) {
        auto kernel2 = program_.getKernel("XargreduceEpilogue");

        kernel2.setArguments(
            max_buffer, static_cast<int>(max_buffer.offset()),
            imax_buffer, static_cast<int>(imax_buffer.offset()),
            y_buffer, static_cast<int>(y_offset));

        auto global2 = std::vector<size_t>{WGS2, m};
        auto local2 = std::vector<size_t>{WGS2, 1};
        RunKernel(kernel2, queue_, device_, global2, local2, event_);
    } else {
        auto shape_buffer = PackShape(y_dims, y_strides, context_, queue_);
        auto kernel2 = program_.getKernel("XargreduceEpilogueStrided");

        kernel2.setArguments(
            static_cast<int>(y_dims.size()), shape_buffer,
            max_buffer, static_cast<int>(max_buffer.offset()),
            imax_buffer, static_cast<int>(imax_buffer.offset()),
            y_buffer, static_cast<int>(y_offset));

        auto global2 = std::vector<size_t>{WGS2, m};
        auto local2 = std::vector<size_t>{WGS2, 1};
        RunKernel(kernel2, queue_, device_, global2, local2, event_);
    }
}

template class Xargreduce<int16_t>;
template class Xargreduce<int32_t>;
template class Xargreduce<int64_t>;
template class Xargreduce<half>;
template class Xargreduce<float>;
template class Xargreduce<double>;

}} // namespace gpgpu::dnn
