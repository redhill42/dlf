#include "xreduce.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T, typename R>
Xreduce<T,R>::Xreduce(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xreduce.cl"
    }) {
}

template <typename T, typename R>
void Xreduce<T,R>::DoReduce(
    const size_t m, const size_t n, const T value,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
    Buffer<R>& y_buffer, const size_t y_offset)
{
    if (n <= 2*db_["WGS1"]) {
        DoReduceDirect(m, n, value,
            x_dims, x_strides, x_buffer, x_offset,
            y_dims, y_strides, y_buffer, y_offset);
    } else {
        DoReduceIndirect(m, n, value,
            x_dims, x_strides, x_buffer, x_offset,
            y_dims, y_strides, y_buffer, y_offset);
    }
}

template <typename T, typename R>
void Xreduce<T,R>::DoReduceDirect(
    const size_t m, const size_t n, const T value,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
    Buffer<R>& y_buffer, const size_t y_offset)
{
    Kernel kernel;

    if (IsContiguous(x_dims, x_strides) && IsContiguous(y_dims, y_strides)) {
        kernel = program_.getKernel("XreduceDirect");
        kernel.setArguments(
            static_cast<int>(n), GetRealArg(value),
            x_buffer, static_cast<int>(x_offset),
            y_buffer, static_cast<int>(y_offset));
    } else {
        auto x_shape = PackShape(x_dims, x_strides, context_, queue_);
        auto y_shape = PackShape(y_dims, y_strides, context_, queue_);
        kernel = program_.getKernel("XreduceDirectStrided");
        kernel.setArguments(
            static_cast<int>(n), GetRealArg(value),
            static_cast<int>(x_dims.size()), x_shape,
            x_buffer, static_cast<int>(x_offset),
            static_cast<int>(y_dims.size()), y_shape,
            y_buffer, static_cast<int>(y_offset));
    }

    auto local_size = NextPowerOfTwo(n) / 2;
    kernel.setLocalMemorySize(local_size * sizeof(R));
    auto global = std::vector<size_t>{m * local_size};
    auto local = std::vector<size_t>{local_size};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T, typename R>
void Xreduce<T,R>::DoReduceIndirect(
    const size_t m, const size_t n, const T value,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
    Buffer<R>& y_buffer, const size_t y_offset)
{
    const size_t WGS1 = (m == 1) ? db_["WGS1"] : 32;
    const size_t WGS2 = db_["WGS2"];

    auto temp_size = 2*WGS2;
    auto temp_buffer = context_.getTemporaryBuffer<R>(temp_size * m);

    if (IsContiguous(x_dims, x_strides)) {
        auto kernel1 = program_.getKernel(m == 1 ? "Xreduce" : "XreduceBatched");
        kernel1.setArguments(
            static_cast<int>(n), GetRealArg(value),
            x_buffer, static_cast<int>(x_offset),
            temp_buffer, static_cast<int>(temp_buffer.offset()));

        auto global1 = std::vector<size_t>{WGS1*temp_size, m};
        auto local1 = std::vector<size_t>{WGS1, 1};
        RunKernel(kernel1, queue_, device_, global1, local1, nullptr);
    } else {
        auto shape_buffer = PackShape(x_dims, x_strides, context_, queue_);
        auto kernel1 = program_.getKernel(m == 1 ? "XreduceStrided" : "XreduceStridedBatched");
        kernel1.setArguments(
            static_cast<int>(n), GetRealArg(value),
            static_cast<int>(x_dims.size()), shape_buffer,
            x_buffer, static_cast<int>(x_offset),
            temp_buffer, static_cast<int>(temp_buffer.offset()));

        auto global1 = std::vector<size_t>{WGS1*temp_size, m};
        auto local1 = std::vector<size_t>{WGS1, 1};
        RunKernel(kernel1, queue_, device_, global1, local1, nullptr);
    }

    if (IsContiguous(y_dims, y_strides)) {
        auto kernel2 = program_.getKernel("XreduceEpilogue");
        kernel2.setArguments(
            static_cast<int>(n),
            temp_buffer, static_cast<int>(temp_buffer.offset()),
            y_buffer, static_cast<int>(y_offset));

        auto global2 = std::vector<size_t>{WGS2, m};
        auto local2 = std::vector<size_t>{WGS2, 1};
        RunKernel(kernel2, queue_, device_, global2, local2, event_);
    } else {
        auto shape_buffer = PackShape(y_dims, y_strides, context_, queue_);
        auto kernel2 = program_.getKernel("XreduceEpilogueStrided");
        kernel2.setArguments(
            static_cast<int>(n),
            static_cast<int>(y_dims.size()), shape_buffer,
            temp_buffer, static_cast<int>(temp_buffer.offset()),
            y_buffer, static_cast<int>(y_offset));

        auto global2 = std::vector<size_t>{WGS2, m};
        auto local2 = std::vector<size_t>{WGS2, 1};
        RunKernel(kernel2, queue_, device_, global2, local2, event_);
    }
}

template class Xreduce<half, half>;
template class Xreduce<float, float>;
template class Xreduce<double, double>;
template class Xreduce<float2, float2>;
template class Xreduce<double2, double2>;
template class Xreduce<int32_t, int32_t>;
template class Xreduce<int64_t, int64_t>;

template class Xreduce<half, int>;
template class Xreduce<float, int>;
template class Xreduce<double, int>;
template class Xreduce<float2, int>;
template class Xreduce<double2, int>;
template class Xreduce<int64_t, int>;

}} // namespace gpgpu::dnn
