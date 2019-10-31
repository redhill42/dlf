#include "xsort.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xsort<T>::Xsort(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xsort.cl"
}) {}

template <typename T>
void Xsort<T>::DoSort(
    const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides)
{
    auto n = dims.back();
    assert(n <= 2*db_["WGS"]);
    auto shape = PackShape(dims, x_strides, y_strides, context_, queue_);

    auto kernel = program_.getKernel("DirectSort");
    kernel.setArguments(
        static_cast<int>(n), dir, static_cast<int>(dims.size()), shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        y_buffer, static_cast<int>(y_offset), static_cast<int>(y_strides.back()));

    size_t batch_size = std::accumulate(dims.begin(), dims.end()-1, 1, std::multiplies<>());
    size_t local_size = NextPowerOfTwo(n) / 2;
    kernel.setLocalMemorySize(local_size*2 * sizeof(T));

    auto global = std::vector<size_t>{batch_size * local_size};
    auto local  = std::vector<size_t>{local_size};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xsort<T>::DoArgSort(
    const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
    Buffer<int32_t>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides)
{
    auto n = dims.back();
    assert(n <= 2*db_["WGS"]);
    auto shape = PackShape(dims, x_strides, y_strides, context_, queue_);

    auto kernel = program_.getKernel("DirectArgSort");
    kernel.setArguments(
        static_cast<int>(n), dir, static_cast<int>(dims.size()), shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        y_buffer, static_cast<int>(y_offset), static_cast<int>(y_strides.back()));

    size_t batch_size = std::accumulate(dims.begin(), dims.end()-1, 1, std::multiplies<>());
    size_t local_size = NextPowerOfTwo(n) / 2;
    kernel.setLocalMemorySize(local_size*2 * (sizeof(T) + sizeof(int32_t)));

    auto global = std::vector<size_t>{batch_size * local_size};
    auto local  = std::vector<size_t>{local_size};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xsort<int16_t>;
template class Xsort<int32_t>;
template class Xsort<int64_t>;
template class Xsort<half>;
template class Xsort<float>;
template class Xsort<double>;

}} // namespace gpgpu::dnn
