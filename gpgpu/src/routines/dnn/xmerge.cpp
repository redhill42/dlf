#include "xmerge.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xmerge<T>::Xmerge(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/local_merge.cl"
    #include "../../kernels/dnn/xmerge.cl"
}) {}

#define WPT 4

template <typename T>
void Xmerge<T>::DoMerge(
    const int dir,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
    const Buffer<T>& y_buffer, const size_t y_offset,
    const std::vector<size_t>& z_dims, const std::vector<size_t>& z_strides,
    Buffer<T>& z_buffer, const size_t z_offset)
{
    auto x_len = x_dims.back();
    auto y_len = y_dims.back();

    if (x_len <= WPT*db_["WGS"] && y_len <= WPT*db_["WGS"]) {
        DoDirectMerge(dir,
                      x_dims, x_strides, x_buffer, x_offset,
                      y_dims, y_strides, y_buffer, y_offset,
                      z_dims, z_strides, z_buffer, z_offset);
    } else {
        DoIndirectMerge(dir,
                        x_dims, x_strides, x_buffer, x_offset,
                        y_dims, y_strides, y_buffer, y_offset,
                        z_dims, z_strides, z_buffer, z_offset);
    }
}

template <typename T>
void Xmerge<T>::DoDirectMerge(
    const int dir,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
    const Buffer<T>& y_buffer, const size_t y_offset,
    const std::vector<size_t>& z_dims, const std::vector<size_t>& z_strides,
    Buffer<T>& z_buffer, const size_t z_offset)
{
    auto x_shape = PackShape(x_dims, x_strides, context_, queue_);
    auto y_shape = PackShape(y_dims, y_strides, context_, queue_);
    auto z_shape = PackShape(z_dims, z_strides, context_, queue_);
    auto x_len = x_dims.back(), x_inc = x_strides.back();
    auto y_len = y_dims.back(), y_inc = y_strides.back();
    auto z_inc = z_strides.back();

    size_t batch_size = std::accumulate(z_dims.begin(), z_dims.end()-1, 1, std::multiplies<>());
    size_t local_size = CeilDiv(std::max(x_len, y_len), WPT);

    auto kernel = program_.getKernel("DirectMerge");
    kernel.setArguments(
        dir,
        static_cast<int>(x_dims.size()), x_shape,
        x_buffer, static_cast<int>(x_len), static_cast<int>(x_offset), static_cast<int>(x_inc),
        static_cast<int>(y_dims.size()), y_shape,
        y_buffer, static_cast<int>(y_len), static_cast<int>(y_offset), static_cast<int>(y_inc),
        static_cast<int>(z_dims.size()), z_shape,
        z_buffer, static_cast<int>(z_offset), static_cast<int>(z_inc));

    // Allocate local memory that holds two local arrays
    kernel.setLocalMemorySize((local_size*WPT+1) * 2*sizeof(T));

    auto global = std::vector<size_t>{batch_size * local_size};
    auto local  = std::vector<size_t>{local_size};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xmerge<T>::DoIndirectMerge(
    const int dir,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides,
    const Buffer<T>& y_buffer, const size_t y_offset,
    const std::vector<size_t>& z_dims, const std::vector<size_t>& z_strides,
    Buffer<T>& z_buffer, const size_t z_offset)
{
    auto x_shape = PackShape(x_dims, x_strides, context_, queue_);
    auto y_shape = PackShape(y_dims, y_strides, context_, queue_);
    auto z_shape = PackShape(z_dims, z_strides, context_, queue_);
    auto x_len = x_dims.back(), x_inc = x_strides.back();
    auto y_len = y_dims.back(), y_inc = y_strides.back();
    auto z_inc = z_strides.back();

    const size_t blocks = CeilDiv(x_len + y_len, db_["WGS"] * WPT);
    size_t batch_size = std::accumulate(z_dims.begin(), z_dims.end()-1, 1, std::multiplies<>());
    auto diagonal = context_.getTemporaryBuffer<int>(2 * (blocks + 1) * batch_size);

    auto kernel1 = program_.getKernel("MergePath");
    kernel1.setArguments(
        dir,
        static_cast<int>(x_dims.size()), x_shape,
        x_buffer, static_cast<int>(x_len), static_cast<int>(x_offset), static_cast<int>(x_inc),
        static_cast<int>(y_dims.size()), y_shape,
        y_buffer, static_cast<int>(y_len), static_cast<int>(y_offset), static_cast<int>(y_inc),
        diagonal, static_cast<int>(diagonal.offset()));

    auto global1 = std::vector<size_t>{32 * blocks, batch_size};
    auto local1 = std::vector<size_t>{32, 1};
    RunKernel(kernel1, queue_, device_, global1, local1, nullptr);

    auto kernel2 = program_.getKernel("Merge");
    kernel2.setArguments(
        dir,
        static_cast<int>(x_dims.size()), x_shape,
        x_buffer, static_cast<int>(x_len), static_cast<int>(x_offset), static_cast<int>(x_inc),
        static_cast<int>(y_dims.size()), y_shape,
        y_buffer, static_cast<int>(y_len), static_cast<int>(y_offset), static_cast<int>(y_inc),
        static_cast<int>(z_dims.size()), z_shape,
        z_buffer, static_cast<int>(z_offset), static_cast<int>(z_inc),
        diagonal, static_cast<int>(diagonal.offset()));

    auto global2 = std::vector<size_t>{blocks * db_["WGS"], batch_size};
    auto local2 = std::vector<size_t>{db_["WGS"], 1};
    RunKernel(kernel2, queue_, device_, global2, local2, event_);
}

template class Xmerge<int32_t>;
template class Xmerge<int64_t>;
template class Xmerge<half>;
template class Xmerge<float>;
template class Xmerge<double>;

}} // namespace gpgpu::dnn
