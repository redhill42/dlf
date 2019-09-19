#include "xmerge.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xmerge<T>::Xmerge(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xmerge.cl"
    }) {}

template <typename T>
void Xmerge<T>::DoMerge(
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

    const size_t blocks = db_["WGS"];
    size_t batch_count = std::accumulate(z_dims.begin(), z_dims.end()-1, 1, std::multiplies<>());
    auto diagonal = context_.getTemporaryBuffer<int>(2 * (blocks + 1) * batch_count);

    auto kernel1 = program_.getKernel("IntersectDiagonals");
    kernel1.setArguments(
        static_cast<int>(x_dims.size()), x_shape,
        x_buffer, static_cast<int>(x_len), static_cast<int>(x_offset), static_cast<int>(x_inc),
        static_cast<int>(y_dims.size()), y_shape,
        y_buffer, static_cast<int>(y_len), static_cast<int>(y_offset), static_cast<int>(y_inc),
        diagonal, static_cast<int>(diagonal.offset()));

    auto global1 = std::vector<size_t>{32 * blocks, batch_count};
    auto local1 = std::vector<size_t>{32, 1};
    RunKernel(kernel1, queue_, device_, global1, local1, nullptr);

    auto kernel2 = program_.getKernel("Xmerge");
    kernel2.setArguments(
        static_cast<int>(x_dims.size()), x_shape,
        x_buffer, static_cast<int>(x_len), static_cast<int>(x_offset), static_cast<int>(x_inc),
        static_cast<int>(y_dims.size()), y_shape,
        y_buffer, static_cast<int>(y_len), static_cast<int>(y_offset), static_cast<int>(y_inc),
        static_cast<int>(z_dims.size()), z_shape,
        z_buffer, static_cast<int>(z_offset), static_cast<int>(z_inc),
        diagonal, static_cast<int>(diagonal.offset()));

    auto global2 = std::vector<size_t>{blocks * blocks, batch_count};
    auto local2 = std::vector<size_t>{blocks, 1};
    RunKernel(kernel2, queue_, device_, global2, local2, event_);
}

template class Xmerge<int32_t>;
template class Xmerge<int64_t>;
template class Xmerge<half>;
template class Xmerge<float>;
template class Xmerge<double>;

}} // namespace gpgpu::dnn
