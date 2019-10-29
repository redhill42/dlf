#include "xresize.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xresize<T>::Xresize(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xresize.cl"
}) {}

template <typename T>
void Xresize<T>::DoResize1D(
    const size_t batch_count,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    Buffer<T>& y_buffer, const size_t y_offset,
    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides)
{
    auto x_len = x_dims[x_dims.size()-1];
    auto y_len = y_dims[y_dims.size()-1];

    if (IsContiguous(x_dims, x_strides) && IsContiguous(y_dims, y_strides)) {
        auto kernel = program_.getKernel("Xresize1d");
        kernel.setArguments(
            x_buffer, static_cast<int>(x_offset), static_cast<int>(x_len),
            y_buffer, static_cast<int>(y_offset), static_cast<int>(y_len));

        auto global = std::vector<size_t>{Ceil(y_len, db_["WGS"]), batch_count};
        auto local = std::vector<size_t>{db_["WGS"], 1};
        RunKernel(kernel, queue_, device_, global, local, event_);
    } else {
        auto x_shape = PackShape(x_dims, x_strides, context_, queue_);
        auto y_shape = PackShape(y_dims, y_strides, context_, queue_);

        auto kernel = program_.getKernel("Xresize1dStrided");
        kernel.setArguments(
            static_cast<int>(x_dims.size()),
            x_buffer, static_cast<int>(x_offset), x_shape,
            y_buffer, static_cast<int>(y_offset), y_shape);

        auto global = std::vector<size_t>{Ceil(y_len, db_["WGS"]), batch_count};
        auto local = std::vector<size_t>{db_["WGS"], 1};
        RunKernel(kernel, queue_, device_, global, local, event_);
    }
}

template <typename T>
void Xresize<T>::DoResize2D(
    const size_t batch_count,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    Buffer<T>& y_buffer, const size_t y_offset,
    const std::vector<size_t>& y_dims, const std::vector<size_t>& y_strides)
{
    auto x_width  = x_dims[x_dims.size()-1];
    auto x_height = x_dims[x_dims.size()-2];
    auto y_width  = y_dims[y_dims.size()-1];
    auto y_height = y_dims[y_dims.size()-2];

    if (IsContiguous(x_dims, x_strides) && IsContiguous(y_dims, y_strides)) {
        auto kernel = program_.getKernel("Xresize2d");
        kernel.setArguments(
            x_buffer, static_cast<int>(x_offset),
            static_cast<int>(x_width), static_cast<int>(x_height),
            y_buffer, static_cast<int>(y_offset),
            static_cast<int>(y_width), static_cast<int>(y_height));

        auto global = std::vector<size_t>{Ceil(y_width*y_height, db_["WGS"]), batch_count};
        auto local = std::vector<size_t>{db_["WGS"], 1};
        RunKernel(kernel, queue_, device_, global, local, event_);
    } else {
        auto x_shape = PackShape(x_dims, x_strides, context_, queue_);
        auto y_shape = PackShape(y_dims, y_strides, context_, queue_);

        auto kernel = program_.getKernel("Xresize2dStrided");
        kernel.setArguments(
            static_cast<int>(x_dims.size()),
            x_buffer, static_cast<int>(x_offset), x_shape,
            y_buffer, static_cast<int>(y_offset), y_shape);

        auto global = std::vector<size_t>{Ceil(y_width*y_height, db_["WGS"]), batch_count};
        auto local = std::vector<size_t>{db_["WGS"], 1};
        RunKernel(kernel, queue_, device_, global, local, event_);
    }
}

template class Xresize<half>;
template class Xresize<float>;
template class Xresize<double>;

}} // namespace gpgpu::dnn
