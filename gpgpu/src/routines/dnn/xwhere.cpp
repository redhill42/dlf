#include "xwhere.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xwhere<T>::Xwhere(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xwhere.cl"
    }) {
}

static Buffer<int> pack_shape(
    const Context& context, const Queue& queue,
    const std::vector<size_t>& dim, const std::vector<size_t>& stride)
{
    auto rank = dim.size();
    assert(stride.size() == rank);

    std::vector<int> shape(rank * 2);
    std::copy(dim.begin(), dim.end(), shape.begin());
    std::copy(stride.begin(), stride.end(), shape.begin() + rank);

    return context.getSharedBuffer<int>(shape.data(), shape.size(), queue);
}

template <typename T>
void Xwhere<T>::DoWhere(
    const size_t n, const size_t rank,
    const Buffer<bool>& c_buffer, const size_t c_offset,
    const std::vector<size_t>& c_dim, const std::vector<size_t>& c_stride,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
    const Buffer<T>& y_buffer, const size_t y_offset,
    const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride,
    Buffer<T>& z_buffer, const size_t z_offset)
{
    auto c_shape = pack_shape(context_, queue_, c_dim, c_stride);
    auto x_shape = pack_shape(context_, queue_, x_dim, x_stride);
    auto y_shape = pack_shape(context_, queue_, y_dim, y_stride);

    auto kernel = program_.getKernel("Xwhere");
    kernel.setArguments(
        static_cast<int>(n),
        static_cast<int>(rank),
        c_buffer, c_shape, static_cast<int>(c_offset),
        x_buffer, x_shape, static_cast<int>(x_offset),
        y_buffer, y_shape, static_cast<int>(y_offset),
        z_buffer, static_cast<int>(z_offset));

    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xwhere<int16_t>;
template class Xwhere<int32_t>;
template class Xwhere<int64_t>;
template class Xwhere<half>;
template class Xwhere<float>;
template class Xwhere<double>;

}} // namespace gpgpu::dnn
