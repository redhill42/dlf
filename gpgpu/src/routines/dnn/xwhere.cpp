#include "xwhere.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xwhere<T>::Xwhere(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xwhere.cl"
}) {}

template <typename T>
void Xwhere<T>::DoWhere(
    const size_t n, const std::vector<size_t>& dim,
    const Buffer<bool>& c_buffer, const size_t c_offset, const std::vector<size_t>& c_stride,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
    const Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride,
    Buffer<T>& z_buffer, const size_t z_offset)
{
    auto shape_buffer = PackShape(dim, c_stride, x_stride, y_stride, context_, queue_);
    auto kernel = program_.getKernel("Xwhere");
    kernel.setArguments(
        static_cast<int>(n),
        static_cast<int>(dim.size()), shape_buffer,
        c_buffer, static_cast<int>(c_offset),
        x_buffer, static_cast<int>(x_offset),
        y_buffer, static_cast<int>(y_offset),
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
