#include "xrange.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xrange<T>::Xrange(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xrange.cl"
}) {}

template <typename T>
void Xrange<T>::DoRange(const size_t n, const T start, const T delta,
                        Buffer<T>& x_buffer, const size_t x_offset)
{
    auto kernel = program_.getKernel("Xrange");
    kernel.setArguments(
        static_cast<int>(n), GetRealArg(start), GetRealArg(delta),
        x_buffer, static_cast<int>(x_offset));

    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xrange<T>::DoRangeStrided(
    const size_t n, const T start, const T delta,
    const std::vector<size_t>& dims, const std::vector<size_t>& strides,
    gpgpu::Buffer<T>& x_buffer, const size_t x_offset)
{
    auto shape_buffer = PackShape(dims, strides, context_, queue_);
    auto kernel = program_.getKernel("XrangeStrided");
    kernel.setArguments(
        static_cast<int>(n), GetRealArg(start), GetRealArg(delta),
        static_cast<int>(dims.size()), shape_buffer,
        x_buffer, static_cast<int>(x_offset));

    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xrange<int16_t>;
template class Xrange<int32_t>;
template class Xrange<int64_t>;
template class Xrange<half>;
template class Xrange<float>;
template class Xrange<double>;

}} // namespace gpgpu::dnn
