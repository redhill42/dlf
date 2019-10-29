#include "xrandom.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xrandom<T>::Xrandom(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/mwc64x.cl"
    #include "../../kernels/dnn/xrandom.cl"
}) {}

template <typename T>
void Xrandom<T>::DoRandom(
    const size_t n, const std::vector<size_t>& dims, const std::vector<size_t>& strides,
    Buffer<T>& x_buffer, const size_t x_offset, const uint64_t seed, const T low, const T high)
{
    auto wgs = db_["WGS"], wpt = size_t(4096);
    auto n_ceiled = Ceil(n, wgs * wpt);
    auto global = std::vector<size_t>{n_ceiled/wpt};
    auto local = std::vector<size_t>{wgs};

    if (IsContiguous(dims, strides)) {
        auto kernel = program_.getKernel("Xrandom");
        kernel.setArguments(
            static_cast<int>(n),
            x_buffer, static_cast<int>(x_offset),
            seed, low, high);
        RunKernel(kernel, queue_, device_, global, local, event_);
    } else {
        auto shape_buffer = PackShape(dims, strides, context_, queue_);
        auto kernel = program_.getKernel("XrandomStrided");
        kernel.setArguments(static_cast<int>(n),
            static_cast<int>(dims.size()), shape_buffer,
            x_buffer, static_cast<int>(x_offset),
            seed, low, high);
        RunKernel(kernel, queue_, device_, global, local, event_);
    }
}

template <typename T>
void Xrandom<T>::DoRandomNormal(
    const size_t n, const std::vector<size_t>& dims, const std::vector<size_t>& strides,
    Buffer<T>& x_buffer, const size_t x_offset, const uint64_t seed, const T mean, const T stdev)
{
    auto wgs = db_["WGS"], wpt = size_t(4096);
    auto n_ceiled = Ceil(n, wgs * wpt);
    auto global = std::vector<size_t>{n_ceiled/wpt};
    auto local = std::vector<size_t>{wgs};

    if (IsContiguous(dims, strides)) {
        auto kernel = program_.getKernel("XrandomNormal");
        kernel.setArguments(
            static_cast<int>(n),
            x_buffer, static_cast<int>(x_offset),
            seed, mean, stdev);
        RunKernel(kernel, queue_, device_, global, local, event_);
    } else {
        auto shape_buffer = PackShape(dims, strides, context_, queue_);
        auto kernel = program_.getKernel("XrandomNormalStrided");
        kernel.setArguments(static_cast<int>(n),
            static_cast<int>(dims.size()), shape_buffer,
            x_buffer, static_cast<int>(x_offset),
            seed, mean, stdev);
        RunKernel(kernel, queue_, device_, global, local, event_);
    }
}

template class Xrandom<int16_t>;
template class Xrandom<int32_t>;
template class Xrandom<int64_t>;
template class Xrandom<float>;
template class Xrandom<double>;
template class Xrandom<float2>;
template class Xrandom<double2>;

}} // namespace gpgpu::dnn
