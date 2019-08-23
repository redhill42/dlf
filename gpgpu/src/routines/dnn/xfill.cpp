#include "xfill.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xfill<T>::Xfill(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xfill.cl"
    }) {
}

template <typename T>
void Xfill<T>::DoFill(const size_t n, Buffer<T>& x_buffer, const size_t x_offset,
                      const T value)
{
    auto kernel = program_.getKernel("Xfill");
    kernel.setArguments(
        static_cast<int>(n),
        x_buffer, static_cast<int>(x_offset),
        GetRealArg(value));

    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xfill<T>::DoFillStrided(const size_t n,
    const std::vector<size_t>& dims, const std::vector<size_t>& strides,
    Buffer<T>& x_buffer, const size_t x_offset, const T value)
{
    auto rank = dims.size();
    assert(strides.size() == dims.size());
    std::vector<int> shape_data(rank * 2);
    std::copy(dims.begin(), dims.end(), shape_data.begin());
    std::copy(strides.begin(), strides.end(), shape_data.begin() + rank);
    auto shape_buffer = context_.getSharedBuffer<int>(shape_data.data(), shape_data.size(), queue_);

    auto kernel = program_.getKernel("XfillStrided");
    kernel.setArguments(
        static_cast<int>(n), static_cast<int>(rank),
        shape_buffer, x_buffer, static_cast<int>(x_offset),
        GetRealArg(value));

    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xfill<int16_t>;
template class Xfill<int32_t>;
template class Xfill<int64_t>;
template class Xfill<half>;
template class Xfill<float>;
template class Xfill<double>;
template class Xfill<float2>;
template class Xfill<double2>;

}} // namespace gpgpu::dnn
