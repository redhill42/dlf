#include "xhardmax.hpp"
#include "xargreduce.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xhardmax<T>::Xhardmax(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xhardmax.cl"
}) {}

template <typename T>
void Xhardmax<T>::DoHardmax(
    const size_t m, const size_t n,
    Buffer<T>& x_buffer, const size_t x_offset)
{
    auto arg_buffer = context_.getTemporaryBuffer<int>(m);

    auto argmax = Xargreduce<T>(queue_, nullptr, "argmax");
    argmax.DoArgReduce(m, n,
        {m, n}, {n, 1}, x_buffer, x_offset,
        {m, 1}, {1, 1}, arg_buffer, arg_buffer.offset());

    auto kernel = program_.getKernel("Xhardmax");
    kernel.setArguments(
        static_cast<int>(n),
        x_buffer, static_cast<int>(x_offset),
        arg_buffer, static_cast<int>(arg_buffer.offset()));

    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"], m};
    auto local = std::vector<size_t>{db_["WGS"], 1};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xhardmax<half>;
template class Xhardmax<float>;
template class Xhardmax<double>;
template class Xhardmax<int32_t>;
template class Xhardmax<int64_t>;

}} // namespace gpgpu::dnn
