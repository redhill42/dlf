#include "xsoftmax.hpp"
#include "xreduce.hpp"
#include "xtransform_b.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xsoftmax<T>::Xsoftmax(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xsoftmax.cl"
}) {}

template <typename T>
void Xsoftmax<T>::DoSoftmax(
    const size_t m, const size_t n,
    Buffer<T>& x_buffer, const size_t x_offset)
{
    auto bias_buffer = context_.getTemporaryBuffer<T>(m);

    auto WGS1 = db_["WGS1"], WGS2 = db_["WGS2"];
    auto temp_size = 2*WGS2;
    auto temp_buffer = context_.getTemporaryBuffer<T>(temp_size * m);

    auto reduce_max = Xreduce<T>(queue_, nullptr, "reduce_max");
    reduce_max.DoReduce(m, n,
        {m, n}, {n, 1}, x_buffer, x_offset,
        {m, 1}, {1, 1}, bias_buffer, bias_buffer.offset());

    auto kernel1 = program_.getKernel("Xsoftmax");
    kernel1.setArguments(
        static_cast<int>(n),
        x_buffer, static_cast<int>(x_offset),
        bias_buffer, static_cast<int>(bias_buffer.offset()),
        temp_buffer, static_cast<int>(temp_buffer.offset()));

    auto global1 = std::vector<size_t>{WGS1*temp_size, m};
    auto local1 = std::vector<size_t>{WGS1, 1};
    RunKernel(kernel1, queue_, device_, global1, local1, nullptr);

    auto kernel2 = program_.getKernel("XsoftmaxEpilogue");
    kernel2.setArguments(
        static_cast<int>(n),
        temp_buffer, static_cast<int>(temp_buffer.offset()),
        bias_buffer, static_cast<int>(bias_buffer.offset()));

    auto global2 = std::vector<size_t>{WGS2, m};
    auto local2 = std::vector<size_t>{WGS2, 1};
    RunKernel(kernel2, queue_, device_, global2, local2, event_);

    auto xform_name = routine_name_ == "softmax" ? "div" : "sub";
    auto xform = Xtransform_b<T,T>(queue_, event_, xform_name);
    if (m == 1) {
        xform.DoTransform(
            m * n, x_buffer, x_offset,
            m, bias_buffer, bias_buffer.offset(),
            x_buffer, x_offset);
    } else {
        xform.DoTransformChannel(
            m, n, m,
            x_buffer, x_offset,
            bias_buffer, bias_buffer.offset(),
            x_buffer, x_offset);
    }
}

template class Xsoftmax<half>;
template class Xsoftmax<float>;
template class Xsoftmax<double>;

}} // namespace gpgpu::dnn
