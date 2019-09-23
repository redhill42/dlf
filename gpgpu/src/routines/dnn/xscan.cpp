#include "xscan.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xscan<T>::Xscan(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xdot"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xscan.cl"
    }) {}

template <typename T>
void Xscan<T>::DoScan(
    const size_t m, const size_t n, const bool exclusive, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
    Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides)
{
    auto shape_buffer = PackShape(dims, x_strides, y_strides, context_, queue_);
    auto x_inc = x_strides.back();
    auto y_inc = y_strides.back();

    auto block_size = db_["WGS1"];
    auto block_counts = std::vector<size_t>();
    auto block_offsets = std::vector<size_t>();
    auto temp_size = size_t(0);

    // Compute block counts and offsets
    for (size_t remainder = n;;) {
        auto count = CeilDiv(remainder, block_size*2);
        block_counts.push_back(count);
        block_offsets.push_back(temp_size);
        remainder = count;
        temp_size += count * m;
        if (count == 1)
            break;
    }

    // Allocate temporary buffer
    auto temp_buffer = context_.getTemporaryBuffer<T>(temp_size);

    // Prescan the input elements
    auto kernel1 = program_.getKernel("PreScan");
    kernel1.setArguments(
        static_cast<int>(n), static_cast<int>(!exclusive),
        static_cast<int>(dims.size()), shape_buffer,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_inc),
        y_buffer, static_cast<int>(y_offset), static_cast<int>(y_inc),
        temp_buffer, static_cast<int>(temp_buffer.offset()));

    auto global1 = std::vector<size_t>{block_counts[0] * block_size, m};
    auto local1  = std::vector<size_t>{block_size, 1};
    RunKernel(kernel1, queue_, device_, global1, local1,
              block_counts[0] == 1 ? event_ : nullptr);

    // Scan partial sums
    for (size_t i = 0; i < block_counts.size()-1; i++) {
        auto kernel2 = program_.getKernel("ScanPartialSums");
        kernel2.setArguments(
            static_cast<int>(block_counts[i]),
            temp_buffer, static_cast<int>(temp_buffer.offset() + block_offsets[i]),
            temp_buffer, static_cast<int>(temp_buffer.offset() + block_offsets[i+1]));

        auto global2 = std::vector<size_t>{block_counts[i+1] * block_size, m};
        auto local2  = std::vector<size_t>{block_size, 1};
        RunKernel(kernel2, queue_, device_, global2, local2, nullptr);
    }

    // Add back to individual blocks (ignoring the last sum)
    for (int i = static_cast<int>(block_counts.size())-2; i > 0; i--) {
        auto kernel3 = program_.getKernel("AddPartialSums");
        kernel3.setArguments(
            static_cast<int>(block_counts[i-1]),
            temp_buffer, static_cast<int>(temp_buffer.offset() + block_offsets[i-1]),
            temp_buffer, static_cast<int>(temp_buffer.offset() + block_offsets[i]));

        auto global3 = std::vector<size_t>{block_counts[i] * block_size, m};
        auto local3  = std::vector<size_t>{block_size, 1};
        RunKernel(kernel3, queue_, device_, global3, local3, nullptr);
    }

    // Add final partial sums to output elements
    if (block_counts[0] > 1) {
        auto kernel4 = program_.getKernel("FinalScan");
        kernel4.setArguments(
            static_cast<int>(n), static_cast<int>(dims.size()), shape_buffer,
            y_buffer, static_cast<int>(y_offset), static_cast<int>(y_inc),
            temp_buffer, static_cast<int>(temp_buffer.offset()));

        auto global4 = std::vector<size_t>{block_counts[0] * block_size, m};
        auto local4  = std::vector<size_t>{block_size, 1};
        RunKernel(kernel4, queue_, device_, global4, local4, event_);
    }
}

template class Xscan<half>;
template class Xscan<float>;
template class Xscan<double>;
template class Xscan<float2>;
template class Xscan<double2>;
template class Xscan<int32_t>;
template class Xscan<int64_t>;

}} // namespace gpgpu::dnn
